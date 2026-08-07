#!/usr/bin/env python3
"""使用静态分片收集完整轨迹与检查点特征。

本脚本有意与具体调度器解耦。可以在每台设备上用不同的 ``--shard-index``
各启动一个副本，也可以只在单张 GPU 上运行。脚本只写入不可变样本产物，
绝不创建共享任务队列。
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.final_paper_cache import (
    artifact_matches,
    cache_paths,
    protocol_fingerprint,
    schedules_for_trace,
    tail_mean,
    task_seed,
)
from src.final_paper_inference import (
    PositionHiddenCapture,
    atomic_torch_save,
    demonstrations_by_subject,
    gold_for,
    prediction_for,
    prompt_messages,
    read_jsonl,
    render_prompt,
    resolved_generation,
    success_for,
)
from src.qwen3_reasoning import generate_trace, load_qwen3
from src.utils import atomic_json, load_yaml, seed_everything


def selected_records(
    path: Path,
    ids_path: Path | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    if ids_path is not None:
        requested = [str(value) for value in json.loads(ids_path.read_text())]
        if len(requested) != len(set(requested)):
            raise ValueError(f"duplicate IDs in {ids_path}")
        by_id = {str(row["problem_id"]): row for row in records}
        missing = sorted(set(requested) - set(by_id))
        if missing:
            raise KeyError(f"IDs absent from prepared split: {missing[:10]}")
        records = [by_id[value] for value in requested]
    if limit is not None:
        records = records[:limit]
    return records


def warmup(model, tokenizer, device: torch.device, config: dict[str, Any]) -> None:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Warm-up."},
            {"role": "user", "content": "Return 1."},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        generate_trace(
            model,
            tokenizer,
            encoded.input_ids.to(device),
            encoded.attention_mask.to(device),
            resolved_generation(config, 8),
            task_seed(20260803, "warmup", "warmup", "reference", "dense"),
            measure_timing=False,
        )
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("probe_train", "calibration", "heldout"),
        required=True,
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--sample-ids", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--measure-timing",
        action="store_true",
        help="记录同步后的单请求 CUDA 与端到端计时；普通缓存收集时不要启用。",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= index < num-shards")

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    manifest_path = (
        args.split_manifest
        if args.split_manifest.is_absolute()
        else ROOT / args.split_manifest
    )
    cache_root = args.cache_root if args.cache_root.is_absolute() else ROOT / args.cache_root
    config = load_yaml(config_path)
    if int(config["seed"]) != 20260803 or config["model"]["dtype"] != "float16":
        raise ValueError("the published protocol requires seed=20260803 and float16")
    model_root = args.model_path or Path(config["model"]["local_path"])
    if not model_root.is_absolute():
        model_root = ROOT / model_root
    fingerprint = protocol_fingerprint(config_path, manifest_path, model_root)
    prepared = Path(config["dataset"]["prepared_root"])
    if not prepared.is_absolute():
        prepared = ROOT / prepared
    ids_path = args.sample_ids
    if ids_path is not None and not ids_path.is_absolute():
        ids_path = ROOT / ids_path
    records = selected_records(
        prepared / f"{args.split}.jsonl",
        ids_path,
        args.limit,
    )
    demonstrations = (
        demonstrations_by_subject(prepared / "demonstrations.jsonl")
        if args.dataset == "mmlu"
        else None
    )

    seed_everything(int(config["seed"]))
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model, tokenizer, model_audit = load_qwen3(
        model_root,
        device,
        config["model"]["dtype"],
        config["model"]["attention_backend"],
    )
    warmup(model, tokenizer, device, config)
    capture_layers = [int(value) for value in config["generation"]["capture_layers"]]
    if capture_layers != [20]:
        raise ValueError("the main protocol captures only layer 20")
    capture = PositionHiddenCapture(model, capture_layers)
    generation = resolved_generation(
        config,
        int(config["generation"]["dense_max_new_tokens"]),
    )
    completed = skipped = failures = 0
    started = time.time()

    for index, record in enumerate(records):
        if index % args.num_shards != args.shard_index:
            continue
        problem_id = str(record["problem_id"])
        destination = cache_paths(cache_root, args.split, problem_id)["dense"]
        try:
            if args.resume and artifact_matches(
                destination,
                problem_id=problem_id,
                fingerprint=fingerprint,
            ):
                skipped += 1
                continue
            if destination.exists():
                raise RuntimeError(
                    f"incompatible artifact is preserved and will not be overwritten: {destination}"
                )
            messages = prompt_messages(
                args.dataset,
                record,
                config,
                demonstrations,
            )
            prompt_text = render_prompt(
                tokenizer,
                messages,
                enable_thinking=bool(config["model"]["enable_thinking"]),
            )
            encoded = tokenizer(prompt_text, return_tensors="pt")
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            generation_seed = task_seed(
                int(config["seed"]),
                args.dataset,
                args.split,
                problem_id,
                "dense",
            )
            with torch.inference_mode():
                trace = generate_trace(
                    model,
                    tokenizer,
                    input_ids,
                    attention_mask,
                    generation,
                    generation_seed,
                    measure_timing=args.measure_timing,
                )
            eos_value = tokenizer.eos_token_id
            eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
            content_ids = list(
                trace.tokens[:-1]
                if trace.tokens and trace.tokens[-1] in eos
                else trace.tokens
            )
            protocol = config["checkpoint_protocol"]
            schedules, decoded_boundary_prefix = schedules_for_trace(
                tokenizer,
                content_ids,
                minimum=int(protocol["minimum"]),
                maximum=int(protocol["maximum"]),
                sentence_gap=int(protocol["sentence_minimum_gap"]),
                fixed=config["generation"]["fixed_budgets"],
            )
            union = sorted(set(schedules["sentence"]) | set(schedules["fixed"]))
            hidden = torch.empty(
                (0, 1, model_audit["hidden_size"]), dtype=torch.float16
            )
            if union:
                maximum = max(union)
                teacher_ids = torch.cat(
                    [
                        input_ids,
                        torch.tensor(
                            [content_ids[:maximum]],
                            dtype=torch.long,
                            device=device,
                        ),
                    ],
                    dim=1,
                )
                positions = [
                    int(input_ids.shape[1]) + checkpoint - 1
                    for checkpoint in union
                ]
                capture.begin(positions, device)
                with torch.inference_mode():
                    model.model(
                        input_ids=teacher_ids,
                        attention_mask=torch.ones_like(teacher_ids),
                        use_cache=False,
                        return_dict=True,
                    )
                hidden = capture.finish_cpu().to(torch.float16)
                del teacher_ids

            dense_text = tokenizer.decode(trace.tokens, skip_special_tokens=True)
            gold = gold_for(args.dataset, record)
            prediction = prediction_for(args.dataset, dense_text)
            success = success_for(args.dataset, gold, prediction)
            cumulative_times: list[float] | None = None
            if args.measure_timing:
                cumulative = float(trace.prefill_cuda_ms)
                cumulative_times = [cumulative]
                for value in trace.decode_cuda_ms:
                    cumulative += float(value)
                    cumulative_times.append(cumulative)
            rows = []
            for checkpoint in union:
                row = {
                    "problem_id": problem_id,
                    "dataset": args.dataset,
                    "split": args.split,
                    "seed": int(config["seed"]),
                    "subject": record.get("subject"),
                    "category": record.get("category"),
                    "checkpoint": checkpoint,
                    "checkpoint_schedules": [
                        name
                        for name, values in schedules.items()
                        if checkpoint in values
                    ],
                    "is_sentence_checkpoint": checkpoint in schedules["sentence"],
                    "is_fixed_checkpoint": checkpoint in schedules["fixed"],
                    "gold_answer": gold,
                    "dense_prediction": prediction,
                    "dense_success": bool(success),
                    "dense_tokens": len(trace.tokens),
                    "dense_content_tokens": len(content_ids),
                    "prompt_tokens": int(input_ids.shape[1]),
                    "prefix_context_tokens": int(input_ids.shape[1]) + checkpoint,
                    "prefix_mean_entropy_tail8": tail_mean(
                        trace.entropies,
                        checkpoint,
                    ),
                    "prefix_token_ids": content_ids[:checkpoint],
                }
                if cumulative_times is not None:
                    row["prefix_cumulative_cuda_ms"] = float(
                        cumulative_times[
                            min(checkpoint - 1, len(cumulative_times) - 1)
                        ]
                    )
                rows.append(row)
            artifact = {
                "schema_version": 3,
                "status": "complete",
                "protocol_id": config["protocol_id"],
                "protocol_fingerprint": fingerprint,
                "dataset": args.dataset,
                "split": args.split,
                "seed": int(config["seed"]),
                "problem_id": problem_id,
                "generation_seed": generation_seed,
                "record": record,
                "gold_answer": gold,
                "model_audit": model_audit,
                "dtype": config["model"]["dtype"],
                "attention_backend": config["model"]["attention_backend"],
                "collection_device": torch.cuda.get_device_name(args.gpu),
                "collection_device_index": args.gpu,
                "timing_valid": bool(args.measure_timing),
                "timing_mode": (
                    "single_request_cuda_and_wall"
                    if args.measure_timing
                    else "not_collected"
                ),
                "prompt_text": prompt_text,
                "prompt_tokens": int(input_ids.shape[1]),
                "dense": {
                    "tokens": trace.tokens,
                    "content_tokens": content_ids,
                    "text": dense_text,
                    "prediction": prediction,
                    "success": bool(success),
                    "reasoning_tokens": len(trace.tokens),
                    "reached_max_tokens": len(trace.tokens)
                    >= int(generation["max_new_tokens"]),
                    "prefill_cuda_ms": trace.prefill_cuda_ms,
                    "decode_cuda_ms": trace.decode_cuda_ms,
                    "wall_ms": trace.wall_ms,
                    "logps": trace.logps,
                    "margins": trace.margins,
                    "entropies_top20": trace.entropies,
                },
                "checkpoint_protocol": protocol,
                "schedules": schedules,
                "decoded_prefix_for_boundaries": decoded_boundary_prefix,
                "rows": rows,
                "capture_layers": capture_layers,
                "hidden": hidden,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            if len(rows) != int(hidden.shape[0]):
                raise RuntimeError(f"row/vector mismatch for {problem_id}")
            atomic_torch_save(artifact, destination)
            completed += 1
            elapsed = max(time.time() - started, 1e-9)
            print(
                json.dumps(
                    {
                        "problem_id": problem_id,
                        "dense_tokens": len(trace.tokens),
                        "sentence_checkpoints": len(schedules["sentence"]),
                        "fixed_checkpoints": len(schedules["fixed"]),
                        "completed": completed,
                        "samples_per_hour": 3600.0 * completed / elapsed,
                    }
                ),
                flush=True,
            )
            del input_ids, attention_mask, trace, hidden
            gc.collect()
        except Exception as error:
            failures += 1
            print(
                json.dumps(
                    {
                        "problem_id": problem_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                ),
                flush=True,
            )
    capture.close()
    summary = {
        "status": "complete" if failures == 0 else "failed",
        "dataset": args.dataset,
        "split": args.split,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "completed": completed,
        "skipped": skipped,
        "failures": failures,
        "timing_valid": bool(args.measure_timing),
        "elapsed_seconds": time.time() - started,
    }
    summary_path = (
        cache_root
        / "dense"
        / args.split
        / f"summary_shard{args.shard_index:03d}.json"
    )
    atomic_json(summary, summary_path)
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

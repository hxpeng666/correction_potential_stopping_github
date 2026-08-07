#!/usr/bin/env python3
"""使用静态任务分片收集直接作答与强制作答分支。"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.final_paper_cache import (
    BRANCH_DIRECT,
    artifact_matches,
    branch_path,
    cache_paths,
    task_seed,
)
from src.final_paper_inference import (
    atomic_torch_save,
    demonstrations_by_subject,
    prediction_for,
    prompt_messages,
    render_prompt,
    resolved_generation,
    success_for,
)
from src.qwen3_reasoning import generate_trace, load_qwen3
from src.utils import atomic_json, load_yaml, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu"), required=True)
    parser.add_argument("--config", type=Path, required=True)
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
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= index < num-shards")

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cache_root = args.cache_root if args.cache_root.is_absolute() else ROOT / args.cache_root
    config = load_yaml(config_path)
    model_root = args.model_path or Path(config["model"]["local_path"])
    if not model_root.is_absolute():
        model_root = ROOT / model_root
    prepared = Path(config["dataset"]["prepared_root"])
    if not prepared.is_absolute():
        prepared = ROOT / prepared
    demonstrations = (
        demonstrations_by_subject(prepared / "demonstrations.jsonl")
        if args.dataset == "mmlu"
        else None
    )
    dense_paths = sorted(
        (cache_root / "dense" / args.split).glob("sample_*.pt")
    )
    tasks: list[tuple[Path, int]] = []
    for dense_path in dense_paths:
        source = torch.load(dense_path, map_location="cpu", weights_only=False)
        checkpoints = [int(row["checkpoint"]) for row in source["rows"]]
        tasks.extend((dense_path, value) for value in [BRANCH_DIRECT, *checkpoints])
    if args.limit_tasks is not None:
        tasks = tasks[: args.limit_tasks]

    seed_everything(int(config["seed"]))
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model, tokenizer, model_audit = load_qwen3(
        model_root,
        device,
        config["model"]["dtype"],
        config["model"]["attention_backend"],
    )
    generation = resolved_generation(
        config,
        int(config["generation"]["force_answer_max_new_tokens"]),
    )
    completed = skipped = failures = oom = 0
    started = time.time()

    for index, (dense_path, checkpoint) in enumerate(tasks):
        if index % args.num_shards != args.shard_index:
            continue
        source = torch.load(dense_path, map_location="cpu", weights_only=False)
        problem_id = str(source["problem_id"])
        fingerprint = str(source["protocol_fingerprint"])
        destination = branch_path(cache_root, args.split, problem_id, checkpoint)
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
                    f"incompatible branch is preserved and will not be overwritten: {destination}"
                )
            if not artifact_matches(
                cache_paths(cache_root, args.split, problem_id)["dense"],
                problem_id=problem_id,
                fingerprint=fingerprint,
            ):
                raise RuntimeError(f"incompatible Dense source: {dense_path}")
            if checkpoint == BRANCH_DIRECT:
                messages = prompt_messages(
                    args.dataset,
                    source["record"],
                    config,
                    demonstrations,
                )
                instruction = (
                    " Do not show any reasoning. Output only the final result as \\boxed{number}."
                    if args.dataset == "gsm8k"
                    else (
                        " Do not show any reasoning. Output only one option letter as "
                        "\\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."
                    )
                )
                messages = [dict(message) for message in messages]
                messages[0]["content"] += instruction
                answer_prefix = str(config["generation"]["direct_answer_prefix"])
                prompt_text = (
                    render_prompt(tokenizer, messages, enable_thinking=False)
                    + answer_prefix
                )
                encoded = tokenizer(prompt_text, return_tensors="pt")
                input_ids = encoded.input_ids.to(device)
                attention_mask = encoded.attention_mask.to(device)
                kind = "direct"
                seed_checkpoint: int | str = "direct"
            else:
                row_by_checkpoint = {
                    int(row["checkpoint"]): row for row in source["rows"]
                }
                if checkpoint not in row_by_checkpoint:
                    raise KeyError(f"checkpoint {checkpoint} absent from {problem_id}")
                prompt_ids = tokenizer(
                    source["prompt_text"], return_tensors="pt"
                ).input_ids
                prefix_ids = torch.tensor(
                    [source["dense"]["content_tokens"][:checkpoint]],
                    dtype=torch.long,
                )
                answer_prefix = str(config["generation"]["force_answer_suffix"])
                suffix_ids = tokenizer(
                    answer_prefix,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids
                input_ids = torch.cat(
                    [prompt_ids, prefix_ids, suffix_ids], dim=1
                ).to(device)
                attention_mask = torch.ones_like(input_ids)
                kind = "forced"
                seed_checkpoint = checkpoint
            generation_seed = task_seed(
                int(config["seed"]),
                args.dataset,
                args.split,
                problem_id,
                seed_checkpoint,
            )
            with torch.inference_mode():
                trace = generate_trace(
                    model,
                    tokenizer,
                    input_ids,
                    attention_mask,
                    generation,
                    generation_seed,
                    measure_timing=False,
                )
            generated = tokenizer.decode(trace.tokens, skip_special_tokens=True)
            branch_text = answer_prefix + generated
            prediction = prediction_for(args.dataset, branch_text)
            success = success_for(args.dataset, source["gold_answer"], prediction)
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
                "checkpoint": checkpoint,
                "kind": kind,
                "collection_device": torch.cuda.get_device_name(args.gpu),
                "dtype": config["model"]["dtype"],
                "context_tokens": int(input_ids.shape[1]),
                "tokens": trace.tokens,
                "generated_tokens": len(trace.tokens),
                "generated_text": generated,
                "text": branch_text,
                "answer_prefix": answer_prefix,
                "prediction": prediction,
                "success": bool(success),
                "timing_excluded_from_replay": True,
                "model_audit": model_audit,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_torch_save(artifact, destination)
            completed += 1
            if completed == 1 or completed % 50 == 0:
                elapsed = max(time.time() - started, 1e-9)
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "problem_id": problem_id,
                            "checkpoint": checkpoint,
                            "branches_per_hour": 3600.0 * completed / elapsed,
                        }
                    ),
                    flush=True,
                )
        except torch.cuda.OutOfMemoryError as error:
            oom += 1
            print(
                json.dumps(
                    {
                        "problem_id": problem_id,
                        "checkpoint": checkpoint,
                        "requires_larger_gpu": True,
                        "error": str(error),
                    }
                ),
                flush=True,
            )
        except Exception as error:
            failures += 1
            print(
                json.dumps(
                    {
                        "problem_id": problem_id,
                        "checkpoint": checkpoint,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                ),
                flush=True,
            )
        finally:
            for name in ("input_ids", "attention_mask", "trace", "source"):
                if name in locals():
                    del locals()[name]
            gc.collect()
            torch.cuda.empty_cache()
    summary = {
        "status": "complete" if failures == 0 and oom == 0 else "incomplete",
        "dataset": args.dataset,
        "split": args.split,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "tasks_visible": len(tasks),
        "completed": completed,
        "skipped": skipped,
        "oom_requires_larger_gpu": oom,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(
        summary,
        cache_root
        / "branches"
        / args.split
        / f"summary_shard{args.shard_index:03d}.json",
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failures or oom:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

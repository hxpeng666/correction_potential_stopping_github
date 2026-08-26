#!/usr/bin/env python3
"""动态领取配对样本，在独立目录生成 BF16 Dense、hidden、entropy 和 forced branches。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_paper_inference import (
    PositionHiddenCapture,
    atomic_torch_save,
    prediction_for,
    resolved_generation,
    success_for,
)
from src.final_paper_protocol import canonical_fingerprint
from src.final_paper_replay_cache import (
    claim_task,
    finish_task,
    schedules_for_trace,
    tail_mean,
    task_seed,
)
from src.qwen3_reasoning import generate_trace, load_qwen3
from src.utils import atomic_json, load_yaml, seed_everything


def output_path(output_root: Path, dataset: str, split: str, problem_id: str) -> Path:
    return output_root / dataset / split / f"sample_{problem_id}.pt"


def pair_fingerprint(
    config: dict[str, Any], selection_fingerprint: str, model_audit: dict[str, Any]
) -> str:
    return canonical_fingerprint(
        {
            "protocol_id": config["protocol_id"],
            "selection_fingerprint": selection_fingerprint,
            "global_seed": config["seed"]["global"],
            "task_seed_derivation": config["seed"]["task_derivation"],
            "model": {
                **config["model"],
                "dtype": "bfloat16",
                "metadata_fingerprint": model_audit["metadata_fingerprint"],
            },
            "generation": config["generation"],
            "checkpoint": config["checkpoint"],
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "collector": "collect_dtype_pair_bf16_primary_v1.py",
            "schema_version": 1,
        }
    )


def complete_and_compatible(
    path: Path, problem_id: str, fingerprint: str, dense_seed: int
) -> bool:
    if not path.is_file():
        return False
    value = torch.load(path, map_location="cpu", weights_only=False)
    return (
        value.get("status") == "complete"
        and value.get("problem_id") == problem_id
        and value.get("pair_protocol_fingerprint") == fingerprint
        and value.get("dtype") == "bfloat16"
        and int(value.get("dense_task_seed", -1)) == dense_seed
    )


def warmup(model, tokenizer, device: torch.device, config: dict[str, Any], worker_id: str) -> None:
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": "Warm-up."}, {"role": "user", "content": "Return 1."}],
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
            task_seed(int(config["seed"]["global"]), "warmup", "warmup", worker_id, "dense"),
        )
    torch.cuda.synchronize(device)


def collect_one(
    task: dict[str, Any],
    config: dict[str, Any],
    model,
    tokenizer,
    model_audit: dict[str, Any],
    capture: PositionHiddenCapture,
    device: torch.device,
    worker_id: str,
) -> tuple[Path, int]:
    source_path = Path(task["source_fp16_artifact"])
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    dataset = str(task["dataset"])
    split = str(task["split"])
    problem_id = str(task["problem_id"])
    global_seed = int(task["global_seed"])
    if split == "heldout":
        raise ValueError("dtype 配对任务禁止使用 heldout")
    if source.get("status") != "complete" or source.get("dtype") != "float16":
        raise ValueError(f"无效 FP16 配对源：{source_path}")
    if source.get("dataset") != dataset or source.get("split") != split:
        raise ValueError(f"配对源 dataset/split 错位：{source_path}")
    if source.get("problem_id") != problem_id or int(source.get("seed", -1)) != global_seed:
        raise ValueError(f"配对源 sample ID/seed 错位：{source_path}")
    prompt_text = str(source["prompt_text"])
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    dense_seed = task_seed(global_seed, dataset, split, problem_id, "dense")
    fingerprint = pair_fingerprint(config, str(task["selection_fingerprint"]), model_audit)
    destination = output_path(Path(task["output_root"]), dataset, split, problem_id)
    if complete_and_compatible(destination, problem_id, fingerprint, dense_seed):
        return destination, 0
    if destination.exists():
        raise RuntimeError(f"拒绝覆盖不兼容的 BF16 配对结果：{destination}")

    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    generation = resolved_generation(config, int(config["generation"]["dense_max_new_tokens"]))
    with torch.inference_mode():
        trace = generate_trace(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            generation,
            dense_seed,
        )
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    content_ids = list(
        trace.tokens[:-1] if trace.tokens and trace.tokens[-1] in eos else trace.tokens
    )
    checkpoint = config["checkpoint"]
    schedules, decoded_boundary_prefix = schedules_for_trace(
        tokenizer,
        content_ids,
        minimum=int(checkpoint["minimum"]),
        maximum=int(checkpoint["maximum"]),
        sentence_gap=int(checkpoint["sentence_minimum_gap"]),
        fixed=config["generation"]["fixed_budgets"],
    )
    sentence = [int(value) for value in schedules["sentence"]]
    hidden = torch.empty((0, 1, int(model_audit["hidden_size"])), dtype=torch.bfloat16)
    if sentence:
        maximum = max(sentence)
        teacher_ids = torch.cat(
            [
                input_ids,
                torch.tensor([content_ids[:maximum]], dtype=torch.long, device=device),
            ],
            dim=1,
        )
        positions = [int(input_ids.shape[1]) + value - 1 for value in sentence]
        capture.begin(positions, device)
        with torch.inference_mode():
            model.model(
                input_ids=teacher_ids,
                attention_mask=torch.ones_like(teacher_ids),
                use_cache=False,
                return_dict=True,
            )
        hidden = capture.finish_cpu().to(torch.bfloat16)
        del teacher_ids

    dense_text = tokenizer.decode(trace.tokens, skip_special_tokens=True)
    gold = source["gold_answer"]
    dense_prediction = prediction_for(dataset, dense_text)
    dense_success = success_for(dataset, gold, dense_prediction)
    suffix_text = str(config["generation"]["force_answer_suffix"])
    suffix_ids_cpu = tokenizer(
        suffix_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    branch_generation = resolved_generation(
        config, int(config["generation"]["force_answer_max_new_tokens"])
    )
    rows: list[dict[str, Any]] = []
    branch_seeds: dict[str, int] = {}
    for value in sentence:
        prefix_ids_cpu = torch.tensor([content_ids[:value]], dtype=torch.long)
        full_ids = torch.cat([encoded.input_ids, prefix_ids_cpu, suffix_ids_cpu], dim=1).to(device)
        branch_seed = task_seed(global_seed, dataset, split, problem_id, value)
        branch_seeds[str(value)] = branch_seed
        with torch.inference_mode():
            branch = generate_trace(
                model,
                tokenizer,
                full_ids,
                torch.ones_like(full_ids),
                branch_generation,
                branch_seed,
            )
        generated = tokenizer.decode(branch.tokens, skip_special_tokens=True)
        branch_text = suffix_text + generated
        prediction = prediction_for(dataset, branch_text)
        current_success = success_for(dataset, gold, prediction)
        rows.append(
            {
                "problem_id": problem_id,
                "dataset": dataset,
                "split": split,
                "subject": source.get("record", {}).get("subject"),
                "category": source.get("record", {}).get("category"),
                "checkpoint": value,
                "gold_answer": gold,
                "dense_prediction": dense_prediction,
                "dense_success": bool(dense_success),
                "dense_tokens": len(trace.tokens),
                "dense_content_tokens": len(content_ids),
                "prefix_context_tokens": int(encoded.input_ids.shape[1]) + value,
                "prefix_mean_entropy_tail8": tail_mean(trace.entropies, value),
                "prefix_token_ids": content_ids[:value],
                "current_prediction": prediction,
                "current_success": bool(current_success),
                "consistency": (
                    prediction is not None
                    and dense_prediction is not None
                    and prediction == dense_prediction
                ),
                "correction": (not current_success) and bool(dense_success),
                "damage": bool(current_success) and (not dense_success),
                "transition": ("C" if current_success else "W")
                + "_to_"
                + ("C" if dense_success else "W"),
                "branch_task_seed": branch_seed,
                "branch_tokens": len(branch.tokens),
                "branch_generated_text": generated,
                "branch_text": branch_text,
                "forced_context_tokens": int(full_ids.shape[1]),
            }
        )
        del prefix_ids_cpu, full_ids, branch
    if len(rows) != int(hidden.shape[0]):
        raise RuntimeError(f"row/vector mismatch：{problem_id}")
    if hidden.numel() and not torch.isfinite(hidden.float()).all():
        raise RuntimeError(f"hidden 出现 NaN/Inf：{problem_id}")
    artifact = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "pair_protocol_fingerprint": fingerprint,
        "selection_fingerprint": task["selection_fingerprint"],
        "dataset": dataset,
        "split": split,
        "problem_id": problem_id,
        "global_seed": global_seed,
        "dense_task_seed": dense_seed,
        "branch_task_seeds": branch_seeds,
        "source_fp16_artifact": str(source_path.resolve()),
        "source_fp16_protocol_fingerprint": source["protocol_fingerprint"],
        "record": source["record"],
        "gold_answer": gold,
        "dtype": "bfloat16",
        "attention_backend": config["model"]["attention_backend"],
        "model_revision": config["model"]["revision"],
        "model_audit": model_audit,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "worker_id": worker_id,
        "worker_gpu": int(device.index),
        "prompt_text": prompt_text,
        "prompt_sha256": prompt_hash,
        "prompt_tokens": int(encoded.input_ids.shape[1]),
        "dense": {
            "tokens": trace.tokens,
            "content_tokens": content_ids,
            "text": dense_text,
            "prediction": dense_prediction,
            "success": bool(dense_success),
            "reasoning_tokens": len(trace.tokens),
            "reached_max_tokens": len(trace.tokens) >= int(generation["max_new_tokens"]),
            "prefill_cuda_ms_diagnostic_only": float(trace.prefill_cuda_ms),
            "decode_cuda_ms_diagnostic_only": trace.decode_cuda_ms,
            "wall_ms_diagnostic_only": float(trace.wall_ms),
            "entropies_top20": trace.entropies,
        },
        "checkpoint_protocol": {
            "minimum": int(checkpoint["minimum"]),
            "maximum": int(checkpoint["maximum"]),
            "sentence_minimum_gap": int(checkpoint["sentence_minimum_gap"]),
        },
        "schedules": schedules,
        "decoded_prefix_for_boundaries": decoded_boundary_prefix,
        "capture_layers": [20],
        "rows": rows,
        "hidden": hidden,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_torch_save(artifact, destination)
    return destination, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/final_paper_primary_v1.yaml"))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    queue_root = args.queue_root if args.queue_root.is_absolute() else ROOT / args.queue_root
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_yaml(config_path)
    if transformers.__version__ != str(config["software"]["transformers"]):
        raise RuntimeError(
            f"Transformers 漂移：{transformers.__version__} != {config['software']['transformers']}"
        )
    if str(config["model"]["attention_backend"]) != "sdpa":
        raise ValueError("配对审计只允许 SDPA")
    seed_everything(int(config["seed"]["global"]))
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model, tokenizer, model_audit = load_qwen3(
        ROOT / config["model"]["local_path"],
        device,
        "bfloat16",
        config["model"]["attention_backend"],
    )
    if model_audit["metadata_fingerprint"] != config["model"]["metadata_fingerprint"]:
        raise RuntimeError("模型 metadata fingerprint 漂移")
    warmup(model, tokenizer, device, config, args.worker_id)
    capture = PositionHiddenCapture(model, [20])
    processed = skipped = failures = branches = 0
    started = time.time()
    while args.max_tasks is None or processed + skipped + failures < args.max_tasks:
        claimed = claim_task(queue_root, "dense", args.worker_id)
        if claimed is None:
            break
        task, claimed_path = claimed
        try:
            destination, branch_count = collect_one(
                task, config, model, tokenizer, model_audit, capture, device, args.worker_id
            )
            if branch_count:
                processed += 1
                branches += branch_count
            else:
                skipped += 1
            finish_task(claimed_path, task, queue_root, "done")
            elapsed = max(time.time() - started, 1e-9)
            state = {
                "status": "running",
                "worker_id": args.worker_id,
                "gpu": args.gpu,
                "processed": processed,
                "skipped": skipped,
                "failures": failures,
                "forced_branches": branches,
                "elapsed_seconds": elapsed,
                "samples_per_hour": 3600.0 * processed / elapsed,
                "last_output": str(destination),
            }
            atomic_json(state, queue_root / "workers" / f"pair_{args.worker_id}.json")
            print(json.dumps(state, ensure_ascii=False), flush=True)
        except Exception as error:
            failures += 1
            task["error_type"] = type(error).__name__
            task["error"] = str(error)
            finish_task(claimed_path, task, queue_root, "failed")
            print(
                json.dumps(
                    {
                        "worker_id": args.worker_id,
                        "problem_id": task.get("problem_id"),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    capture.close()
    elapsed = max(time.time() - started, 1e-9)
    final = {
        "status": "complete",
        "worker_id": args.worker_id,
        "gpu": args.gpu,
        "processed": processed,
        "skipped": skipped,
        "failures": failures,
        "forced_branches": branches,
        "elapsed_seconds": elapsed,
        "samples_per_hour": 3600.0 * processed / elapsed,
    }
    atomic_json(final, queue_root / "workers" / f"pair_{args.worker_id}.complete.json")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""只补 exact-hybrid schedule 缺失的 hidden 与 forced-answer sidecar。"""
from __future__ import annotations

import argparse
import gc
import json
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
from src.final_paper_replay_cache import claim_task, finish_task, tail_mean, task_seed
from src.qwen3_reasoning import generate_trace, load_qwen3
from src.utils import atomic_json, load_yaml, seed_everything


def destination(root: Path, task: dict[str, Any]) -> Path:
    return root / task["dataset"] / task["split"] / f"sample_{task['problem_id']}.pt"


def extension_fingerprint(
    config: dict[str, Any], requirements: dict[str, Any], model_audit: dict[str, Any]
) -> str:
    return canonical_fingerprint(
        {
            "protocol_id": config["protocol_id"],
            "requirements_fingerprint": requirements["requirements_fingerprint"],
            "dtype": config["model"]["dtype"],
            "model_revision": config["model"]["revision"],
            "model_metadata_fingerprint": model_audit["metadata_fingerprint"],
            "attention_backend": config["model"]["attention_backend"],
            "generation": config["generation"],
            "task_seed_derivation": config["seed"]["task_derivation"],
            "collector": "collect_hybrid_extension_primary_v1.py",
            "schema_version": 1,
        }
    )


def compatible(path: Path, task: dict[str, Any], fingerprint: str) -> bool:
    if not path.is_file():
        return False
    value = torch.load(path, map_location="cpu", weights_only=False)
    return (
        value.get("status") == "complete"
        and value.get("problem_id") == task["problem_id"]
        and value.get("extension_fingerprint") == fingerprint
        and value.get("missing_checkpoints") == task["missing_checkpoints"]
    )


def warmup(model, tokenizer, device: torch.device, config: dict[str, Any], worker: str) -> None:
    encoded = tokenizer("Warm-up.", return_tensors="pt")
    with torch.inference_mode():
        generate_trace(
            model,
            tokenizer,
            encoded.input_ids.to(device),
            encoded.attention_mask.to(device),
            resolved_generation(config, 4),
            task_seed(int(config["seed"]["global"]), "warmup", "hybrid", worker, "branch"),
        )
    torch.cuda.synchronize(device)


def collect_one(
    task: dict[str, Any],
    config: dict[str, Any],
    requirements: dict[str, Any],
    model,
    tokenizer,
    model_audit: dict[str, Any],
    capture: PositionHiddenCapture,
    device: torch.device,
    worker_id: str,
) -> tuple[Path, int]:
    source_path = Path(task["source_artifact"])
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    missing = sorted(set(map(int, task["missing_checkpoints"])))
    if not missing:
        raise ValueError("hybrid extension task 不应包含空 missing checkpoint")
    dataset, split, problem_id = task["dataset"], task["split"], task["problem_id"]
    dtype_name = str(config["model"]["dtype"])
    expected_source = config["datasets"][dataset]
    if source.get("status") != "complete" or source.get("dtype") != dtype_name:
        raise ValueError(f"source status/dtype 错位：{source_path}")
    if source.get("protocol_fingerprint") != expected_source["selected_cache_protocol_fingerprint"]:
        raise ValueError(f"source protocol fingerprint 错位：{source_path}")
    if source.get("problem_id") != problem_id or source.get("dataset") != dataset or source.get("split") != split:
        raise ValueError(f"source sample key 错位：{source_path}")
    existing = {int(row["checkpoint"]) for row in source["rows"]}
    if existing & set(missing):
        raise ValueError(f"missing checkpoint 已存在：{problem_id}")
    fingerprint = extension_fingerprint(config, requirements, model_audit)
    path = destination(Path(task["output_root"]), task)
    if compatible(path, task, fingerprint):
        return path, 0
    if path.exists():
        raise RuntimeError(f"拒绝覆盖不同指纹 hybrid sidecar：{path}")

    encoded = tokenizer(source["prompt_text"], return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    content_ids = list(source["dense"]["content_tokens"])
    if max(missing) > len(content_ids):
        raise ValueError(f"checkpoint 超出 Dense content：{problem_id}")
    teacher_ids = torch.cat(
        [input_ids, torch.tensor([content_ids[:max(missing)]], dtype=torch.long, device=device)],
        dim=1,
    )
    positions = [int(input_ids.shape[1]) + checkpoint - 1 for checkpoint in missing]
    capture.begin(positions, device)
    with torch.inference_mode():
        model.model(
            input_ids=teacher_ids,
            attention_mask=torch.ones_like(teacher_ids),
            use_cache=False,
            return_dict=True,
        )
    hidden_dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    hidden = capture.finish_cpu().to(hidden_dtype)
    del teacher_ids

    suffix = str(config["generation"]["force_answer_suffix"])
    suffix_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt").input_ids
    generation = resolved_generation(config, int(config["generation"]["force_answer_max_new_tokens"]))
    dense = source["dense"]
    rows = []
    branch_seeds = {}
    for checkpoint in missing:
        branch_seed = task_seed(
            int(config["seed"]["global"]), dataset, split, problem_id, checkpoint
        )
        branch_seeds[str(checkpoint)] = branch_seed
        full_ids = torch.cat(
            [encoded.input_ids, torch.tensor([content_ids[:checkpoint]], dtype=torch.long), suffix_ids],
            dim=1,
        ).to(device)
        with torch.inference_mode():
            trace = generate_trace(
                model,
                tokenizer,
                full_ids,
                torch.ones_like(full_ids),
                generation,
                branch_seed,
            )
        generated = tokenizer.decode(trace.tokens, skip_special_tokens=True)
        text = suffix + generated
        prediction = prediction_for(dataset, text)
        current_success = success_for(dataset, source["gold_answer"], prediction)
        dense_prediction = dense.get("prediction")
        dense_success = bool(dense.get("success"))
        rows.append(
            {
                "problem_id": problem_id,
                "dataset": dataset,
                "split": split,
                "seed": int(config["seed"]["global"]),
                "subject": source.get("record", {}).get("subject"),
                "category": source.get("record", {}).get("category"),
                "checkpoint": checkpoint,
                "checkpoint_schedules": ["hybrid"],
                "is_sentence_checkpoint": False,
                "is_fixed_checkpoint": False,
                "is_hybrid_checkpoint": True,
                "gold_answer": source["gold_answer"],
                "dense_prediction": dense_prediction,
                "dense_success": dense_success,
                "dense_tokens": int(dense["reasoning_tokens"]),
                "dense_content_tokens": len(content_ids),
                "prefix_context_tokens": int(encoded.input_ids.shape[1]) + checkpoint,
                "prefix_mean_entropy_tail8": tail_mean(dense["entropies_top20"], checkpoint),
                "prefix_token_ids": content_ids[:checkpoint],
                "current_prediction": prediction,
                "current_success": bool(current_success),
                "consistency": prediction is not None and dense_prediction is not None and prediction == dense_prediction,
                "correction": (not current_success) and dense_success,
                "damage": bool(current_success) and (not dense_success),
                "transition": ("C" if current_success else "W") + "_to_" + ("C" if dense_success else "W"),
                "branch_task_seed": branch_seed,
                "branch_tokens": len(trace.tokens),
                "branch_generated_text": generated,
                "branch_text": text,
                "forced_context_tokens": int(full_ids.shape[1]),
                "branch_worker_timing_excluded": True,
            }
        )
        del full_ids, trace
    if len(rows) != int(hidden.shape[0]) or (hidden.numel() and not torch.isfinite(hidden.float()).all()):
        raise RuntimeError(f"hybrid row/vector 非法：{problem_id}")
    artifact = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "extension_fingerprint": fingerprint,
        "requirements_fingerprint": requirements["requirements_fingerprint"],
        "source_protocol_fingerprint": source["protocol_fingerprint"],
        "source_artifact": str(source_path.resolve()),
        "dataset": dataset,
        "split": split,
        "problem_id": problem_id,
        "dtype": dtype_name,
        "attention_backend": config["model"]["attention_backend"],
        "model_audit": model_audit,
        "global_seed": int(config["seed"]["global"]),
        "branch_task_seeds": branch_seeds,
        "missing_checkpoints": missing,
        "capture_layers": [20],
        "rows": rows,
        "hidden": hidden,
        "worker_id": worker_id,
        "worker_gpu": int(device.index),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_torch_save(artifact, path)
    return path, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/final_paper_primary_v1.yaml"))
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    config = load_yaml(args.config if args.config.is_absolute() else ROOT / args.config)
    if config.get("primary") is not True or config.get("runnable") is not True:
        raise RuntimeError("主 dtype 尚未冻结，禁止启动 hybrid extension")
    requirements = json.loads(args.requirements.read_text(encoding="utf-8"))
    if requirements.get("status") != "frozen":
        raise ValueError("hybrid requirements 未冻结")
    dtype_name = str(config["model"]["dtype"])
    seed_everything(int(config["seed"]["global"]))
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model, tokenizer, model_audit = load_qwen3(
        ROOT / config["model"]["local_path"],
        device,
        dtype_name,
        config["model"]["attention_backend"],
    )
    if model_audit["metadata_fingerprint"] != config["model"]["metadata_fingerprint"]:
        raise RuntimeError("模型 fingerprint 漂移")
    warmup(model, tokenizer, device, config, args.worker_id)
    capture = PositionHiddenCapture(model, [20])
    processed = skipped = failures = checkpoints = 0
    started = time.time()
    while args.max_tasks is None or processed + skipped + failures < args.max_tasks:
        claimed = claim_task(args.queue_root, "dense", args.worker_id)
        if claimed is None:
            break
        task, claimed_path = claimed
        task["output_root"] = str(args.output_root.resolve())
        try:
            path, count = collect_one(
                task, config, requirements, model, tokenizer, model_audit, capture, device, args.worker_id
            )
            processed += int(count > 0); skipped += int(count == 0); checkpoints += count
            finish_task(claimed_path, task, args.queue_root, "done")
            elapsed = max(time.time() - started, 1e-9)
            atomic_json(
                {"status": "running", "processed": processed, "skipped": skipped, "failures": failures, "checkpoints": checkpoints, "samples_per_hour": 3600 * processed / elapsed, "last_output": str(path)},
                args.queue_root / "workers" / f"hybrid_{args.worker_id}.json",
            )
        except Exception as error:
            failures += 1
            task["error_type"] = type(error).__name__; task["error"] = str(error)
            finish_task(claimed_path, task, args.queue_root, "failed")
            print(json.dumps({"worker": args.worker_id, "error": repr(error), "task": task.get("problem_id")}), flush=True)
        gc.collect(); torch.cuda.empty_cache()
    capture.close()
    print(json.dumps({"status": "complete", "worker": args.worker_id, "processed": processed, "skipped": skipped, "failures": failures, "checkpoints": checkpoints}, indent=2))


if __name__ == "__main__":
    main()

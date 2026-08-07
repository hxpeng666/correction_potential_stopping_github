#!/usr/bin/env python3
"""在不修改缓存的前提下创建派生的单请求延迟回放视图。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer

from src.final_paper_inference import atomic_torch_save
from src.final_paper_cache import artifact_matches
from src.utils import atomic_json, load_yaml


def predict(model: dict[str, Any], context: float) -> float:
    scale = float(model["context_scale"])
    value = float(context) / scale
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    result = float(coefficients @ np.asarray([1.0, value, value * value]))
    return max(result, float(model["minimum_prediction_ms"]))


def autoregressive_cost(model: dict[str, Any], context: int, steps: int) -> float:
    return float(sum(predict(model, context + offset) for offset in range(max(0, steps))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cost-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--probe-overhead-ms", type=float, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--cost-device",
        help="应用于所有回放样本的设备模型键；成本文件含多个设备时必须指定。",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    cache_root = args.cache_root if args.cache_root.is_absolute() else ROOT / args.cache_root
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    cost_path = args.cost_model if args.cost_model.is_absolute() else ROOT / args.cost_model
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    cost_model_fingerprint = hashlib.sha256(cost_path.read_bytes()).hexdigest()
    device_models = cost["models_by_device"]
    if args.cost_device is not None:
        cost_device = str(args.cost_device)
    elif len(device_models) == 1:
        cost_device = next(iter(device_models))
    else:
        raise ValueError(
            "cost model contains multiple devices; select the target deployment device with --cost-device"
        )
    if cost_device not in device_models:
        raise KeyError(f"cost device {cost_device!r} absent from model")
    model_root = args.model_path or Path(config["model"]["local_path"])
    if not model_root.is_absolute():
        model_root = ROOT / model_root
    tokenizer = AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    suffix_tokens = len(
        tokenizer(
            config["generation"]["force_answer_suffix"],
            add_special_tokens=False,
        ).input_ids
    )
    completed = skipped = 0
    for split in ("probe_train", "calibration", "heldout"):
        for path in sorted((cache_root / "merged" / split).glob("sample_*.pt")):
            source = torch.load(path, map_location="cpu", weights_only=False)
            problem_id = str(source["problem_id"])
            fingerprint = str(source["protocol_fingerprint"])
            destination = output_root / split / f"sample_{problem_id}.pt"
            if args.resume and artifact_matches(
                destination, problem_id=problem_id, fingerprint=fingerprint
            ):
                existing = torch.load(
                    destination, map_location="cpu", weights_only=False
                )
                if (
                    existing.get("cost_model_fingerprint")
                    == cost_model_fingerprint
                    and existing.get("cost_device") == cost_device
                    and float(existing.get("probe_overhead_ms", -1.0))
                    == float(args.probe_overhead_ms)
                ):
                    skipped += 1
                    continue
            if destination.exists():
                raise RuntimeError(f"incompatible replay view preserved: {destination}")
            local = device_models[cost_device]
            decode_model = local["decode_token"]
            prefill_model = local["prefill"]
            prompt_tokens = int(source["prompt_tokens"])
            dense_steps = int(source["dense"]["reasoning_tokens"])
            predicted_prefill = predict(prefill_model, prompt_tokens)
            dense_cuda = predicted_prefill + autoregressive_cost(
                decode_model, prompt_tokens + 1, max(0, dense_steps - 1)
            )
            direct = dict(source["direct"])
            direct_context = int(direct["context_tokens"])
            direct_tokens = int(direct["generated_tokens"])
            direct_cost = predict(prefill_model, direct_context) + autoregressive_cost(
                decode_model, direct_context + 1, max(0, direct_tokens - 1)
            )
            direct["worker_wall_ms_excluded"] = direct.get("wall_ms")
            direct["wall_ms"] = direct_cost
            direct["replay_wall_ms"] = direct_cost
            direct["latency_source"] = cost["latency_label"]
            dense = dict(source["dense"])
            dense["measured_collection_wall_ms"] = dense.get("wall_ms")
            dense["wall_ms"] = dense_cuda
            dense["replay_estimated_ms"] = dense_cuda
            rows = []
            for original in source["rows"]:
                row = dict(original)
                checkpoint = int(row["checkpoint"])
                prefix_total = predicted_prefill + autoregressive_cost(
                    decode_model, prompt_tokens + 1, max(0, checkpoint - 1)
                )
                answer_cost = autoregressive_cost(
                    decode_model,
                    int(row["prefix_context_tokens"]),
                    suffix_tokens + int(row["branch_tokens"]),
                )
                row["dense_measured_collection_wall_ms"] = source["dense"].get(
                    "wall_ms"
                )
                row["dense_wall_ms"] = dense_cuda
                row["dense_prefill_cuda_ms"] = predicted_prefill
                row["prefix_decode_cuda_ms"] = max(
                    0.0, prefix_total - predicted_prefill
                )
                row["answer_replay_ms"] = answer_cost
                row["probe_replay_ms"] = float(args.probe_overhead_ms)
                row["probe_cumulative_sentence_ms"] = float(args.probe_overhead_ms) * sum(
                    int(value) <= checkpoint for value in source["schedules"]["sentence"]
                )
                row["probe_cumulative_fixed_ms"] = float(args.probe_overhead_ms) * sum(
                    int(value) <= checkpoint for value in source["schedules"]["fixed"]
                )
                # 分支成本不包含停止器检查。自适应回放会加入对应检查计划的
                # 累计开销，固定预算则不加入该项。
                row["branch_wall_ms"] = answer_cost
                row["replay_latency_label"] = cost["latency_label"]
                row.pop("prefix_token_ids", None)
                row.pop("branch_text", None)
                row.pop("branch_generated_text", None)
                rows.append(row)
            replay: dict[str, Any] = dict(source)
            replay.update(
                {
                    "source_common_cache_artifact": str(path),
                    "source_dense_artifact": str(path),
                    "dense": dense,
                    "direct": direct,
                    "rows": rows,
                    "latency_label": cost["latency_label"],
                    "cost_model": str(cost_path),
                    "cost_model_fingerprint": cost_model_fingerprint,
                    "cost_device": cost_device,
                    "probe_overhead_ms": float(args.probe_overhead_ms),
                    "replay_view_created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            atomic_torch_save(replay, destination)
            completed += 1
    summary = {
        "status": "complete",
        "dataset": args.dataset,
        "completed_now": completed,
        "skipped": skipped,
        "latency_label": cost["latency_label"],
        "cost_model": str(cost_path),
        "cost_model_fingerprint": cost_model_fingerprint,
        "cost_device": cost_device,
        "probe_overhead_ms": args.probe_overhead_ms,
    }
    atomic_json(summary, output_root / "materialize.complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

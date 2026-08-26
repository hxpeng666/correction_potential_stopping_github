#!/usr/bin/env python3
"""把冻结 A100 单请求成本模型应用到 MMLU-Pro held-out 公共缓存。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_paper_inference import atomic_torch_save
from src.final_paper_protocol import canonical_fingerprint
from src.utils import atomic_json, load_yaml

LABEL = "A100 single-request replay-estimated latency"


def design(prompt: float, reasoning: float) -> np.ndarray:
    p, r = prompt / 4096.0, reasoning / 4096.0
    return np.asarray([1, p, p*p, p*p*p, r, p*r, r*r, p*p*r, p*r*r, r*r*r], dtype=np.float64)


def predict(coefficients: np.ndarray, prompt: float, reasoning: float) -> float:
    return max(float(design(prompt, reasoning) @ coefficients), 0.001)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_paper_mmlu_pro_transfer_v1.yaml")
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(ROOT / args.config)
    cost_path = ROOT / config["replay"]["cost_model"]
    overhead_path = ROOT / config["replay"]["checkpoint_overhead"]
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    overhead = json.loads(overhead_path.read_text(encoding="utf-8"))
    if cost.get("status") != "frozen" or not cost.get("validation_gate", {}).get("passed"):
        raise ValueError("A100 成本模型未冻结或验证未通过")
    if overhead.get("status") != "complete" or not overhead.get("exclusive_gpu_verified"):
        raise ValueError("checkpoint overhead benchmark 无效")
    coefficients = np.asarray(cost["unified_cumulative_prefix_model"]["coefficients"], dtype=np.float64)
    boundary_ms = float(overhead["boundary_check_per_reasoning_token"]["mean_ms"])
    entropy_ms = float(overhead["entropy_top20_per_reasoning_token_wall"]["mean_ms"])
    sampling_ms = float(overhead["sampling_per_generated_token_wall"]["mean_ms"])
    stopper_ms = float(overhead["stopper_feature_mlp_per_sentence_checkpoint_wall"]["mean_ms"])
    view_fingerprint = canonical_fingerprint({
        "protocol_id": config["protocol_id"], "dataset": "mmlu_pro", "dtype": "float16",
        "timing_selection_fingerprint": cost["timing_selection_fingerprint"],
        "checkpoint_benchmark_fingerprint": overhead["benchmark_fingerprint"],
        "formula": "model_forward+sampling+boundary+entropy+feature_mlp+final_answer",
    })
    source_paths = sorted((args.merged_root / "heldout").glob("sample_*.pt"))
    if not source_paths:
        raise FileNotFoundError("MMLU-Pro merged heldout 为空")
    output = args.output_root / "heldout"; output.mkdir(parents=True, exist_ok=True)
    completed = skipped = 0
    for source_path in source_paths:
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        problem_id = str(source["problem_id"]); destination = output / f"sample_{problem_id}.pt"
        if args.resume and destination.is_file():
            previous = torch.load(destination, map_location="cpu", weights_only=False)
            if previous.get("status") == "complete" and previous.get("primary_replay_view_fingerprint") == view_fingerprint:
                skipped += 1; continue
            raise RuntimeError(f"拒绝 resume 不兼容 replay view：{destination}")
        if destination.exists(): raise RuntimeError(f"拒绝覆盖：{destination}")
        if source.get("status") != "complete" or source.get("dtype") != "float16": raise ValueError(source_path)
        prompt = int(source["prompt_tokens"]); dense_steps = int(source["dense"]["reasoning_tokens"])
        dense_content = len(source["dense"].get("content_tokens", [])) or dense_steps
        dense_cost = predict(coefficients, prompt, dense_steps) + sampling_ms * dense_steps
        sentence_positions = sorted(int(row["checkpoint"]) for row in source["rows"] if row.get("is_sentence_checkpoint"))
        fixed_positions = sorted(int(row["checkpoint"]) for row in source["rows"] if row.get("is_fixed_checkpoint"))
        inspected_fallback = min(dense_content, 768)
        sentence_fallback_overhead = (boundary_ms + entropy_ms) * inspected_fallback + stopper_ms * len(sentence_positions)
        fixed_fallback_overhead = entropy_ms * inspected_fallback + stopper_ms * len(fixed_positions)
        dense = dict(source["dense"]); dense.update({
            "collection_timing_excluded": True, "measured_collection_wall_ms": dense.get("wall_ms"),
            "wall_ms": dense_cost, "replay_wall_ms": dense_cost,
            "adaptive_fallback_wall_ms": dense_cost + sentence_fallback_overhead,
            "sentence_adaptive_fallback_wall_ms": dense_cost + sentence_fallback_overhead,
            "fixed_adaptive_fallback_wall_ms": dense_cost + fixed_fallback_overhead,
            "latency_source": LABEL,
        })
        direct = dict(source["direct"]); direct_generated = int(direct["generated_tokens"])
        direct_cost = predict(coefficients, int(direct["context_tokens"]), direct_generated) + sampling_ms * direct_generated
        direct.update({"worker_timing_excluded": True, "wall_ms": direct_cost, "replay_wall_ms": direct_cost, "latency_source": LABEL})
        rows = []
        for original in source["rows"]:
            row = dict(original); checkpoint = int(row["checkpoint"])
            prefix_forward = predict(coefficients, prompt, checkpoint)
            forced_context = int(row.get("forced_context_tokens", prompt + checkpoint))
            suffix_tokens = max(0, forced_context - prompt - checkpoint)
            answer_steps = suffix_tokens + int(row["branch_tokens"])
            answer_forward = max(0.0, predict(coefficients, prompt, checkpoint + answer_steps) - prefix_forward)
            prefix_sampling = sampling_ms * checkpoint; answer_sampling = sampling_ms * int(row["branch_tokens"])
            visited = sum(value <= checkpoint for value in sentence_positions)
            fixed_visited = sum(value <= checkpoint for value in fixed_positions)
            inspected = min(checkpoint, 768)
            sentence_overhead = (boundary_ms + entropy_ms) * inspected + stopper_ms * visited
            fixed_overhead = entropy_ms * inspected + stopper_ms * fixed_visited
            base_stop = prefix_forward + prefix_sampling + answer_forward + answer_sampling
            row.update({
                "dense_wall_ms": dense_cost, "dense_reference_wall_ms": dense_cost,
                "adaptive_fallback_wall_ms": dense_cost + sentence_fallback_overhead,
                "sentence_adaptive_fallback_wall_ms": dense_cost + sentence_fallback_overhead,
                "fixed_adaptive_fallback_wall_ms": dense_cost + fixed_fallback_overhead,
                "dense_prefill_cuda_ms": 0.0, "prefix_decode_cuda_ms": prefix_forward,
                "prefix_replay_a100_ms": prefix_forward + prefix_sampling,
                "answer_replay_a100_ms": answer_forward + answer_sampling,
                "boundary_tokens_inspected": inspected, "boundary_check_replay_ms": boundary_ms * inspected,
                "entropy_tokens_inspected": inspected, "entropy_top20_replay_ms": entropy_ms * inspected,
                "checkpoint_checks_incurred": visited, "checkpoint_check_replay_ms": stopper_ms * visited,
                "branch_wall_ms": answer_forward + answer_sampling,
                "fixed_replay_stop_wall_ms": base_stop, "fixed_adaptive_replay_stop_wall_ms": base_stop + fixed_overhead,
                "sentence_replay_stop_wall_ms": base_stop + sentence_overhead, "replay_stop_wall_ms": base_stop + sentence_overhead,
                "replay_latency_label": LABEL,
            }); rows.append(row)
        replay = dict(source); replay.update({
            "schema_version": 1, "source_common_cache_artifact": str(source_path.resolve()),
            "source_dense_artifact": str(destination.resolve()), "dense": dense, "direct": direct, "rows": rows,
            "latency_label": LABEL, "cost_model": str(cost_path.resolve()), "checkpoint_overhead_benchmark": str(overhead_path.resolve()),
            "policy_cost_mode": "includes_boundary_sampling_top20_entropy_hidden_tail8_scaler_mlp",
            "primary_replay_view_fingerprint": view_fingerprint, "replay_view_created_at": datetime.now(timezone.utc).isoformat(),
        })
        atomic_torch_save(replay, destination); completed += 1
    summary = {"status":"complete","dataset":"mmlu_pro","source_files":len(source_paths),"completed_now":completed,"skipped":skipped,
               "view_fingerprint":view_fingerprint,"latency_label":LABEL,"policy_cost_mode":"includes_boundary_sampling_top20_entropy_hidden_tail8_scaler_mlp"}
    atomic_json(summary, args.output_root / "materialize.complete"); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

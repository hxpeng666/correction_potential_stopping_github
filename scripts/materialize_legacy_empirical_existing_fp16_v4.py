#!/usr/bin/env python3
"""Apply the frozen A100 cost model to selected, immutable existing FP16 artifacts."""
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
from src.utils import atomic_json

LABEL = "A100 single-request replay-estimated latency"


def design(prompt: float, reasoning: float) -> np.ndarray:
    p = float(prompt) / 4096.0
    r = float(reasoning) / 4096.0
    return np.asarray([1.0, p, p*p, p*p*p, r, p*r, r*r, p*p*r, p*r*r, r*r*r], dtype=np.float64)


def predict(coefficients: np.ndarray, prompt: float, reasoning: float) -> float:
    return max(float(design(prompt, reasoning) @ coefficients), 0.001)


def reusable(path: Path, problem_id: str, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    value = torch.load(path, map_location="cpu", weights_only=False)
    return value.get("status") == "complete" and str(value.get("problem_id")) == problem_id and value.get("legacy_existing_fp16_view_fingerprint") == fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu"), required=True)
    parser.add_argument("--selected-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cost-model", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cost = json.loads(args.cost_model.read_text(encoding="utf-8"))
    if cost.get("status") != "frozen" or not cost.get("validation_gate", {}).get("passed"):
        raise ValueError("A100 cost model is not frozen/passed")
    coefficients = np.asarray(cost["unified_cumulative_prefix_model"]["coefficients"], dtype=np.float64)
    view_fingerprint = canonical_fingerprint({
        "protocol": "legacy_empirical_v4_existing_fp16_seed_override",
        "cost_timing_fingerprint": cost["timing_selection_fingerprint"],
        "formula": "prefix_cost(prompt,checkpoint)+answer_increment; checkpoint/probe overhead=0",
    })
    completed = skipped = 0
    for split in ("probe_train", "calibration", "heldout"):
        source_paths = sorted((args.selected_cache_root / "merged" / split).glob("sample_*.pt"))
        if not source_paths:
            raise FileNotFoundError(f"no selected artifacts for {args.dataset}/{split}")
        output = args.output_root / split
        output.mkdir(parents=True, exist_ok=True)
        for source_path in source_paths:
            source = torch.load(source_path, map_location="cpu", weights_only=False)
            problem_id = str(source["problem_id"])
            destination = output / f"sample_{problem_id}.pt"
            if args.resume and reusable(destination, problem_id, view_fingerprint):
                skipped += 1
                continue
            if destination.exists():
                raise RuntimeError(f"refusing to overwrite incompatible replay artifact: {destination}")
            if source.get("status") != "complete" or source.get("dtype") != "float16":
                raise ValueError(f"invalid existing FP16 artifact: {source_path}")
            if int(source.get("seed", -1)) != 20260803:
                raise ValueError(f"generation seed mismatch: {source_path}")
            if 20 not in [int(value) for value in source.get("capture_layers", [])]:
                raise ValueError(f"layer 20 absent: {source_path}")
            prompt_tokens = int(source["prompt_tokens"])
            dense_steps = int(source["dense"]["reasoning_tokens"])
            dense_cost = predict(coefficients, prompt_tokens, dense_steps)
            dense = dict(source["dense"])
            dense.update({
                "collection_timing_excluded": True,
                "measured_collection_wall_ms": dense.get("wall_ms"),
                "wall_ms": dense_cost,
                "replay_wall_ms": dense_cost,
                "adaptive_fallback_wall_ms": dense_cost,
                "latency_source": LABEL,
            })
            direct = dict(source["direct"])
            direct_context = int(direct.get("context_tokens", direct.get("prompt_tokens")))
            direct_cost = predict(coefficients, direct_context, int(direct["generated_tokens"]))
            direct.update({
                "worker_timing_excluded": True,
                "measured_collection_wall_ms": direct.get("wall_ms"),
                "wall_ms": direct_cost,
                "replay_wall_ms": direct_cost,
                "latency_source": LABEL,
            })
            rows = []
            for original in source["rows"]:
                row = dict(original)
                checkpoint = int(row["checkpoint"])
                prefix_cost = predict(coefficients, prompt_tokens, checkpoint)
                forced_context = int(row.get("forced_context_tokens", prompt_tokens + checkpoint))
                suffix_tokens = max(0, forced_context - prompt_tokens - checkpoint)
                answer_steps = suffix_tokens + int(row["branch_tokens"])
                answer_cost = max(0.0, predict(coefficients, prompt_tokens, checkpoint + answer_steps) - prefix_cost)
                row.update({
                    "dense_wall_ms": dense_cost,
                    "dense_reference_wall_ms": dense_cost,
                    "adaptive_fallback_wall_ms": dense_cost,
                    "dense_prefill_cuda_ms": 0.0,
                    "prefix_decode_cuda_ms": prefix_cost,
                    "prefix_replay_a100_ms": prefix_cost,
                    "answer_replay_a100_ms": answer_cost,
                    "checkpoint_checks_incurred": 0,
                    "checkpoint_check_replay_ms": 0.0,
                    "branch_wall_ms": answer_cost,
                    "replay_stop_wall_ms": prefix_cost + answer_cost,
                    "replay_latency_label": LABEL,
                    "policy_cost_mode": "legacy_no_probe_overhead",
                })
                rows.append(row)
            replay = dict(source)
            replay.update({
                "schema_version": 4,
                "source_common_cache_artifact": str(source_path.resolve()),
                "source_dense_artifact": str(destination.resolve()),
                "dense": dense,
                "direct": direct,
                "rows": rows,
                "latency_label": LABEL,
                "cost_model": str(args.cost_model.resolve()),
                "policy_cost_mode": "legacy_no_probe_overhead",
                "checkpoint_cost_mean_ms": 0.0,
                "legacy_existing_fp16_view_fingerprint": view_fingerprint,
                "replay_view_created_at": datetime.now(timezone.utc).isoformat(),
            })
            atomic_torch_save(replay, destination)
            completed += 1
    summary = {
        "status": "complete", "dataset": args.dataset,
        "completed_now": completed, "skipped": skipped,
        "view_fingerprint": view_fingerprint, "latency_label": LABEL,
        "checkpoint_probe_overhead_ms": 0.0,
    }
    atomic_json(summary, args.output_root / "materialize.complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

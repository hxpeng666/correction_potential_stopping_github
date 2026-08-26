#!/usr/bin/env python3
"""从三次预声明独占 A100 重复中选择最接近逐组件中位数的一次。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.final_paper_protocol import canonical_fingerprint
from src.utils import atomic_json

COMPONENTS = (
    "boundary_check_per_reasoning_token",
    "entropy_top20_per_reasoning_token_wall",
    "sampling_per_generated_token_wall",
    "stopper_feature_mlp_per_sentence_checkpoint_wall",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-relative-range", type=float, default=0.10)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("拒绝覆盖既有 overhead selection")
    values = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if any(value.get("status") != "complete" or value.get("exclusive_gpu_verified") is not True for value in values):
        raise ValueError("候选 overhead benchmark 非 complete/exclusive")
    protected = {(value["device"], value["hidden_dtype"], tuple(value["architecture"]), value["iterations"], value["warmup"]) for value in values}
    if len(protected) != 1:
        raise ValueError("overhead 重复配置不一致")
    matrix = np.asarray([[float(value[key]["mean_ms"]) for key in COMPONENTS] for value in values])
    medians = np.median(matrix, axis=0)
    relative_ranges = (matrix.max(axis=0) - matrix.min(axis=0)) / np.maximum(medians, 1e-12)
    passed = bool(np.all(relative_ranges <= args.maximum_relative_range))
    distances = np.sum(np.abs(matrix - medians) / np.maximum(medians, 1e-12), axis=1)
    selected_index = int(np.argmin(distances))
    selected = dict(values[selected_index])
    selected.update(
        {
            "status": "complete" if passed else "failed_repeatability_gate",
            "selection_rule": "predeclared three repeats; choose replicate with minimum componentwise relative L1 distance to medians",
            "candidate_files": [str(path.resolve()) for path in args.inputs],
            "selected_candidate_index": selected_index,
            "component_order": list(COMPONENTS),
            "component_mean_matrix_ms": matrix.tolist(),
            "component_medians_ms": medians.tolist(),
            "component_relative_ranges": relative_ranges.tolist(),
            "maximum_relative_range": args.maximum_relative_range,
            "repeatability_gate_passed": passed,
        }
    )
    selected.pop("benchmark_fingerprint", None)
    selected["benchmark_fingerprint"] = canonical_fingerprint(selected)
    atomic_json(selected, args.output)
    print(json.dumps({"status": selected["status"], "selected": selected_index, "relative_ranges": relative_ranges.tolist()}, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()

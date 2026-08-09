#!/usr/bin/env python3
"""Focused regression tests for legacy empirical threshold/replay semantics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.legacy_empirical_probe_v4 import (
    calibrate_policies,
    fit_validation_masks,
    fit_validation_problem_ids,
    method_direction,
    simulate_policy,
    threshold_grid,
)


def frame() -> pd.DataFrame:
    rows = []
    for problem in range(12):
        dense_success = problem % 3 != 0
        for checkpoint, current_success in ((64, problem % 4 == 0), (96, problem % 2 == 0), (128, dense_success)):
            rows.append({
                "problem_id": f"p{problem:02d}",
                "checkpoint": checkpoint,
                "current_success": current_success,
                "dense_success": dense_success,
                "current_prediction": "1" if current_success else "0",
                "dense_prediction": "1" if dense_success else "0",
                "gold_answer": "1",
                "dense_tokens": 256,
                "branch_tokens": 3,
                "dense_wall_ms": 100.0,
                "dense_prefill_cuda_ms": 10.0,
                "prefix_decode_cuda_ms": checkpoint / 2.0,
                "branch_wall_ms": 2.0,
                "prefix_mean_entropy_tail8": 0.5,
                "subject": "s",
                "category": "c",
            })
    return pd.DataFrame(rows)


def main() -> None:
    values = frame()
    correction_scores = np.tile(np.asarray([0.1, 0.4, 0.9]), 12)
    high_scores = 1.0 - correction_scores
    assert method_direction("correction") == "low"
    assert method_direction("correctness") == "high"
    low = simulate_policy(values, correction_scores, "low", 0.5, include_records=True)
    high = simulate_policy(values, high_scores, "high", 0.5, include_records=True)
    assert all(row["checkpoint"] == 64 for row in low["records"])
    assert all(row["checkpoint"] == 64 for row in high["records"])
    grid = threshold_grid(correction_scores, "low", 101)
    assert grid[0][1] is True and len(grid) <= 102
    calibrated = calibrate_policies(
        values,
        correction_scores,
        "low",
        grid_size=101,
        empirical_budgets=[0, 1, 2, 4, 10],
        coverage_targets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    assert set(calibrated) == {"grid_declared_size", "grid_realized_size", "curve", "empirical_B", "coverage"}
    assert set(calibrated["empirical_B"]) == {"0", "1", "2", "4", "10"}
    assert set(calibrated["coverage"]) == {"30", "40", "50", "60", "70", "80", "90"}
    fallback = simulate_policy(values, correction_scores, "low", -1.0, force_dense=True)
    assert fallback["coverage"] == 0.0
    assert fallback["token_reduction"] == 0.0
    assert fallback["replay_wall_reduction"] == 0.0
    assert fallback["fallback"] == 12
    assert sum(fallback["counts"].values()) == 0
    assert fallback["accuracy"] == fallback["dense_accuracy"]
    identity = 100.0 * (fallback["counts"]["W_to_C"] - fallback["counts"]["C_to_W"]) / fallback["problems"]
    assert abs(fallback["accuracy_drop_pp"] - identity) < 1e-12
    fit, validation = fit_validation_masks(values, "mmlu", seed=0)
    assert len(set(values.loc[fit, "problem_id"]) & set(values.loc[validation, "problem_id"])) == 0
    assert len(set(values.loc[fit, "problem_id"])) == 9
    fit_ids, validation_ids = fit_validation_problem_ids(
        values, "mmlu", seed=0, additional_problem_ids=["fallback_only"]
    )
    assert len(fit_ids) == 10 and len(validation_ids) == 3
    assert "fallback_only" in fit_ids | validation_ids
    assert not (fit_ids & validation_ids)
    print("legacy empirical protocol regression tests: PASS")


if __name__ == "__main__":
    main()

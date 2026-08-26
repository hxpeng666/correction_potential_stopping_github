#!/usr/bin/env python3
"""论文主协议的轻量回归测试；不加载基础模型。"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_paper_protocol import parse_mcq_answer
from src.final_paper_replay_cache import task_seed
from src.legacy_empirical_probe_v4 import (
    _last_switch_flags,
    add_targets,
    build_features,
    calibrate_policies,
    correction_loss,
    fit_validation_masks,
    method_direction,
    simulate_policy,
    threshold_grid,
)
from src.utils import load_yaml
from scripts.train_legacy_empirical_probe_v4 import calibration_value, global_seed


def replay_frame() -> pd.DataFrame:
    rows = []
    for problem in range(12):
        dense_success = problem % 3 != 0
        for checkpoint, current_success in (
            (64, problem % 4 == 0),
            (96, problem % 2 == 0),
            (128, dense_success),
        ):
            rows.append(
                {
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
                    "adaptive_fallback_wall_ms": 101.5,
                    "dense_prefill_cuda_ms": 10.0,
                    "prefix_decode_cuda_ms": checkpoint / 2.0,
                    "branch_wall_ms": 2.0,
                    "replay_stop_wall_ms": checkpoint / 2.0 + 12.75,
                    "prefix_mean_entropy_tail8": 0.5,
                    "subject": "s",
                    "category": "c",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    config = load_yaml(ROOT / "configs/final_paper_primary_v1.yaml")
    assert config["primary"] is True
    assert config["protocol_id"] == "cps_qwen3_4b_sentence_empiricalB_seed20260803_train1000_cal500_mmlu1k_v1"
    assert config["checkpoint"]["primary_schedule"] == "sentence"
    assert config["calibration"]["named_workpoints"] == {"strict": 1, "balanced": 2, "aggressive": 4}
    assert config["datasets"]["gsm8k"]["heldout"] == 1319
    assert config["datasets"]["mmlu"]["report_label"] == "MMLU-1k distribution-shift"
    assert global_seed(config) == 20260803
    assert calibration_value(config, "quantile_grid_size", "threshold_quantiles") == 101
    assert calibration_value(config, "historical_workpoints", "named_workpoints") == {
        "strict": 1, "balanced": 2, "aggressive": 4
    }

    assert parse_mcq_answer(r"reasoning A then \boxed{C}") == "C"
    assert parse_mcq_answer("Final answer: B") == "B"
    assert parse_mcq_answer("reasoning mentions A B C D") is None

    labels = pd.DataFrame(
        [
            {"problem_id": "missing", "checkpoint": 64, "current_prediction": None, "dense_prediction": None, "current_success": False, "dense_success": False},
            {"problem_id": "switch", "checkpoint": 64, "current_prediction": "A", "dense_prediction": "B", "current_success": False, "dense_success": True},
            {"problem_id": "switch", "checkpoint": 96, "current_prediction": "B", "dense_prediction": "B", "current_success": True, "dense_success": True},
            {"problem_id": "final_change", "checkpoint": 64, "current_prediction": "A", "dense_prediction": "B", "current_success": False, "dense_success": True},
            {"problem_id": "final_change", "checkpoint": 96, "current_prediction": "A", "dense_prediction": "B", "current_success": False, "dense_success": True},
        ]
    )
    targets = add_targets(labels)
    assert not bool(targets.loc[targets.problem_id == "missing", "target_consistency"].iloc[0])
    assert _last_switch_flags(["A", "B"], "B") == [False, True]
    assert _last_switch_flags(["A", "A"], "B") == [False, False]
    final_change = targets[targets.problem_id == "final_change"].sort_values("checkpoint")
    assert final_change.target_last_switch.tolist() == [False, False]
    assert targets.loc[(targets.problem_id == "switch") & (targets.checkpoint == 64), "target_correction"].item()

    values = replay_frame()
    correction_scores = np.tile(np.asarray([0.1, 0.4, 0.9]), 12)
    assert method_direction("correction") == "low"
    assert method_direction("correctness") == "high"
    low = simulate_policy(values, correction_scores, "low", 0.5, include_records=True)
    high = simulate_policy(values, 1.0 - correction_scores, "high", 0.5, include_records=True)
    assert all(row["checkpoint"] == 64 for row in low["records"])
    assert all(row["checkpoint"] == 64 for row in high["records"])
    assert all(abs(row["replay_wall_ms"] - 44.75) < 1e-12 for row in low["records"])

    grid = threshold_grid(correction_scores, "low", 101)
    assert grid[0][1] is True
    calibrated = calibrate_policies(
        values,
        correction_scores,
        "low",
        grid_size=101,
        empirical_budgets=[0, 1, 2, 4, 10],
        coverage_targets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    assert set(calibrated["empirical_B"]) == {"0", "1", "2", "4", "10"}
    assert set(calibrated["coverage"]) == {"30", "40", "50", "60", "70", "80", "90"}
    for budget in (1, 2, 4):
        assert calibrated["empirical_B"][str(budget)]["lost_correct_count"] <= budget

    sentinel = simulate_policy(values, correction_scores, "low", -1.0, force_dense=True)
    assert sentinel["coverage"] == 0.0
    assert sentinel["token_reduction"] == 0.0
    assert sentinel["replay_wall_reduction"] == 0.0
    assert sentinel["fallback"] == 12
    assert sentinel["accuracy"] == sentinel["dense_accuracy"]
    no_hit = simulate_policy(values, correction_scores, "low", -1.0)
    assert no_hit["coverage"] == 0.0
    assert no_hit["mean_replay_wall_ms"] == 101.5
    assert no_hit["replay_wall_reduction"] < 0.0

    fallback = {
        "problem_id": "fallback_only",
        "subject": "s",
        "category": "c",
        "gold_answer": "1",
        "dense_prediction": "1",
        "dense_success": True,
        "dense_tokens": 80,
        "dense_wall_ms": 40.0,
        "adaptive_fallback_wall_ms": 40.0,
    }
    with_fallback = simulate_policy(
        values,
        correction_scores,
        "low",
        0.5,
        fallback_records=[fallback],
    )
    assert with_fallback["problems"] == 13
    assert with_fallback["fallback"] == 1

    fit, validation = fit_validation_masks(values, "mmlu", seed=0, additional_problem_ids=["fallback_only"])
    assert not set(values.loc[fit, "problem_id"]) & set(values.loc[validation, "problem_id"])

    hidden = np.zeros((len(values), 1, 2560), dtype=np.float32)
    hidden[:, 0, 0] = np.arange(len(values), dtype=np.float32)
    features = build_features(values, hidden, [20], layer=20, feature_kind="full")
    assert features.shape == (len(values), 5126)
    assert np.isfinite(features).all()

    logits = torch.tensor([0.2, -0.1, 0.4], requires_grad=True)
    target = torch.tensor([1.0, 0.0, 1.0])
    remaining = torch.tensor([0.8, 0.5, 0.2])
    total, point, trajectory = correction_loss(
        logits, target, remaining, np.asarray([0, 2, 3]), beta=0.5, trajectory=True
    )
    assert torch.isfinite(total) and trajectory > 0 and torch.allclose(total, point + trajectory)
    bce_only, point_only, no_trajectory = correction_loss(
        logits, target, remaining, np.asarray([0, 2, 3]), beta=0.5, trajectory=False
    )
    assert torch.allclose(bce_only, point_only) and float(no_trajectory) == 0.0

    seed_a = task_seed(20260803, "gsm8k", "probe_train", "p", "dense")
    seed_b = task_seed(20260803, "gsm8k", "probe_train", "p", 64)
    assert seed_a == task_seed(20260803, "gsm8k", "probe_train", "p", "dense")
    assert seed_a != seed_b

    manifest_path = ROOT / "results/final_paper_primary_v1/dtype_pair_audit_v2/PAIR_SAMPLE_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert len(manifest["samples"]) == 400
        assert not manifest["heldout_used"]
        assert all(row["split"] in {"probe_train", "calibration"} for row in manifest["samples"])
        assert len({(row["dataset"], row["sample_id"]) for row in manifest["samples"]}) == 400
        mmlu = [row for row in manifest["samples"] if row["dataset"] == "mmlu"]
        assert len({row["subject"] for row in mmlu}) == 57
        assert sum(row["reached_4096_fp16"] for row in manifest["samples"] if row["dataset"] == "gsm8k") == 10
        assert sum(row["reached_4096_fp16"] for row in mmlu) <= 4

    print("primary protocol v1 regression tests: PASS")


if __name__ == "__main__":
    main()

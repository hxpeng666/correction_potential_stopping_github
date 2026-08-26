#!/usr/bin/env python3
"""四状态 probe 的核心语义回归测试。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.four_state_utility_probe_v1 import (
    FourStateUtilityProbe,
    four_state_targets,
    legacy_weighted_multiclass_trajectory_loss,
    predict_probabilities,
    simulate_zero_utility_policy,
)


def frame() -> pd.DataFrame:
    rows = []
    for problem_id, dense_success, current in (
        ("a", True, (False, True)),
        ("b", False, (True, False)),
    ):
        for checkpoint, current_success in zip((64, 96), current):
            rows.append({
                "problem_id": problem_id,
                "checkpoint": checkpoint,
                "current_success": current_success,
                "dense_success": dense_success,
                "current_prediction": "X",
                "dense_prediction": "Y",
                "gold_answer": "Y" if dense_success else "Z",
                "branch_tokens": 3,
                "dense_tokens": 200,
                "dense_prefill_cuda_ms": 1.0,
                "prefix_decode_cuda_ms": float(checkpoint),
                "branch_wall_ms": 2.0,
                "dense_wall_ms": 200.0,
                "adaptive_fallback_wall_ms": 201.0,
            })
    return pd.DataFrame(rows)


def test_four_state_targets() -> None:
    labels = four_state_targets(frame())
    assert labels.tolist() == [0, 3, 1, 2]


def test_softmax_sums_to_one() -> None:
    model = FourStateUtilityProbe(5)
    probabilities = predict_probabilities(model, np.zeros((7, 5), dtype=np.float32), torch.device("cpu"))
    assert probabilities.shape == (7, 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_first_hit_strict_negative_and_equality_continues() -> None:
    probabilities = np.asarray([
        [0.30, 0.30, 0.20, 0.20],  # a@64：差为0，必须继续
        [0.10, 0.40, 0.20, 0.30],  # a@96：首次小于0，停止
        [0.40, 0.20, 0.20, 0.20],  # b@64：继续
        [0.50, 0.10, 0.20, 0.20],  # b@96：继续，Dense fallback
    ])
    result = simulate_zero_utility_policy(frame(), probabilities, include_records=True)
    records = {row["problem_id"]: row for row in result["records"]}
    assert records["a"]["checkpoint"] == 96
    assert not records["a"]["fallback"]
    assert records["b"]["fallback"]
    assert result["coverage"] == 0.5


def test_legacy_trajectory_loss_uses_wc_minus_cw_margin() -> None:
    targets = torch.tensor([0, 0, 3], dtype=torch.long)
    remaining = torch.tensor([0.8, 0.5, 0.2])
    offsets = np.asarray([0, 3])
    safe = torch.tensor([[3.0, -1.0, 0.0, 0.0], [2.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]])
    unsafe = safe.clone()
    unsafe[1, 0], unsafe[1, 1] = -2.0, 2.0
    safe_loss = legacy_weighted_multiclass_trajectory_loss(safe, targets, remaining, offsets)[0]
    unsafe_loss = legacy_weighted_multiclass_trajectory_loss(unsafe, targets, remaining, offsets)[0]
    assert torch.isfinite(safe_loss) and torch.isfinite(unsafe_loss)
    assert unsafe_loss > safe_loss


if __name__ == "__main__":
    test_four_state_targets()
    test_softmax_sums_to_one()
    test_first_hit_strict_negative_and_equality_continues()
    test_legacy_trajectory_loss_uses_wc_minus_cw_margin()
    print("四状态概率与停止规则测试通过")

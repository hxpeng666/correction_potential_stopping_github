#!/usr/bin/env python3
"""动态Bellman停止、风险上界和token成本语义测试。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.dynamic_optimal_stopping_v1 import (
    DYNAMIC_FEATURE_KINDS,
    backward_value_targets,
    build_dynamic_features,
    clopper_pearson_upper,
    simulate_dynamic_policy,
    transition_token_costs,
    one_step_value_targets,
    dense_endpoint_value_targets,
)


def wcw_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"problem_id": "wcw", "checkpoint": 64, "current_success": False, "dense_success": False, "current_prediction": "A", "dense_prediction": "A", "gold_answer": "B", "branch_tokens": 9, "dense_tokens": 256},
        {"problem_id": "wcw", "checkpoint": 96, "current_success": True, "dense_success": False, "current_prediction": "B", "dense_prediction": "A", "gold_answer": "B", "branch_tokens": 9, "dense_tokens": 256},
        {"problem_id": "wcw", "checkpoint": 128, "current_success": False, "dense_success": False, "current_prediction": "A", "dense_prediction": "A", "gold_answer": "B", "branch_tokens": 9, "dense_tokens": 256},
    ])


def test_backward_dynamic_programming_values_intermediate_correct_state() -> None:
    frame = wcw_frame()
    stop = np.asarray([0.10, 0.90, 0.10])
    risk = np.zeros(3)
    targets, values = backward_value_targets(
        frame, stop, risk,
        np.asarray([0.2]), np.asarray([0.0]), cost_unit_tokens=4096,
    )
    assert targets.shape == (3, 1)
    assert values[1, 0] > values[2, 0]
    assert targets[0, 0] == values[1, 0]


def test_online_policy_continues_then_stops_at_wcw_middle() -> None:
    frame = wcw_frame()
    stop = np.asarray([0.10, 0.90, 0.10])
    risk = np.zeros(3)
    continuation = np.asarray([0.90, 0.10, 0.0])
    result = simulate_dynamic_policy(
        frame, stop, risk, continuation,
        lambda_value=0.2, mu_value=0.0, cost_unit_tokens=4096,
        include_records=True,
    )
    record = result["records"][0]
    assert record["checkpoint"] == 96
    assert record["method_success"] is True
    assert record["dense_success"] is False
    assert record["method_tokens"] == 96  # branch_tokens=9被明确忽略
    assert result["counts"]["C_to_W"] == 1


def test_transition_cost_last_step_reaches_dense_endpoint() -> None:
    costs = transition_token_costs(wcw_frame(), 4096)
    assert np.allclose(costs, np.asarray([32, 32, 128]) / 4096)
    remaining = transition_token_costs(wcw_frame(), 4096, cost_mode="remaining_to_dense")
    assert np.allclose(remaining, np.asarray([192, 160, 128]) / 4096)


def test_one_step_and_endpoint_targets() -> None:
    frame = wcw_frame()
    stop = np.asarray([0.1, 0.9, 0.1])
    risk = np.asarray([0.2, 0.0, 0.3])
    one_step = one_step_value_targets(frame, stop, risk, np.asarray([0.0, 1.0]))
    assert np.allclose(one_step[0], [0.9, 0.9])
    assert np.allclose(one_step[1], [0.1, -0.2])
    endpoint = dense_endpoint_value_targets(frame, 2)
    assert np.allclose(endpoint, 0.0)


def test_simultaneous_cp_upper_zero_events() -> None:
    upper = clopper_pearson_upper(0, 500, 0.05 / 48)
    assert 0.01 < upper < 0.02


def test_all_feature_ablations_have_expected_columns() -> None:
    frame = wcw_frame()
    frame["prefix_mean_entropy_tail8"] = [0.4, 0.5, 0.6]
    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(3, 1, 2560)).astype(np.float32)
    expected = {
        "full": 5126, "h_only": 2560, "delta_only": 2560, "scalars_only": 6,
        "h_delta": 5120, "full_no_position": 5123, "full_no_geometry": 5124,
        "full_no_hidden": 2566, "full_no_delta": 2566,
    }
    for kind in DYNAMIC_FEATURE_KINDS:
        values = build_dynamic_features(frame, hidden, [20], layer=20, feature_kind=kind)
        assert values.shape[0] == len(frame)
        assert np.isfinite(values).all()
        if kind in expected:
            assert values.shape[1] == expected[kind]
        elif kind.startswith("h_delta_plus_"):
            assert values.shape[1] == 5121
        elif kind.startswith("full_no_"):
            assert values.shape[1] == 5125


if __name__ == "__main__":
    test_backward_dynamic_programming_values_intermediate_correct_state()
    test_online_policy_continues_then_stops_at_wcw_middle()
    test_transition_cost_last_step_reaches_dense_endpoint()
    test_one_step_and_endpoint_targets()
    test_simultaneous_cp_upper_zero_events()
    test_all_feature_ablations_have_expected_columns()
    print("动态最优停止核心语义测试通过")

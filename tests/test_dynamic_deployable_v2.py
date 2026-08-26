#!/usr/bin/env python3
"""可部署动态停止、受控 OS-Pruner 与因果隔离测试。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.dynamic_optimal_stopping_deployable_v2 import (
    candidate_feasibility_counts,
    decide_current_action,
    dense_endpoint_q_continue_targets,
    deterministic_action_uniform,
    one_step_q_continue_targets,
    os_pruner_expected_utility,
    recursive_q_continue_targets,
    simulate_deployable_dynamic_policy,
    simulate_os_pruner_policy,
)


def wcw_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"problem_id": "wcw", "checkpoint": 64, "current_success": False, "dense_success": False, "current_prediction": "A", "dense_prediction": "A", "gold_answer": "B", "branch_tokens": 9, "dense_tokens": 256},
        {"problem_id": "wcw", "checkpoint": 96, "current_success": True, "dense_success": False, "current_prediction": "B", "dense_prediction": "A", "gold_answer": "B", "branch_tokens": 9, "dense_tokens": 256},
        {"problem_id": "wcw", "checkpoint": 128, "current_success": False, "dense_success": False, "current_prediction": "A", "dense_prediction": "A", "gold_answer": "B", "branch_tokens": 9, "dense_tokens": 256},
    ])


def test_q_target_includes_future_increment_cost() -> None:
    frame = wcw_frame()
    stop = np.asarray([0.10, 0.90, 0.10])
    risk = np.zeros(3)
    lambdas = np.asarray([1.0])
    targets, values = recursive_q_continue_targets(
        frame, stop, risk, lambdas, np.asarray([0.0]), cost_unit_tokens=256,
    )
    # 最后一步继续到 Dense：0 - (256-128)/256 = -0.5。
    assert np.allclose(targets[2, 0], -0.5)
    # 第一个 checkpoint 的 target 是包含 32-token 成本的下一状态价值。
    assert np.allclose(targets[0, 0], values[1, 0] - 32.0 / 256.0)


def test_deployable_policy_continues_then_stops_without_future_cost_argument() -> None:
    frame = wcw_frame()
    stop = np.asarray([0.10, 0.90, 0.10])
    risk = np.zeros(3)
    q_continue = np.asarray([0.80, 0.20, 0.00])
    result = simulate_deployable_dynamic_policy(
        frame, stop, risk, q_continue, mu_value=0.0, include_records=True,
    )
    record = result["records"][0]
    assert record["checkpoint"] == 96
    assert record["method_success"] is True
    assert result["future_fields_used_by_action"] is False


def test_future_mutation_cannot_change_current_action() -> None:
    frame = wcw_frame()
    stop = np.asarray([0.10, 0.90, 0.10])
    risk = np.zeros(3)
    q_continue = np.asarray([0.80, 0.20, 0.00])
    original = simulate_deployable_dynamic_policy(
        frame, stop, risk, q_continue, mu_value=0.0, include_records=True,
    )
    mutated = frame.copy()
    # 保持 checkpoint 顺序，但任意改变未来位置和 Dense endpoint。
    mutated.loc[1:, "checkpoint"] = [700, 767]
    mutated["dense_tokens"] = 4096
    replayed = simulate_deployable_dynamic_policy(
        mutated, stop, risk, q_continue, mu_value=0.0, include_records=True,
    )
    # 第一个动作 continue，第二个动作 stop；future mutation 不改变动作序列。
    assert decide_current_action(stop[0], q_continue[0]) is False
    assert decide_current_action(stop[1], q_continue[1]) is True
    assert original["records"][0]["fallback"] == replayed["records"][0]["fallback"]
    assert original["records"][0]["method_success"] == replayed["records"][0]["method_success"]


def test_one_step_and_endpoint_targets_also_include_cost() -> None:
    frame = wcw_frame()
    stop = np.asarray([0.1, 0.9, 0.1])
    risk = np.asarray([0.2, 0.0, 0.3])
    lambdas = np.asarray([0.0, 1.0])
    mus = np.asarray([0.0, 1.0])
    one_step = one_step_q_continue_targets(
        frame, stop, risk, lambdas, mus, cost_unit_tokens=256,
    )
    assert np.allclose(one_step[0], [0.9, 0.9 - 32.0 / 256.0])
    assert np.allclose(one_step[1], [0.1, -0.2 - 32.0 / 256.0])
    endpoint = dense_endpoint_q_continue_targets(frame, lambdas, cost_unit_tokens=256)
    assert np.allclose(endpoint[:, 0], 0.0)
    assert np.allclose(endpoint[:, 1], -np.asarray([192, 160, 128]) / 256.0)


def test_os_pruner_expected_first_hit_utility() -> None:
    frame = wcw_frame().iloc[:2].copy()
    positions = np.asarray([0, 1])
    offsets = np.asarray([0, 2])
    logits = torch.zeros((2, 1), requires_grad=True)  # s=[0.5,0.5]
    loss, parts = os_pruner_expected_utility(
        logits, frame, positions, offsets,
        torch.tensor([0.0]), torch.tensor([0.0]), cost_unit_tokens=256,
    )
    # stop masses 0.5/0.25，fallback 0.25；correctness 0/1，Dense=0。
    assert torch.allclose(loss, torch.tensor(-0.25))
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.isfinite(parts["mean_candidate_utility"])


def test_os_action_randomness_is_order_invariant() -> None:
    first = deterministic_action_uniform(20260803, "gsm8k", "calibration", "x", 64)
    second = deterministic_action_uniform(20260803, "gsm8k", "calibration", "x", 64)
    assert first == second and 0.0 < first < 1.0
    frame = wcw_frame()
    probability = np.ones(len(frame))
    result = simulate_os_pruner_policy(
        frame, probability, dataset="gsm8k", split="calibration",
        action_seed=20260803, include_records=True,
    )
    assert result["records"][0]["checkpoint"] == 64


def test_candidate_failure_mechanisms_are_separated() -> None:
    curve = [
        {"coverage": 0.0, "accuracy": 0.8, "lost_correct_count": 0},
        {"coverage": 0.5, "accuracy": 0.7, "lost_correct_count": 1},
        {"coverage": 0.4, "accuracy": 0.8, "lost_correct_count": 5},
    ]
    counts = candidate_feasibility_counts(curve, 0.8, 0.01, [0, 4, 10])
    assert counts["nonzero_stopping_candidates"] == 2
    assert counts["accuracy_feasible_candidates"] == 2
    assert counts["accuracy_and_budget_feasible"] == {"0": 1, "4": 1, "10": 2}


if __name__ == "__main__":
    test_q_target_includes_future_increment_cost()
    test_deployable_policy_continues_then_stops_without_future_cost_argument()
    test_future_mutation_cannot_change_current_action()
    test_one_step_and_endpoint_targets_also_include_cost()
    test_os_pruner_expected_first_hit_utility()
    test_os_action_randomness_is_order_invariant()
    test_candidate_failure_mechanisms_are_separated()
    print("可部署动态停止与OS-Pruner核心语义测试通过")

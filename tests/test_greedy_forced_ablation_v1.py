#!/usr/bin/env python3
"""Small invariants needed by the greedy forced-answer ablation pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dynamic_optimal_stopping_v1 import build_dynamic_features
from src.legacy_empirical_probe_v4 import _finite_cost, build_features, simulate_policy


def test_static_and_dynamic_2566_features_are_identical() -> None:
    frame = pd.DataFrame(
        {
            "problem_id": ["a", "a", "b"],
            "checkpoint": [64, 80, 72],
            "prefix_mean_entropy_tail8": [0.1, 0.2, 0.3],
        }
    )
    rng = np.random.default_rng(7)
    hidden = rng.normal(size=(3, 1, 2560)).astype(np.float32)
    static = build_features(
        frame, hidden, [20], layer=20, feature_kind="full_no_delta"
    )
    dynamic = build_dynamic_features(
        frame, hidden, [20], layer=20, feature_kind="full_no_delta"
    )
    assert static.shape == (3, 2566)
    np.testing.assert_allclose(static, dynamic, rtol=0.0, atol=0.0)


def test_missing_timing_falls_back_to_token_cost() -> None:
    assert _finite_cost(None, 123) == 123.0
    assert _finite_cost(float("nan"), 123) == 123.0
    assert _finite_cost(4.5, 123) == 4.5


def test_greedy_policy_uses_tokens_for_both_cost_sides() -> None:
    frame = pd.DataFrame(
        [
            {
                "problem_id": "x",
                "checkpoint": 64,
                "branch_tokens": 3,
                "dense_tokens": 100,
                "dense_wall_ms": 9999.0,
                "replay_stop_wall_ms": None,
                "adaptive_fallback_wall_ms": 8888.0,
                "forced_answer_decoding": "greedy_argmax",
                "current_success": True,
                "dense_success": True,
                "current_prediction": "1",
                "dense_prediction": "1",
                "gold_answer": "1",
            }
        ]
    )
    value = simulate_policy(frame, np.asarray([1.0]), "high", 0.5)
    assert value["mean_replay_wall_ms"] == 64.0
    assert value["mean_dense_wall_ms"] == 100.0
    assert np.isclose(value["replay_wall_reduction"], 0.36)


if __name__ == "__main__":
    test_static_and_dynamic_2566_features_are_identical()
    test_missing_timing_falls_back_to_token_cost()
    test_greedy_policy_uses_tokens_for_both_cost_sides()
    print("test_greedy_forced_ablation_v1: passed")

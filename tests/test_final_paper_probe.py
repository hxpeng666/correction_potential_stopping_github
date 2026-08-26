from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from src.final_paper_probe import (
    add_targets,
    binomial_upper_simultaneous,
    build_features,
    build_online_feature,
    calibration_curve,
    correction_loss,
    select_formal_bound,
    simulate_policy,
)


class FeatureAndTargetTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "problem_id": ["p1", "p1", "p1", "p2", "p2"],
                "checkpoint": [64, 80, 96, 64, 96],
                "prefix_mean_entropy_tail8": [1, 2, 3, 4, 5],
                "current_prediction": ["A", "B", "B", "A", "A"],
                "dense_prediction": ["B", "B", "B", "B", "B"],
                "current_success": [False, True, True, True, True],
                "dense_success": [True, True, True, False, False],
            }
        )
        self.frame = add_targets(self.frame)
        self.hidden = np.zeros((5, 3, 2560), dtype=np.float32)
        self.hidden[:, 1, 0] = [1, 3, 6, 10, 14]

    def test_last_switch_definition(self):
        self.assertEqual(
            self.frame.target_last_switch.tolist(),
            [False, True, True, False, False],
        )
        self.assertEqual(
            self.frame.target_correction.tolist(),
            [True, False, False, False, False],
        )

    def test_feature_dimensions_and_trajectory_reset(self):
        full = build_features(self.frame, self.hidden, [8, 20, 35])
        self.assertEqual(full.shape, (5, 5126))
        # First checkpoint of each problem has a zero hidden delta.
        self.assertEqual(float(full[0, 2560]), 0.0)
        self.assertEqual(float(full[3, 2560]), 0.0)
        self.assertEqual(build_features(
            self.frame, self.hidden, [8, 20, 35], feature_kind="h_only"
        ).shape[1], 2560)
        self.assertEqual(build_features(
            self.frame, self.hidden, [8, 20, 35], feature_kind="h_delta"
        ).shape[1], 5120)
        self.assertEqual(build_features(
            self.frame, self.hidden, [8, 20, 35], feature_kind="full_no_entropy"
        ).shape[1], 5125)
        self.assertEqual(build_features(
            self.frame, self.hidden, [8, 20, 35], feature_kind="full_no_position"
        ).shape[1], 5123)

    def test_online_feature_matches_offline_ordering(self):
        offline = build_features(self.frame, self.hidden, [8, 20, 35])
        first = build_online_feature(
            self.hidden[0, 1], None, 64, 0,
            float(self.frame.iloc[0].prefix_mean_entropy_tail8),
        )
        second = build_online_feature(
            self.hidden[1, 1], self.hidden[0, 1], 80, 64,
            float(self.frame.iloc[1].prefix_mean_entropy_tail8),
        )
        np.testing.assert_allclose(first[0], offline[0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(second[0], offline[1], rtol=0, atol=1e-6)




class LossAndCalibrationTest(unittest.TestCase):
    def test_trajectory_soft_min_adds_protection_and_backpropagates(self):
        logits = torch.tensor([-2.0, 1.0, -1.0, 0.5], requires_grad=True)
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        remaining = torch.tensor([0.9, 0.5, 0.2, 0.1])
        offsets = np.asarray([0, 2, 4])
        total, point, trajectory = correction_loss(
            logits, target, remaining, offsets, beta=0.5, trajectory=True
        )
        self.assertGreater(float(trajectory), 0.0)
        self.assertAlmostEqual(float(total), float(point + trajectory), places=6)
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_simulation_uses_first_eligible_checkpoint(self):
        frame = pd.DataFrame(
            {
                "problem_id": ["p1", "p1", "p2"],
                "checkpoint": [64, 96, 64],
                "current_success": [False, True, True],
                "dense_success": [True, True, False],
                "current_prediction": ["1", "2", "A"],
                "dense_prediction": ["2", "2", "B"],
                "gold_answer": ["2", "2", "A"],
                "dense_tokens": [200, 200, 150],
                "branch_tokens": [2, 2, 2],
                "dense_wall_ms": [100, 100, 80],
                "dense_prefill_cuda_ms": [10, 10, 8],
                "prefix_decode_cuda_ms": [20, 30, 18],
                "branch_wall_ms": [2, 2, 2],
                "subject": [None, None, None],
                "category": [None, None, None],
            }
        )
        result = simulate_policy(
            frame, np.asarray([0.1, 0.05, 0.9]), "low", 0.1, include_records=True
        )
        self.assertEqual(result["records"][0]["checkpoint"], 64)
        self.assertEqual(result["counts"]["W_to_C"], 1)
        self.assertEqual(result["counts"]["C_to_W"], 0)


    def test_fallback_only_problem_remains_in_denominator(self):
        frame = pd.DataFrame(
            {
                "problem_id": ["p1"],
                "checkpoint": [64],
                "current_success": [True],
                "dense_success": [True],
                "current_prediction": ["1"],
                "dense_prediction": ["1"],
                "gold_answer": ["1"],
                "dense_tokens": [100],
                "branch_tokens": [2],
                "dense_wall_ms": [50],
                "dense_prefill_cuda_ms": [5],
                "prefix_decode_cuda_ms": [10],
                "branch_wall_ms": [2],
                "subject": [None],
                "category": [None],
            }
        )
        result = simulate_policy(
            frame,
            np.asarray([0.1]),
            "low",
            0.2,
            fallback_records=[{
                "problem_id": "p2",
                "subject": None,
                "category": None,
                "gold_answer": "2",
                "dense_prediction": "2",
                "dense_success": True,
                "dense_tokens": 80,
                "dense_wall_ms": 40,
            }],
        )
        self.assertEqual(result["problems"], 2)
        self.assertEqual(result["fallback"], 1)
        self.assertEqual(result["coverage"], 0.5)


    def test_bonferroni_bound_and_dense_fallback(self):
        bound = binomial_upper_simultaneous(
            0, 1000, confidence=0.95, grid_size=101
        )
        self.assertGreater(bound, 0.0)
        frame = pd.DataFrame(
            {
                "problem_id": ["p1", "p2"],
                "checkpoint": [64, 64],
                "current_success": [False, False],
                "dense_success": [True, True],
                "current_prediction": ["0", "0"],
                "dense_prediction": ["1", "1"],
                "gold_answer": ["1", "1"],
                "dense_tokens": [100, 100],
                "branch_tokens": [2, 2],
                "dense_wall_ms": [50, 50],
                "dense_prefill_cuda_ms": [5, 5],
                "prefix_decode_cuda_ms": [10, 10],
                "branch_wall_ms": [2, 2],
                "subject": [None, None],
                "category": [None, None],
            }
        )
        curve = calibration_curve(
            frame, np.asarray([0.1, 0.2]), "low", grid_size=5, confidence=0.95
        )
        selected = select_formal_bound(curve, alpha=0.01)
        self.assertTrue(selected["dense_fallback"])
        self.assertEqual(selected["coverage"], 0.0)
        self.assertEqual(selected["simultaneous_upper_95"], 0.0)


if __name__ == "__main__":
    unittest.main()

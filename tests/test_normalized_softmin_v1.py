#!/usr/bin/env python3
"""Regression tests for count-normalized W->C trajectory protection."""

from __future__ import annotations

import numpy as np
import torch

from src.legacy_empirical_probe_normalized_v1 import correction_loss


def evaluate(count: int, *, normalized: bool, weight: float = 1.0):
    logits = torch.zeros(count, dtype=torch.float32, requires_grad=True)
    target = torch.ones(count, dtype=torch.float32)
    remaining = torch.zeros(count, dtype=torch.float32)
    total, point, trajectory = correction_loss(
        logits,
        target,
        remaining,
        np.asarray([0, count]),
        beta=0.5,
        trajectory=True,
        normalize_by_count=normalized,
        trajectory_weight=weight,
    )
    total.backward()
    assert torch.isfinite(logits.grad).all()
    return float(total), float(point), float(trajectory)


def main() -> None:
    normalized = [evaluate(count, normalized=True) for count in (1, 2, 8, 48)]
    trajectory_values = [row[2] for row in normalized]
    assert max(trajectory_values) - min(trajectory_values) < 1e-6, trajectory_values

    unnormalized = [evaluate(count, normalized=False)[2] for count in (1, 2, 8, 48)]
    assert all(left < right for left, right in zip(unnormalized, unnormalized[1:])), unnormalized

    total, point, trajectory = evaluate(8, normalized=True, weight=0.0)
    assert abs(total - point) < 1e-6
    assert trajectory > 0

    try:
        correction_loss(
            torch.zeros(1), torch.ones(1), torch.zeros(1), np.asarray([0, 1]),
            beta=0.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("beta=0 must be rejected")

    print({
        "status": "passed",
        "normalized_equal_logit_trajectory_loss": trajectory_values,
        "unnormalized_equal_logit_trajectory_loss": unnormalized,
        "trajectory_weight_zero_matches_point": True,
    })


if __name__ == "__main__":
    main()

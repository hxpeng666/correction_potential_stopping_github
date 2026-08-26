#!/usr/bin/env python3
"""Paired problem bootstrap for GSM8K B=2 policy outcomes."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from src.utils import load_yaml


SCHEDULES = ("sentence", "fixed_budget", "prefix_stride", "lynx_cue", "paragraph", "hybrid")
TARGETS = ("correctness", "consistency", "last_switch", "correction_bce")


def arrays(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    records = sorted(artifact["records"]["empirical_B"]["2"], key=lambda row: str(row["problem_id"]))
    return (
        [str(row["problem_id"]) for row in records],
        np.asarray([bool(row["method_success"]) for row in records], dtype=float),
        np.asarray([bool(row["dense_success"]) for row in records], dtype=float),
        np.asarray([float(row["method_tokens"]) for row in records], dtype=float),
        np.asarray([float(row["dense_tokens"]) for row in records], dtype=float),
    )


def interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [.025, .975])]


def bootstrap_single(method_success: np.ndarray, dense_success: np.ndarray, method_tokens: np.ndarray, dense_tokens: np.ndarray, seed: int, replicates: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    accuracy_delta = np.empty(replicates, dtype=float)
    token_reduction = np.empty(replicates, dtype=float)
    n = len(method_success)
    cursor = 0
    while cursor < replicates:
        width = min(250, replicates - cursor)
        index = rng.integers(0, n, size=(width, n))
        accuracy_delta[cursor : cursor + width] = (method_success[index] - dense_success[index]).mean(axis=1)
        token_reduction[cursor : cursor + width] = 1.0 - method_tokens[index].mean(axis=1) / dense_tokens[index].mean(axis=1)
        cursor += width
    return {
        "problems": n,
        "accuracy_delta_pp": 100.0 * float((method_success - dense_success).mean()),
        "accuracy_delta_pp_ci95": [100.0 * value for value in interval(accuracy_delta)],
        "token_reduction": float(1.0 - method_tokens.mean() / dense_tokens.mean()),
        "token_reduction_ci95": interval(token_reduction),
    }


def bootstrap_pair(left: tuple, right: tuple, seed: int, replicates: int) -> dict[str, Any]:
    left_ids, left_success, _dense_success, left_tokens, _dense_tokens = left
    right_ids, right_success, _right_dense, right_tokens, _right_dense_tokens = right
    if left_ids != right_ids:
        raise ValueError("paired policy problem ids differ")
    rng = np.random.default_rng(seed)
    n = len(left_ids)
    accuracy = np.empty(replicates, dtype=float)
    tokens = np.empty(replicates, dtype=float)
    cursor = 0
    while cursor < replicates:
        width = min(250, replicates - cursor)
        index = rng.integers(0, n, size=(width, n))
        accuracy[cursor : cursor + width] = (left_success[index] - right_success[index]).mean(axis=1)
        tokens[cursor : cursor + width] = (left_tokens[index] - right_tokens[index]).mean(axis=1)
        cursor += width
    return {
        "problems": n,
        "left_minus_right_accuracy_pp": 100.0 * float((left_success - right_success).mean()),
        "left_minus_right_accuracy_pp_ci95": [100.0 * value for value in interval(accuracy)],
        "left_minus_right_mean_tokens": float((left_tokens - right_tokens).mean()),
        "left_minus_right_mean_tokens_ci95": interval(tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    output_root = ROOT / config["output_root"]
    matrix_rows = list(csv.DictReader((output_root / "checkpoint_probe_matrix.csv").open(encoding="utf-8")))
    calibration_b2 = [row for row in matrix_rows if int(row["budget_B"]) == 2]
    best_by_target = {
        target: min(
            (row for row in calibration_b2 if row["target"] == target),
            key=lambda row: (float(row["calibration_mean_reasoning_and_answer_tokens"]), float(row["mean_available_checkpoints_calibration"]), row["schedule"]),
        )["schedule"]
        for target in TARGETS
    }
    cached: dict[tuple[str, str], tuple] = {}
    per_combination = {}
    seed = int(config["seed"]["bootstrap"])
    for schedule_index, schedule in enumerate(SCHEDULES):
        for target_index, target in enumerate(TARGETS):
            values = arrays(output_root / "probes" / schedule / target / "policy_records.pt")
            cached[schedule, target] = values
            per_combination[f"{schedule}/{target}"] = bootstrap_single(*values[1:], seed + 101 * schedule_index + target_index, args.replicates)
    versus_sentence = {}
    for target_index, target in enumerate(TARGETS):
        winner = best_by_target[target]
        versus_sentence[target] = {
            "calibration_selected_schedule": winner,
            "comparison": bootstrap_pair(cached[winner, target], cached["sentence", target], seed + 10000 + target_index, args.replicates),
        }
    output = {
        "status": "complete",
        "budget_B": 2,
        "replicates": args.replicates,
        "seed": seed,
        "selection": "calibration only",
        "per_combination": per_combination,
        "calibration_winner_versus_sentence": versus_sentence,
    }
    (output_root / "checkpoint_probe_bootstrap.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

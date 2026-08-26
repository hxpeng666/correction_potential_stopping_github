#!/usr/bin/env python3
"""Summarize normalized BCE+trajectory schedule probes and compare with BCE."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = (
    "sentence",
    "fixed_budget",
    "prefix_stride",
    "lynx_cue",
    "paragraph",
    "hybrid",
)
BUDGETS = (0, 1, 2, 4, 10)
TARGET = "correction_trajectory_normalized"
METRICS = (
    "accuracy",
    "dense_accuracy",
    "accuracy_drop_pp",
    "coverage",
    "fallback_rate",
    "lost_correct_count",
    "lost_correct_rate",
    "mean_reasoning_and_answer_tokens",
    "mean_dense_reasoning_tokens",
    "token_reduction",
    "threshold",
)


def ordinal_ranks(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda name: (values[name], name))
    return {name: index + 1 for index, name in enumerate(ordered)}


def policy_check_count(scores: dict[str, Any], records: list[dict[str, Any]]) -> float:
    by_id: dict[str, list[int]] = {}
    for problem_id, checkpoint in zip(
        scores["problem_ids"]["heldout"],
        scores["checkpoints"]["heldout"],
    ):
        by_id.setdefault(str(problem_id), []).append(int(checkpoint))
    for values in by_id.values():
        values.sort()
    counts = []
    for row in records:
        checkpoints = by_id.get(str(row["problem_id"]), [])
        if row.get("fallback") or row.get("checkpoint") is None:
            counts.append(len(checkpoints))
        else:
            stop = int(row["checkpoint"])
            counts.append(sum(value <= stop for value in checkpoints))
    return float(statistics.mean(counts))


def transitions(records: list[dict[str, Any]]) -> dict[str, int]:
    result = {key: 0 for key in ("W_to_C", "C_to_W", "W_to_W", "C_to_C", "fallback")}
    for row in records:
        key = str(row.get("transition", "fallback"))
        result[key] = result.get(key, 0) + 1
    return result


def bootstrap_pair(
    proposed: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    proposed = sorted(proposed, key=lambda row: str(row["problem_id"]))
    baseline = sorted(baseline, key=lambda row: str(row["problem_id"]))
    proposed_ids = [str(row["problem_id"]) for row in proposed]
    baseline_ids = [str(row["problem_id"]) for row in baseline]
    if proposed_ids != baseline_ids:
        raise ValueError("paired normalized-trajectory/BCE problem ids differ")
    p_success = np.asarray([bool(row["method_success"]) for row in proposed], dtype=float)
    b_success = np.asarray([bool(row["method_success"]) for row in baseline], dtype=float)
    p_tokens = np.asarray([float(row["method_tokens"]) for row in proposed], dtype=float)
    b_tokens = np.asarray([float(row["method_tokens"]) for row in baseline], dtype=float)
    dense_tokens = np.asarray([float(row["dense_tokens"]) for row in proposed], dtype=float)
    rng = np.random.default_rng(seed)
    n = len(proposed)
    accuracy_delta = np.empty(replicates)
    token_delta = np.empty(replicates)
    reduction_delta = np.empty(replicates)
    cursor = 0
    while cursor < replicates:
        width = min(250, replicates - cursor)
        index = rng.integers(0, n, size=(width, n))
        accuracy_delta[cursor : cursor + width] = (
            p_success[index] - b_success[index]
        ).mean(axis=1)
        token_delta[cursor : cursor + width] = (
            p_tokens[index] - b_tokens[index]
        ).mean(axis=1)
        reduction_delta[cursor : cursor + width] = (
            b_tokens[index].mean(axis=1) - p_tokens[index].mean(axis=1)
        ) / dense_tokens[index].mean(axis=1)
        cursor += width
    interval = lambda value: [float(x) for x in np.quantile(value, [.025, .975])]
    return {
        "problems": n,
        "trajectory_minus_bce_accuracy_pp": 100.0 * float((p_success - b_success).mean()),
        "trajectory_minus_bce_accuracy_pp_ci95": [
            100.0 * value for value in interval(accuracy_delta)
        ],
        "trajectory_minus_bce_mean_tokens": float((p_tokens - b_tokens).mean()),
        "trajectory_minus_bce_mean_tokens_ci95": interval(token_delta),
        "trajectory_minus_bce_token_reduction": float(
            (b_tokens.mean() - p_tokens.mean()) / dense_tokens.mean()
        ),
        "trajectory_minus_bce_token_reduction_ci95": interval(reduction_delta),
    }


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    run_root = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
    baseline_root = (
        args.baseline_root if args.baseline_root.is_absolute() else ROOT / args.baseline_root
    )
    old_rows = list(csv.DictReader(
        (baseline_root / "checkpoint_probe_matrix.csv").open(encoding="utf-8")
    ))
    schedule_metadata = {
        schedule: next(row for row in old_rows if row["schedule"] == schedule)
        for schedule in SCHEDULES
    }

    rows: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    errors: list[str] = []
    for schedule_index, schedule in enumerate(SCHEDULES):
        directory = run_root / "probes" / schedule / TARGET
        probe = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
        records_artifact = torch.load(
            directory / "policy_records.pt", map_location="cpu", weights_only=False
        )
        scores = torch.load(directory / "scores.pt", map_location="cpu", weights_only=False)
        spec = probe["run_spec"]
        expected_spec = {
            "actual_schedule_label": schedule,
            "loss": "bce_traj",
            "trajectory_aggregation": "normalized_softmin",
            "trajectory_normalize_by_count": True,
            "trajectory_softmin_beta": 0.5,
            "trajectory_weight": 1.0,
            "calibration_accuracy_epsilon": 0.01,
        }
        for key, expected in expected_spec.items():
            if spec.get(key) != expected:
                errors.append(f"{schedule}: run_spec {key}={spec.get(key)!r}, expected {expected!r}")
        if probe["split_counts"]["heldout"]["problems"] != 1319:
            errors.append(f"{schedule}: heldout problem count mismatch")

        for budget in BUDGETS:
            result = probe["frozen_policy_results"]["empirical_B"][str(budget)]
            local_records = records_artifact["records"]["empirical_B"][str(budget)]
            ids = [str(row["problem_id"]) for row in local_records]
            if len(ids) != 1319 or len(set(ids)) != 1319:
                errors.append(f"{schedule}/B={budget}: policy record identity mismatch")
            row: dict[str, Any] = {
                "schedule": schedule,
                "target": TARGET,
                "budget_B": budget,
                "label_ap_heldout_descriptive": probe["heldout_label_ap_descriptive"],
                "label_auc_heldout_descriptive": probe["heldout_label_auc_descriptive"],
                "mean_available_checkpoints_calibration": float(
                    schedule_metadata[schedule]["mean_available_checkpoints_calibration"]
                ),
                "zero_checkpoint_problems_all": int(
                    schedule_metadata[schedule]["zero_checkpoint_problems_all"]
                ),
                "mean_policy_checks_heldout": policy_check_count(scores, local_records),
            }
            for split in ("calibration", "heldout"):
                for metric in METRICS:
                    row[f"{split}_{metric}"] = result[split][metric]
            if not finite_tree(row):
                errors.append(f"{schedule}/B={budget}: non-finite metric")
            rows.append(row)

        proposed = records_artifact["records"]["empirical_B"]["2"]
        baseline_artifact = torch.load(
            baseline_root / "probes" / schedule / "correction_bce" / "policy_records.pt",
            map_location="cpu",
            weights_only=False,
        )
        baseline = baseline_artifact["records"]["empirical_B"]["2"]
        comparison = bootstrap_pair(
            proposed,
            baseline,
            seed=args.seed + schedule_index,
            replicates=args.replicates,
        )
        comparison["normalized_transitions"] = transitions(proposed)
        comparison["bce_transitions"] = transitions(baseline)
        comparisons[schedule] = comparison

    if errors:
        raise RuntimeError("; ".join(errors))

    matrix_path = run_root / "normalized_trajectory_schedule_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    extended_path = run_root / "checkpoint_probe_matrix_five_targets.csv"
    fieldnames = list(old_rows[0])
    with extended_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerows(rows)

    b2 = [row for row in rows if row["budget_B"] == 2]
    normalized_ranks = ordinal_ranks({
        row["schedule"]: float(row["calibration_mean_reasoning_and_answer_tokens"])
        for row in b2
    })
    normalized_winner = min(
        b2,
        key=lambda row: (
            float(row["calibration_mean_reasoning_and_answer_tokens"]),
            float(row["mean_available_checkpoints_calibration"]),
            row["schedule"],
        ),
    )["schedule"]

    all_rows: list[dict[str, Any]] = [*old_rows, *rows]
    targets = (
        "correctness",
        "consistency",
        "last_switch",
        "correction_bce",
        TARGET,
    )
    five_target_ranks = {
        target: ordinal_ranks({
            row["schedule"]: float(row["calibration_mean_reasoning_and_answer_tokens"])
            for row in all_rows
            if int(row["budget_B"]) == 2 and row["target"] == target
        })
        for target in targets
    }
    five_target_mean_rank = {
        schedule: statistics.mean(five_target_ranks[target][schedule] for target in targets)
        for schedule in SCHEDULES
    }
    five_target_ranking = sorted(
        SCHEDULES,
        key=lambda schedule: (
            five_target_mean_rank[schedule],
            float(schedule_metadata[schedule]["mean_available_checkpoints_calibration"]),
            schedule,
        ),
    )

    summary = {
        "status": "complete",
        "selection_data": "GSM8K calibration only",
        "target": TARGET,
        "trajectory_aggregation": "normalized_softmin",
        "trajectory_beta": 0.5,
        "trajectory_weight": 1.0,
        "primary_budget_B": 2,
        "normalized_target_winner": normalized_winner,
        "normalized_target_ranking": sorted(
            SCHEDULES, key=lambda schedule: normalized_ranks[schedule]
        ),
        "normalized_target_ranks": normalized_ranks,
        "five_target_overall_ranking": five_target_ranking,
        "five_target_mean_ranks": five_target_mean_rank,
        "five_target_ranks_by_target": five_target_ranks,
        "B2_rows": {row["schedule"]: row for row in b2},
        "paired_normalized_trajectory_vs_bce_B2": comparisons,
        "bootstrap_replicates": args.replicates,
        "bootstrap_seed": args.seed,
        "matrix_csv": str(matrix_path),
        "extended_five_target_matrix_csv": str(extended_path),
    }
    (run_root / "normalized_trajectory_schedule_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

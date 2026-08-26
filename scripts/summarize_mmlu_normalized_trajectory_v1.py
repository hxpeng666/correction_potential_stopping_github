#!/usr/bin/env python3
"""Summarize the frozen MMLU-Pro normalized-trajectory follow-up."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = ("paragraph", "sentence", "lynx_cue")
TARGET = "correction_trajectory_normalized"
BUDGETS = (0, 1, 2, 4, 10)
METRICS = (
    "accuracy", "dense_accuracy", "accuracy_drop_pp", "coverage",
    "fallback_rate", "lost_correct_count", "lost_correct_rate",
    "mean_reasoning_and_answer_tokens", "mean_dense_reasoning_tokens",
    "token_reduction", "threshold",
)


def bootstrap_pair(
    proposed: list[dict[str, Any]], baseline: list[dict[str, Any]],
    *, seed: int, replicates: int,
) -> dict[str, Any]:
    proposed = sorted(proposed, key=lambda row: str(row["problem_id"]))
    baseline = sorted(baseline, key=lambda row: str(row["problem_id"]))
    if [str(x["problem_id"]) for x in proposed] != [str(x["problem_id"]) for x in baseline]:
        raise ValueError("paired MMLU-Pro problem ids differ")
    ps = np.asarray([bool(x["method_success"]) for x in proposed], dtype=float)
    bs = np.asarray([bool(x["method_success"]) for x in baseline], dtype=float)
    pt = np.asarray([float(x["method_tokens"]) for x in proposed], dtype=float)
    bt = np.asarray([float(x["method_tokens"]) for x in baseline], dtype=float)
    dt = np.asarray([float(x["dense_tokens"]) for x in proposed], dtype=float)
    n = len(proposed)
    rng = np.random.default_rng(seed)
    accuracy = np.empty(replicates)
    token_delta = np.empty(replicates)
    reduction_delta = np.empty(replicates)
    cursor = 0
    while cursor < replicates:
        width = min(250, replicates - cursor)
        index = rng.integers(0, n, size=(width, n))
        accuracy[cursor:cursor + width] = (ps[index] - bs[index]).mean(axis=1)
        token_delta[cursor:cursor + width] = (pt[index] - bt[index]).mean(axis=1)
        reduction_delta[cursor:cursor + width] = (
            bt[index].mean(axis=1) - pt[index].mean(axis=1)
        ) / dt[index].mean(axis=1)
        cursor += width
    interval = lambda x: [float(v) for v in np.quantile(x, [.025, .975])]
    return {
        "problems": n,
        "trajectory_minus_bce_accuracy_pp": 100.0 * float((ps - bs).mean()),
        "trajectory_minus_bce_accuracy_pp_ci95": [100.0 * x for x in interval(accuracy)],
        "trajectory_minus_bce_mean_tokens": float((pt - bt).mean()),
        "trajectory_minus_bce_mean_tokens_ci95": interval(token_delta),
        "trajectory_minus_bce_token_reduction": float((bt.mean() - pt.mean()) / dt.mean()),
        "trajectory_minus_bce_token_reduction_ci95": interval(reduction_delta),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--gsm-summary", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    run_root = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
    baseline_root = args.baseline_root if args.baseline_root.is_absolute() else ROOT / args.baseline_root
    gsm_summary_path = args.gsm_summary if args.gsm_summary.is_absolute() else ROOT / args.gsm_summary
    gsm_summary = json.loads(gsm_summary_path.read_text(encoding="utf-8"))
    if gsm_summary.get("normalized_target_winner") != "paragraph":
        raise RuntimeError("GSM8K normalized-trajectory candidate selection changed")

    rows: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    for schedule_index, schedule in enumerate(SCHEDULES):
        directory = run_root / "probes" / schedule / TARGET
        probe = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
        records = torch.load(directory / "policy_records.pt", map_location="cpu", weights_only=False)
        spec = probe["run_spec"]
        expected = {
            "actual_schedule_label": schedule,
            "loss": "bce_traj",
            "trajectory_aggregation": "normalized_softmin",
            "trajectory_normalize_by_count": True,
            "trajectory_softmin_beta": 0.5,
            "trajectory_weight": 1.0,
            "calibration_accuracy_epsilon": 0.01,
        }
        if any(spec.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"{schedule}: normalized trajectory run spec mismatch")
        if probe["split_counts"]["heldout"]["problems"] != 1000:
            raise RuntimeError(f"{schedule}: heldout size mismatch")
        for budget in BUDGETS:
            result = probe["frozen_policy_results"]["empirical_B"][str(budget)]
            row: dict[str, Any] = {
                "schedule": schedule, "target": TARGET, "budget_B": budget,
                "label_ap_heldout_descriptive": probe["heldout_label_ap_descriptive"],
                "label_auc_heldout_descriptive": probe["heldout_label_auc_descriptive"],
            }
            for split in ("calibration", "heldout"):
                for metric in METRICS:
                    row[f"{split}_{metric}"] = result[split][metric]
            rows.append(row)
        proposed = records["records"]["empirical_B"]["2"]
        baseline_records = torch.load(
            baseline_root / "probes" / schedule / "correction_bce" / "policy_records.pt",
            map_location="cpu", weights_only=False,
        )["records"]["empirical_B"]["2"]
        comparisons[schedule] = bootstrap_pair(
            proposed, baseline_records,
            seed=args.seed + schedule_index, replicates=args.replicates,
        )

    matrix_path = run_root / "mmlu_pro_normalized_trajectory_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    b2 = {row["schedule"]: row for row in rows if row["budget_B"] == 2}
    summary = {
        "status": "complete",
        "selection_frozen_on": "GSM8K calibration only",
        "candidate_rationale": {
            "paragraph": "GSM8K normalized-trajectory calibration winner",
            "sentence": "current sentence-checkpoint method anchor",
            "lynx_cue": "lowest-checkpoint literature-style anchor",
        },
        "no_mmlu_pro_candidate_reselection": True,
        "target": TARGET,
        "trajectory_aggregation": "normalized_softmin",
        "trajectory_beta": 0.5,
        "trajectory_weight": 1.0,
        "primary_budget_B": 2,
        "B2_rows": b2,
        "paired_normalized_trajectory_vs_bce_B2": comparisons,
        "bootstrap_replicates": args.replicates,
        "bootstrap_seed": args.seed,
        "matrix_csv": str(matrix_path),
    }
    (run_root / "mmlu_pro_normalized_trajectory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

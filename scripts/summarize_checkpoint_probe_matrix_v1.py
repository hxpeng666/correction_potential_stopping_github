#!/usr/bin/env python3
"""Summarize the frozen checkpoint matrix and select MMLU-Pro candidates."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.utils import load_yaml


SCHEDULES = ("sentence", "fixed_budget", "prefix_stride", "lynx_cue", "paragraph", "hybrid")
TARGETS = ("correctness", "consistency", "last_switch", "correction_bce")
BUDGETS = (0, 1, 2, 4, 10)


def metric(source: dict[str, Any], key: str) -> Any:
    value = source.get(key)
    return value if value is not None else float("nan")


def ordinal_ranks(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda name: (values[name], name))
    return {name: index + 1 for index, name in enumerate(ordered)}


CHECKPOINT_CACHE: dict[str, dict[str, list[int]]] = {}


def policy_check_count(cache_root: Path, records: list[dict[str, Any]]) -> float:
    key = str(cache_root.resolve())
    if key not in CHECKPOINT_CACHE:
        by_id: dict[str, list[int]] = {}
        for path in (cache_root / "heldout").glob("sample_*.pt"):
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            by_id[str(artifact["problem_id"])] = [int(row["checkpoint"]) for row in artifact["rows"]]
        CHECKPOINT_CACHE[key] = by_id
    by_id = CHECKPOINT_CACHE[key]
    counts = []
    for row in records:
        checkpoints = by_id[str(row["problem_id"])]
        if row.get("fallback") or row.get("checkpoint") is None:
            counts.append(len(checkpoints))
        else:
            stop = int(row["checkpoint"])
            counts.append(sum(value <= stop for value in checkpoints))
    return float(statistics.mean(counts)) if counts else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    output_root = ROOT / config["output_root"]
    audit = json.loads((output_root / "cache_audit.json").read_text(encoding="utf-8"))
    matrix_summary = json.loads((output_root / "probe_matrix_summary.json").read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or matrix_summary.get("status") != "complete":
        raise RuntimeError("cache/probe matrix is not complete")

    rows: list[dict[str, Any]] = []
    probes: dict[tuple[str, str], dict[str, Any]] = {}
    for schedule in SCHEDULES:
        for target in TARGETS:
            directory = output_root / "probes" / schedule / target
            probe = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
            if probe.get("status") != "complete" or probe["split_counts"]["heldout"]["problems"] != 1319:
                raise RuntimeError(f"incomplete probe: {schedule}/{target}")
            probes[schedule, target] = probe
            record_artifact = torch.load(directory / "policy_records.pt", map_location="cpu", weights_only=False)
            for budget in BUDGETS:
                result = probe["frozen_policy_results"]["empirical_B"][str(budget)]
                heldout_records = record_artifact["records"]["empirical_B"][str(budget)]
                row = {
                    "schedule": schedule,
                    "target": target,
                    "budget_B": budget,
                    "label_ap_heldout_descriptive": probe["heldout_label_ap_descriptive"],
                    "label_auc_heldout_descriptive": probe["heldout_label_auc_descriptive"],
                    "mean_available_checkpoints_calibration": audit["schedules"][schedule]["splits"]["calibration"]["checkpoint_count"]["mean"],
                    "zero_checkpoint_problems_all": audit["schedules"][schedule]["zero_checkpoint_problems"],
                    "mean_policy_checks_heldout": policy_check_count(output_root / "cache" / schedule, heldout_records),
                }
                for split in ("calibration", "heldout"):
                    values = result[split]
                    for key in ("accuracy", "dense_accuracy", "accuracy_drop_pp", "coverage", "fallback_rate", "lost_correct_count", "lost_correct_rate", "mean_reasoning_and_answer_tokens", "mean_dense_reasoning_tokens", "token_reduction", "threshold"):
                        row[f"{split}_{key}"] = metric(values, key)
                rows.append(row)

    csv_path = output_root / "checkpoint_probe_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    b2 = [row for row in rows if row["budget_B"] == 2]
    ranks_by_target: dict[str, dict[str, int]] = {}
    for target in TARGETS:
        ranks_by_target[target] = ordinal_ranks({row["schedule"]: float(row["calibration_mean_reasoning_and_answer_tokens"]) for row in b2 if row["target"] == target})
    mean_rank = {schedule: statistics.mean(ranks_by_target[target][schedule] for target in TARGETS) for schedule in SCHEDULES}
    calibration_checks = {schedule: float(audit["schedules"][schedule]["splits"]["calibration"]["checkpoint_count"]["mean"]) for schedule in SCHEDULES}
    b1_cost = {schedule: statistics.mean(float(row["calibration_mean_reasoning_and_answer_tokens"]) for row in rows if row["schedule"] == schedule and row["budget_B"] == 1) for schedule in SCHEDULES}
    ranking = sorted(SCHEDULES, key=lambda schedule: (mean_rank[schedule], calibration_checks[schedule], b1_cost[schedule], schedule))
    top_three = ranking[:3]
    low_check = min(top_three, key=lambda schedule: (calibration_checks[schedule], mean_rank[schedule]))
    ordered_pairs = sorted(
        b2,
        key=lambda row: (
            float(row["calibration_mean_reasoning_and_answer_tokens"]),
            float(row["calibration_accuracy_drop_pp"]),
            float(row["mean_available_checkpoints_calibration"]),
            row["schedule"],
            row["target"],
        ),
    )
    best_by_target = {
        target: min(
            (row for row in b2 if row["target"] == target),
            key=lambda row: (
                float(row["calibration_mean_reasoning_and_answer_tokens"]),
                float(row["mean_available_checkpoints_calibration"]),
                row["schedule"],
            ),
        )
        for target in TARGETS
    }
    proposed_pairs = [
        *ordered_pairs[:2],
        best_by_target["correction_bce"],
        best_by_target["correctness"],
        {"schedule": "sentence", "target": "correction_bce"},
        {"schedule": "lynx_cue", "target": "correctness"},
    ]
    mmlu_combinations = []
    seen_pairs = set()
    for row in proposed_pairs:
        pair = (str(row["schedule"]), str(row["target"]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            mmlu_combinations.append({"schedule": pair[0], "target": pair[1]})
    mmlu_candidates = []
    for row in mmlu_combinations:
        if row["schedule"] not in mmlu_candidates:
            mmlu_candidates.append(row["schedule"])

    ranking_rows = []
    for index, schedule in enumerate(ranking, 1):
        local_b2 = [row for row in b2 if row["schedule"] == schedule]
        ranking_rows.append({
            "overall_rank": index,
            "schedule": schedule,
            "mean_target_rank_calibration_B2": mean_rank[schedule],
            "target_ranks": {target: ranks_by_target[target][schedule] for target in TARGETS},
            "mean_available_checkpoints_calibration": calibration_checks[schedule],
            "mean_calibration_tokens_B2": statistics.mean(float(row["calibration_mean_reasoning_and_answer_tokens"]) for row in local_b2),
            "mean_heldout_tokens_B2_descriptive": statistics.mean(float(row["heldout_mean_reasoning_and_answer_tokens"]) for row in local_b2),
            "mean_heldout_accuracy_drop_pp_B2_descriptive": statistics.mean(float(row["heldout_accuracy_drop_pp"]) for row in local_b2),
            "mean_policy_checks_heldout_B2_descriptive": statistics.mean(float(row["mean_policy_checks_heldout"]) for row in local_b2),
        })
    mean_calibration_tokens = {
        row["schedule"]: float(row["mean_calibration_tokens_B2"])
        for row in ranking_rows
    }
    best_mean_tokens = min(mean_calibration_tokens.values())
    lightweight_feasible = [
        schedule for schedule in SCHEDULES
        if mean_calibration_tokens[schedule] <= 1.05 * best_mean_tokens
    ]
    lightweight_winner = min(
        lightweight_feasible,
        key=lambda schedule: (calibration_checks[schedule], mean_calibration_tokens[schedule], schedule),
    )
    summary = {
        "status": "complete",
        "selection_data": "GSM8K calibration only",
        "primary_budget_B": 2,
        "ranking_rule": config["comparison"]["schedule_ranking"],
        "winner": ranking[0],
        "lightweight_winner": lightweight_winner,
        "lightweight_feasible_schedules": lightweight_feasible,
        "ranking": ranking_rows,
        "mmlu_pro_schedule_candidates_frozen": mmlu_candidates,
        "mmlu_pro_combinations_frozen": mmlu_combinations,
        "heldout_is_descriptive_not_used_for_selection": True,
        "matrix_csv": str(csv_path),
    }
    (output_root / "checkpoint_schedule_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed audit for the MMLU-Pro normalized trajectory follow-up."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = ("paragraph", "sentence", "lynx_cue")
TARGET = "correction_trajectory_normalized"
BUDGETS = (0, 1, 2, 4, 10)


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
    args = parser.parse_args()
    run_root = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
    errors: list[str] = []
    controller = json.loads((run_root / "probe_matrix_summary.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "mmlu_pro_normalized_trajectory_summary.json").read_text(encoding="utf-8"))
    if controller.get("status") != "complete" or controller.get("failures"):
        errors.append("controller incomplete or has failures")
    if controller.get("selection_frozen_on") != "GSM8K calibration only":
        errors.append("candidate selection source mismatch")
    if summary.get("status") != "complete" or not summary.get("no_mmlu_pro_candidate_reselection"):
        errors.append("summary selection invariant mismatch")

    probes = []
    for schedule in SCHEDULES:
        directory = run_root / "probes" / schedule / TARGET
        required = [directory / name for name in (
            "phase.complete", "probe.json", "probe.pt", "scores.pt", "policy_records.pt"
        )]
        if any(not path.is_file() for path in required):
            errors.append(f"{schedule}: missing probe artifacts")
            continue
        marker = json.loads((directory / "phase.complete").read_text(encoding="utf-8"))
        probe = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
        records = torch.load(directory / "policy_records.pt", map_location="cpu", weights_only=False)
        expected_spec = {
            "actual_schedule_label": schedule,
            "method": "correction",
            "loss": "bce_traj",
            "trajectory_aggregation": "normalized_softmin",
            "trajectory_normalize_by_count": True,
            "trajectory_softmin_beta": 0.5,
            "trajectory_weight": 1.0,
            "calibration_accuracy_epsilon": 0.01,
        }
        if marker.get("status") != "complete":
            errors.append(f"{schedule}: incomplete phase marker")
        for key, expected in expected_spec.items():
            if probe["run_spec"].get(key) != expected:
                errors.append(f"{schedule}: run spec {key} mismatch")
        expected_counts = {"probe_train": 1000, "calibration": 500, "heldout": 1000}
        actual_counts = {
            split: probe["split_counts"][split]["problems"] for split in expected_counts
        }
        if actual_counts != expected_counts:
            errors.append(f"{schedule}: split counts mismatch")
        for budget in BUDGETS:
            local = records["records"]["empirical_B"][str(budget)]
            ids = [str(row["problem_id"]) for row in local]
            if len(ids) != 1000 or len(set(ids)) != 1000:
                errors.append(f"{schedule}/B={budget}: heldout record mismatch")
            if not finite_tree(probe["frozen_policy_results"]["empirical_B"][str(budget)]):
                errors.append(f"{schedule}/B={budget}: non-finite metrics")
        probes.append({
            "schedule": schedule,
            "best_epoch": probe["best_epoch"],
            "run_spec_fingerprint": probe["run_spec_fingerprint"],
            "valid": True,
        })

    rows = list(csv.DictReader(
        (run_root / "mmlu_pro_normalized_trajectory_matrix.csv").open(encoding="utf-8")
    ))
    if len(rows) != 15:
        errors.append(f"matrix has {len(rows)} rows, expected 15")
    pattern = re.compile(
        r"Traceback|CUDA out of memory|RuntimeError|AssertionError|\bnan\b|\binf\b",
        re.IGNORECASE,
    )
    log_hits = []
    for path in sorted((run_root / "logs").glob("probe_*.log")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                log_hits.append({"file": str(path), "line": line_number, "text": line[:300]})
    if log_hits:
        errors.append(f"training logs contain {len(log_hits)} error-like lines")
    report = {
        "status": "complete" if not errors else "failed",
        "protocol": "MMLU-Pro GSM8K-frozen normalized correction BCE+trajectory v1",
        "checks": {
            "probes": probes,
            "matrix_rows": len(rows),
            "bootstrap_replicates": summary.get("bootstrap_replicates"),
            "no_mmlu_pro_candidate_reselection": summary.get("no_mmlu_pro_candidate_reselection"),
            "log_error_hits": log_hits,
        },
        "errors": errors,
    }
    (run_root / "FINAL_AUDIT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

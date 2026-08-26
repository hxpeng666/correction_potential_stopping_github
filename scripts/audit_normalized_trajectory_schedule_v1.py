#!/usr/bin/env python3
"""Fail-closed audit for the normalized trajectory schedule ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

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
TARGET = "correction_trajectory_normalized"
BUDGETS = (0, 1, 2, 4, 10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    matrix_summary = json.loads(
        (run_root / "probe_matrix_summary.json").read_text(encoding="utf-8")
    )
    result_summary = json.loads(
        (run_root / "normalized_trajectory_schedule_summary.json").read_text(encoding="utf-8")
    )
    if matrix_summary.get("status") != "complete" or matrix_summary.get("failures"):
        errors.append("probe controller did not complete cleanly")
    if result_summary.get("status") != "complete":
        errors.append("result summarizer did not complete cleanly")

    probe_checks = []
    for schedule in SCHEDULES:
        directory = run_root / "probes" / schedule / TARGET
        required = [
            directory / "phase.complete",
            directory / "probe.json",
            directory / "probe.pt",
            directory / "scores.pt",
            directory / "policy_records.pt",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            errors.append(f"{schedule}: missing {missing}")
            continue
        marker = json.loads((directory / "phase.complete").read_text(encoding="utf-8"))
        probe = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
        records = torch.load(
            directory / "policy_records.pt", map_location="cpu", weights_only=False
        )
        spec = probe.get("run_spec", {})
        exact = {
            "actual_schedule_label": schedule,
            "method": "correction",
            "loss": "bce_traj",
            "trajectory_aggregation": "normalized_softmin",
            "trajectory_normalize_by_count": True,
            "trajectory_softmin_beta": 0.5,
            "trajectory_weight": 1.0,
            "calibration_accuracy_epsilon": 0.01,
        }
        for key, expected in exact.items():
            if spec.get(key) != expected:
                errors.append(f"{schedule}: {key} mismatch")
        if marker.get("status") != "complete":
            errors.append(f"{schedule}: phase marker incomplete")
        expected_counts = {"probe_train": 1000, "calibration": 500, "heldout": 1319}
        actual_counts = {
            split: probe["split_counts"][split]["problems"] for split in expected_counts
        }
        if actual_counts != expected_counts:
            errors.append(f"{schedule}: split counts {actual_counts}")
        for budget in BUDGETS:
            rows = records["records"]["empirical_B"][str(budget)]
            ids = [str(row["problem_id"]) for row in rows]
            if len(ids) != 1319 or len(set(ids)) != 1319:
                errors.append(f"{schedule}/B={budget}: invalid heldout records")
            metrics = probe["frozen_policy_results"]["empirical_B"][str(budget)]
            if not finite_tree(metrics):
                errors.append(f"{schedule}/B={budget}: non-finite metrics")
        probe_checks.append({
            "schedule": schedule,
            "best_epoch": probe["best_epoch"],
            "run_spec_fingerprint": probe["run_spec_fingerprint"],
            "valid": True,
        })

    matrix_rows = list(csv.DictReader(
        (run_root / "normalized_trajectory_schedule_matrix.csv").open(encoding="utf-8")
    ))
    extended_rows = list(csv.DictReader(
        (run_root / "checkpoint_probe_matrix_five_targets.csv").open(encoding="utf-8")
    ))
    if len(matrix_rows) != 30:
        errors.append(f"normalized matrix has {len(matrix_rows)} rows, expected 30")
    if len(extended_rows) != 150:
        errors.append(f"five-target matrix has {len(extended_rows)} rows, expected 150")

    error_pattern = re.compile(
        r"Traceback|CUDA out of memory|RuntimeError|AssertionError|\bnan\b|\binf\b",
        re.IGNORECASE,
    )
    log_hits = []
    for path in sorted((run_root / "logs").glob("probe_*.log")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if error_pattern.search(line):
                log_hits.append({"file": str(path), "line": line_number, "text": line[:300]})
    if log_hits:
        errors.append(f"training logs contain {len(log_hits)} error-like lines")

    source_files = {
        "trainer": ROOT / "scripts/train_controlled_label_normalized_trajectory_v1.py",
        "normalized_loss": ROOT / "src/legacy_empirical_probe_normalized_v1.py",
        "normalization_test": ROOT / "tests/test_normalized_softmin_v1.py",
        "controller": ROOT / "scripts/run_normalized_trajectory_schedule_matrix_v1.py",
        "summarizer": ROOT / "scripts/summarize_normalized_trajectory_schedule_v1.py",
    }
    report = {
        "status": "complete" if not errors else "failed",
        "protocol": "GSM8K six-schedule normalized correction BCE+trajectory v1",
        "checks": {
            "probes": probe_checks,
            "normalized_matrix_rows": len(matrix_rows),
            "extended_five_target_matrix_rows": len(extended_rows),
            "bootstrap_replicates": result_summary.get("bootstrap_replicates"),
            "log_error_hits": log_hits,
            "source_sha256": {
                name: sha256(path) for name, path in source_files.items()
            },
        },
        "errors": errors,
    }
    (run_root / "FINAL_AUDIT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

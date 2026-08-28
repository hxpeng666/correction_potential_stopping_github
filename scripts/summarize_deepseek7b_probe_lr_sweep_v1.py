#!/usr/bin/env python3
"""Audit and summarize the threshold-free DeepSeek-7B probe LR sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DATASETS = ("gsm8k", "math")
METHODS = ("bce", "bce_trajectory")
LEARNING_RATES = (0.000025, 0.00005, 0.0001, 0.0002, 0.0004)


def lr_tag(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".").replace(".", "p")


def forbidden_key_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if "empirical_b" in lowered or "validation_b0" in lowered:
                found.append(path)
            found.extend(forbidden_key_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_key_paths(child, f"{prefix}[{index}]"))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-problems", type=int, default=24)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    audit_errors: list[str] = []
    identities: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    inputs: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}

    for dataset in DATASETS:
        for method in METHODS:
            for learning_rate in LEARNING_RATES:
                run_dir = (
                    args.output_root
                    / dataset
                    / method
                    / f"lr_{lr_tag(learning_rate)}"
                    / "seed_0"
                )
                probe_path = run_dir / "probe.json"
                marker_path = run_dir / "phase.complete"
                if not probe_path.is_file() or not marker_path.is_file():
                    audit_errors.append(f"missing complete run: {run_dir}")
                    continue
                payload = json.loads(probe_path.read_text())
                if payload.get("status") != "complete":
                    audit_errors.append(f"non-complete payload: {probe_path}")
                    continue
                invocation = payload["invocation"]
                if invocation.get("selection_rule") != "validation_objective":
                    audit_errors.append(f"wrong selection rule: {probe_path}")
                actual_lr = float(invocation.get("learning_rate"))
                if not math.isclose(actual_lr, learning_rate, rel_tol=0.0, abs_tol=1e-15):
                    audit_errors.append(f"wrong learning rate: {probe_path}")
                if payload.get("screen_only") is not True:
                    audit_errors.append(f"heldout-capable run in screen: {probe_path}")
                forbidden = forbidden_key_paths(payload)
                if forbidden:
                    audit_errors.append(
                        f"legacy empirical calibration fields in {probe_path}: {forbidden}"
                    )
                history = payload["history"]
                if len(history) != 48:
                    audit_errors.append(
                        f"expected 48 epochs, got {len(history)}: {probe_path}"
                    )
                best_epoch = int(payload["best_epoch"])
                selected = next(row for row in history if int(row["epoch"]) == best_epoch)
                objective_min = min(float(row["validation_objective"]) for row in history)
                if not math.isclose(
                    float(selected["validation_objective"]),
                    objective_min,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    audit_errors.append(f"best epoch is not objective minimum: {probe_path}")
                fit_problems = len(payload["split"]["fit_problem_ids"])
                steps_per_epoch = math.ceil(fit_problems / args.batch_problems)
                reproducibility = payload["reproducibility"]
                identities[dataset].add(reproducibility["initial_state_sha256"])
                inputs[dataset].add(reproducibility["input"]["features_sha256"])
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "learning_rate": learning_rate,
                        "training_seed": invocation["training_seed"],
                        "split_seed": invocation["split_seed"],
                        "best_epoch_zero_based": best_epoch,
                        "best_optimizer_step": (best_epoch + 1) * steps_per_epoch,
                        "validation_objective": float(selected["validation_objective"]),
                        "validation_point_loss": float(selected["validation_point_loss"]),
                        "validation_protect_loss": float(selected["validation_protect_loss"]),
                        "validation_ap": float(selected["validation_ap"]),
                        "validation_auc": float(selected["validation_auc"]),
                        "initial_state_sha256": reproducibility["initial_state_sha256"],
                        "final_state_sha256": reproducibility["final_state_sha256"],
                        "invocation_fingerprint": payload["invocation_fingerprint"],
                    }
                )

    for dataset in DATASETS:
        if len(identities[dataset]) != 1:
            audit_errors.append(
                f"initialization mismatch for {dataset}: {len(identities[dataset])} hashes"
            )
        if len(inputs[dataset]) != 1:
            audit_errors.append(
                f"feature-input mismatch for {dataset}: {len(inputs[dataset])} hashes"
            )

    expected = len(DATASETS) * len(METHODS) * len(LEARNING_RATES)
    if len(rows) != expected:
        audit_errors.append(f"expected {expected} rows, got {len(rows)}")
    status = "complete" if not audit_errors else "failed"
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "RESULTS_VALIDATION.csv"
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    (args.output_root / "RESULTS_VALIDATION.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_root / "AUDIT.json").write_text(
        json.dumps(
            {
                "status": status,
                "expected_runs": expected,
                "completed_runs": len(rows),
                "errors": audit_errors,
                "selection_uses_calibration": False,
                "selection_uses_heldout": False,
                "selection_uses_empirical_B": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    if audit_errors:
        raise SystemExit("; ".join(audit_errors))
    print(json.dumps({"status": status, "runs": len(rows), "csv": str(csv_path)}))


if __name__ == "__main__":
    main()

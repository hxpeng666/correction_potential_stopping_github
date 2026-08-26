#!/usr/bin/env python3
"""Report frozen MMLU-Pro follow-up combinations without re-selection."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from src.utils import load_yaml


def bootstrap(records: list[dict], seed: int, replicates: int = 10000) -> dict:
    records = sorted(records, key=lambda row: str(row["problem_id"]))
    method_success = np.asarray([bool(row["method_success"]) for row in records], dtype=float)
    dense_success = np.asarray([bool(row["dense_success"]) for row in records], dtype=float)
    method_tokens = np.asarray([float(row["method_tokens"]) for row in records], dtype=float)
    dense_tokens = np.asarray([float(row["dense_tokens"]) for row in records], dtype=float)
    rng = np.random.default_rng(seed)
    accuracy = np.empty(replicates)
    reduction = np.empty(replicates)
    cursor = 0
    while cursor < replicates:
        width = min(250, replicates - cursor)
        index = rng.integers(0, len(records), size=(width, len(records)))
        accuracy[cursor : cursor + width] = (method_success[index] - dense_success[index]).mean(axis=1)
        reduction[cursor : cursor + width] = 1.0 - method_tokens[index].mean(axis=1) / dense_tokens[index].mean(axis=1)
        cursor += width
    return {
        "accuracy_delta_pp_ci95": [100.0 * float(x) for x in np.quantile(accuracy, [.025, .975])],
        "token_reduction_ci95": [float(x) for x in np.quantile(reduction, [.025, .975])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gsm-matrix", type=Path, required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    gsm_matrix_path = args.gsm_matrix if args.gsm_matrix.is_absolute() else ROOT / args.gsm_matrix
    config = load_yaml(config_path)
    output_root = ROOT / config["output_root"]
    matrix = json.loads((output_root / "probe_matrix_summary.json").read_text(encoding="utf-8"))
    if matrix.get("status") != "complete" or matrix.get("dataset") != "mmlu_pro":
        raise RuntimeError("MMLU-Pro probe matrix is incomplete")
    gsm_rows = list(csv.DictReader(gsm_matrix_path.open(encoding="utf-8")))
    rows = []
    details = {}
    for index, combination in enumerate(config["comparison"]["mmlu_pro_combinations"]):
        schedule = str(combination["schedule"])
        target = str(combination["target"])
        directory = output_root / "probes" / schedule / target
        probe = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
        policies = torch.load(directory / "policy_records.pt", map_location="cpu", weights_only=False)
        result = probe["frozen_policy_results"]["empirical_B"]["2"]
        gsm = next(row for row in gsm_rows if row["schedule"] == schedule and row["target"] == target and int(row["budget_B"]) == 2)
        row = {"schedule": schedule, "target": target}
        for split in ("calibration", "heldout"):
            for key in ("accuracy", "dense_accuracy", "accuracy_drop_pp", "coverage", "lost_correct_count", "mean_reasoning_and_answer_tokens", "mean_dense_reasoning_tokens", "token_reduction"):
                row[f"mmlu_pro_{split}_{key}"] = result[split][key]
        row["gsm8k_heldout_accuracy_drop_pp_descriptive"] = float(gsm["heldout_accuracy_drop_pp"])
        row["gsm8k_heldout_token_reduction_descriptive"] = float(gsm["heldout_token_reduction"])
        ci = bootstrap(policies["records"]["empirical_B"]["2"], int(config["seed"]["bootstrap"]) + index)
        row.update(ci)
        rows.append(row)
        details[f"{schedule}/{target}"] = {"metrics": row, "selection_source": config["comparison"]["selection_source"]}
    csv_path = output_root / "mmlu_pro_followup.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = {
        "status": "complete",
        "selection_frozen_on": "GSM8K calibration only",
        "no_mmlu_pro_reselection": True,
        "budget_B": 2,
        "combinations": details,
        "csv": str(csv_path),
    }
    (output_root / "mmlu_pro_followup_summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

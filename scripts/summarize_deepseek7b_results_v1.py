#!/usr/bin/env python3
"""Compile all five targets and all empirical-B workpoints into one report."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path


METHODS = ("correctness", "consistency", "last_switch", "bce", "bce_traj")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for dataset in ("gsm8k", "math500", "aime"):
        for method in METHODS:
            path = args.output_root / "probes" / dataset / method / "probe.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            for budget, result in payload["frozen_policy_results"]["empirical_B"].items():
                calibration = result["calibration"]
                heldout = result["heldout"]
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "budget_B": int(budget),
                        "threshold": calibration["threshold"],
                        "calibration_lost_correct": calibration["lost_correct_count"],
                        "test_problems": heldout["problems"],
                        "dense_accuracy": heldout["dense_accuracy"],
                        "accuracy": heldout["accuracy"],
                        "accuracy_delta_pp": 100 * (heldout["accuracy"] - heldout["dense_accuracy"]),
                        "token_reduction": heldout["token_reduction"],
                        "lost_correct_count": heldout["lost_correct_count"],
                        "coverage": heldout["coverage"],
                        "mean_tokens": heldout["mean_reasoning_and_answer_tokens"],
                        "mean_dense_tokens": heldout["mean_dense_reasoning_tokens"],
                    }
                )
    json_path = args.output_root / "RESULTS_ALL_B.json"
    csv_path = args.output_root / "RESULTS_ALL_B.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    b2 = [row for row in rows if row["budget_B"] == 2]
    lines = [
        "# DeepSeek-R1-Distill-Qwen-7B main ablation",
        "",
        "Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        f"- Origin Date: {date.today().isoformat()}",
        "- Verification Evidence: COMPLETION_AUDIT.json (required before final reporting)",
        "- Version: exp_result_v1",
        "",
        "MATH-500 and AIME use the category-balanced Hendrycks MATH fit/calibration split and are OOD held-out evaluations.",
        "",
        "| Dataset | Method | Dense Acc | Acc | Delta pp | Token reduction | Lost | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in b2:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['dense_accuracy']:.4f} | "
            f"{row['accuracy']:.4f} | {row['accuracy_delta_pp']:+.2f} | "
            f"{100*row['token_reduction']:.2f}% | {row['lost_correct_count']} | "
            f"{100*row['coverage']:.2f}% |"
        )
    (args.output_root / "RESULTS_B2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(rows), "b2_rows": len(b2)}, indent=2))


if __name__ == "__main__":
    main()

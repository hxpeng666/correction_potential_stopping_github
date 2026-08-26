#!/usr/bin/env python3
"""Compare paired forced-answer caches for a suffix-only ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def compare_split(reference: Path, candidate: Path) -> dict[str, Any]:
    reference_paths = {path.name: path for path in reference.glob("*.pt")}
    candidate_paths = {path.name: path for path in candidate.glob("*.pt")}
    common = sorted(reference_paths.keys() & candidate_paths.keys())
    totals = {
        "rows": 0,
        "parsed_reference": 0,
        "parsed_candidate": 0,
        "candidate_parse_gain": 0,
        "candidate_parse_loss": 0,
        "answer_changed": 0,
        "correct_reference": 0,
        "correct_candidate": 0,
        "candidate_correctness_gain": 0,
        "candidate_correctness_loss": 0,
        "correction_reference": 0,
        "correction_candidate": 0,
        "hidden_mismatch_files": 0,
    }
    reference_branch_tokens: list[float] = []
    candidate_branch_tokens: list[float] = []
    for name in common:
        old = torch.load(reference_paths[name], map_location="cpu", weights_only=False)
        new = torch.load(candidate_paths[name], map_location="cpu", weights_only=False)
        old_rows = {int(row["checkpoint"]): row for row in old["rows"]}
        new_rows = {int(row["checkpoint"]): row for row in new["rows"]}
        if old_rows.keys() != new_rows.keys():
            raise ValueError(f"checkpoint mismatch: {name}")
        if not torch.equal(old["hidden"], new["hidden"]):
            totals["hidden_mismatch_files"] += 1
        for checkpoint in sorted(old_rows):
            left, right = old_rows[checkpoint], new_rows[checkpoint]
            old_prediction = left.get("current_prediction")
            new_prediction = right.get("current_prediction")
            old_parsed = old_prediction is not None
            new_parsed = new_prediction is not None
            old_correct = bool(left.get("current_success"))
            new_correct = bool(right.get("current_success"))
            old_correction = bool(left.get("correction"))
            new_correction = bool(right.get("correction"))
            totals["rows"] += 1
            totals["parsed_reference"] += int(old_parsed)
            totals["parsed_candidate"] += int(new_parsed)
            totals["candidate_parse_gain"] += int(new_parsed and not old_parsed)
            totals["candidate_parse_loss"] += int(old_parsed and not new_parsed)
            totals["answer_changed"] += int(old_prediction != new_prediction)
            totals["correct_reference"] += int(old_correct)
            totals["correct_candidate"] += int(new_correct)
            totals["candidate_correctness_gain"] += int(new_correct and not old_correct)
            totals["candidate_correctness_loss"] += int(old_correct and not new_correct)
            totals["correction_reference"] += int(old_correction)
            totals["correction_candidate"] += int(new_correction)
            reference_branch_tokens.append(float(left.get("branch_tokens", 0)))
            candidate_branch_tokens.append(float(right.get("branch_tokens", 0)))
    rows = max(1, totals["rows"])
    return {
        "reference_files": len(reference_paths),
        "candidate_files": len(candidate_paths),
        "paired_files": len(common),
        **totals,
        "parse_rate_reference": totals["parsed_reference"] / rows,
        "parse_rate_candidate": totals["parsed_candidate"] / rows,
        "answer_change_rate": totals["answer_changed"] / rows,
        "checkpoint_accuracy_reference": totals["correct_reference"] / rows,
        "checkpoint_accuracy_candidate": totals["correct_candidate"] / rows,
        "correction_rate_reference": totals["correction_reference"] / rows,
        "correction_rate_candidate": totals["correction_candidate"] / rows,
        "mean_branch_tokens_reference": mean(reference_branch_tokens),
        "mean_branch_tokens_candidate": mean(candidate_branch_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["probe_train", "calibration", "heldout"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "reference_root": str(args.reference_root.resolve()),
        "candidate_root": str(args.candidate_root.resolve()),
        "splits": {
            split: compare_split(args.reference_root / split, args.candidate_root / split)
            for split in args.splits
        },
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

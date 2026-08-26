#!/usr/bin/env python3
"""Audit and summarize the frozen-v2 OOD calibration replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def fmt(row: dict) -> str:
    return (
        f"| {row['method']} | {row['dataset']} | "
        f"{row.get('budget_B') if row.get('budget_B') is not None else row.get('confidence_c')} | "
        f"{row['test_accuracy_delta_pp']:+.2f} | "
        f"{100.0 * row['test_token_reduction']:.2f}% | "
        f"{row['test_lost_correct_count']} | {100.0 * row['test_coverage']:.2f}% |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    source = args.source_root.resolve()
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    rows = json.loads((root / "RESULTS.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    if len(rows) != 134:
        errors.append(f"expected 134 rows, found {len(rows)}")
    identities = {
        (
            row["family"],
            row["method"],
            row["dataset"],
            row.get("budget_B"),
            row.get("confidence_c"),
        )
        for row in rows
    }
    if len(identities) != len(rows):
        errors.append("duplicate result identities")
    if sha256(root / "RESULTS.json") != manifest["results_json_sha256"]:
        errors.append("RESULTS.json hash mismatch")
    if sha256(root / "RESULTS.csv") != manifest["results_csv_sha256"]:
        errors.append("RESULTS.csv hash mismatch")
    if sha256(source / "COMPLETION_AUDIT.json") != manifest["source_completion_audit_sha256"]:
        errors.append("source v2 completion audit changed")
    for method, files in manifest["source_files"].items():
        for label in ("probe_json", "math_scores", "aime_scores"):
            path = Path(files[label])
            if sha256(path) != files[f"{label}_sha256"]:
                errors.append(f"source hash mismatch: {method}/{label}")

    expected_family_counts = {
        "empirical_B_only": 52,
        "empirical_B_with_1pp_guard": 52,
        "lynx_conformal_correctness": 10,
        "lynx_conformal_correction_adapted": 20,
    }
    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in expected_family_counts
    }
    if family_counts != expected_family_counts:
        errors.append(f"family counts mismatch: {family_counts}")

    for row in rows:
        if row["family"] == "empirical_B_only" and row["calibration_lost_correct_count"] > row["budget_B"]:
            errors.append(f"B violation: {row['method']}/{row['dataset']}/B={row['budget_B']}")
        if row["family"] == "empirical_B_with_1pp_guard":
            if row["calibration_lost_correct_count"] > row["budget_B"]:
                errors.append(f"guarded B violation: {row['method']}/{row['dataset']}/B={row['budget_B']}")
            if row["calibration_accuracy_delta_pp"] < -1.000001:
                errors.append(f"1pp guard violation: {row['method']}/{row['dataset']}/B={row['budget_B']}")

    highlights = [
        row
        for row in rows
        if (
            row["family"] == "empirical_B_only"
            and row["budget_B"] in (10, 20, 35)
        )
        or (
            row["family"] == "lynx_conformal_correctness"
            and row["confidence_c"] in (0.97, 0.90, 0.70)
        )
    ]
    lines = [
        "# DeepSeek-7B 13K OOD calibration sweep",
        "",
        "| Method | Dataset | B or c | Δ accuracy (pp) | Token reduction | Lost-correct | Exit coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
        *[fmt(row) for row in highlights],
        "",
        "The LYNX rows transplant only the public class-conditional conformal calibration rule onto the existing paragraph correctness probe; they are not a native-cue LYNX reproduction.",
        "",
    ]
    atomic_text("\n".join(lines), root / "SUMMARY.md")
    audit = {
        "status": "complete" if not errors else "failed",
        "errors": errors,
        "rows": len(rows),
        "unique_rows": len(identities),
        "family_counts": family_counts,
        "source_v2_completion_audit_sha256": sha256(source / "COMPLETION_AUDIT.json"),
        "source_v2_results_all_b_sha256": sha256(source / "RESULTS_ALL_B.json"),
        "summary_sha256": sha256(root / "SUMMARY.md"),
    }
    atomic_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", root / "AUDIT.json")
    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

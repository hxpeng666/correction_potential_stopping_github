#!/usr/bin/env python3
"""Evaluate the frozen Qwen3-14B five-method suite with trajectory-envelope LTT."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from recalibrate_deepseek7b_method_exploration_ltt_v1 import (
    compact_metrics,
    load_replay_data,
    replay_curve,
    select_ltt,
    threshold_grid,
)


METHODS = ("correctness", "consistency", "last_switch", "bce", "bce_traj")
ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def align(data, payload: dict[str, Any], split: str) -> np.ndarray:
    scores = np.asarray(payload["scores"][split], dtype=np.float64)
    if "keys" in payload:
        keys = payload["keys"][split]
        ids = [str(value) for value in keys["problem_ids"]]
        checkpoints = np.asarray(keys["checkpoints"], dtype=np.int64)
    else:
        ids = [str(value) for value in payload["problem_ids"][split]]
        checkpoints = np.asarray(payload["checkpoints"][split], dtype=np.int64)
    if ids != data.row_problem_ids or not np.array_equal(checkpoints, data.row_checkpoints):
        raise AssertionError(f"score/frame mismatch for split {split}")
    return scores


def alpha_slug(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=101)
    args = parser.parse_args()

    probe_root = args.probe_root.resolve()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    data = {
        "gsm8k": {
            "probe_train": load_replay_data(data_root / "gsm8k" / "probe_train", "paragraph"),
            "calibration": load_replay_data(data_root / "gsm8k" / "calibration", "paragraph"),
            "heldout": load_replay_data(data_root / "gsm8k" / "heldout", "paragraph"),
        },
        "math": {
            "probe_train": load_replay_data(data_root / "math" / "probe_train", "paragraph"),
            "calibration": load_replay_data(data_root / "math" / "calibration", "paragraph"),
            "heldout": load_replay_data(data_root / "math500" / "heldout", "paragraph"),
            "ood": load_replay_data(data_root / "aime" / "heldout", "paragraph"),
        },
    }
    expected = {
        "gsm8k": {"probe_train": 1000, "calibration": 500, "heldout": 1319},
        "math": {"probe_train": 1400, "calibration": 700, "heldout": 500, "ood": 30},
    }
    for dataset, splits in expected.items():
        for split, count in splits.items():
            if data[dataset][split].problems != count:
                raise AssertionError(f"{dataset}/{split}: expected {count} problems")

    frozen: dict[tuple[str, str], dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for method in METHODS:
        for dataset in ("gsm8k", "math"):
            model_name = "gsm8k" if dataset == "gsm8k" else "math500"
            model_dir = probe_root / "probes" / model_name / method
            report_path = model_dir / "probe.json"
            scores_path = model_dir / "scores.pt"
            marker_path = model_dir / "phase.complete"
            for path in (report_path, scores_path, marker_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
                source_hashes[str(path)] = sha256(path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            scores = torch.load(scores_path, map_location="cpu", weights_only=False)
            direction = (
                str(report["run_spec"]["stop_direction"])
                if report.get("run_spec") is not None
                else "low"
            )
            aligned = {
                split: align(data[dataset][split], scores, split)
                for split in ("probe_train", "calibration", "heldout")
            }
            if dataset == "math":
                if "ood" in scores["scores"]:
                    aligned["ood"] = align(data[dataset]["ood"], scores, "ood")
                else:
                    ood_path = probe_root / "probes" / "aime" / method / "scores.pt"
                    source_hashes[str(ood_path)] = sha256(ood_path)
                    ood_scores = torch.load(ood_path, map_location="cpu", weights_only=False)
                    aligned["ood"] = align(data[dataset]["ood"], ood_scores, "heldout")
            frozen[(method, dataset)] = {
                "direction": direction,
                "scores": aligned,
                "grid": threshold_grid(aligned["probe_train"], args.grid_size, direction),
            }

    all_rows: list[dict[str, Any]] = []
    all_details: dict[str, Any] = {}
    for alpha in ALPHAS:
        rows: list[dict[str, Any]] = []
        details: dict[str, Any] = {}
        for method in METHODS:
            details[method] = {}
            for dataset in ("gsm8k", "math"):
                item = frozen[(method, dataset)]
                direction = item["direction"]
                scores = item["scores"]
                grid = item["grid"]
                calibration_curve, calibration_lost = replay_curve(
                    data[dataset]["calibration"], scores["calibration"], grid, 0, direction
                )
                selected_index, certificate = select_ltt(
                    calibration_curve, calibration_lost, alpha, args.delta
                )
                selected = calibration_curve[selected_index]
                details[method][dataset] = {
                    "direction": direction,
                    "selected_grid_index": selected_index,
                    "selected_threshold": selected["threshold"],
                    "certificate": certificate,
                    "calibration": selected,
                    "tests": {},
                }
                for split in ["heldout"] + (["ood"] if dataset == "math" else []):
                    test_curve, _ = replay_curve(
                        data[dataset][split], scores[split], [grid[selected_index]], 0, direction
                    )
                    test = test_curve[0]
                    details[method][dataset]["tests"][split] = test
                    reported_dataset = (
                        "gsm8k" if dataset == "gsm8k"
                        else ("math500" if split == "heldout" else "aime2024")
                    )
                    rows.append(
                        {
                            "method": method,
                            "primary_method": method == "bce_traj",
                            "dataset": reported_dataset,
                            "calibrator": "trajectory_envelope_ltt",
                            "alpha": alpha,
                            "delta": args.delta,
                            "threshold": selected["threshold"],
                            "selected_grid_index": selected_index,
                            "certified_prefix_end": certificate["certified_prefix_end"],
                            "allowed_first_failure_boundaries": certificate["allowed_first_failure_boundaries"],
                            "calibration_first_failure_boundaries": certificate["selected_first_failure_boundaries"],
                            "calibration_ucb": certificate["trajectory_envelope_upper"],
                            **compact_metrics("calibration", selected),
                            **compact_metrics("test", test),
                        }
                    )
        if len(rows) != 15:
            raise AssertionError(f"alpha={alpha}: expected 15 rows, got {len(rows)}")
        for method in METHODS:
            transferred = [
                row for row in rows
                if row["method"] == method and row["dataset"] in {"math500", "aime2024"}
            ]
            if len(transferred) != 2 or transferred[0]["threshold"] != transferred[1]["threshold"]:
                raise AssertionError(f"MATH threshold not reused on AIME for {method}")
        alpha_dir = output / f"alpha_{alpha_slug(alpha)}"
        write_csv(rows, alpha_dir / "RESULTS_LTT.csv")
        atomic_json(rows, alpha_dir / "RESULTS_LTT.json")
        atomic_json(details, alpha_dir / "CALIBRATION_DETAILS.json")
        atomic_json({"status": "complete", "alpha": alpha, "rows": 15}, alpha_dir / "AUDIT.json")
        all_rows.extend(rows)
        all_details[str(alpha)] = details

    if len(all_rows) != 90:
        raise AssertionError(f"expected 90 aggregate rows, got {len(all_rows)}")
    if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
        raise AssertionError("a frozen probe artifact changed during evaluation")
    write_csv(all_rows, output / "RESULTS_ALL_ALPHA.csv")
    atomic_json(all_rows, output / "RESULTS_ALL_ALPHA.json")
    atomic_json(all_details, output / "CALIBRATION_ALL_ALPHA.json")
    lines = [
        "# Qwen3-14B deterministic five-method results",
        "",
        "Cells report held-out token reduction and accuracy delta relative to Dense.",
        "",
        "| Method | Dataset | Alpha | Token reduction | Delta acc (pp) | Lost | Helped |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        lines.append(
            f"| {row['method']} | {row['dataset']} | {row['alpha']:.1%} | "
            f"{row['test_deployed_token_reduction']:.2%} | {row['test_accuracy_delta_pp']:+.2f} | "
            f"{row['test_lost_correct']} | {row['test_helped']} |"
        )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "methods": list(METHODS),
            "alphas": list(ALPHAS),
            "rows": len(all_rows),
            "source_files_verified_unchanged": len(source_hashes),
            "checks": {
                "problem_level_first_hit": True,
                "trajectory_envelope_monotone": True,
                "exact_binomial_ucb": True,
                "continuous_fixed_sequence_prefix": True,
                "candidate_thresholds_from_probe_train_only": True,
                "calibration_only_certification": True,
                "heldout_unused_for_selection": True,
                "math_threshold_reused_on_aime": True,
                "fixed_empirical_B_used": False,
                "cost": "reasoning_only_to_match_deepseek7b_final",
            },
        },
        output / "AUDIT.json",
    )
    atomic_json(
        {"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "rows": 90},
        output / "EXPERIMENT_COMPLETE.json",
    )


if __name__ == "__main__":
    main()

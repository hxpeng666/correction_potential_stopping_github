#!/usr/bin/env python3
"""Evaluate the frozen LR=5e-5 original/forced-cap grader probe pair.

Probe checkpoints are selected only by the internal validation objective.  This
script performs the first and only deployment-threshold selection using
trajectory-envelope LTT on the independent calibration split.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recalibrate_deepseek7b_method_exploration_ltt_v1 import (
    align_scores,
    compact_metrics,
    load_replay_data,
    replay_curve,
    select_ltt,
    sha256,
    threshold_grid,
)
from src.reproducibility import code_provenance, strict_reproducibility


CONDITIONS = ("original_grader", "forced_cap_grader")
METHODS = ("bce", "bce_traj")
ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def align_named(data, saved: dict[str, Any], score_split: str) -> np.ndarray:
    return align_scores(
        data,
        {
            "scores": {"aligned": saved["scores"][score_split]},
            "keys": {"aligned": saved["keys"][score_split]},
        },
        "aligned",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--original-data-root", type=Path, required=True)
    parser.add_argument("--forced-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=101)
    args = parser.parse_args()
    reproducibility = strict_reproducibility(seed=0, num_threads=1)
    code_identity = code_provenance(
        ROOT,
        (
            "scripts/evaluate_deepseek7b_grader_pair_lr5e5_ltt_v1.py",
            "scripts/recalibrate_deepseek7b_method_exploration_ltt_v1.py",
            "src/reproducibility.py",
        ),
    )
    roots = {
        "original_grader": args.original_data_root.resolve(),
        "forced_cap_grader": args.forced_data_root.resolve(),
    }
    output = args.output.resolve()
    experiment = args.experiment_root.resolve()
    if output == experiment or experiment in output.parents:
        raise ValueError("LTT output must be separate from frozen probe directories")

    rows: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    for condition in CONDITIONS:
        data_root = roots[condition]
        loaded = {
            "gsm8k": {
                "probe_train": load_replay_data(data_root / "gsm8k/probe_train"),
                "calibration": load_replay_data(data_root / "gsm8k/calibration"),
                "heldout": load_replay_data(data_root / "gsm8k/heldout"),
            },
            "math": {
                "probe_train": load_replay_data(data_root / "math/probe_train"),
                "calibration": load_replay_data(data_root / "math/calibration"),
                "heldout": load_replay_data(data_root / "math500/heldout"),
                "ood": load_replay_data(data_root / "aime/heldout"),
            },
        }
        details[condition] = {}
        for dataset in ("gsm8k", "math"):
            details[condition][dataset] = {}
            for method in METHODS:
                model_dir = experiment / "conditions" / condition / "probes" / dataset / method
                report_path = model_dir / "probe.json"
                scores_path = model_dir / "scores.pt"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                saved = torch.load(scores_path, map_location="cpu", weights_only=False)
                source_hashes[str(report_path)] = sha256(report_path)
                source_hashes[str(scores_path)] = sha256(scores_path)
                invocation = report["invocation"]
                if float(invocation["learning_rate"]) != 5e-5:
                    raise AssertionError(f"unexpected learning rate: {model_dir}")
                if invocation["selection_rule"] != "validation_objective":
                    raise AssertionError(f"non-primary epoch selection: {model_dir}")
                if invocation["deployment_calibration"] != "scores_only":
                    raise AssertionError(f"probe performed legacy calibration: {model_dir}")
                aligned = {
                    "probe_train": align_named(loaded[dataset]["probe_train"], saved, "probe_train"),
                    "calibration": align_named(loaded[dataset]["calibration"], saved, "calibration"),
                    "heldout": align_named(loaded[dataset]["heldout"], saved, "heldout"),
                }
                if dataset == "math":
                    aligned["ood"] = align_named(loaded[dataset]["ood"], saved, "ood")
                grid = threshold_grid(aligned["probe_train"], args.grid_size, "low")
                method_details: dict[str, Any] = {
                    "best_epoch_zero_based": int(report["best_epoch"]),
                    "learning_rate": float(invocation["learning_rate"]),
                    "alphas": {},
                }
                details[condition][dataset][method] = method_details
                for alpha in ALPHAS:
                    calibration_curve, calibration_lost = replay_curve(
                        loaded[dataset]["calibration"], aligned["calibration"], grid, 0, "low"
                    )
                    selected_index, certificate = select_ltt(
                        calibration_curve, calibration_lost, alpha, args.delta
                    )
                    selected = calibration_curve[selected_index]
                    method_details["alphas"][str(alpha)] = {
                        "selected_grid_index": selected_index,
                        "threshold": selected["threshold"],
                        "certificate": certificate,
                        "calibration": selected,
                    }
                    for split in (["heldout"] + (["ood"] if dataset == "math" else [])):
                        test_curve, _ = replay_curve(
                            loaded[dataset][split], aligned[split], [grid[selected_index]], 0, "low"
                        )
                        test = test_curve[0]
                        reported_dataset = (
                            "gsm8k" if dataset == "gsm8k" else
                            ("math500" if split == "heldout" else "aime2024")
                        )
                        rows.append(
                            {
                                "condition": condition,
                                "method": method,
                                "dataset": reported_dataset,
                                "learning_rate": 5e-5,
                                "alpha": alpha,
                                "delta": args.delta,
                                "best_epoch_one_based": int(report["best_epoch"]) + 1,
                                "threshold": selected["threshold"],
                                "selected_grid_index": selected_index,
                                "calibration_ucb": certificate["trajectory_envelope_upper"],
                                **compact_metrics("calibration", selected),
                                **compact_metrics("test", test),
                            }
                        )
                if method == METHODS[0]:
                    for split in (["heldout"] + (["ood"] if dataset == "math" else [])):
                        sentinel, _ = replay_curve(
                            loaded[dataset][split], aligned[split], [grid[0]], 0, "low"
                        )
                        metric = sentinel[0]
                        reported_dataset = (
                            "gsm8k" if dataset == "gsm8k" else
                            ("math500" if split == "heldout" else "aime2024")
                        )
                        dense_rows.append(
                            {
                                "condition": condition,
                                "dataset": reported_dataset,
                                "accuracy": metric["dense_accuracy"],
                                "mean_dense_tokens": metric["mean_dense_tokens"],
                                "problems": loaded[dataset][split].problems,
                            }
                        )

    if len(rows) != len(CONDITIONS) * len(METHODS) * 3 * len(ALPHAS):
        raise AssertionError(f"unexpected LTT row count: {len(rows)}")
    if len(dense_rows) != len(CONDITIONS) * 3:
        raise AssertionError(f"unexpected Dense row count: {len(dense_rows)}")
    for condition in CONDITIONS:
        for method in METHODS:
            for alpha in ALPHAS:
                transferred = [
                    row for row in rows
                    if row["condition"] == condition and row["method"] == method
                    and row["alpha"] == alpha and row["dataset"] in {"math500", "aime2024"}
                ]
                if len(transferred) != 2 or transferred[0]["threshold"] != transferred[1]["threshold"]:
                    raise AssertionError("MATH threshold was not transferred unchanged to AIME")
    if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
        raise AssertionError("frozen probe source changed during LTT evaluation")

    output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output / "RESULTS_LTT.csv")
    write_csv(dense_rows, output / "DENSE_BASELINES.csv")
    atomic_json(rows, output / "RESULTS_LTT.json")
    atomic_json(dense_rows, output / "DENSE_BASELINES.json")
    atomic_json(details, output / "CALIBRATION_DETAILS.json")
    atomic_json(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result_rows": len(rows),
            "dense_rows": len(dense_rows),
            "learning_rate": 5e-5,
            "training_seed": 0,
            "split_seed": 0,
            "model_selection": "minimum_internal_validation_objective",
            "deployment_calibration": "trajectory_envelope_ltt_only",
            "alphas": list(ALPHAS),
            "delta": args.delta,
            "fixed_empirical_B_used": False,
            "math_threshold_reused_on_aime": True,
            "source_files_verified_unchanged": len(source_hashes),
            "reproducibility": reproducibility,
            "code_identity": code_identity,
        },
        output / "AUDIT.json",
    )
    print(json.dumps({"status": "complete", "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

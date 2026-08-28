#!/usr/bin/env python3
"""Apply the frozen trajectory-envelope LTT calibrator to five probe ablations."""
from __future__ import annotations

import argparse
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

from src.reproducibility import code_provenance, strict_reproducibility

from recalibrate_deepseek7b_method_exploration_ltt_v1 import (
    align_scores as _unused_align_scores,
    atomic_json,
    compact_metrics,
    load_replay_data,
    replay_curve,
    select_ltt,
    sha256,
    threshold_grid,
    write_csv,
)


METHODS = ("correctness", "consistency", "last_switch", "bce", "bce_traj")


def align_standard(data, payload: dict[str, Any], split: str) -> np.ndarray:
    scores = np.asarray(payload["scores"][split], dtype=np.float64)
    ids = [str(value) for value in payload["problem_ids"][split]]
    checkpoints = np.asarray(payload["checkpoints"][split], dtype=np.int64)
    if ids != data.row_problem_ids or not np.array_equal(checkpoints, data.row_checkpoints):
        raise AssertionError(f"score/frame mismatch: {split}")
    if len(scores) != len(ids):
        raise AssertionError(f"score length mismatch: {split}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--gsm-data-root", type=Path, required=True)
    parser.add_argument("--math-data-root", type=Path, required=True)
    parser.add_argument("--gsm-correctness-data-root", type=Path)
    parser.add_argument("--math-correctness-data-root", type=Path)
    parser.add_argument("--gsm-heldout-root", type=Path, required=True)
    parser.add_argument("--math-heldout-root", type=Path, required=True)
    parser.add_argument("--aime-heldout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=101)
    args = parser.parse_args()
    reproducibility = strict_reproducibility(seed=0, num_threads=1)
    code_identity = code_provenance(
        ROOT,
        (
            "scripts/recalibrate_deepseek7b_five_ablation_ltt_v1.py",
            "scripts/recalibrate_deepseek7b_method_exploration_ltt_v1.py",
            "src/reproducibility.py",
        ),
    )
    probe_root = args.probe_root.resolve()
    output = args.output.resolve()
    if output == probe_root or probe_root in output.parents:
        raise ValueError("output must not be inside the frozen probe source")

    data = {
        "gsm8k": {
            "probe_train": load_replay_data(args.gsm_data_root / "probe_train"),
            "calibration": load_replay_data(args.gsm_data_root / "calibration"),
            "heldout": load_replay_data(args.gsm_heldout_root / "heldout"),
        },
        "math": {
            "probe_train": load_replay_data(args.math_data_root / "probe_train"),
            "calibration": load_replay_data(args.math_data_root / "calibration"),
            "heldout": load_replay_data(args.math_heldout_root / "heldout"),
            "ood": load_replay_data(args.aime_heldout_root / "heldout"),
        },
    }
    correctness_data = {
        "gsm8k": {
            "probe_train": load_replay_data(
                (args.gsm_correctness_data_root or args.gsm_data_root) / "probe_train"
            ),
            "calibration": load_replay_data(
                (args.gsm_correctness_data_root or args.gsm_data_root) / "calibration"
            ),
        },
        "math": {
            "probe_train": load_replay_data(
                (args.math_correctness_data_root or args.math_data_root) / "probe_train"
            ),
            "calibration": load_replay_data(
                (args.math_correctness_data_root or args.math_data_root) / "calibration"
            ),
        },
    }
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    for method in METHODS:
        details[method] = {}
        for dataset in ("gsm8k", "math"):
            method_data = data[dataset]
            if method == "correctness":
                method_data = {
                    **data[dataset],
                    **correctness_data[dataset],
                }
            model_name = "gsm8k" if dataset == "gsm8k" else "math500"
            model_dir = probe_root / "probes" / model_name / method
            probe_path = model_dir / "probe.json"
            scores_path = model_dir / "scores.pt"
            test_score_path = scores_path
            report = json.loads(probe_path.read_text())
            payload = torch.load(scores_path, map_location="cpu", weights_only=False)
            source_hashes[str(probe_path)] = sha256(probe_path)
            source_hashes[str(scores_path)] = sha256(scores_path)
            direction = str(report["run_spec"]["stop_direction"])
            aligned = {
                split: align_standard(method_data[split], payload, split)
                for split in ("probe_train", "calibration", "heldout")
            }
            if dataset == "math":
                test_score_path = probe_root / "probes" / "aime" / method / "scores.pt"
                ood_payload = torch.load(test_score_path, map_location="cpu", weights_only=False)
                source_hashes[str(test_score_path)] = sha256(test_score_path)
                aligned["ood"] = align_standard(method_data["ood"], ood_payload, "heldout")
            grid = threshold_grid(aligned["probe_train"], args.grid_size, direction)
            calibration_curve, calibration_lost = replay_curve(
                method_data["calibration"],
                aligned["calibration"],
                grid,
                readout_suffix_tokens=0,
                direction=direction,
            )
            selected_index, certificate = select_ltt(
                calibration_curve, calibration_lost, args.alpha, args.delta
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
                    method_data[split],
                    aligned[split],
                    [grid[selected_index]],
                    readout_suffix_tokens=0,
                    direction=direction,
                )
                test = test_curve[0]
                details[method][dataset]["tests"][split] = test
                reported_dataset = (
                    "gsm8k" if dataset == "gsm8k" else ("math500" if split == "heldout" else "aime2024")
                )
                rows.append(
                    {
                        "method": method,
                        "primary_method": method == "bce_traj",
                        "dataset": reported_dataset,
                        "calibrator": "trajectory_envelope_ltt",
                        "alpha": args.alpha,
                        "delta": args.delta,
                        "direction": direction,
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
        raise AssertionError(f"expected 15 rows, got {len(rows)}")
    for method in METHODS:
        transferred = [row for row in rows if row["method"] == method and row["dataset"] in {"math500", "aime2024"}]
        if len(transferred) != 2 or transferred[0]["threshold"] != transferred[1]["threshold"]:
            raise AssertionError(f"MATH threshold was not reused on AIME: {method}")
    if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
        raise AssertionError("frozen source changed during recalibration")

    output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output / "RESULTS_LTT.csv")
    atomic_json(rows, output / "RESULTS_LTT.json")
    atomic_json(details, output / "CALIBRATION_DETAILS.json")
    primary = [row for row in rows if row["primary_method"]]
    atomic_json(
        {
            "status": "frozen",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "primary_method": "bce_traj",
            "main_calibrator": "trajectory_envelope_ltt",
            "alpha": args.alpha,
            "delta": args.delta,
            "selection_objective": "maximize calibration token reduction inside certified prefix",
            "fixed_empirical_B_role": "appendix_only",
            "primary_results": primary,
            "reproducibility": reproducibility,
            "code_identity": code_identity,
        },
        output / "PRIMARY_RESULTS.json",
    )
    atomic_json(
        {
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "probe_root": str(probe_root),
            "source_files_verified_unchanged": len(source_hashes),
            "methods": len(METHODS),
            "result_rows": len(rows),
            "reproducibility": reproducibility,
            "code_identity": code_identity,
            "checks": {
                "problem_level_first_hit": True,
                "trajectory_envelope_monotone": True,
                "exact_binomial_ucb": True,
                "continuous_fixed_sequence_prefix": True,
                "token_only_objective": True,
                "test_unused_for_selection": True,
                "math_threshold_reused_on_aime": True,
                "fixed_B_not_main": True,
            },
        },
        output / "AUDIT.json",
    )
    report = [
        "# Five-ablation results with trajectory-envelope LTT",
        "",
        f"Main calibration: alpha={args.alpha:.2%}, delta={args.delta:.2%}; fixed B is appendix-only.",
        "",
        "| Method | Dataset | Accuracy | Dense | Delta acc (pp) | Token reduction | Lost | Helped | Cal UCB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['method']} | {row['dataset']} | {row['test_accuracy']:.2%} | "
            f"{row['test_dense_accuracy']:.2%} | {row['test_accuracy_delta_pp']:+.2f} | "
            f"{row['test_deployed_token_reduction']:.2%} | {row['test_lost_correct']} | "
            f"{row['test_helped']} | {row['calibration_ucb']:.2%} |"
        )
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    atomic_json(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "main_calibrator": "trajectory_envelope_ltt",
            "primary_method": "bce_traj",
            "result_rows": len(rows),
            "git_commit": code_identity["git"]["commit"],
        },
        output / "EXPERIMENT_COMPLETE.json",
    )
    print(json.dumps({"status": "complete", "primary": primary}, indent=2))


if __name__ == "__main__":
    main()

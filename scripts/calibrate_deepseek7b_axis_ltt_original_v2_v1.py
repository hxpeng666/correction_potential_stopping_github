#!/usr/bin/env python3
"""Calibrate uncensored original-v2 method axes with trajectory-envelope LTT."""
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

from src.reproducibility import code_provenance, strict_reproducibility

from recalibrate_deepseek7b_method_exploration_ltt_v1 import (
    compact_metrics,
    load_replay_data,
    replay_curve,
    select_ltt,
    sha256,
    threshold_grid,
)


AXES = ("weight", "robust", "reach", "feature", "aux_feature")
EXPERIMENT = {
    "weight": "bce_weight",
    "robust": "first_hit_trajectory",
    "reach": "first_hit_trajectory",
    "feature": "feature_construction",
    "aux_feature": "feature_construction",
}


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])
    os.replace(temporary, path)


def align(data, payload: dict[str, Any], split: str) -> np.ndarray:
    scores = np.asarray(payload["scores"][split], dtype=np.float64)
    keys = payload["keys"][split]
    ids = [str(value) for value in keys["problem_ids"]]
    checkpoints = np.asarray(keys["checkpoints"], dtype=np.int64)
    if ids != data.row_problem_ids or not np.array_equal(checkpoints, data.row_checkpoints):
        raise AssertionError(f"external score alignment mismatch: {split}")
    return scores


def align_probe_train(data, payload: dict[str, Any]) -> np.ndarray:
    scores = np.asarray(payload["scores"]["probe_train"], dtype=np.float64)
    keys = payload["keys"]["probe_train"]
    ids = [str(value) for value in keys["problem_ids"]]
    checkpoints = np.asarray(keys["checkpoints"], dtype=np.int64)
    if ids != data.row_problem_ids or not np.array_equal(checkpoints, data.row_checkpoints):
        raise AssertionError("probe-train score alignment mismatch")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--external-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=[0.005, 0.01, 0.02, 0.03, 0.05],
    )
    parser.add_argument("--main-alpha", type=float, default=0.03)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=101)
    args = parser.parse_args()
    reproducibility = strict_reproducibility(seed=0, num_threads=1)
    code_identity = code_provenance(
        ROOT,
        (
            "scripts/calibrate_deepseek7b_axis_ltt_original_v2_v1.py",
            "scripts/recalibrate_deepseek7b_method_exploration_ltt_v1.py",
            "src/reproducibility.py",
        ),
    )
    if args.main_alpha not in args.alphas:
        raise ValueError("main alpha must be in alpha grid")
    source = args.source.resolve()
    external = args.external_scores.resolve()
    output = args.output.resolve()
    data_root = args.data_root.resolve()
    for dataset in ("gsm8k", "math"):
        manifest = json.loads((external / f"{dataset.upper()}_SCORING_MANIFEST.json").read_text())
        if manifest.get("status") != "complete" or int(manifest["models"]) != 149:
            raise ValueError(f"incomplete external scoring: {dataset}")
    data = {
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
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for dataset in ("gsm8k", "math"):
        for axis in AXES:
            model_dirs = sorted((source / "screen" / dataset / axis).glob("*"))
            expected = {"weight": 4, "robust": 25, "reach": 60, "feature": 12, "aux_feature": 48}[axis]
            if len([path for path in model_dirs if (path / "probe.json").is_file()]) != expected:
                raise AssertionError(f"axis model count mismatch: {dataset}/{axis}")
            for model_dir in model_dirs:
                report_path = model_dir / "probe.json"
                checkpoint_path = model_dir / "probe.pt"
                train_scores_path = model_dir / "scores.pt"
                if not report_path.is_file():
                    continue
                ext_path = external / "scores" / axis / model_dir.name / f"{dataset}.pt"
                report = json.loads(report_path.read_text())
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                train_payload = torch.load(train_scores_path, map_location="cpu", weights_only=False)
                ext_payload = torch.load(ext_path, map_location="cpu", weights_only=False)
                if ext_payload.get("status") != "complete":
                    raise ValueError(f"incomplete score payload: {ext_path}")
                for path in (report_path, checkpoint_path, train_scores_path, ext_path):
                    source_hashes[str(path)] = sha256(path)
                train_scores = align_probe_train(data[dataset]["probe_train"], train_payload)
                calibration_scores = align(data[dataset]["calibration"], ext_payload, "calibration")
                heldout_scores = align(data[dataset]["heldout"], ext_payload, "heldout")
                ood_scores = (
                    align(data[dataset]["ood"], ext_payload, "ood") if dataset == "math" else None
                )
                grid = threshold_grid(train_scores, args.grid_size, "low")
                readout_cost = int(checkpoint["readout_suffix_tokens"])
                calibration_curve, calibration_lost = replay_curve(
                    data[dataset]["calibration"],
                    calibration_scores,
                    grid,
                    readout_cost,
                    direction="low",
                )
                for alpha in args.alphas:
                    selected_index, certificate = select_ltt(
                        calibration_curve, calibration_lost, alpha, args.delta
                    )
                    selected = calibration_curve[selected_index]
                    if selected_index > 0 and certificate["trajectory_envelope_upper"] > alpha + 1e-12:
                        raise AssertionError("selected threshold violates declared alpha")
                    tests = [("gsm8k" if dataset == "gsm8k" else "math500", "heldout", heldout_scores)]
                    if dataset == "math":
                        assert ood_scores is not None
                        tests.append(("aime2024", "ood", ood_scores))
                    for reported_dataset, split, split_scores in tests:
                        test_curve, _ = replay_curve(
                            data[dataset][split],
                            split_scores,
                            [grid[selected_index]],
                            readout_cost,
                            direction="low",
                        )
                        test = test_curve[0]
                        rows.append(
                            {
                                "experiment": EXPERIMENT[axis],
                                "axis": axis,
                                "label": model_dir.name,
                                "dataset": reported_dataset,
                                "alpha": alpha,
                                "is_main_alpha": alpha == args.main_alpha,
                                "delta": args.delta,
                                "threshold": selected["threshold"],
                                "selected_grid_index": selected_index,
                                "certified_prefix_end": certificate["certified_prefix_end"],
                                "allowed_first_failure_boundaries": certificate["allowed_first_failure_boundaries"],
                                "calibration_first_failure_boundaries": certificate["selected_first_failure_boundaries"],
                                "calibration_ucb": certificate["trajectory_envelope_upper"],
                                "readout_suffix_tokens_per_checkpoint": readout_cost,
                                "validation_ap": report["internal_validation"]["label_ap"],
                                "validation_auc": report["internal_validation"]["label_auc"],
                                **compact_metrics("calibration", selected),
                                **compact_metrics("test", test),
                            }
                        )
    expected_rows = 149 * len(args.alphas) * 3
    if len(rows) != expected_rows:
        raise AssertionError(f"expected {expected_rows} rows, got {len(rows)}")
    if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
        raise AssertionError("source changed during calibration")

    # Cross-domain main-alpha table.  This is an ablation report, not an extra
    # post-hoc model-selection step; no threshold is changed using test labels.
    main_rows = [row for row in rows if row["is_main_alpha"]]
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in main_rows:
        grouped.setdefault((row["axis"], row["label"]), {})[row["dataset"]] = row
    summary: list[dict[str, Any]] = []
    for (axis, label), values in grouped.items():
        if set(values) != {"gsm8k", "math500", "aime2024"}:
            raise AssertionError(f"incomplete cross-domain values: {axis}/{label}")
        gsm, math, aime = values["gsm8k"], values["math500"], values["aime2024"]
        summary.append(
            {
                "experiment": EXPERIMENT[axis],
                "axis": axis,
                "label": label,
                "alpha": args.main_alpha,
                "gsm8k_accuracy": gsm["test_accuracy"],
                "gsm8k_delta_pp": gsm["test_accuracy_delta_pp"],
                "gsm8k_token_reduction": gsm["test_deployed_token_reduction"],
                "gsm8k_lost": gsm["test_lost_correct"],
                "math500_accuracy": math["test_accuracy"],
                "math500_delta_pp": math["test_accuracy_delta_pp"],
                "math500_token_reduction": math["test_deployed_token_reduction"],
                "math500_lost": math["test_lost_correct"],
                "aime2024_accuracy": aime["test_accuracy"],
                "aime2024_delta_pp": aime["test_accuracy_delta_pp"],
                "aime2024_token_reduction": aime["test_deployed_token_reduction"],
                "aime2024_lost": aime["test_lost_correct"],
                "mean_calibration_token_reduction": 0.5 * (
                    gsm["calibration_deployed_token_reduction"]
                    + math["calibration_deployed_token_reduction"]
                ),
                "min_test_token_reduction_gsm_math": min(
                    gsm["test_deployed_token_reduction"], math["test_deployed_token_reduction"]
                ),
                "mean_test_token_reduction_gsm_math": 0.5 * (
                    gsm["test_deployed_token_reduction"] + math["test_deployed_token_reduction"]
                ),
                "mean_validation_ap": 0.5 * (gsm["validation_ap"] + math["validation_ap"]),
                "probe_parameters_gsm": json.loads((source / "screen/gsm8k" / axis / label / "probe.json").read_text())["probe"]["parameters"],
            }
        )
    summary.sort(key=lambda row: (row["experiment"], -row["mean_test_token_reduction_gsm_math"]))

    output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output / "ALL_ALPHA_RESULTS.csv")
    atomic_json(rows, output / "ALL_ALPHA_RESULTS.json")
    write_csv(summary, output / "MAIN_ALPHA_RESULTS.csv")
    atomic_json(summary, output / "MAIN_ALPHA_RESULTS.json")
    atomic_json(
        {
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scientific_protocol": "uncensored_original_v2",
            "right_censoring": False,
            "reference_method": "legacy_weighted hidden_scalars normalized_softmin beta0.5 lambda1",
            "calibrator": "trajectory_envelope_ltt",
            "alphas": args.alphas,
            "main_alpha": args.main_alpha,
            "delta": args.delta,
            "models_per_dataset": 149,
            "result_rows": len(rows),
            "summary_rows": len(summary),
            "fixed_B_role": "appendix_only",
            "checks": {
                "threshold_grid_probe_train_only": True,
                "calibration_problem_level_first_failure": True,
                "exact_binomial_ucb": True,
                "token_objective_includes_one_step_suffix": True,
                "heldout_ood_unused_for_thresholds": True,
                "math_threshold_reused_on_aime": True,
                "source_files_unchanged": len(source_hashes),
            },
            "interpretation": "Cross-model held-out ordering is descriptive ablation analysis, not an additional deployment-model selection guarantee.",
            "reproducibility": reproducibility,
            "code_identity": code_identity,
            "data_root": str(data_root),
        },
        output / "AUDIT.json",
    )
    atomic_json(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "main_calibrator": "trajectory_envelope_ltt",
            "main_alpha": args.main_alpha,
            "result_rows": len(rows),
            "git_commit": code_identity["git"]["commit"],
        },
        output / "EXPERIMENT_COMPLETE.json",
    )
    print(json.dumps({"status": "complete", "rows": len(rows), "summary": len(summary)}))


if __name__ == "__main__":
    main()

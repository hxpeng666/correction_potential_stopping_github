#!/usr/bin/env python3
"""Apply trajectory-envelope LTT to the deterministic three-axis ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
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
    threshold_grid,
)
from src.reproducibility import code_provenance, strict_reproducibility


ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)
AXIS_VARIANTS = {
    "feature": (
        "primary",
        "feature_last4_mean",
        "feature_paragraph_mean",
        "feature_prefix_mean",
        "feature_pca32_linear",
        "feature_one_step_full",
        "feature_pool_paragraph_pca256",
        "feature_pool_last4_pca128_one_step",
    ),
    "bce_weight": (
        "primary",
        "weight_legacy_remaining",
        "weight_problem_balanced_random",
        "weight_problem_balanced_stratified",
    ),
    "first_hit_trajectory": (
        "primary",
        "trajectory_none",
        "trajectory_hard_min",
        "trajectory_bottomk",
        "trajectory_cvar",
        "first_hit_earliest_safe_cvar",
    ),
}


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write empty result table")
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


def align_named(data: Any, saved: dict[str, Any], split: str) -> np.ndarray:
    return align_scores(
        data,
        {
            "scores": {"aligned": saved["scores"][split]},
            "keys": {"aligned": saved["keys"][split]},
        },
        "aligned",
    )


def markdown_table(rows: list[dict[str, Any]], axis: str, dataset: str) -> str:
    selected = [
        row for row in rows if row["axis"] == axis and row["dataset"] == dataset
    ]
    by_variant = {variant: [] for variant in AXIS_VARIANTS[axis]}
    for row in selected:
        by_variant[row["variant"]].append(row)
    header = ["Variant"] + [f"{100 * alpha:g}%" for alpha in ALPHAS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for variant, values in by_variant.items():
        ordered = sorted(values, key=lambda value: float(value["alpha"]))
        cells = [variant]
        for value in ordered:
            cells.append(
                f"{100 * float(value['test_total_token_reduction']):.2f} / "
                f"{float(value['test_accuracy_delta_pp']):+.2f}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=101)
    args = parser.parse_args()
    strict_reproducibility(seed=0, num_threads=1)
    code_identity = code_provenance(
        ROOT,
        (
            "scripts/evaluate_deepseek7b_deterministic_three_axis_ablation_v1.py",
            "scripts/recalibrate_deepseek7b_method_exploration_ltt_v1.py",
            "src/reproducibility.py",
        ),
    )
    experiment = args.experiment_root.resolve()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    if output == experiment or experiment in output.parents:
        # A sibling is unnecessary here; ltt is intentionally a child of the
        # immutable probe root but never mutates a probe artifact.
        if output != experiment / "ltt":
            raise ValueError("unexpected nested LTT output")
    variants = json.loads((experiment / "VARIANTS.json").read_text(encoding="utf-8"))
    expected_unique = set().union(*map(set, AXIS_VARIANTS.values()))
    if set(variants) != expected_unique or len(variants) != 16:
        raise AssertionError("variant registry mismatch")

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
    rows_by_variant: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    details: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    epochs: dict[str, dict[str, int]] = {}
    for variant_id, spec in variants.items():
        details[variant_id] = {}
        epochs[variant_id] = {}
        for dataset in ("gsm8k", "math"):
            model_dir = experiment / "probes" / variant_id / dataset
            probe_path = model_dir / "probe.json"
            scores_path = model_dir / "scores.pt"
            if not (model_dir / "phase.complete").is_file():
                raise FileNotFoundError(f"incomplete probe: {model_dir}")
            report = json.loads(probe_path.read_text(encoding="utf-8"))
            saved = torch.load(scores_path, map_location="cpu", weights_only=False)
            source_hashes[str(probe_path)] = sha256(probe_path)
            source_hashes[str(scores_path)] = sha256(scores_path)
            invocation = report["invocation"]
            expected_fields = {
                "representation_kind": spec["representation_kind"],
                "feature_kind": spec["feature_kind"],
                "probe_architecture": spec["probe_architecture"],
                "point_loss": spec["point_loss"],
                "stratified_problem_batches": spec["stratified_problem_batches"],
                "trajectory_scope": spec["trajectory_scope"],
                "trajectory_aggregation": spec["trajectory_aggregation"],
                "rho": spec["rho"],
                "lambda_protect": spec["lambda_protect"],
            }
            for field, expected in expected_fields.items():
                if invocation[field] != expected:
                    raise AssertionError(f"{variant_id}/{dataset}: {field} mismatch")
            if float(invocation["learning_rate"]) != 5e-5:
                raise AssertionError("learning-rate drift")
            if invocation["selection_rule"] != "validation_objective":
                raise AssertionError("epoch-selection drift")
            if invocation["deployment_calibration"] != "scores_only":
                raise AssertionError("probe used a deployment threshold")
            if invocation["training_seed"] != 0 or invocation["split_seed"] != 0:
                raise AssertionError("seed drift")
            epochs[variant_id][dataset] = int(report["best_epoch"]) + 1
            expected_splits = {"probe_train", "calibration", "heldout"}
            if dataset == "math":
                expected_splits.add("ood")
            if set(saved["scores"]) != expected_splits:
                raise AssertionError(f"score split mismatch: {model_dir}")
            aligned = {
                split: align_named(loaded[dataset][split], saved, split)
                for split in expected_splits
            }
            grid = threshold_grid(aligned["probe_train"], args.grid_size, "low")
            readout_cost = int(spec["online_readout_token_cost"])
            calibration_curve, calibration_lost = replay_curve(
                loaded[dataset]["calibration"], aligned["calibration"],
                grid, readout_cost, "low",
            )
            details[variant_id][dataset] = {}
            for alpha in ALPHAS:
                selected_index, certificate = select_ltt(
                    calibration_curve, calibration_lost, alpha, args.delta
                )
                selected = calibration_curve[selected_index]
                if selected_index > certificate["certified_prefix_end"]:
                    raise AssertionError("selected threshold is not certified")
                details[variant_id][dataset][str(alpha)] = {
                    "selected_grid_index": selected_index,
                    "threshold": selected["threshold"],
                    "certificate": certificate,
                    "calibration": selected,
                }
                for split in (["heldout"] + (["ood"] if dataset == "math" else [])):
                    test_curve, _ = replay_curve(
                        loaded[dataset][split], aligned[split],
                        [grid[selected_index]], readout_cost, "low",
                    )
                    test = test_curve[0]
                    reported_dataset = (
                        "gsm8k" if dataset == "gsm8k" else
                        ("math500" if split == "heldout" else "aime2024")
                    )
                    row = {
                        "variant": variant_id,
                        "variant_axis": spec["axis"],
                        "dataset": reported_dataset,
                        "alpha": alpha,
                        "delta": args.delta,
                        "best_epoch_one_based": epochs[variant_id][dataset],
                        "threshold": selected["threshold"],
                        "selected_grid_index": selected_index,
                        "calibration_ucb": certificate["trajectory_envelope_upper"],
                        "online_readout_token_cost_per_checkpoint": readout_cost,
                        **compact_metrics("calibration", selected),
                        **compact_metrics("test", test),
                        # Explicit current-paper name: includes online readout
                        # token-equivalent cost where the feature needs it.
                        "calibration_total_token_reduction": selected["deployed_token_reduction"],
                        "test_total_token_reduction": test["deployed_token_reduction"],
                    }
                    rows_by_variant.setdefault(
                        (variant_id, reported_dataset, alpha), []
                    ).append(row)

    rows: list[dict[str, Any]] = []
    for axis, axis_variants in AXIS_VARIANTS.items():
        for variant_id in axis_variants:
            for dataset in ("gsm8k", "math500", "aime2024"):
                for alpha in ALPHAS:
                    candidates = rows_by_variant[(variant_id, dataset, alpha)]
                    if len(candidates) != 1:
                        raise AssertionError("non-unique variant result")
                    rows.append({"axis": axis, **candidates[0]})
    if len(rows) != 324:
        raise AssertionError(f"expected 324 result rows, got {len(rows)}")

    # MATH calibration threshold must transfer unchanged to AIME.
    for axis in AXIS_VARIANTS:
        for variant_id in AXIS_VARIANTS[axis]:
            for alpha in ALPHAS:
                pair = [
                    row for row in rows
                    if row["axis"] == axis and row["variant"] == variant_id
                    and row["alpha"] == alpha
                    and row["dataset"] in {"math500", "aime2024"}
                ]
                if len(pair) != 2 or pair[0]["threshold"] != pair[1]["threshold"]:
                    raise AssertionError("AIME threshold was not transferred from MATH")

    output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output / "RESULTS.csv")
    atomic_json(rows, output / "RESULTS.json")
    atomic_json(details, output / "CALIBRATION_DETAILS.json")
    report = [
        "# Deterministic DeepSeek-7B three-axis ablation",
        "",
        "Each cell is `total token reduction (%) / accuracy delta (pp)`.",
        "Thresholds use trajectory-envelope LTT; no empirical budget B is used.",
    ]
    for axis in AXIS_VARIANTS:
        for dataset in ("gsm8k", "math500", "aime2024"):
            report.extend(
                ["", f"## {axis} — {dataset}", "", markdown_table(rows, axis, dataset)]
            )
    (output / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    gate_audits = {}
    for dataset in ("gsm8k", "math"):
        path = experiment / "determinism_gate" / dataset / "AUDIT.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("all_exact") is not True and value.get("status") != "complete":
            raise AssertionError(f"failed determinism gate: {path}")
        gate_audits[str(path)] = {"sha256": sha256(path), "payload": value}
    audit = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": code_identity["git"]["commit"],
        "code_identity": code_identity,
        "determinism_gates": gate_audits,
        "unique_variants": 16,
        "axis_rows": {axis: len(values) for axis, values in AXIS_VARIANTS.items()},
        "result_rows": len(rows),
        "alphas": list(ALPHAS),
        "delta": args.delta,
        "candidate_grid_size": args.grid_size,
        "candidate_threshold_source": "probe_train scores only",
        "epoch_selection": "minimum internal validation objective",
        "calibrator": "trajectory-envelope LTT",
        "selection_objective": "maximize calibration total generated-token reduction inside certified prefix",
        "fixed_empirical_B_used": False,
        "heldout_used_for_selection": False,
        "aime_retrained_or_recalibrated": False,
        "source_hashes": source_hashes,
        "best_epochs_one_based": epochs,
    }
    atomic_json(audit, output / "AUDIT.json")
    atomic_json(
        {
            "status": "complete",
            "completed_at": audit["completed_at"],
            "result_rows": len(rows),
            "audit": str(output / "AUDIT.json"),
        },
        output / "EXPERIMENT_COMPLETE.json",
    )


if __name__ == "__main__":
    main()

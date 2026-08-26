#!/usr/bin/env python3
"""用probe-train内部validation拟合Q_continue全局Huber截距并冻结回放。

该升级不增加输入、输出头或候选网格，也不读取policy calibration或heldout
标签来拟合偏置。每个(lambda, mu)候选只增加一个由内部validation估计的标量。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.dynamic_optimal_stopping_deployable_v2 import (
    candidate_feasibility_counts,
    recursive_q_continue_targets,
    simulate_deployable_dynamic_policy,
)
from src.final_paper_inference import atomic_torch_save
from src.final_paper_protocol import canonical_fingerprint
from src.legacy_empirical_probe_v4 import load_checkpoint_split
from src.utils import atomic_json, load_yaml


def strip_records(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = dict(payload)
    return summary, summary.pop("records")


def fit_huber_intercept(residual: np.ndarray, beta: float) -> float:
    """求argmin_b mean huber(pred+b,target)，不依赖scipy。"""
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Huber截距残差无效")
    low = float(values.min() - beta - 1.0)
    high = float(values.max() + beta + 1.0)
    # derivative wrt b is clip(b - residual, -beta, beta)
    for _ in range(100):
        middle = (low + high) / 2.0
        derivative = float(np.clip(middle - values, -beta, beta).mean())
        if derivative > 0:
            high = middle
        else:
            low = middle
    return (low + high) / 2.0


def huber_loss(error: np.ndarray, beta: float) -> float:
    absolute = np.abs(np.asarray(error, dtype=np.float64))
    return float(np.where(
        absolute <= beta,
        0.5 * absolute * absolute / beta,
        absolute - 0.5 * beta,
    ).mean())


def choose_empirical(
    curve: list[dict[str, Any]], dense: dict[str, Any], budget: int, epsilon: float,
) -> dict[str, Any]:
    feasible = [
        row for row in curve
        if int(row["lost_correct_count"]) <= int(budget)
        and float(row["accuracy"]) >= float(dense["dense_accuracy"]) - epsilon
    ]
    if feasible:
        selected = dict(min(feasible, key=lambda row: (
            float(row["mean_reasoning_tokens"]),
            -float(row["token_reduction"]),
            -float(row["coverage"]),
            int(row["candidate_index"]),
        )))
        selected["dense_fallback"] = False
    else:
        selected = dict(dense)
        selected.update({"candidate_index": None, "selected_candidate": "dense", "dense_fallback": True})
    selected.update({
        "selection_family": "empirical_B_with_accuracy_constraint",
        "budget_B": int(budget),
        "accuracy_epsilon": float(epsilon),
    })
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--full-probe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    marker = args.output / "phase.complete"
    if args.resume and marker.is_file():
        print(json.dumps({"status": "skipped_complete", "output": str(args.output)}))
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"拒绝覆盖非空目录：{args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    config = load_yaml(args.config)
    dynamic = config["dynamic_policy"]
    dataset_config = config["datasets"][args.dataset]
    source_probe = json.loads((args.full_probe_root / "probe.json").read_text(encoding="utf-8"))
    predictions = torch.load(
        args.full_probe_root / "predictions.pt", map_location="cpu", weights_only=False
    )
    candidates = source_probe["run_spec"]["candidate_grid"]
    lambdas = np.asarray([float(row["lambda"]) for row in candidates], dtype=np.float64)
    mus = np.asarray([float(row["mu"]) for row in candidates], dtype=np.float64)

    frames = {}
    fallbacks = {}
    for split in ("probe_train", "calibration", "heldout"):
        frame, _, _, fallback = load_checkpoint_split(args.raw_root / split, "sentence")
        expected_ids = [str(value) for value in predictions["problem_ids"][split]]
        expected_checkpoints = [int(value) for value in predictions["checkpoints"][split]]
        if expected_ids != frame.problem_id.astype(str).tolist():
            raise ValueError(f"{split} problem ID/row不对齐")
        if expected_checkpoints != frame.checkpoint.astype(int).tolist():
            raise ValueError(f"{split} checkpoint不对齐")
        expected_count = int(dataset_config[split])
        actual_count = int(frame.problem_id.nunique()) + len(fallback)
        if actual_count != expected_count:
            raise ValueError(f"{split}题数{actual_count} != {expected_count}")
        frames[split] = frame
        fallbacks[split] = fallback

    stop_probability = {
        split: predictions["stop_probability"][split].numpy().astype(np.float64)
        for split in frames
    }
    risk_probability = {
        split: predictions["risk_probability"][split].numpy().astype(np.float64)
        for split in frames
    }
    q_original = {
        split: predictions["q_continue_values"][split].numpy().astype(np.float64)
        for split in frames
    }

    targets, _ = recursive_q_continue_targets(
        frames["probe_train"],
        stop_probability["probe_train"],
        risk_probability["probe_train"],
        lambdas,
        mus,
        cost_unit_tokens=float(dynamic["cost_unit_tokens"]),
    )
    validation_ids = set(str(value) for value in source_probe["validation_problem_ids"])
    validation_mask = frames["probe_train"].problem_id.astype(str).isin(validation_ids).to_numpy()
    if int(validation_mask.sum()) == 0:
        raise ValueError("内部validation没有checkpoint")
    beta = float(config["probe"]["value_huber_beta"])
    biases = np.zeros(len(candidates), dtype=np.float64)
    diagnostics = []
    for candidate in candidates:
        index = int(candidate["candidate_index"])
        target = targets[validation_mask, index].astype(np.float64)
        prediction = q_original["probe_train"][validation_mask, index]
        residual = target - prediction
        bias = fit_huber_intercept(residual, beta)
        biases[index] = bias
        diagnostics.append({
            **candidate,
            "q_bias": float(bias),
            "validation_rows": int(len(residual)),
            "validation_mean_residual_before": float(residual.mean()),
            "validation_median_residual_before": float(np.median(residual)),
            "validation_mae_before": float(np.abs(residual).mean()),
            "validation_mae_after": float(np.abs(residual - bias).mean()),
            "validation_huber_before": huber_loss(residual, beta),
            "validation_huber_after": huber_loss(residual - bias, beta),
        })
    q_corrected = {
        split: q_original[split] + biases[None, :]
        for split in frames
    }

    dense_calibration = simulate_deployable_dynamic_policy(
        frames["calibration"], stop_probability["calibration"], risk_probability["calibration"],
        np.zeros(len(frames["calibration"]), dtype=np.float64), mu_value=0.0,
        fallback_records=fallbacks["calibration"], force_dense=True,
    )
    dense_calibration.update({
        "selected_candidate": "dense", "candidate_index": None,
        "lambda": None, "mu": None,
    })
    calibration_curve = []
    for candidate in candidates:
        index = int(candidate["candidate_index"])
        row = simulate_deployable_dynamic_policy(
            frames["calibration"], stop_probability["calibration"], risk_probability["calibration"],
            q_corrected["calibration"][:, index], mu_value=float(candidate["mu"]),
            fallback_records=fallbacks["calibration"],
        )
        row.update(candidate)
        row["q_bias"] = float(biases[index])
        calibration_curve.append(row)

    epsilon = float(dynamic["accuracy_epsilon"])
    selected = {
        str(int(budget)): choose_empirical(
            calibration_curve, dense_calibration, int(budget), epsilon
        )
        for budget in dynamic["empirical_B"]
    }
    results = {}
    records_by_budget = {}
    for key, calibration_selection in selected.items():
        if calibration_selection.get("dense_fallback"):
            evaluated = simulate_deployable_dynamic_policy(
                frames["heldout"], stop_probability["heldout"], risk_probability["heldout"],
                np.zeros(len(frames["heldout"]), dtype=np.float64), mu_value=0.0,
                fallback_records=fallbacks["heldout"], include_records=True, force_dense=True,
            )
        else:
            index = int(calibration_selection["candidate_index"])
            candidate = candidates[index]
            evaluated = simulate_deployable_dynamic_policy(
                frames["heldout"], stop_probability["heldout"], risk_probability["heldout"],
                q_corrected["heldout"][:, index], mu_value=float(candidate["mu"]),
                fallback_records=fallbacks["heldout"], include_records=True,
            )
        summary, records = strip_records(evaluated)
        results[key] = {"calibration": calibration_selection, "heldout": summary}
        records_by_budget[key] = records

    run_spec = {
        "dataset": args.dataset,
        "method": "dynamic_full_plus_validation_huber_q_intercept",
        "source_run_spec_fingerprint": source_probe["run_spec_fingerprint"],
        "q_bias_fit_scope": "probe_train_internal_validation_only",
        "q_bias_fit_loss": f"Huber(beta={beta}) scalar intercept per frozen candidate",
        "policy_calibration_used_for_q_bias": False,
        "heldout_used_for_q_bias": False,
        "candidate_grid_unchanged": True,
        "candidate_grid": candidates,
        "empirical_B": [int(value) for value in dynamic["empirical_B"]],
    }
    payload = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_spec": run_spec,
        "run_spec_fingerprint": canonical_fingerprint(run_spec),
        "source_full_probe": str(args.full_probe_root),
        "fit_validation_problem_ids": sorted(validation_ids),
        "q_bias_diagnostics": diagnostics,
        "calibration": {
            "dense_sentinel": dense_calibration,
            "curve": calibration_curve,
            "selected": selected,
            "candidate_feasibility": candidate_feasibility_counts(
                calibration_curve, float(dense_calibration["dense_accuracy"]), epsilon,
                [int(value) for value in dynamic["empirical_B"]],
            ),
        },
        "frozen_policy_results": results,
        "heldout_used_for_selection": False,
    }
    atomic_json(payload, args.output / "q_bias_upgrade.json")
    atomic_torch_save({
        "status": "complete",
        "q_bias": torch.from_numpy(biases.astype(np.float32)),
        "records": {"empirical_B": records_by_budget},
    }, args.output / "policy_records.pt")
    atomic_json({
        "status": "complete",
        "run_spec_fingerprint": payload["run_spec_fingerprint"],
        "artifacts": ["q_bias_upgrade.json", "policy_records.pt"],
    }, marker)
    print(json.dumps({
        "status": "complete", "dataset": args.dataset,
        "mean_abs_q_bias": float(np.abs(biases).mean()),
        "selected": selected,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replay frozen DeepSeek-7B 13K probes under broader OOD calibration policies.

This script is deliberately read-only with respect to the completed v2 experiment.
It reuses cached probe scores and forced-answer trajectories, writes to a separate
output directory, and evaluates:

1. Expanded problem-level lost-correct budgets, with and without the historical
   one-percentage-point calibration-accuracy guard.
2. The class-conditional split-conformal decision rule used by the public LYNX
   implementation, applied to the paragraph correctness probe and, as an
   explicitly adapted wrapper, to the correction-risk probes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.train_deepseek7b_ablation_v1 import apply_semantic_answer_targets
from src.legacy_empirical_probe_normalized_v1 import (
    load_checkpoint_split,
    select_empirical_budget,
    simulate_policy,
    target_values,
)


LYNX_CONFIDENCES = (0.97, 0.95, 0.90, 0.80, 0.70)
EXPANDED_BUDGETS = (0, 1, 2, 4, 10, 20, 21, 35, 50, 70, 100, 140, 210)
METHODS = ("correctness", "bce", "bce_traj")
TEST_DATASETS = ("math500", "aime")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def conformal_quantile(scores: np.ndarray, delta: float) -> float:
    """Exact finite-sample quantile from the public LYNX implementation."""
    values = np.asarray(scores, dtype=np.float64)
    n = int(len(values))
    if n <= 0:
        return float("inf")
    k = int(math.ceil((n + 1) * (1.0 - float(delta))))
    k = max(1, min(k, n))
    return float(np.sort(values)[k - 1])


def prepare_frame(root: Path, split: str, schedule: str):
    frame, _hidden, _layers, fallbacks = load_checkpoint_split(root / split, schedule)
    del _hidden
    frame = apply_semantic_answer_targets(frame)
    frame["branch_tokens"] = 0
    frame["replay_stop_wall_ms"] = frame.checkpoint.astype(float)
    frame["dense_wall_ms"] = frame.dense_tokens.astype(float)
    frame["adaptive_fallback_wall_ms"] = frame.dense_tokens.astype(float)
    for fallback in fallbacks:
        fallback["dense_wall_ms"] = float(fallback["dense_tokens"])
        fallback["adaptive_fallback_wall_ms"] = float(fallback["dense_tokens"])
    return frame, fallbacks


def load_score_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete score payload: {path}")
    return payload


def checked_scores(
    payload: dict[str, Any], split: str, frame, *, source: Path
) -> np.ndarray:
    scores = payload["scores"][split].detach().cpu().numpy().astype(np.float64)
    ids = [str(value) for value in payload["problem_ids"][split]]
    checkpoints = [int(value) for value in payload["checkpoints"][split]]
    frame_ids = frame.problem_id.astype(str).tolist()
    frame_checkpoints = frame.checkpoint.astype(int).tolist()
    if ids != frame_ids or checkpoints != frame_checkpoints or len(scores) != len(frame):
        raise ValueError(f"score/frame alignment mismatch: {source}:{split}")
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite scores: {source}:{split}")
    return scores


def fastest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(
        min(
            rows,
            key=lambda row: (
                float(row["mean_replay_wall_ms"]),
                -float(row["token_reduction"]),
                -float(row["coverage"]),
                float(row["threshold"]),
            ),
        )
    )


def select_budget_with_guard(
    curve: list[dict[str, Any]], budget: int, epsilon: float
) -> dict[str, Any]:
    dense_row = next(row for row in curve if row.get("is_no_stop_sentinel"))
    feasible = [
        row
        for row in curve
        if int(row["lost_correct_count"]) <= int(budget)
        and float(row["accuracy"])
        >= float(dense_row["dense_accuracy"]) - float(epsilon)
    ]
    chosen = dict(dense_row) if not feasible else fastest(feasible)
    chosen["budget_B"] = int(budget)
    chosen["accuracy_epsilon"] = float(epsilon)
    return chosen


def metric_projection(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_problems": int(metrics["problems"]),
        f"{prefix}_accuracy": float(metrics["accuracy"]),
        f"{prefix}_dense_accuracy": float(metrics["dense_accuracy"]),
        f"{prefix}_accuracy_delta_pp": 100.0
        * (float(metrics["accuracy"]) - float(metrics["dense_accuracy"])),
        f"{prefix}_token_reduction": float(metrics["token_reduction"]),
        f"{prefix}_coverage": float(metrics["coverage"]),
        f"{prefix}_fallback_rate": float(metrics["fallback_rate"]),
        f"{prefix}_lost_correct_count": int(metrics["lost_correct_count"]),
        f"{prefix}_lost_correct_rate": float(metrics["lost_correct_rate"]),
        f"{prefix}_mean_tokens": float(metrics["mean_reasoning_and_answer_tokens"]),
        f"{prefix}_mean_dense_tokens": float(metrics["mean_dense_reasoning_tokens"]),
    }


def result_row(
    *,
    family: str,
    method: str,
    dataset: str,
    threshold: float,
    calibration: dict[str, Any],
    heldout: dict[str, Any],
    budget: int | None = None,
    confidence: float | None = None,
    delta: float | None = None,
    q_stop: float | None = None,
    q_continue: float | None = None,
    accuracy_guard_pp: float | None = None,
) -> dict[str, Any]:
    return {
        "family": family,
        "method": method,
        "dataset": dataset,
        "budget_B": budget,
        "budget_rate_over_700": None if budget is None else float(budget / 700.0),
        "confidence_c": confidence,
        "delta": delta,
        "threshold": float(threshold),
        "q_stop_class": q_stop,
        "q_continue_class": q_continue,
        "accuracy_guard_pp": accuracy_guard_pp,
        **metric_projection(calibration, "calibration"),
        **metric_projection(heldout, "test"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requested-gpu", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    source_root = args.source_root.resolve()
    probes_root = source_root / "probes"
    cache_root = source_root / "cache"
    schedule = "sentence"  # Stored loader key; actual checkpoint label is paragraph.

    frames: dict[str, Any] = {}
    fallbacks: dict[str, Any] = {}
    frames["calibration"], fallbacks["calibration"] = prepare_frame(
        cache_root / "math", "calibration", schedule
    )
    frames["math500"], fallbacks["math500"] = prepare_frame(
        cache_root / "math500", "heldout", schedule
    )
    frames["aime"], fallbacks["aime"] = prepare_frame(
        cache_root / "aime", "heldout", schedule
    )

    reports: dict[str, dict[str, Any]] = {}
    score_sets: dict[str, dict[str, np.ndarray]] = {}
    source_files: dict[str, Any] = {}
    for method in METHODS:
        math_dir = probes_root / "math500" / method
        aime_dir = probes_root / "aime" / method
        report_path = math_dir / "probe.json"
        math_score_path = math_dir / "scores.pt"
        aime_score_path = aime_dir / "scores.pt"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise ValueError(f"incomplete probe report: {report_path}")
        reports[method] = report
        math_payload = load_score_payload(math_score_path)
        aime_payload = load_score_payload(aime_score_path)
        score_sets[method] = {
            "calibration": checked_scores(
                math_payload, "calibration", frames["calibration"], source=math_score_path
            ),
            "math500": checked_scores(
                math_payload, "heldout", frames["math500"], source=math_score_path
            ),
            "aime": checked_scores(
                aime_payload, "heldout", frames["aime"], source=aime_score_path
            ),
        }
        source_files[method] = {
            "probe_json": str(report_path.resolve()),
            "probe_json_sha256": sha256(report_path),
            "math_scores": str(math_score_path.resolve()),
            "math_scores_sha256": sha256(math_score_path),
            "aime_scores": str(aime_score_path.resolve()),
            "aime_scores_sha256": sha256(aime_score_path),
        }

    rows: list[dict[str, Any]] = []

    # Expanded B sweep for the two correction-potential objectives.
    for method in ("bce", "bce_traj"):
        report = reports[method]
        curve = report["calibration"]["curve"]
        direction = str(report["run_spec"]["stop_direction"])
        epsilon = float(report["run_spec"]["calibration_accuracy_epsilon"])
        for budget in EXPANDED_BUDGETS:
            selections = {
                "empirical_B_only": select_empirical_budget(curve, budget),
                "empirical_B_with_1pp_guard": select_budget_with_guard(
                    curve, budget, epsilon
                ),
            }
            for family, selected in selections.items():
                threshold = float(selected["threshold"])
                for dataset in TEST_DATASETS:
                    heldout = simulate_policy(
                        frames[dataset],
                        score_sets[method][dataset],
                        direction,
                        threshold,
                        fallback_records=fallbacks[dataset],
                        force_dense=bool(selected.get("is_no_stop_sentinel", False)),
                    )
                    rows.append(
                        result_row(
                            family=family,
                            method=method,
                            dataset=dataset,
                            threshold=threshold,
                            calibration=selected,
                            heldout=heldout,
                            budget=budget,
                            accuracy_guard_pp=(100.0 * epsilon if "guard" in family else None),
                        )
                    )

    # Public LYNX class-conditional split conformal rule.
    for method in METHODS:
        report = reports[method]
        direction = str(report["run_spec"]["stop_direction"])
        calibration_scores = score_sets[method]["calibration"]
        target_method = "correctness" if method == "correctness" else "correction"
        labels = target_values(frames["calibration"], target_method).astype(np.int64)
        for confidence in LYNX_CONFIDENCES:
            delta = 1.0 - confidence
            if method == "correctness":
                # LYNX native semantics: class 1 means a forced exit is correct.
                q_stop = conformal_quantile(1.0 - calibration_scores[labels == 1], delta)
                q_continue = conformal_quantile(calibration_scores[labels == 0], delta)
                threshold = max(
                    1.0 - q_stop,
                    float(np.nextafter(q_continue, np.inf)),
                )
                family = "lynx_conformal_correctness"
            else:
                # Adaptation to our label: class 0 is safe-to-stop, class 1 is W->C danger.
                q_stop = conformal_quantile(calibration_scores[labels == 0], delta)
                q_continue = conformal_quantile(1.0 - calibration_scores[labels == 1], delta)
                threshold = min(
                    q_stop,
                    float(np.nextafter(1.0 - q_continue, -np.inf)),
                )
                family = "lynx_conformal_correction_adapted"
            calibration = simulate_policy(
                frames["calibration"],
                calibration_scores,
                direction,
                threshold,
                fallback_records=fallbacks["calibration"],
            )
            calibration["conformal_n_stop"] = int((labels == (1 if method == "correctness" else 0)).sum())
            calibration["conformal_n_continue"] = int((labels == (0 if method == "correctness" else 1)).sum())
            for dataset in TEST_DATASETS:
                heldout = simulate_policy(
                    frames[dataset],
                    score_sets[method][dataset],
                    direction,
                    threshold,
                    fallback_records=fallbacks[dataset],
                )
                rows.append(
                    result_row(
                        family=family,
                        method=method,
                        dataset=dataset,
                        threshold=threshold,
                        calibration=calibration,
                        heldout=heldout,
                        confidence=confidence,
                        delta=delta,
                        q_stop=q_stop,
                        q_continue=q_continue,
                    )
                )

    expected_rows = (
        2 * len(EXPANDED_BUDGETS) * 2 * len(TEST_DATASETS)
        + len(METHODS) * len(LYNX_CONFIDENCES) * len(TEST_DATASETS)
    )
    if len(rows) != expected_rows:
        raise AssertionError((len(rows), expected_rows))

    json_path = args.output / "RESULTS.json"
    csv_path = args.output / "RESULTS.csv"
    atomic_json(rows, json_path)
    fieldnames = list(rows[0].keys())
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)

    manifest = {
        "status": "complete",
        "created_at": now(),
        "source_root": str(source_root),
        "source_completion_audit_sha256": sha256(source_root / "COMPLETION_AUDIT.json"),
        "source_files": source_files,
        "requested_gpu": int(args.requested_gpu),
        "execution_note": (
            "Replay consumes cached CPU tensors only; CUDA_VISIBLE_DEVICES may bind GPU0 "
            "but no base-model generation or GPU allocation is required."
        ),
        "checkpoint_protocol": "paragraph (stored loader key: sentence)",
        "dense_budget": 13000,
        "forced_answer_max_new_tokens": 48,
        "expanded_budgets": list(EXPANDED_BUDGETS),
        "lynx_confidences": list(LYNX_CONFIDENCES),
        "lynx_rule": {
            "implementation": "finite-sample class-conditional split conformal",
            "quantile_rank": "ceil((n+1)*(1-delta)), clipped to [1,n]",
            "exit_condition": "binary predictive set equals the stop/safe class singleton",
            "paper": "https://arxiv.org/abs/2512.05325",
            "code": "https://github.com/farukakgul/LYNX",
        },
        "scientific_caveat": (
            "The correctness row uses the LYNX calibration rule on our paragraph probe. "
            "The correction rows are an adapted conformal wrapper around W->C risk, not native LYNX."
        ),
        "results_rows": len(rows),
        "results_json_sha256": sha256(json_path),
        "results_csv_sha256": sha256(csv_path),
    }
    atomic_json(manifest, args.output / "RUN_MANIFEST.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

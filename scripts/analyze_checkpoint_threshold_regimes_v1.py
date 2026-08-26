#!/usr/bin/env python3
"""Replay frozen checkpoint probes under three threshold-selection regimes."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SCHEDULES = (
    "sentence",
    "fixed_budget",
    "prefix_stride",
    "lynx_cue",
    "paragraph",
    "hybrid",
)
POINT_TARGETS = ("correctness", "consistency", "last_switch", "correction_bce")
TRAJECTORY_TARGET = "correction_trajectory_normalized"
PERCENT_RATES = (0.0001, 0.0002, 0.0004)
PRIMARY_PERCENT_RATE = 0.0002
FIXED_THRESHOLD = 0.5
FIXED_THRESHOLDS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-filter", choices=("gsm8k", "mmlu_pro"))
    parser.add_argument("--schedule-filter", choices=SCHEDULES)
    parser.add_argument("--ours-only", action="store_true")
    return parser.parse_args()


def fastest_with_constraints(
    curve: list[dict[str, Any]], budget: int, epsilon: float
) -> dict[str, Any]:
    dense = next(row for row in curve if row.get("is_no_stop_sentinel"))
    feasible = [
        row
        for row in curve
        if int(row["lost_correct_count"]) <= budget
        and float(row["accuracy"]) >= float(dense["dense_accuracy"]) - epsilon
    ]
    if not feasible:
        selected = dict(dense)
    else:
        selected = dict(
            min(
                feasible,
                key=lambda row: (
                    float(row["mean_reasoning_and_answer_tokens"]),
                    -float(row["token_reduction"]),
                    -float(row["coverage"]),
                    float(row["threshold"]),
                ),
            )
        )
    selected["budget_B"] = int(budget)
    selected["accuracy_epsilon"] = float(epsilon)
    return selected


def normalize_cost_frame(frame, fallbacks: list[dict[str, Any]]) -> None:
    frame["branch_tokens"] = 0
    frame["replay_stop_wall_ms"] = frame.checkpoint.astype(float)
    frame["dense_wall_ms"] = frame.dense_tokens.astype(float)
    frame["adaptive_fallback_wall_ms"] = frame.dense_tokens.astype(float)
    for fallback in fallbacks:
        fallback["dense_wall_ms"] = float(fallback["dense_tokens"])
        fallback["adaptive_fallback_wall_ms"] = float(fallback["dense_tokens"])


def load_checkpoint_frame_only(
    directory: Path,
    schedule: str,
):
    """Match the training frame order without retaining the large hidden tensor."""
    paths = sorted(directory.glob("sample_*.pt"))
    if not paths:
        raise FileNotFoundError(directory)
    all_rows: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    required = (
        "problem_id",
        "checkpoint",
        "current_success",
        "dense_success",
        "dense_tokens",
    )
    for index, path in enumerate(paths):
        artifact = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
        if artifact.get("status") != "complete":
            raise ValueError(f"incomplete checkpoint artifact: {path}")
        selected = [
            {key: row.get(key) for key in required}
            for row in artifact.get("rows", [])
            if schedule in row.get("checkpoint_schedules", [])
        ]
        all_rows.extend(selected)
        if not selected:
            dense = artifact["dense"]
            fallbacks.append(
                {
                    "problem_id": str(artifact["problem_id"]),
                    "dense_success": bool(dense["success"]),
                    "dense_tokens": int(dense["reasoning_tokens"]),
                }
            )
        del artifact, selected
        if index % 128 == 0:
            gc.collect()
    if not all_rows:
        raise ValueError(f"no legal checkpoints in {directory}")
    frame = pd.DataFrame(all_rows)
    frame = frame.sort_values(
        ["problem_id", "checkpoint"], kind="stable"
    ).reset_index(drop=True)
    if frame.duplicated(["problem_id", "checkpoint"]).any():
        raise ValueError(f"duplicate problem/checkpoint rows in {directory}")
    frame["problem_id"] = frame.problem_id.astype(str)
    if set(frame.problem_id) & {row["problem_id"] for row in fallbacks}:
        raise ValueError(f"scorable/fallback overlap in {directory}")
    if int(frame.problem_id.nunique()) + len(fallbacks) != len(paths):
        raise ValueError(f"problem accounting mismatch in {directory}")
    return frame, fallbacks


def simulate_compact(
    frame: pd.DataFrame,
    scores: np.ndarray,
    direction: str,
    threshold: float,
    *,
    fallback_records: list[dict[str, Any]] | None = None,
    force_dense: bool = False,
) -> dict[str, Any]:
    """First-hit replay equivalent for the frozen reasoning-token-only protocol."""
    if len(frame) != len(scores):
        raise ValueError("frame/score mismatch")
    scores = np.asarray(scores, dtype=np.float64)
    method_success: list[bool] = []
    dense_success: list[bool] = []
    method_tokens: list[int] = []
    dense_tokens: list[int] = []
    stopped: list[bool] = []
    for _, positions in frame.groupby("problem_id", sort=False).indices.items():
        idx = np.asarray(positions, dtype=np.int64)
        group = frame.iloc[idx]
        first = group.iloc[0]
        if force_dense:
            eligible = np.empty(0, dtype=np.int64)
        elif direction == "high":
            eligible = idx[scores[idx] >= threshold]
        else:
            eligible = idx[scores[idx] <= threshold]
        if len(eligible):
            chosen = frame.iloc[int(eligible[0])]
            method_success.append(bool(chosen.current_success))
            method_tokens.append(min(int(chosen.checkpoint), int(chosen.dense_tokens)))
            stopped.append(True)
        else:
            method_success.append(bool(first.dense_success))
            method_tokens.append(int(first.dense_tokens))
            stopped.append(False)
        dense_success.append(bool(first.dense_success))
        dense_tokens.append(int(first.dense_tokens))
    for fallback in fallback_records or []:
        method_success.append(bool(fallback["dense_success"]))
        dense_success.append(bool(fallback["dense_success"]))
        method_tokens.append(int(fallback["dense_tokens"]))
        dense_tokens.append(int(fallback["dense_tokens"]))
        stopped.append(False)
    method_success_a = np.asarray(method_success, dtype=bool)
    dense_success_a = np.asarray(dense_success, dtype=bool)
    method_tokens_a = np.asarray(method_tokens, dtype=np.float64)
    dense_tokens_a = np.asarray(dense_tokens, dtype=np.float64)
    stopped_a = np.asarray(stopped, dtype=bool)
    n = len(method_success_a)
    lost = int(np.sum(dense_success_a & ~method_success_a))
    return {
        "problems": n,
        "fallback": int(np.sum(~stopped_a)),
        "fallback_rate": float(np.mean(~stopped_a)),
        "coverage": float(np.mean(stopped_a)),
        "accuracy": float(np.mean(method_success_a)),
        "dense_accuracy": float(np.mean(dense_success_a)),
        "accuracy_drop_pp": float(
            100.0 * (np.mean(dense_success_a) - np.mean(method_success_a))
        ),
        "lost_correct_count": lost,
        "lost_correct_rate": float(lost / n),
        "mean_reasoning_and_answer_tokens": float(np.mean(method_tokens_a)),
        "mean_dense_reasoning_tokens": float(np.mean(dense_tokens_a)),
        "token_reduction": float(
            1.0 - np.mean(method_tokens_a) / np.mean(dense_tokens_a)
        ),
        "mean_replay_wall_ms": float(np.mean(method_tokens_a)),
        "mean_dense_wall_ms": float(np.mean(dense_tokens_a)),
        "threshold": float(threshold),
    }


def evaluate_threshold(
    frame,
    scores: np.ndarray,
    direction: str,
    threshold: float,
    fallbacks: list[dict[str, Any]],
    *,
    force_dense: bool = False,
) -> dict[str, Any]:
    return simulate_compact(
        frame,
        scores,
        direction,
        float(threshold),
        fallback_records=fallbacks,
        force_dense=force_dense,
    )


def flattened_row(
    *,
    dataset: str,
    schedule: str,
    target: str,
    regime: str,
    calibration: dict[str, Any],
    heldout: dict[str, Any],
    checkpoint_rows: int,
    checkpoint_rate: float | None = None,
    requested_budget: int | None = None,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "schedule": schedule,
        "target": target,
        "regime": regime,
        "calibration_checkpoint_rows": int(checkpoint_rows),
        "checkpoint_percentage": checkpoint_rate,
        "requested_budget_B": requested_budget,
        "threshold": float(calibration["threshold"]),
        "calibration_lost_correct": int(calibration["lost_correct_count"]),
        "calibration_lost_correct_rate": float(calibration["lost_correct_rate"]),
        "calibration_accuracy": float(calibration["accuracy"]),
        "calibration_dense_accuracy": float(calibration["dense_accuracy"]),
        "calibration_accuracy_delta_pp": 100.0
        * (float(calibration["accuracy"]) - float(calibration["dense_accuracy"])),
        "calibration_token_reduction": float(calibration["token_reduction"]),
        "calibration_coverage": float(calibration["coverage"]),
        "heldout_lost_correct": int(heldout["lost_correct_count"]),
        "heldout_lost_correct_rate": float(heldout["lost_correct_rate"]),
        "heldout_accuracy": float(heldout["accuracy"]),
        "heldout_dense_accuracy": float(heldout["dense_accuracy"]),
        "heldout_accuracy_delta_pp": 100.0
        * (float(heldout["accuracy"]) - float(heldout["dense_accuracy"])),
        "heldout_mean_tokens": float(heldout["mean_reasoning_and_answer_tokens"]),
        "heldout_mean_dense_tokens": float(heldout["mean_dense_reasoning_tokens"]),
        "heldout_token_reduction": float(heldout["token_reduction"]),
        "heldout_coverage": float(heldout["coverage"]),
    }


def probe_sources(project_root: Path, dataset: str) -> dict[tuple[str, str], Path]:
    sources: dict[tuple[str, str], Path] = {}
    if dataset == "gsm8k":
        base = project_root / "results/gsm8k_full_checkpoint_schedule_ablation_v1/probes"
        traj = project_root / "results/gsm8k_checkpoint_schedule_normalized_trajectory_v1/probes"
        for schedule in SCHEDULES:
            for target in POINT_TARGETS:
                sources[(schedule, target)] = base / schedule / target
            sources[(schedule, TRAJECTORY_TARGET)] = traj / schedule / TRAJECTORY_TARGET
    else:
        base = project_root / "results/mmlu_pro_checkpoint_followup_v1/probes"
        traj = project_root / "results/mmlu_pro_checkpoint_schedule_normalized_trajectory_v1/probes"
        for schedule in ("sentence", "paragraph", "lynx_cue"):
            for target in POINT_TARGETS:
                candidate = base / schedule / target
                if (candidate / "probe.json").is_file():
                    sources[(schedule, target)] = candidate
            candidate = traj / schedule / TRAJECTORY_TARGET
            if (candidate / "probe.json").is_file():
                sources[(schedule, TRAJECTORY_TARGET)] = candidate
    return sources


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = project_root / output_root
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project_root))

    rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    fixed_threshold_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    print(json.dumps({"status": "started", "output": str(output_root)}), flush=True)

    datasets = (args.dataset_filter,) if args.dataset_filter else ("gsm8k", "mmlu_pro")
    for dataset in datasets:
        sources = probe_sources(project_root, dataset)
        schedules = [s for s in SCHEDULES if any(key[0] == s for key in sources)]
        if args.schedule_filter:
            schedules = [s for s in schedules if s == args.schedule_filter]
        for schedule in schedules:
            print(
                json.dumps(
                    {"status": "loading", "dataset": dataset, "schedule": schedule}
                ),
                flush=True,
            )
            local_sources = {
                target: path
                for (source_schedule, target), path in sources.items()
                if source_schedule == schedule
            }
            if args.ours_only:
                local_sources = {
                    target: path
                    for target, path in local_sources.items()
                    if target in ("correction_bce", TRAJECTORY_TARGET)
                }
            if not local_sources:
                continue
            sample_json = json.loads(
                next(iter(local_sources.values())).joinpath("probe.json").read_text()
            )
            raw_root = Path(sample_json["input"]["root"])
            calibration_frame, calibration_fallbacks = load_checkpoint_frame_only(
                raw_root / "calibration",
                "sentence",
            )
            gc.collect()
            heldout_frame, heldout_fallbacks = load_checkpoint_frame_only(
                raw_root / "heldout",
                "sentence",
            )
            gc.collect()
            checkpoint_rows = len(calibration_frame)

            for target, probe_path in sorted(local_sources.items()):
                probe_json = json.loads((probe_path / "probe.json").read_text())
                score_payload = torch.load(
                    probe_path / "scores.pt", map_location="cpu", weights_only=False
                )
                cal_scores = np.asarray(score_payload["scores"]["calibration"])
                test_scores = np.asarray(score_payload["scores"]["heldout"])
                del score_payload
                if len(cal_scores) != len(calibration_frame):
                    raise ValueError(
                        f"calibration alignment failure {dataset}/{schedule}/{target}: "
                        f"{len(cal_scores)} != {len(calibration_frame)}"
                    )
                if len(test_scores) != len(heldout_frame):
                    raise ValueError(
                        f"heldout alignment failure {dataset}/{schedule}/{target}: "
                        f"{len(test_scores)} != {len(heldout_frame)}"
                    )
                direction = str(probe_json["run_spec"]["stop_direction"])
                epsilon = float(probe_json["run_spec"]["calibration_accuracy_epsilon"])
                curve = probe_json["calibration"]["curve"]

                # Regime 1: existing fixed problem-level B=2.
                fixed_b = probe_json["frozen_policy_results"]["empirical_B"]["2"]
                rows.append(
                    flattened_row(
                        dataset=dataset,
                        schedule=schedule,
                        target=target,
                        regime="fixed_problem_B2",
                        calibration=fixed_b["calibration"],
                        heldout=fixed_b["heldout"],
                        checkpoint_rows=checkpoint_rows,
                        requested_budget=2,
                    )
                )

                # Regime 2: B scales with the number of calibration checkpoint rows.
                for rate in PERCENT_RATES:
                    budget = int(math.floor(rate * checkpoint_rows + 0.5))
                    selected = fastest_with_constraints(curve, budget, epsilon)
                    heldout = evaluate_threshold(
                        heldout_frame,
                        test_scores,
                        direction,
                        float(selected["threshold"]),
                        heldout_fallbacks,
                        force_dense=bool(selected.get("is_no_stop_sentinel", False)),
                    )
                    record = flattened_row(
                        dataset=dataset,
                        schedule=schedule,
                        target=target,
                        regime=f"checkpoint_scaled_B_{100.0 * rate:.3f}pct",
                        calibration=selected,
                        heldout=heldout,
                        checkpoint_rows=checkpoint_rows,
                        checkpoint_rate=rate,
                        requested_budget=budget,
                    )
                    sensitivity_rows.append(record)
                    if rate == PRIMARY_PERCENT_RATE:
                        rows.append(record)

                # Regime 3: no calibration; fixed sigmoid thresholds shared by schedules.
                for fixed_threshold in FIXED_THRESHOLDS:
                    cal_fixed = evaluate_threshold(
                        calibration_frame,
                        cal_scores,
                        direction,
                        fixed_threshold,
                        calibration_fallbacks,
                    )
                    test_fixed = evaluate_threshold(
                        heldout_frame,
                        test_scores,
                        direction,
                        fixed_threshold,
                        heldout_fallbacks,
                    )
                    fixed_record = flattened_row(
                        dataset=dataset,
                        schedule=schedule,
                        target=target,
                        regime=f"fixed_threshold_{fixed_threshold:g}",
                        calibration=cal_fixed,
                        heldout=test_fixed,
                        checkpoint_rows=checkpoint_rows,
                    )
                    fixed_threshold_rows.append(fixed_record)
                    if fixed_threshold == FIXED_THRESHOLD:
                        rows.append(fixed_record)
                audits.append(
                    {
                        "dataset": dataset,
                        "schedule": schedule,
                        "target": target,
                        "probe_path": str(probe_path),
                        "calibration_rows": len(cal_scores),
                        "heldout_rows": len(test_scores),
                        "direction": direction,
                    }
                )
                print(
                    json.dumps(
                        {
                            "status": "replayed",
                            "dataset": dataset,
                            "schedule": schedule,
                            "target": target,
                        }
                    ),
                    flush=True,
                )
                del cal_scores, test_scores, probe_json
                gc.collect()

    write_csv(output_root / "threshold_regime_primary.csv", rows)
    write_csv(output_root / "checkpoint_percentage_sensitivity.csv", sensitivity_rows)
    write_csv(output_root / "fixed_threshold_sensitivity.csv", fixed_threshold_rows)
    payload = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "definitions": {
            "fixed_problem_B2": "existing schedule-specific calibration with at most 2 lost-correct calibration problems and Dense-1pp accuracy floor",
            "checkpoint_scaled_B": "B_s=round(rho * number of calibration checkpoint rows), retaining the Dense-1pp accuracy floor",
            "checkpoint_scaled_primary_rate": PRIMARY_PERCENT_RATE,
            "checkpoint_scaled_sensitivity_rates": list(PERCENT_RATES),
            "fixed_threshold": FIXED_THRESHOLD,
            "fixed_threshold_rule": "high-direction targets stop at score>=0.5; low-direction targets stop at score<=0.5; no B or accuracy floor",
        },
        "rows": len(rows),
        "sensitivity_rows": len(sensitivity_rows),
        "fixed_threshold_rows": len(fixed_threshold_rows),
        "audit": audits,
    }
    (output_root / "RUN_AUDIT.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

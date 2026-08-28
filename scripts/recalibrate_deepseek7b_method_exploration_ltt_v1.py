#!/usr/bin/env python3
"""Recalibrate frozen method-exploration finalists with trajectory-envelope LTT.

This is deliberately a read-only post-processing pass over already-frozen probes.
Candidate thresholds are constructed from probe-train *scores only*.  Labels from
the independent calibration split are then used to certify a conservative-to-
aggressive prefix with an exact one-sided binomial test.  Held-out and OOD labels
are never consulted during threshold selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import beta as beta_distribution
from scipy.stats import binom


@dataclass
class ReplayData:
    problem_ids: list[str]
    row_problem_ids: list[str]
    row_checkpoints: np.ndarray
    row_current_success: np.ndarray
    groups: list[tuple[int, int, bool, int]]
    fallbacks: list[tuple[str, bool, int]]

    @property
    def problems(self) -> int:
        return len(self.groups) + len(self.fallbacks)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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


def dense_fields(artifact: dict[str, Any]) -> tuple[bool, int]:
    dense = artifact.get("dense", {})
    if "success" in dense and "reasoning_tokens" in dense:
        return bool(dense["success"]), int(dense["reasoning_tokens"])
    rows = artifact.get("rows", [])
    if rows:
        return bool(rows[0]["dense_success"]), int(rows[0]["dense_tokens"])
    source_path = Path(artifact["source_dense_artifact"])
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    return bool(source["dense"]["success"]), int(source["dense"]["reasoning_tokens"])


def load_replay_data(directory: Path, schedule: str = "sentence") -> ReplayData:
    paths = sorted(directory.glob("sample_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no artifacts: {directory}")
    rows: list[tuple[str, int, bool, bool, int]] = []
    fallbacks: list[tuple[str, bool, int]] = []
    for path in paths:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("status") != "complete":
            raise ValueError(f"incomplete artifact: {path}")
        problem_id = str(artifact["problem_id"])
        selected = [
            row
            for row in artifact.get("rows", [])
            if schedule in row.get("checkpoint_schedules", [])
        ]
        if not selected:
            dense_success, dense_tokens = dense_fields(artifact)
            fallbacks.append((problem_id, dense_success, dense_tokens))
            continue
        for row in selected:
            rows.append(
                (
                    problem_id,
                    int(row["checkpoint"]),
                    bool(row["current_success"]),
                    bool(row["dense_success"]),
                    int(row["dense_tokens"]),
                )
            )
    rows.sort(key=lambda value: (value[0], value[1]))
    row_problem_ids = [value[0] for value in rows]
    checkpoints = np.asarray([value[1] for value in rows], dtype=np.int64)
    groups: list[tuple[int, int, bool, int]] = []
    problem_ids: list[str] = []
    start = 0
    while start < len(rows):
        problem_id = rows[start][0]
        end = start + 1
        while end < len(rows) and rows[end][0] == problem_id:
            end += 1
        dense_successes = {rows[index][3] for index in range(start, end)}
        dense_tokens = {rows[index][4] for index in range(start, end)}
        if len(dense_successes) != 1 or len(dense_tokens) != 1:
            raise AssertionError(f"inconsistent dense fields for {problem_id}")
        groups.append((start, end, dense_successes.pop(), dense_tokens.pop()))
        problem_ids.append(problem_id)
        start = end
    fallbacks.sort(key=lambda value: value[0])
    if len(problem_ids) + len(fallbacks) != len(paths):
        raise AssertionError(f"problem accounting mismatch: {directory}")
    return ReplayData(
        problem_ids,
        row_problem_ids,
        checkpoints,
        np.asarray([value[2] for value in rows], dtype=bool),
        groups,
        fallbacks,
    )


def threshold_grid(
    scores: np.ndarray, size: int, direction: str = "low"
) -> list[dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("invalid scores")
    quantiles = [float(x) for x in np.unique(np.quantile(values, np.linspace(0, 1, size)))]
    if direction == "low":
        thresholds = [-1.0] + quantiles
        if thresholds[-1] != 2.0:
            thresholds.append(2.0)
    elif direction == "high":
        thresholds = [2.0] + list(reversed(quantiles))
        if thresholds[-1] != -1.0:
            thresholds.append(-1.0)
    else:
        raise ValueError(f"unknown stop direction: {direction}")
    return [
        {"grid_index": index, "threshold": value, "is_no_stop_sentinel": index == 0}
        for index, value in enumerate(thresholds)
    ]


def align_scores(data: ReplayData, saved: dict[str, Any], split: str) -> np.ndarray:
    scores = np.asarray(saved["scores"][split], dtype=np.float64)
    keys = saved["keys"][split]
    ids = [str(value) for value in keys["problem_ids"]]
    checkpoints = np.asarray(keys["checkpoints"], dtype=np.int64)
    if ids != data.row_problem_ids or not np.array_equal(checkpoints, data.row_checkpoints):
        raise AssertionError(f"score/frame alignment mismatch for {split}")
    if len(scores) != len(ids):
        raise AssertionError(f"score length mismatch for {split}")
    return scores


def replay_curve(
    data: ReplayData,
    scores: np.ndarray,
    grid: list[dict[str, Any]],
    readout_suffix_tokens: int,
    direction: str = "low",
) -> tuple[list[dict[str, Any]], np.ndarray]:
    thresholds = np.asarray([row["threshold"] for row in grid], dtype=np.float64)
    force_dense = np.asarray(
        [bool(row["is_no_stop_sentinel"]) for row in grid], dtype=bool
    )
    candidates = len(grid)
    problems = data.problems
    correct = np.zeros(candidates, dtype=np.int64)
    lost = np.zeros((candidates, problems), dtype=bool)
    helped = np.zeros(candidates, dtype=np.int64)
    stopped = np.zeros(candidates, dtype=np.int64)
    total_reasoning = np.zeros(candidates, dtype=np.float64)
    total_deployed = np.zeros(candidates, dtype=np.float64)
    total_readout = np.zeros(candidates, dtype=np.float64)
    total_dense = 0.0
    current_success = data.row_current_success
    for problem_index, (start, end, dense_success, dense_tokens) in enumerate(data.groups):
        local_scores = scores[start:end]
        local_success = current_success[start:end]
        local_checkpoints = data.row_checkpoints[start:end]
        if direction == "low":
            mask = local_scores[None, :] <= thresholds[:, None]
        elif direction == "high":
            mask = local_scores[None, :] >= thresholds[:, None]
        else:
            raise ValueError(f"unknown stop direction: {direction}")
        any_stop = mask.any(axis=1)
        any_stop[force_dense] = False
        earliest = np.argmax(mask, axis=1)
        chosen_success = np.where(any_stop, local_success[earliest], dense_success)
        reasoning = np.where(any_stop, local_checkpoints[earliest], dense_tokens).astype(np.float64)
        visited = np.where(any_stop, earliest + 1, end - start).astype(np.float64)
        readout = visited * readout_suffix_tokens
        # The sentinel means the policy is disabled, so it performs no readout.
        readout[force_dense] = 0.0
        correct += chosen_success.astype(np.int64)
        lost[:, problem_index] = dense_success & (~chosen_success)
        helped += ((not dense_success) & chosen_success).astype(np.int64)
        stopped += any_stop.astype(np.int64)
        total_reasoning += reasoning
        total_readout += readout
        total_deployed += reasoning + readout
        total_dense += dense_tokens
    fallback_offset = len(data.groups)
    for offset, (_, dense_success, dense_tokens) in enumerate(data.fallbacks):
        correct += int(dense_success)
        total_reasoning += dense_tokens
        total_deployed += dense_tokens
        total_dense += dense_tokens
        lost[:, fallback_offset + offset] = False
    dense_correct = sum(int(group[2]) for group in data.groups) + sum(int(row[1]) for row in data.fallbacks)
    curve: list[dict[str, Any]] = []
    for index, spec in enumerate(grid):
        curve.append(
            {
                **spec,
                "problems": problems,
                "accuracy": float(correct[index] / problems),
                "dense_accuracy": float(dense_correct / problems),
                "accuracy_delta_pp": float(100 * (correct[index] - dense_correct) / problems),
                "lost_correct_count": int(lost[index].sum()),
                "lost_correct_rate": float(lost[index].mean()),
                "helped_count": int(helped[index]),
                "coverage": float(stopped[index] / problems),
                "mean_reasoning_tokens": float(total_reasoning[index] / problems),
                "mean_dense_tokens": float(total_dense / problems),
                "token_reduction": float(1 - total_reasoning[index] / total_dense),
                "mean_one_step_suffix_tokens": float(total_readout[index] / problems),
                "mean_deployed_token_equivalent": float(total_deployed[index] / problems),
                "deployed_token_reduction": float(1 - total_deployed[index] / total_dense),
                "readout_suffix_tokens_per_checkpoint": int(readout_suffix_tokens),
            }
        )
    return curve, lost


def first_failure_counts(lost: np.ndarray) -> np.ndarray:
    active = lost[1:]
    any_failure = active.any(axis=0)
    first = np.argmax(active, axis=0) + 1
    first = np.where(any_failure, first, lost.shape[0])
    return np.asarray([(first <= index).sum() for index in range(lost.shape[0])], dtype=np.int64)


def exact_allowed_k(n: int, alpha: float, delta: float) -> int:
    valid = [k for k in range(n + 1) if float(binom.cdf(k, n, alpha)) <= delta]
    return max(valid) if valid else -1


def cp_upper(k: int, n: int, delta: float) -> float:
    if k >= n:
        return 1.0
    return float(beta_distribution.ppf(1.0 - delta, k + 1, n - k))


def select_ltt(
    curve: list[dict[str, Any]], lost: np.ndarray, alpha: float, delta: float
) -> tuple[int, dict[str, Any]]:
    critical = first_failure_counts(lost)
    allowed = exact_allowed_k(int(curve[0]["problems"]), alpha, delta)
    certified = [0]
    first_failed = None
    for index in range(1, len(curve)):
        if int(critical[index]) <= allowed:
            certified.append(index)
        else:
            first_failed = index
            break
    selected = min(
        certified,
        key=lambda index: (
            -float(curve[index]["deployed_token_reduction"]),
            -float(curve[index]["coverage"]),
            index,
        ),
    )
    upper = 0.0 if selected == 0 else cp_upper(
        int(critical[selected]), int(curve[0]["problems"]), delta
    )
    return selected, {
        "allowed_first_failure_boundaries": int(allowed),
        "selected_first_failure_boundaries": int(critical[selected]),
        "trajectory_envelope_upper": float(upper),
        "certified_prefix_end": int(certified[-1]),
        "first_failed_index": first_failed,
        "candidate_count": len(curve),
    }


def compact_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_dense_accuracy": metrics["dense_accuracy"],
        f"{prefix}_accuracy_delta_pp": metrics["accuracy_delta_pp"],
        f"{prefix}_lost_correct": metrics["lost_correct_count"],
        f"{prefix}_helped": metrics["helped_count"],
        f"{prefix}_token_reduction": metrics["token_reduction"],
        f"{prefix}_deployed_token_reduction": metrics["deployed_token_reduction"],
        f"{prefix}_coverage": metrics["coverage"],
        f"{prefix}_mean_one_step_suffix_tokens": metrics["mean_one_step_suffix_tokens"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=101)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError("output must be separate from frozen source")
    final_audit = json.loads((source / "FINAL_AUDIT.json").read_text())
    if final_audit.get("status") != "complete":
        raise ValueError("source FINAL_AUDIT is incomplete")
    freeze = json.loads((source / "FINALIST_SELECTION_FREEZE.json").read_text())
    if freeze.get("calibration_or_test_used_for_selection") is not False:
        raise ValueError("source model selection is not leakage-free")
    finalist_ids = [item["combination"]["id"] for item in freeze["finalists"]]
    if len(finalist_ids) != 17 or len(set(finalist_ids)) != 17:
        raise AssertionError("expected exactly 17 frozen finalists")

    roots = {
        "gsm8k": {
            "probe_train": Path("results/deepseek7b_censored13k_v1/views/gsm8k/final_dependent/probe_train"),
            "calibration": Path("results/deepseek7b_censored13k_v1/views/gsm8k/final_dependent/calibration"),
            "heldout": Path("results/deepseek7b_main_v2/cache/gsm8k/heldout"),
        },
        "math": {
            "probe_train": Path("results/deepseek7b_censored13k_v1/views/math/final_dependent/probe_train"),
            "calibration": Path("results/deepseek7b_censored13k_v1/views/math/final_dependent/calibration"),
            "heldout": Path("results/deepseek7b_main_v2/cache/math500/heldout"),
            "ood": Path("results/deepseek7b_main_v2/cache/aime/heldout"),
        },
    }
    loaded = {
        dataset: {split: load_replay_data(path) for split, path in splits.items()}
        for dataset, splits in roots.items()
    }
    source_hashes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for finalist_id in finalist_ids:
        details[finalist_id] = {}
        for dataset in ("gsm8k", "math"):
            model_dir = source / "finalists" / finalist_id / dataset
            probe_path = model_dir / "probe.json"
            scores_path = model_dir / "scores.pt"
            source_hashes[str(probe_path)] = sha256(probe_path)
            source_hashes[str(scores_path)] = sha256(scores_path)
            probe = json.loads(probe_path.read_text())
            saved = torch.load(scores_path, map_location="cpu", weights_only=False)
            expected_splits = {"probe_train", "calibration", "heldout"}
            if dataset == "math":
                expected_splits.add("ood")
            if set(saved["scores"]) != expected_splits:
                raise AssertionError(f"unexpected score splits: {model_dir}")
            aligned = {
                split: align_scores(loaded[dataset][split], saved, split)
                for split in expected_splits
            }
            grid = threshold_grid(aligned["probe_train"], args.grid_size)
            readout_cost = 6 if "one_step" in probe["invocation"]["feature_kind"] else 0
            calibration_curve, calibration_lost = replay_curve(
                loaded[dataset]["calibration"], aligned["calibration"], grid, readout_cost
            )
            selected_index, certificate = select_ltt(
                calibration_curve, calibration_lost, args.alpha, args.delta
            )
            selected = calibration_curve[selected_index]
            test_splits = ["heldout"] + (["ood"] if dataset == "math" else [])
            details[finalist_id][dataset] = {
                "alpha": args.alpha,
                "delta": args.delta,
                "candidate_grid_source": "probe_train scores only",
                "candidate_order": "conservative_to_aggressive",
                "selection_objective": "maximize calibration deployed-token reduction",
                "selected_grid_index": selected_index,
                "selected_threshold": selected["threshold"],
                "certificate": certificate,
                "calibration": selected,
                "test": {},
            }
            for split in test_splits:
                test_curve, _ = replay_curve(
                    loaded[dataset][split], aligned[split], [grid[selected_index]], readout_cost
                )
                test = test_curve[0]
                details[finalist_id][dataset]["test"][split] = test
                reported_dataset = (
                    "gsm8k" if dataset == "gsm8k" else ("math500" if split == "heldout" else "aime2024")
                )
                rows.append(
                    {
                        "combination": finalist_id,
                        "primary_predeclared": finalist_id == freeze["primary_recommended_id"],
                        "dataset": reported_dataset,
                        "calibrator": "trajectory_envelope_ltt",
                        "alpha": args.alpha,
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

    # AIME must reuse the exact MATH calibration threshold and certificate.
    for finalist_id in finalist_ids:
        math_rows = [row for row in rows if row["combination"] == finalist_id and row["dataset"] in {"math500", "aime2024"}]
        if len(math_rows) != 2 or math_rows[0]["threshold"] != math_rows[1]["threshold"]:
            raise AssertionError(f"MATH/AIME threshold reuse failed for {finalist_id}")
    if len(rows) != 17 * 3:
        raise AssertionError(f"result row count mismatch: {len(rows)}")
    if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
        raise AssertionError("frozen source changed during recalibration")

    output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output / "FINAL_RESULTS_LTT.csv")
    atomic_json(rows, output / "FINAL_RESULTS_LTT.json")
    atomic_json(details, output / "CALIBRATION_DETAILS.json")
    primary_rows = [row for row in rows if row["primary_predeclared"]]
    atomic_json(
        {
            "status": "frozen",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "primary_model": freeze["primary_recommended_id"],
            "model_selection": "pre-existing leakage-free freeze; external calibration/test not used to reselect architecture",
            "main_calibrator": "trajectory_envelope_ltt",
            "alpha": args.alpha,
            "delta": args.delta,
            "candidate_grid_source": "probe_train scores only",
            "threshold_selection": "maximize calibration deployed-token reduction inside the certified prefix",
            "fixed_empirical_B_role": "appendix_only",
            "primary_results": primary_rows,
        },
        output / "PRIMARY_SELECTION_FREEZE.json",
    )
    atomic_json(
        {
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "source_final_audit_sha256": sha256(source / "FINAL_AUDIT.json"),
            "source_files_verified_unchanged": len(source_hashes),
            "finalists": len(finalist_ids),
            "result_rows": len(rows),
            "checks": {
                "problem_level_first_hit": True,
                "trajectory_envelope_monotone": True,
                "exact_binomial_ucb": True,
                "continuous_fixed_sequence_prefix": True,
                "token_only_objective_including_one_step_suffix": True,
                "heldout_ood_unused_for_selection": True,
                "math_threshold_reused_on_aime": True,
                "fixed_B_not_main": True,
            },
        },
        output / "AUDIT.json",
    )
    report = [
        "# DeepSeek-7B method exploration: LTT-calibrated main results",
        "",
        f"Frozen primary model: `{freeze['primary_recommended_id']}`.",
        f"Main calibrator: trajectory-envelope LTT with alpha={args.alpha:.2%}, delta={args.delta:.2%}.",
        "The threshold maximizes calibration deployed-token reduction within the certified prefix.",
        "Fixed empirical B is retained only as an appendix reference in the frozen source output.",
        "",
        "| Dataset | Accuracy | Dense accuracy | Delta acc (pp) | Deployed token reduction | Lost | Helped | Calibration UCB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_rows:
        report.append(
            f"| {row['dataset']} | {row['test_accuracy']:.2%} | {row['test_dense_accuracy']:.2%} | "
            f"{row['test_accuracy_delta_pp']:+.2f} | {row['test_deployed_token_reduction']:.2%} | "
            f"{row['test_lost_correct']} | {row['test_helped']} | {row['calibration_ucb']:.2%} |"
        )
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    atomic_json(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "primary_model": freeze["primary_recommended_id"],
            "main_calibrator": "trajectory_envelope_ltt",
            "fixed_B_role": "appendix_only",
            "result_rows": len(rows),
        },
        output / "EXPERIMENT_COMPLETE.json",
    )
    print(json.dumps({"status": "complete", "rows": len(rows), "primary": primary_rows}, indent=2))


if __name__ == "__main__":
    main()

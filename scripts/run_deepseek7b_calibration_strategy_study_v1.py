#!/usr/bin/env python3
"""Compare calibration policies on frozen DeepSeek-7B v2 trajectories.

The source experiment is read-only.  Candidate score thresholds are derived
only from probe-training scores; calibration questions are used only to select
or certify a threshold; MATH-500 and AIME are never used for selection.
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

import numpy as np
import torch
from scipy.stats import beta, binom

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_deepseek7b_v2_ood_calibration_sweep_v1 import (  # noqa: E402
    checked_scores,
    conformal_quantile,
    load_score_payload,
    prepare_frame,
)
from src.legacy_empirical_probe_normalized_v1 import (  # noqa: E402
    select_empirical_budget,
    simulate_policy,
    target_values,
)


METHODS = ("correctness", "consistency", "last_switch", "bce", "bce_traj")
DATASETS = ("math500", "aime")
EMPIRICAL_BUDGETS = (0, 1, 2, 4, 10, 20)
FORMAL_FAMILIES = (
    "bonferroni_cp",
    "fixed_sequence_ltt",
    "trajectory_first_failure_conformal",
    "trajectory_envelope_ltt",
)
ALL_ALPHA_FAMILIES = ("lynx_split_conformal",) + FORMAL_FAMILIES


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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: json_ready(row.get(key)) for key in fields} for row in rows])
    os.replace(temporary, path)


def training_threshold_grid(scores: np.ndarray, direction: str, size: int) -> list[dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("invalid probe-training scores")
    if np.min(values) < -1e-7 or np.max(values) > 1.0 + 1e-7:
        raise ValueError("expected sigmoid scores in [0,1]")
    quantiles = [float(x) for x in np.unique(np.quantile(values, np.linspace(0, 1, size)))]
    ordered = quantiles if direction == "low" else list(reversed(quantiles))
    sentinel = -1.0 if direction == "low" else 2.0
    full_stop = 2.0 if direction == "low" else -1.0
    thresholds = [sentinel] + ordered
    if thresholds[-1] != full_stop:
        thresholds.append(full_stop)
    return [
        {
            "grid_index": index,
            "threshold": float(threshold),
            "is_no_stop_sentinel": index == 0,
        }
        for index, threshold in enumerate(thresholds)
    ]


def record_arrays(records: list[dict[str, Any]], expected_ids: list[str] | None = None):
    ids = [str(row["problem_id"]) for row in records]
    if expected_ids is not None and ids != expected_ids:
        raise ValueError("policy record alignment changed across thresholds")
    arrays = {
        "success": np.asarray([row["method_success"] for row in records], dtype=np.float64),
        "lost": np.asarray([row["transition"] == "W_to_C" for row in records], dtype=np.float64),
        "tokens": np.asarray([row["method_tokens"] for row in records], dtype=np.float64),
        "dense_tokens": np.asarray([row["dense_tokens"] for row in records], dtype=np.float64),
        "coverage": np.asarray([not row["fallback"] for row in records], dtype=np.float64),
        "wall": np.asarray([row["replay_wall_ms"] for row in records], dtype=np.float64),
        "dense_wall": np.asarray([row["dense_wall_ms"] for row in records], dtype=np.float64),
    }
    return ids, arrays


def replay_candidate_curve(frame, scores, direction: str, grid, fallbacks):
    summaries: list[dict[str, Any]] = []
    matrices: dict[str, list[np.ndarray]] = {
        key: [] for key in ("success", "lost", "tokens", "dense_tokens", "coverage", "wall", "dense_wall")
    }
    ids: list[str] | None = None
    for spec in grid:
        metrics = simulate_policy(
            frame,
            scores,
            direction,
            float(spec["threshold"]),
            include_records=True,
            fallback_records=fallbacks,
            force_dense=bool(spec["is_no_stop_sentinel"]),
        )
        records = metrics.pop("records")
        current_ids, arrays = record_arrays(records, ids)
        if ids is None:
            ids = current_ids
        row = {**metrics, **spec}
        summaries.append(row)
        for key, value in arrays.items():
            matrices[key].append(value)
    stacked = {key: np.stack(value, axis=0) for key, value in matrices.items()}
    if not np.all(stacked["lost"][0] == 0) or not np.all(stacked["coverage"][0] == 0):
        raise AssertionError("no-stop sentinel is not dense fallback")
    return summaries, stacked, ids or []


def first_failure_indices(lost_matrix: np.ndarray) -> np.ndarray:
    # Candidate index is ordered from conservative to aggressive.  K denotes
    # that no candidate in the declared family ever causes a lost-correct exit.
    active = lost_matrix[1:] > 0.5
    any_failure = active.any(axis=0)
    first = np.argmax(active, axis=0) + 1
    return np.where(any_failure, first, lost_matrix.shape[0]).astype(np.int64)


def fastest_index(curve: list[dict[str, Any]], candidates: list[int]) -> int:
    return min(
        candidates,
        key=lambda index: (
            float(curve[index]["mean_replay_wall_ms"]),
            -float(curve[index]["token_reduction"]),
            -float(curve[index]["coverage"]),
            index,
        ),
    )


def exact_allowed_k(n: int, alpha: float, test_level: float) -> int:
    valid = [k for k in range(n + 1) if float(binom.cdf(k, n, alpha)) <= test_level]
    return max(valid) if valid else -1


def cp_upper(k: int, n: int, failure_probability: float) -> float:
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - failure_probability, k + 1, n - k))


def select_bonferroni(curve, lost_counts, alpha: float, delta: float):
    non_sentinel = max(1, len(curve) - 1)
    allowed = exact_allowed_k(int(curve[0]["problems"]), alpha, delta / non_sentinel)
    feasible = [0] + [index for index in range(1, len(curve)) if int(lost_counts[index]) <= allowed]
    index = fastest_index(curve, feasible)
    upper = 0.0 if index == 0 else cp_upper(
        int(lost_counts[index]), int(curve[index]["problems"]), delta / non_sentinel
    )
    return index, {
        "allowed_lost": allowed,
        "simultaneous_upper": upper,
        "candidate_tests": non_sentinel,
    }


def select_fixed_sequence(curve, lost_counts, alpha: float, delta: float):
    allowed = exact_allowed_k(int(curve[0]["problems"]), alpha, delta)
    certified = [0]
    first_failed = None
    for index in range(1, len(curve)):
        if int(lost_counts[index]) <= allowed:
            certified.append(index)
        else:
            first_failed = index
            break
    index = fastest_index(curve, certified)
    upper = 0.0 if index == 0 else cp_upper(
        int(lost_counts[index]), int(curve[index]["problems"]), delta
    )
    return index, {
        "allowed_lost": allowed,
        "pointwise_upper": upper,
        "certified_prefix_end": certified[-1],
        "first_failed_index": first_failed,
    }


def select_first_failure_conformal(curve, critical_counts, alpha: float):
    n = int(curve[0]["problems"])
    # Selecting strictly before the k-th calibration first-failure boundary
    # yields the finite-sample marginal rank guarantee k/(n+1) <= alpha.
    allowed = max(-1, int(math.floor(alpha * (n + 1))) - 1)
    feasible = [0] + [
        index for index in range(1, len(curve)) if int(critical_counts[index]) <= allowed
    ]
    index = fastest_index(curve, feasible)
    return index, {
        "allowed_first_failure_boundaries": allowed,
        "marginal_rank_bound": float((allowed + 1) / (n + 1)) if allowed >= 0 else 0.0,
    }


def select_envelope_ltt(curve, critical_counts, alpha: float, delta: float):
    allowed = exact_allowed_k(int(curve[0]["problems"]), alpha, delta)
    certified = [0]
    first_failed = None
    for index in range(1, len(curve)):
        if int(critical_counts[index]) <= allowed:
            certified.append(index)
        else:
            first_failed = index
            break
    index = fastest_index(curve, certified)
    upper = 0.0 if index == 0 else cp_upper(
        int(critical_counts[index]), int(curve[index]["problems"]), delta
    )
    return index, {
        "allowed_first_failure_boundaries": allowed,
        "trajectory_envelope_upper": upper,
        "certified_prefix_end": certified[-1],
        "first_failed_index": first_failed,
    }


def lynx_threshold(scores: np.ndarray, labels: np.ndarray, direction: str, alpha: float):
    if direction == "high":
        q_stop = conformal_quantile(1.0 - scores[labels == 1], alpha)
        q_continue = conformal_quantile(scores[labels == 0], alpha)
        threshold = max(1.0 - q_stop, float(np.nextafter(q_continue, np.inf)))
    else:
        q_stop = conformal_quantile(scores[labels == 0], alpha)
        q_continue = conformal_quantile(1.0 - scores[labels == 1], alpha)
        threshold = min(q_stop, float(np.nextafter(1.0 - q_continue, -np.inf)))
    return float(threshold), float(q_stop), float(q_continue)


def metric_projection(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_problems": int(metrics["problems"]),
        f"{prefix}_accuracy": float(metrics["accuracy"]),
        f"{prefix}_dense_accuracy": float(metrics["dense_accuracy"]),
        f"{prefix}_accuracy_delta_pp": 100.0
        * (float(metrics["accuracy"]) - float(metrics["dense_accuracy"])),
        f"{prefix}_lost_correct_count": int(metrics["lost_correct_count"]),
        f"{prefix}_lost_correct_rate": float(metrics["lost_correct_rate"]),
        f"{prefix}_token_reduction": float(metrics["token_reduction"]),
        f"{prefix}_replay_wall_reduction": float(metrics["replay_wall_reduction"]),
        f"{prefix}_coverage": float(metrics["coverage"]),
        f"{prefix}_mean_tokens": float(metrics["mean_reasoning_and_answer_tokens"]),
        f"{prefix}_mean_dense_tokens": float(metrics["mean_dense_reasoning_tokens"]),
    }


def bootstrap_test(records: list[dict[str, Any]], weights: np.ndarray):
    _ids, arrays = record_arrays(records)
    n = len(records)
    accuracy = weights @ arrays["success"] / n
    lost = weights @ arrays["lost"] / n
    coverage = weights @ arrays["coverage"] / n
    mean_tokens = weights @ arrays["tokens"] / n
    mean_dense = weights @ arrays["dense_tokens"] / n
    reduction = 1.0 - mean_tokens / mean_dense
    result = {}
    for name, values in (
        ("accuracy", accuracy),
        ("lost_correct_rate", lost),
        ("coverage", coverage),
        ("token_reduction", reduction),
    ):
        low, high = np.quantile(values, [0.025, 0.975])
        result[f"test_{name}_ci_low"] = float(low)
        result[f"test_{name}_ci_high"] = float(high)
    return result


def bootstrap_selection(
    curve,
    matrices,
    critical_matrix,
    alphas,
    delta: float,
    resamples: int,
    rng: np.random.Generator,
):
    n = matrices["lost"].shape[1]
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=resamples).astype(np.float64)
    lost = weights @ matrices["lost"].T
    critical = weights @ critical_matrix.T
    mean_wall = weights @ matrices["wall"].T / n
    outputs = []

    def choose_fastest(feasible: np.ndarray, wall: np.ndarray) -> int:
        indices = np.flatnonzero(feasible)
        return int(indices[np.argmin(wall[indices])]) if len(indices) else 0

    for alpha in alphas:
        allowed_ltt = exact_allowed_k(n, float(alpha), delta)
        allowed_bonf = exact_allowed_k(n, float(alpha), delta / max(1, len(curve) - 1))
        allowed_tffc = max(-1, int(math.floor(float(alpha) * (n + 1))) - 1)
        selections = {family: [] for family in FORMAL_FAMILIES}
        for sample in range(resamples):
            selections["bonferroni_cp"].append(
                choose_fastest(
                    np.r_[True, lost[sample, 1:] <= allowed_bonf], mean_wall[sample]
                )
            )
            failed = np.flatnonzero(lost[sample, 1:] > allowed_ltt)
            end = int(failed[0]) if len(failed) else len(curve) - 1
            mask = np.arange(len(curve)) <= end
            selections["fixed_sequence_ltt"].append(choose_fastest(mask, mean_wall[sample]))
            selections["trajectory_first_failure_conformal"].append(
                choose_fastest(
                    np.r_[True, critical[sample, 1:] <= allowed_tffc], mean_wall[sample]
                )
            )
            failed = np.flatnonzero(critical[sample, 1:] > allowed_ltt)
            end = int(failed[0]) if len(failed) else len(curve) - 1
            mask = np.arange(len(curve)) <= end
            selections["trajectory_envelope_ltt"].append(choose_fastest(mask, mean_wall[sample]))
        for family, indices_list in selections.items():
            indices = np.asarray(indices_list, dtype=np.int64)
            thresholds = np.asarray([curve[index]["threshold"] for index in indices])
            unique, counts = np.unique(indices, return_counts=True)
            mode_pos = int(np.argmax(counts))
            outputs.append(
                {
                    "family": family,
                    "alpha": float(alpha),
                    "resamples": int(resamples),
                    "selection_index_median": float(np.median(indices)),
                    "selection_index_iqr_low": float(np.quantile(indices, 0.25)),
                    "selection_index_iqr_high": float(np.quantile(indices, 0.75)),
                    "threshold_median": float(np.median(thresholds)),
                    "threshold_iqr_low": float(np.quantile(thresholds, 0.25)),
                    "threshold_iqr_high": float(np.quantile(thresholds, 0.75)),
                    "dense_fallback_rate": float(np.mean(indices == 0)),
                    "mode_index": int(unique[mode_pos]),
                    "mode_frequency": float(counts[mode_pos] / resamples),
                }
            )
    return outputs


def result_row(
    *, family, guarantee, method, dataset, direction, threshold, calibration, test,
    alpha=None, delta=None, budget=None, selection=None, test_ci=None,
):
    return {
        "family": family,
        "guarantee_scope": guarantee,
        "method": method,
        "dataset": dataset,
        "direction": direction,
        "alpha": alpha,
        "delta": delta,
        "budget_B": budget,
        "threshold": float(threshold),
        "dense_fallback": bool(selection and selection.get("selected_index") == 0)
        or bool(calibration.get("is_no_stop_sentinel", False)),
        **metric_projection(calibration, "calibration"),
        **metric_projection(test, "test"),
        **(selection or {}),
        **(test_ci or {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    if not all(0 < value < 1 for value in args.alphas):
        raise ValueError("alphas must be in (0,1)")
    source = args.source_root.resolve()
    if not (source / "COMPLETION_AUDIT.json").exists():
        raise FileNotFoundError("source completion audit is missing")
    source_audit = json.loads((source / "COMPLETION_AUDIT.json").read_text())
    if source_audit.get("status") != "complete":
        raise ValueError("source experiment is not complete")

    schedule = "sentence"  # loader key; scientific schedule is paragraph
    frames = {}
    fallbacks = {}
    for name, rel, split in (
        ("probe_train", "math", "probe_train"),
        ("calibration", "math", "calibration"),
        ("math500", "math500", "heldout"),
        ("aime", "aime", "heldout"),
    ):
        frames[name], fallbacks[name] = prepare_frame(source / "cache" / rel, split, schedule)

    rng = np.random.default_rng(args.seed)
    test_weights = {
        name: rng.multinomial(
            int(frame.problem_id.nunique()) + len(fallbacks[name]),
            np.full(
                int(frame.problem_id.nunique()) + len(fallbacks[name]),
                1.0 / (int(frame.problem_id.nunique()) + len(fallbacks[name])),
            ),
            size=args.bootstrap_resamples,
        ).astype(np.float64)
        for name, frame in (("math500", frames["math500"]), ("aime", frames["aime"]))
    }

    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    method_manifests = {}
    source_hashes = {"COMPLETION_AUDIT.json": sha256(source / "COMPLETION_AUDIT.json")}

    for method in METHODS:
        math_dir = source / "probes" / "math500" / method
        aime_dir = source / "probes" / "aime" / method
        report_path = math_dir / "probe.json"
        math_scores_path = math_dir / "scores.pt"
        aime_scores_path = aime_dir / "scores.pt"
        report = json.loads(report_path.read_text())
        direction = str(report["run_spec"]["stop_direction"])
        math_payload = load_score_payload(math_scores_path)
        aime_payload = load_score_payload(aime_scores_path)
        scores = {
            "probe_train": checked_scores(math_payload, "probe_train", frames["probe_train"], source=math_scores_path),
            "calibration": checked_scores(math_payload, "calibration", frames["calibration"], source=math_scores_path),
            "math500": checked_scores(math_payload, "heldout", frames["math500"], source=math_scores_path),
            "aime": checked_scores(aime_payload, "heldout", frames["aime"], source=aime_scores_path),
        }
        grid = training_threshold_grid(scores["probe_train"], direction, args.grid_size)
        curve, matrices, calibration_ids = replay_candidate_curve(
            frames["calibration"], scores["calibration"], direction, grid, fallbacks["calibration"]
        )
        first_failure = first_failure_indices(matrices["lost"])
        critical_matrix = (
            first_failure[np.newaxis, :] <= np.arange(len(curve))[:, np.newaxis]
        ).astype(np.float64)
        lost_counts = matrices["lost"].sum(axis=1).astype(np.int64)
        critical_counts = critical_matrix.sum(axis=1).astype(np.int64)
        if any(int(row["lost_correct_count"]) != int(lost_counts[index]) for index, row in enumerate(curve)):
            raise AssertionError("curve lost counts do not match policy records")

        local_bootstrap = bootstrap_selection(
            curve, matrices, critical_matrix, args.alphas, args.delta,
            args.bootstrap_resamples, rng,
        )
        for item in local_bootstrap:
            bootstrap_rows.append({"method": method, **item})

        selection_table = {}
        for alpha in args.alphas:
            alpha = float(alpha)
            selectors = {
                "bonferroni_cp": select_bonferroni(curve, lost_counts, alpha, args.delta),
                "fixed_sequence_ltt": select_fixed_sequence(curve, lost_counts, alpha, args.delta),
                "trajectory_first_failure_conformal": select_first_failure_conformal(
                    curve, critical_counts, alpha
                ),
                "trajectory_envelope_ltt": select_envelope_ltt(
                    curve, critical_counts, alpha, args.delta
                ),
            }
            for family, (index, details) in selectors.items():
                selection_table[(family, alpha)] = (index, details)

        target_method = "correction" if method in ("bce", "bce_traj") else method
        labels = target_values(frames["calibration"], target_method).astype(np.int64)
        lynx_selections = {}
        for alpha in args.alphas:
            threshold, q_stop, q_continue = lynx_threshold(
                scores["calibration"], labels, direction, float(alpha)
            )
            calibration = simulate_policy(
                frames["calibration"], scores["calibration"], direction, threshold,
                fallback_records=fallbacks["calibration"],
            )
            lynx_selections[float(alpha)] = (
                threshold,
                calibration,
                {"q_stop": q_stop, "q_continue": q_continue},
            )

        test_cache = {}

        def evaluate_test(dataset: str, threshold: float, force_dense: bool):
            key = (dataset, float(threshold), bool(force_dense))
            if key not in test_cache:
                metrics = simulate_policy(
                    frames[dataset], scores[dataset], direction, float(threshold),
                    include_records=True, fallback_records=fallbacks[dataset],
                    force_dense=force_dense,
                )
                records = metrics.pop("records")
                ci = bootstrap_test(records, test_weights[dataset])
                test_cache[key] = (metrics, ci)
            return test_cache[key]

        for alpha in args.alphas:
            alpha = float(alpha)
            threshold, calibration, extra = lynx_selections[alpha]
            for dataset in DATASETS:
                test, ci = evaluate_test(dataset, threshold, False)
                rows.append(
                    result_row(
                        family="lynx_split_conformal",
                        guarantee="checkpoint_class_conditional_marginal",
                        method=method,
                        dataset=dataset,
                        direction=direction,
                        threshold=threshold,
                        calibration=calibration,
                        test=test,
                        alpha=alpha,
                        delta=None,
                        selection={"confidence": 1.0 - alpha, **extra},
                        test_ci=ci,
                    )
                )

            for family in FORMAL_FAMILIES:
                index, details = selection_table[(family, alpha)]
                calibration = curve[index]
                guarantee = {
                    "bonferroni_cp": "problem_level_pac_simultaneous",
                    "fixed_sequence_ltt": "problem_level_pac_fixed_sequence",
                    "trajectory_first_failure_conformal": "problem_level_marginal_first_failure",
                    "trajectory_envelope_ltt": "problem_level_pac_trajectory_envelope",
                }[family]
                selection = {
                    "selected_index": int(index),
                    "candidate_grid_size": len(curve),
                    "candidate_grid_source": "probe_train_score_quantiles",
                    "critical_boundary_count": int(critical_counts[index]),
                    **details,
                }
                for dataset in DATASETS:
                    test, ci = evaluate_test(
                        dataset,
                        float(calibration["threshold"]),
                        bool(calibration["is_no_stop_sentinel"]),
                    )
                    rows.append(
                        result_row(
                            family=family,
                            guarantee=guarantee,
                            method=method,
                            dataset=dataset,
                            direction=direction,
                            threshold=float(calibration["threshold"]),
                            calibration=calibration,
                            test=test,
                            alpha=alpha,
                            delta=float(args.delta) if family != "trajectory_first_failure_conformal" else None,
                            selection=selection,
                            test_ci=ci,
                        )
                    )

        # Preserve the exact historical empirical-B operating points as a reference.
        for budget in EMPIRICAL_BUDGETS:
            calibration = select_empirical_budget(report["calibration"]["curve"], int(budget))
            for dataset in DATASETS:
                test, ci = evaluate_test(
                    dataset,
                    float(calibration["threshold"]),
                    bool(calibration.get("is_no_stop_sentinel", False)),
                )
                rows.append(
                    result_row(
                        family="empirical_B",
                        guarantee="problem_level_empirical_only",
                        method=method,
                        dataset=dataset,
                        direction=direction,
                        threshold=float(calibration["threshold"]),
                        calibration=calibration,
                        test=test,
                        budget=int(budget),
                        selection={"historical_calibration_grid": True},
                        test_ci=ci,
                    )
                )

        method_manifests[method] = {
            "direction": direction,
            "candidate_grid_size": len(curve),
            "calibration_questions": len(calibration_ids),
            "calibration_checkpoints": len(scores["calibration"]),
            "first_failure_questions": int(np.sum(first_failure < len(curve))),
        }
        for path in (report_path, math_scores_path, aime_scores_path):
            source_hashes[str(path.relative_to(source))] = sha256(path)

    expected_rows = len(METHODS) * len(DATASETS) * (
        len(args.alphas) * len(ALL_ALPHA_FAMILIES) + len(EMPIRICAL_BUDGETS)
    )
    if len(rows) != expected_rows:
        raise AssertionError((len(rows), expected_rows))
    expected_bootstrap = len(METHODS) * len(FORMAL_FAMILIES) * len(args.alphas)
    if len(bootstrap_rows) != expected_bootstrap:
        raise AssertionError((len(bootstrap_rows), expected_bootstrap))

    # Select the main calibrator without using either OOD test set: among formal
    # problem-level PAC procedures, maximize mean calibration token reduction for
    # BCE+trajectory over the predeclared 1--5% risk grid; break ties by bootstrap
    # mode frequency and lower dense-fallback rate.
    formal_candidates = (
        "bonferroni_cp",
        "fixed_sequence_ltt",
        "trajectory_envelope_ltt",
    )
    rank_rows = []
    rank_alphas = [value for value in args.alphas if value >= 0.01]
    for family in formal_candidates:
        selected_rows = [
            row for row in rows
            if row["method"] == "bce_traj"
            and row["dataset"] == "math500"
            and row["family"] == family
            and row["alpha"] in rank_alphas
        ]
        boot = [
            row for row in bootstrap_rows
            if row["method"] == "bce_traj"
            and row["family"] == family
            and row["alpha"] in rank_alphas
        ]
        rank_rows.append(
            {
                "family": family,
                "mean_calibration_token_reduction": float(np.mean([
                    row["calibration_token_reduction"] for row in selected_rows
                ])),
                "mean_bootstrap_mode_frequency": float(np.mean([
                    row["mode_frequency"] for row in boot
                ])),
                "mean_bootstrap_dense_fallback_rate": float(np.mean([
                    row["dense_fallback_rate"] for row in boot
                ])),
            }
        )
    recommendation = max(
        rank_rows,
        key=lambda row: (
            row["mean_calibration_token_reduction"],
            row["mean_bootstrap_mode_frequency"],
            -row["mean_bootstrap_dense_fallback_rate"],
        ),
    )
    chosen_family = recommendation["family"]
    recommendation_payload = {
        "status": "complete",
        "selection_uses_ood_test": False,
        "recommended_primary_calibrator": chosen_family,
        "recommended_primary_alpha": 0.01,
        "recommended_balanced_alpha": 0.03,
        "delta": float(args.delta),
        "ranking": sorted(
            rank_rows,
            key=lambda row: row["mean_calibration_token_reduction"],
            reverse=True,
        ),
        "rationale": (
            "Only problem-level PAC strategies were eligible. Ranking used BCE+trajectory "
            "calibration efficiency over the predeclared alpha grid, then bootstrap stability. "
            "MATH-500 and AIME were reporting-only and did not select the strategy."
        ),
        "ood_caveat": (
            "MATH calibration to MATH-500/AIME is zero-shot transfer. Formal exchangeability-based "
            "risk guarantees do not automatically survive this distribution shift."
        ),
    }

    results_json = args.output / "RESULTS.json"
    results_csv = args.output / "RESULTS.csv"
    bootstrap_json = args.output / "BOOTSTRAP_SELECTION.json"
    bootstrap_csv = args.output / "BOOTSTRAP_SELECTION.csv"
    atomic_json(rows, results_json)
    atomic_csv(rows, results_csv)
    atomic_json(bootstrap_rows, bootstrap_json)
    atomic_csv(bootstrap_rows, bootstrap_csv)
    atomic_json(recommendation_payload, args.output / "RECOMMENDATION.json")

    summary_lines = [
        "# DeepSeek-7B Calibration Strategy Study",
        "",
        f"Recommended primary calibrator: **{chosen_family}**",
        "",
        "Selection was calibration-only; MATH-500 and AIME were not used to choose the strategy.",
        "",
        "## Formal strategy ranking",
        "",
        "| Strategy | Mean calibration token reduction | Bootstrap mode frequency | Bootstrap Dense fallback |",
        "|---|---:|---:|---:|",
    ]
    for item in recommendation_payload["ranking"]:
        summary_lines.append(
            f"| {item['family']} | {100*item['mean_calibration_token_reduction']:.2f}% | "
            f"{100*item['mean_bootstrap_mode_frequency']:.2f}% | "
            f"{100*item['mean_bootstrap_dense_fallback_rate']:.2f}% |"
        )
    summary_lines += [
        "",
        "## OOD caveat",
        "",
        recommendation_payload["ood_caveat"],
        "",
    ]
    (args.output / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    # Source integrity is checked again after all reads.
    source_hashes_after = {
        key: sha256(source / key) for key in source_hashes
    }
    audit = {
        "status": "complete" if source_hashes_after == source_hashes else "failed",
        "created_at": now(),
        "source_root": str(source),
        "source_unchanged": source_hashes_after == source_hashes,
        "source_hashes": source_hashes,
        "heldout_used_for_selection": False,
        "candidate_grid_source": "probe_train scores only",
        "calibration_unit": "problem-level full first-exit replay",
        "methods": list(METHODS),
        "datasets": list(DATASETS),
        "alphas": list(map(float, args.alphas)),
        "delta": float(args.delta),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "expected_result_rows": expected_rows,
        "actual_result_rows": len(rows),
        "expected_bootstrap_rows": expected_bootstrap,
        "actual_bootstrap_rows": len(bootstrap_rows),
        "method_manifests": method_manifests,
        "output_hashes": {
            "RESULTS.json": sha256(results_json),
            "RESULTS.csv": sha256(results_csv),
            "BOOTSTRAP_SELECTION.json": sha256(bootstrap_json),
            "BOOTSTRAP_SELECTION.csv": sha256(bootstrap_csv),
            "RECOMMENDATION.json": sha256(args.output / "RECOMMENDATION.json"),
            "SUMMARY.md": sha256(args.output / "SUMMARY.md"),
        },
    }
    atomic_json(audit, args.output / "AUDIT.json")
    manifest = {
        "status": audit["status"],
        "created_at": now(),
        "command": " ".join(sys.argv),
        "source_root": str(source),
        "output": str(args.output.resolve()),
        "execution": "cached CPU replay; no model generation and no CUDA allocation",
        "methods": list(METHODS),
        "families": ["empirical_B", *ALL_ALPHA_FAMILIES],
        "method_manifests": method_manifests,
        "recommendation": recommendation_payload,
    }
    atomic_json(manifest, args.output / "RUN_MANIFEST.json")
    if audit["status"] != "complete":
        raise RuntimeError("source integrity audit failed")
    print(json.dumps({
        "status": "complete",
        "results_rows": len(rows),
        "bootstrap_rows": len(bootstrap_rows),
        "recommended": chosen_family,
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

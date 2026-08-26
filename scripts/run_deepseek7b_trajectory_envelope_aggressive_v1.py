#!/usr/bin/env python3
"""Aggressive trajectory-envelope LTT sweep on frozen DeepSeek-7B v2 data.

The study is calibration-only for threshold selection, uses token reduction as
the sole cost objective, and evaluates all five probe targets on GSM8K and on
MATH-to-MATH-500/AIME transfer. Held-out sets never select a threshold or alpha.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_deepseek7b_calibration_strategy_token_v2 import (  # noqa: E402
    atomic_csv,
    atomic_json,
    bootstrap_test,
    checked_scores,
    cp_upper,
    exact_allowed_k,
    first_failure_indices,
    load_score_payload,
    most_token_efficient_index,
    prepare_frame,
    replay_candidate_curve,
    result_row,
    sha256,
    training_threshold_grid,
)
from src.legacy_empirical_probe_normalized_v1 import simulate_policy  # noqa: E402


METHODS = ("correctness", "consistency", "last_switch", "bce", "bce_traj")
DOMAINS = {
    "gsm8k": {
        "train_cache": "gsm8k",
        "probe_dataset": "gsm8k",
        "tests": {"gsm8k": ("gsm8k", "gsm8k")},
    },
    "math": {
        "train_cache": "math",
        "probe_dataset": "math500",
        "tests": {
            "math500": ("math500", "math500"),
            "aime": ("aime", "aime"),
        },
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bootstrap_envelope_selection(
    curve: list[dict[str, Any]],
    matrices: dict[str, np.ndarray],
    critical_matrix: np.ndarray,
    alphas: list[float],
    delta: float,
    resamples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    n = int(matrices["lost"].shape[1])
    weights = rng.multinomial(
        n, np.full(n, 1.0 / n), size=resamples
    ).astype(np.float64)
    critical = weights @ critical_matrix.T
    mean_tokens = weights @ matrices["tokens"].T / n
    mean_dense = weights @ matrices["dense_tokens"].T / n
    reduction = 1.0 - mean_tokens / mean_dense
    coverage = weights @ matrices["coverage"].T / n
    outputs = []
    for alpha in alphas:
        allowed = exact_allowed_k(n, float(alpha), float(delta))
        selected = []
        for sample in range(resamples):
            failed = np.flatnonzero(critical[sample, 1:] > allowed)
            end = int(failed[0]) if len(failed) else len(curve) - 1
            feasible = list(range(end + 1))
            index = min(
                feasible,
                key=lambda value: (
                    -float(reduction[sample, value]),
                    -float(coverage[sample, value]),
                    value,
                ),
            )
            selected.append(index)
        indices = np.asarray(selected, dtype=np.int64)
        thresholds = np.asarray(
            [float(curve[index]["threshold"]) for index in indices], dtype=np.float64
        )
        unique, counts = np.unique(indices, return_counts=True)
        mode_position = int(np.argmax(counts))
        outputs.append(
            {
                "alpha": float(alpha),
                "resamples": int(resamples),
                "selection_index_median": float(np.median(indices)),
                "selection_index_iqr_low": float(np.quantile(indices, 0.25)),
                "selection_index_iqr_high": float(np.quantile(indices, 0.75)),
                "threshold_median": float(np.median(thresholds)),
                "threshold_iqr_low": float(np.quantile(thresholds, 0.25)),
                "threshold_iqr_high": float(np.quantile(thresholds, 0.75)),
                "dense_fallback_rate": float(np.mean(indices == 0)),
                "mode_index": int(unique[mode_position]),
                "mode_frequency": float(counts[mode_position] / resamples),
            }
        )
    return outputs


def select_envelope_token(
    curve: list[dict[str, Any]],
    critical_counts: np.ndarray,
    alpha: float,
    delta: float,
) -> tuple[int, dict[str, Any]]:
    n = int(curve[0]["problems"])
    allowed = exact_allowed_k(n, float(alpha), float(delta))
    certified = [0]
    first_failed = None
    for index in range(1, len(curve)):
        if int(critical_counts[index]) <= allowed:
            certified.append(index)
        else:
            first_failed = index
            break
    index = most_token_efficient_index(curve, certified)
    upper = 0.0 if index == 0 else cp_upper(
        int(critical_counts[index]), n, float(delta)
    )
    return index, {
        "allowed_first_failure_boundaries": int(allowed),
        "trajectory_envelope_upper": float(upper),
        "certified_prefix_end": int(certified[-1]),
        "first_failed_index": first_failed,
    }


def load_domain(source: Path, name: str):
    spec = DOMAINS[name]
    schedule = "sentence"  # loader tag; frozen scientific schedule is paragraph
    train, train_fallback = prepare_frame(
        source / "cache" / spec["train_cache"], "probe_train", schedule
    )
    calibration, calibration_fallback = prepare_frame(
        source / "cache" / spec["train_cache"], "calibration", schedule
    )
    frames = {"probe_train": train, "calibration": calibration}
    fallbacks = {
        "probe_train": train_fallback,
        "calibration": calibration_fallback,
    }
    for dataset, (cache_name, _probe_name) in spec["tests"].items():
        frames[dataset], fallbacks[dataset] = prepare_frame(
            source / "cache" / cache_name, "heldout", schedule
        )
    return frames, fallbacks


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
    alphas = [float(value) for value in args.alphas]
    if not all(0.0 < value < 1.0 for value in alphas):
        raise ValueError("alphas must lie in (0,1)")
    if sorted(set(alphas)) != alphas:
        raise ValueError("alphas must be unique and increasing")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    source = args.source_root.resolve()
    completion = source / "COMPLETION_AUDIT.json"
    if not completion.exists():
        raise FileNotFoundError(completion)
    completion_payload = json.loads(completion.read_text())
    if completion_payload.get("status") != "complete":
        raise ValueError("source completion audit is not complete")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    source_hashes = {"COMPLETION_AUDIT.json": sha256(completion)}

    for domain in DOMAINS:
        frames, fallbacks = load_domain(source, domain)
        spec = DOMAINS[domain]
        test_weights = {}
        for dataset in spec["tests"]:
            n = int(frames[dataset].problem_id.nunique()) + len(fallbacks[dataset])
            test_weights[dataset] = rng.multinomial(
                n, np.full(n, 1.0 / n), size=args.bootstrap_resamples
            ).astype(np.float64)

        for method in METHODS:
            probe_root = source / "probes" / spec["probe_dataset"] / method
            report_path = probe_root / "probe.json"
            main_scores_path = probe_root / "scores.pt"
            report = json.loads(report_path.read_text())
            direction = str(report["run_spec"]["stop_direction"])
            main_payload = load_score_payload(main_scores_path)
            scores = {
                "probe_train": checked_scores(
                    main_payload, "probe_train", frames["probe_train"],
                    source=main_scores_path,
                ),
                "calibration": checked_scores(
                    main_payload, "calibration", frames["calibration"],
                    source=main_scores_path,
                ),
            }
            for dataset, (_cache_name, probe_name) in spec["tests"].items():
                score_path = source / "probes" / probe_name / method / "scores.pt"
                payload = main_payload if score_path == main_scores_path else load_score_payload(score_path)
                scores[dataset] = checked_scores(
                    payload, "heldout", frames[dataset], source=score_path
                )
                source_hashes[str(score_path.relative_to(source))] = sha256(score_path)

            grid = training_threshold_grid(
                scores["probe_train"], direction, args.grid_size
            )
            curve, matrices, calibration_ids = replay_candidate_curve(
                frames["calibration"], scores["calibration"], direction,
                grid, fallbacks["calibration"],
            )
            first_failure = first_failure_indices(matrices["lost"])
            critical_matrix = (
                first_failure[np.newaxis, :] <= np.arange(len(curve))[:, np.newaxis]
            ).astype(np.float64)
            critical_counts = critical_matrix.sum(axis=1).astype(np.int64)
            local_bootstrap = bootstrap_envelope_selection(
                curve, matrices, critical_matrix, alphas, float(args.delta),
                int(args.bootstrap_resamples), rng,
            )
            for item in local_bootstrap:
                bootstrap_rows.append(
                    {"domain": domain, "method": method, **item}
                )

            test_cache: dict[tuple[str, float, bool], tuple[dict[str, Any], dict[str, Any]]] = {}

            def evaluate(dataset: str, threshold: float, force_dense: bool):
                key = (dataset, float(threshold), bool(force_dense))
                if key not in test_cache:
                    metrics = simulate_policy(
                        frames[dataset], scores[dataset], direction, threshold,
                        include_records=True,
                        fallback_records=fallbacks[dataset],
                        force_dense=force_dense,
                    )
                    records = metrics.pop("records")
                    ci = bootstrap_test(records, test_weights[dataset])
                    test_cache[key] = (metrics, ci)
                return test_cache[key]

            for alpha in alphas:
                index, details = select_envelope_token(
                    curve, critical_counts, alpha, float(args.delta)
                )
                calibration = curve[index]
                selection = {
                    "selected_index": int(index),
                    "candidate_grid_size": len(curve),
                    "candidate_grid_source": "probe_train_score_quantiles",
                    "critical_boundary_count": int(critical_counts[index]),
                    "selection_objective": "maximize_calibration_token_reduction",
                    **details,
                }
                if index != 0 and float(details["trajectory_envelope_upper"]) > alpha + 1e-12:
                    raise AssertionError("selected threshold violates declared alpha")
                for dataset in spec["tests"]:
                    test, ci = evaluate(
                        dataset,
                        float(calibration["threshold"]),
                        bool(calibration["is_no_stop_sentinel"]),
                    )
                    row = result_row(
                        family="trajectory_envelope_ltt",
                        guarantee="problem_level_pac_trajectory_envelope",
                        method=method,
                        dataset=dataset,
                        direction=direction,
                        threshold=float(calibration["threshold"]),
                        calibration=calibration,
                        test=test,
                        alpha=alpha,
                        delta=float(args.delta),
                        selection={"calibration_domain": domain, **selection},
                        test_ci=ci,
                    )
                    if any("wall" in key.lower() for key in row):
                        raise AssertionError("wall-time metric leaked into token-only output")
                    rows.append(row)

            manifests[f"{domain}/{method}"] = {
                "direction": direction,
                "candidate_grid_size": len(curve),
                "calibration_questions": len(calibration_ids),
                "calibration_checkpoints": len(scores["calibration"]),
                "first_failure_questions": int(np.sum(first_failure < len(curve))),
            }
            source_hashes[str(report_path.relative_to(source))] = sha256(report_path)
            source_hashes[str(main_scores_path.relative_to(source))] = sha256(main_scores_path)

    expected_results = len(METHODS) * len(alphas) * sum(
        len(spec["tests"]) for spec in DOMAINS.values()
    )
    expected_bootstrap = len(METHODS) * len(alphas) * len(DOMAINS)
    if len(rows) != expected_results:
        raise AssertionError((len(rows), expected_results))
    if len(bootstrap_rows) != expected_bootstrap:
        raise AssertionError((len(bootstrap_rows), expected_bootstrap))

    results_json = args.output / "RESULTS.json"
    results_csv = args.output / "RESULTS.csv"
    bootstrap_json = args.output / "BOOTSTRAP_SELECTION.json"
    bootstrap_csv = args.output / "BOOTSTRAP_SELECTION.csv"
    atomic_json(rows, results_json)
    atomic_csv(rows, results_csv)
    atomic_json(bootstrap_rows, bootstrap_json)
    atomic_csv(bootstrap_rows, bootstrap_csv)

    source_after = {key: sha256(source / key) for key in source_hashes}
    audit = {
        "status": "complete" if source_after == source_hashes else "failed",
        "created_at": now(),
        "source_root": str(source),
        "source_unchanged": source_after == source_hashes,
        "heldout_used_for_threshold_or_alpha_selection": False,
        "strategy": "trajectory_envelope_ltt",
        "calibration_unit": "problem_level_full_first_exit_replay",
        "threshold_selection_objective": "maximize_calibration_token_reduction",
        "reported_cost_metric": "reasoning_token_reduction",
        "wall_time_used": False,
        "alphas": alphas,
        "delta": float(args.delta),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "methods": list(METHODS),
        "domains": DOMAINS,
        "expected_result_rows": expected_results,
        "actual_result_rows": len(rows),
        "expected_bootstrap_rows": expected_bootstrap,
        "actual_bootstrap_rows": len(bootstrap_rows),
        "manifests": manifests,
        "source_hashes": source_hashes,
        "output_hashes": {
            "RESULTS.json": sha256(results_json),
            "RESULTS.csv": sha256(results_csv),
            "BOOTSTRAP_SELECTION.json": sha256(bootstrap_json),
            "BOOTSTRAP_SELECTION.csv": sha256(bootstrap_csv),
        },
    }
    atomic_json(audit, args.output / "AUDIT.json")
    manifest = {
        "status": audit["status"],
        "created_at": now(),
        "command": " ".join(sys.argv),
        "execution": "cached CPU replay; no model generation and no CUDA allocation",
        "source_root": str(source),
        "output": str(args.output.resolve()),
        "strategy": "trajectory_envelope_ltt",
        "threshold_selection_objective": "maximize_calibration_token_reduction",
        "reported_cost_metric": "reasoning_token_reduction",
        "wall_time_used": False,
        "primary_alpha": 0.01,
        "balanced_alpha": 0.03,
        "aggressive_alphas_are_reporting_only": [value for value in alphas if value > 0.05],
        "heldout_used_for_threshold_or_alpha_selection": False,
        "methods": list(METHODS),
        "domains": DOMAINS,
    }
    atomic_json(manifest, args.output / "RUN_MANIFEST.json")
    if audit["status"] != "complete":
        raise RuntimeError("source integrity audit failed")
    print(json.dumps({
        "status": "complete",
        "results_rows": len(rows),
        "bootstrap_rows": len(bootstrap_rows),
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

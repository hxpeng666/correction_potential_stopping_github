#!/usr/bin/env python3
"""复用冻结动态预测，审计500题calibration的候选选择稳定性。

候选去重严格只读取probe-train内部validation动作和标签；calibration用于
经验B选点/重采样，heldout仅在选点完成后映射冻结候选结果。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from src.dynamic_optimal_stopping_deployable_v2 import simulate_deployable_dynamic_policy
from src.legacy_empirical_probe_v4 import _artifact_rows, add_targets
from src.utils import atomic_json


def load_frame_only(directory: Path, schedule: str = "sentence") -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paths = sorted(directory.glob("sample_*.pt"))
    if not paths:
        raise FileNotFoundError(directory)
    rows: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    layers = None
    for path in paths:
        local_rows, _, local_layers, fallback = _artifact_rows(path, schedule)
        if layers is None:
            layers = local_layers
        elif layers != local_layers:
            raise ValueError(f"capture layers不一致: {path}")
        rows.extend(local_rows)
        if fallback is not None:
            fallbacks.append(fallback)
    frame = pd.DataFrame(rows).sort_values(["problem_id", "checkpoint"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["problem_id", "checkpoint"]).any():
        raise ValueError(f"重复problem/checkpoint: {directory}")
    return add_targets(frame), fallbacks


def rows_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["problem_id"]): row for row in records}
    if len(result) != len(records):
        raise ValueError("policy records存在重复problem_id")
    return result


def candidate_arrays(candidate_records: list[list[dict[str, Any]]]) -> tuple[list[str], dict[str, np.ndarray]]:
    indexed = [rows_by_id(rows) for rows in candidate_records]
    ids = sorted(indexed[0])
    if any(set(rows) != set(ids) for rows in indexed):
        raise ValueError("候选之间problem ID不一致")
    fields = {
        "success": np.asarray([[bool(rows[x]["method_success"]) for x in ids] for rows in indexed], dtype=np.float64),
        "dense_success": np.asarray([[bool(rows[x]["dense_success"]) for x in ids] for rows in indexed], dtype=np.float64),
        "used": np.asarray([[float(rows[x]["method_tokens"]) for x in ids] for rows in indexed], dtype=np.float64),
        "dense_tokens": np.asarray([[float(rows[x]["dense_tokens"]) for x in ids] for rows in indexed], dtype=np.float64),
        "stopped": np.asarray([[not bool(rows[x]["fallback"]) for x in ids] for rows in indexed], dtype=np.float64),
        "checkpoint": np.asarray([[-1 if rows[x]["fallback"] else int(rows[x]["checkpoint"]) for x in ids] for rows in indexed], dtype=np.int64),
    }
    if not np.all(fields["dense_success"] == fields["dense_success"][0]):
        raise ValueError("候选dense_success不一致")
    if not np.all(fields["dense_tokens"] == fields["dense_tokens"][0]):
        raise ValueError("候选dense_tokens不一致")
    return ids, fields


def summaries_from_arrays(values: dict[str, np.ndarray], indices: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if indices is None:
        success = values["success"]
        dense = values["dense_success"]
        used = values["used"]
        dense_tokens = values["dense_tokens"]
        stopped = values["stopped"]
    else:
        success = values["success"][:, indices]
        dense = values["dense_success"][:, indices]
        used = values["used"][:, indices]
        dense_tokens = values["dense_tokens"][:, indices]
        stopped = values["stopped"][:, indices]
    accuracy = success.mean(axis=-1)
    dense_accuracy = dense.mean(axis=-1)
    lost = ((dense > 0.5) & (success < 0.5)).sum(axis=-1)
    gained = ((dense < 0.5) & (success > 0.5)).sum(axis=-1)
    mean_used = used.mean(axis=-1)
    mean_dense = dense_tokens.mean(axis=-1)
    return {
        "accuracy": accuracy,
        "dense_accuracy": dense_accuracy,
        "lost": lost,
        "gained": gained,
        "mean_used": mean_used,
        "token_reduction": 1.0 - mean_used / mean_dense,
        "coverage": stopped.mean(axis=-1),
    }


def simulate_candidates(
    frame: pd.DataFrame,
    fallbacks: list[dict[str, Any]],
    predictions: dict[str, Any],
    split: str,
    mus: np.ndarray,
    row_mask: np.ndarray | None = None,
    allowed_fallback_ids: set[str] | None = None,
) -> tuple[list[list[dict[str, Any]]], dict[str, np.ndarray]]:
    stop = np.asarray(predictions["stop_probability"][split], dtype=np.float64)
    risk = np.asarray(predictions["risk_probability"][split], dtype=np.float64)
    q_values = np.asarray(predictions["q_continue_values"][split], dtype=np.float64)
    saved_ids = np.asarray(predictions["problem_ids"][split]).astype(str)
    saved_checkpoints = np.asarray(predictions["checkpoints"][split], dtype=np.int64)
    if not np.array_equal(frame.problem_id.astype(str).to_numpy(), saved_ids):
        raise ValueError(f"{split}: frame/prediction problem ID错位")
    if not np.array_equal(frame.checkpoint.to_numpy(dtype=np.int64), saved_checkpoints):
        raise ValueError(f"{split}: frame/prediction checkpoint错位")
    if row_mask is not None:
        frame = frame.loc[row_mask].copy().reset_index(drop=True)
        stop = stop[row_mask]
        risk = risk[row_mask]
        q_values = q_values[row_mask]
    if allowed_fallback_ids is not None:
        fallbacks = [row for row in fallbacks if str(row["problem_id"]) in allowed_fallback_ids]
    all_records = []
    summaries = []
    for candidate in range(q_values.shape[1]):
        result = simulate_deployable_dynamic_policy(
            frame, stop, risk, q_values[:, candidate], mu_value=float(mus[candidate]),
            fallback_records=fallbacks, include_records=True,
        )
        all_records.append(result.pop("records"))
        summaries.append(result)
    return all_records, {key: np.asarray([row[key] for row in summaries]) for key in (
        "accuracy", "dense_accuracy", "lost_correct_count", "gained_correct_count",
        "mean_reasoning_tokens", "token_reduction", "coverage",
    )}


def similarity_matrices(checkpoint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stopped = checkpoint >= 0
    count = checkpoint.shape[0]
    exact = np.eye(count, dtype=np.float64)
    stop_agree = np.eye(count, dtype=np.float64)
    for i in range(count):
        for j in range(i):
            exact[i, j] = exact[j, i] = np.mean(checkpoint[i] == checkpoint[j])
            stop_agree[i, j] = stop_agree[j, i] = np.mean(stopped[i] == stopped[j])
    return exact, stop_agree


def validation_deduplicate(values: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    summary = summaries_from_arrays(values)
    exact_agree, stop_agree = similarity_matrices(values["checkpoint"])
    exact_min = float(config["exact_checkpoint_agreement_min"])
    stop_min = float(config["stop_fallback_agreement_min"])
    dense_acc = float(summary["dense_accuracy"][0])
    priority = sorted(range(values["checkpoint"].shape[0]), key=lambda index: (
        max(0.0, dense_acc - 0.01 - float(summary["accuracy"][index])),
        float(summary["lost"][index]) / values["checkpoint"].shape[1],
        float(summary["mean_used"][index]), index,
    ))
    clusters: list[list[int]] = []
    representatives: list[int] = []
    assignment: dict[int, int] = {}
    for candidate in priority:
        matching = None
        for cluster_index, representative in enumerate(representatives):
            if exact_agree[candidate, representative] >= exact_min and stop_agree[candidate, representative] >= stop_min:
                matching = cluster_index
                break
        if matching is None:
            matching = len(clusters)
            clusters.append([])
            representatives.append(candidate)
        clusters[matching].append(candidate)
        assignment[candidate] = matching

    pareto = []
    for candidate in representatives:
        dominated = False
        for other in representatives:
            if other == candidate:
                continue
            weak = (
                summary["accuracy"][other] >= summary["accuracy"][candidate]
                and summary["lost"][other] <= summary["lost"][candidate]
                and summary["mean_used"][other] <= summary["mean_used"][candidate]
            )
            strict = (
                summary["accuracy"][other] > summary["accuracy"][candidate]
                or summary["lost"][other] < summary["lost"][candidate]
                or summary["mean_used"][other] < summary["mean_used"][candidate]
            )
            if weak and strict:
                dominated = True
                break
        if not dominated:
            pareto.append(candidate)
    retained = sorted(pareto if config.get("use_validation_pareto_filter", True) else representatives)
    return {
        "retained_candidates": retained,
        "representatives": representatives,
        "clusters": [sorted(cluster) for cluster in clusters],
        "candidate_to_cluster": {str(key): int(value) for key, value in assignment.items()},
        "exact_checkpoint_agreement": exact_agree,
        "stop_fallback_agreement": stop_agree,
        "validation_summary": summary,
    }


def bootstrap_matrices(values: dict[str, np.ndarray], replicates: int, seed: int) -> dict[str, np.ndarray]:
    candidates, problems = values["success"].shape
    rng = np.random.default_rng(seed)
    output = {key: np.empty((replicates, candidates), dtype=np.float64) for key in (
        "accuracy", "lost", "gained", "mean_used", "token_reduction", "coverage",
    )}
    output["dense_accuracy"] = np.empty(replicates, dtype=np.float64)
    batch = 250
    for start in range(0, replicates, batch):
        end = min(start + batch, replicates)
        indices = rng.integers(0, problems, size=(end - start, problems))
        dense0 = values["dense_success"][0]
        dense_tokens0 = values["dense_tokens"][0]
        output["dense_accuracy"][start:end] = dense0[indices].mean(axis=1)
        mean_dense = dense_tokens0[indices].mean(axis=1)
        for candidate in range(candidates):
            success = values["success"][candidate][indices]
            dense = values["dense_success"][candidate][indices]
            used = values["used"][candidate][indices]
            stopped = values["stopped"][candidate][indices]
            output["accuracy"][start:end, candidate] = success.mean(axis=1)
            output["lost"][start:end, candidate] = ((dense > 0.5) & (success < 0.5)).sum(axis=1)
            output["gained"][start:end, candidate] = ((dense < 0.5) & (success > 0.5)).sum(axis=1)
            output["mean_used"][start:end, candidate] = used.mean(axis=1)
            output["token_reduction"][start:end, candidate] = 1.0 - used.mean(axis=1) / mean_dense
            output["coverage"][start:end, candidate] = stopped.mean(axis=1)
    return output


def select_bootstrap(
    matrices: dict[str, np.ndarray], candidate_set: list[int], budgets: list[int], epsilon: float,
) -> dict[int, np.ndarray]:
    chosen = {}
    for budget in budgets:
        selected = np.full(len(matrices["dense_accuracy"]), -1, dtype=np.int64)
        for replicate in range(len(selected)):
            feasible = [candidate for candidate in candidate_set if (
                matrices["lost"][replicate, candidate] <= budget
                and matrices["accuracy"][replicate, candidate] >= matrices["dense_accuracy"][replicate] - epsilon
            )]
            if feasible:
                selected[replicate] = min(feasible, key=lambda candidate: (
                    matrices["mean_used"][replicate, candidate],
                    -matrices["token_reduction"][replicate, candidate],
                    -matrices["coverage"][replicate, candidate],
                    candidate,
                ))
        chosen[budget] = selected
    return chosen


def nominal_select(summary: dict[str, np.ndarray], candidates: list[int], budget: int, epsilon: float) -> int:
    dense_accuracy = float(summary["dense_accuracy"][0])
    feasible = [candidate for candidate in candidates if (
        summary["lost"][candidate] <= budget
        and summary["accuracy"][candidate] >= dense_accuracy - epsilon
    )]
    if not feasible:
        return -1
    return min(feasible, key=lambda candidate: (
        summary["mean_used"][candidate], -summary["token_reduction"][candidate],
        -summary["coverage"][candidate], candidate,
    ))


def quantiles(values: np.ndarray, points: list[float]) -> dict[str, float]:
    result = np.quantile(values, points)
    return {f"q{int(round(point * 100)):02d}": float(value) for point, value in zip(points, result)}


def expected_agreement(probability: np.ndarray, matrix: np.ndarray) -> float:
    return float(probability @ matrix @ probability)


def analyze_selection_family(
    name: str,
    selected: dict[int, np.ndarray],
    matrices: dict[str, np.ndarray],
    calibration_values: dict[str, np.ndarray],
    heldout_values: dict[str, np.ndarray],
    cluster_map: dict[int, int],
    budgets: list[int],
    quantile_points: list[float],
    gate: dict[str, Any],
) -> dict[str, Any]:
    candidates = calibration_values["checkpoint"].shape[0]
    heldout_summary = summaries_from_arrays(heldout_values)
    dense_checkpoint = np.full((1, calibration_values["checkpoint"].shape[1]), -1, dtype=np.int64)
    actions = np.concatenate([calibration_values["checkpoint"], dense_checkpoint], axis=0)
    exact, stop_agree = similarity_matrices(actions)
    output = {}
    for budget in budgets:
        chosen = selected[budget]
        mapped = np.where(chosen < 0, candidates, chosen)
        counts = Counter(map(int, chosen))
        frequencies = {str(key): value / len(chosen) for key, value in sorted(counts.items())}
        cluster_counts = Counter("dense" if value < 0 else str(cluster_map[value]) for value in chosen)
        cluster_frequencies = {key: value / len(chosen) for key, value in sorted(cluster_counts.items())}
        probability = np.bincount(mapped, minlength=candidates + 1).astype(np.float64) / len(mapped)
        selected_metrics = {}
        heldout_metrics = {}
        dense_acc_boot = matrices["dense_accuracy"]
        for metric in ("accuracy", "accuracy_delta", "lost", "token_reduction", "coverage"):
            if metric == "accuracy":
                values = np.where(chosen < 0, dense_acc_boot, matrices[metric][np.arange(len(chosen)), np.maximum(chosen, 0)])
            elif metric == "accuracy_delta":
                chosen_accuracy = np.where(chosen < 0, dense_acc_boot, matrices["accuracy"][np.arange(len(chosen)), np.maximum(chosen, 0)])
                values = chosen_accuracy - dense_acc_boot
            elif metric == "lost":
                values = np.where(chosen < 0, 0.0, matrices[metric][np.arange(len(chosen)), np.maximum(chosen, 0)])
            else:
                values = np.where(chosen < 0, 0.0, matrices[metric][np.arange(len(chosen)), np.maximum(chosen, 0)])
            selected_metrics[metric] = quantiles(values, quantile_points)
            if metric in ("accuracy", "accuracy_delta", "lost", "token_reduction", "coverage"):
                if metric == "accuracy":
                    hdense = heldout_values["dense_success"][0].mean()
                    hvalues = np.asarray([hdense if value < 0 else heldout_summary["accuracy"][value] for value in chosen])
                elif metric == "accuracy_delta":
                    hdense = heldout_values["dense_success"][0].mean()
                    hvalues = np.asarray([0.0 if value < 0 else heldout_summary["accuracy"][value] - hdense for value in chosen])
                elif metric == "lost":
                    hvalues = np.asarray([0.0 if value < 0 else heldout_summary["lost"][value] for value in chosen])
                else:
                    hvalues = np.asarray([0.0 if value < 0 else heldout_summary[metric][value] for value in chosen])
                heldout_metrics[metric] = quantiles(hvalues, quantile_points)
        modal_candidate_frequency = max(frequencies.values())
        modal_cluster_frequency = max(cluster_frequencies.values())
        pair_exact = expected_agreement(probability, exact)
        pair_stop = expected_agreement(probability, stop_agree)
        token_width = selected_metrics["token_reduction"]["q95"] - selected_metrics["token_reduction"]["q05"]
        accuracy_width = selected_metrics["accuracy_delta"]["q95"] - selected_metrics["accuracy_delta"]["q05"]
        passed = (
            max(modal_candidate_frequency, modal_cluster_frequency) >= float(gate["modal_candidate_or_cluster_frequency_min"])
            and pair_stop >= float(gate["expected_pairwise_stop_fallback_agreement_min"])
            and token_width <= float(gate["calibration_token_reduction_p90_width_max"])
            and accuracy_width <= float(gate["calibration_paired_accuracy_delta_p90_width_max"])
        )
        output[str(budget)] = {
            "family": name,
            "selection_frequencies": frequencies,
            "cluster_frequencies": cluster_frequencies,
            "modal_candidate_frequency": modal_candidate_frequency,
            "modal_cluster_frequency": modal_cluster_frequency,
            "expected_pairwise_exact_checkpoint_agreement": pair_exact,
            "expected_pairwise_stop_fallback_agreement": pair_stop,
            "calibration_selected_metrics": selected_metrics,
            "frozen_heldout_metrics_over_calibration_resamples": heldout_metrics,
            "stability_gate_passed": bool(passed),
        }
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze_dataset(dataset: str, config: dict[str, Any], source: dict[str, Any], output_root: Path) -> dict[str, Any]:
    source_root = ROOT / config["source_result_root"] / dataset
    probe = json.load((source_root / "probe.json").open())
    predictions = torch.load(source_root / "predictions.pt", map_location="cpu", weights_only=False)
    replay_root = ROOT / source["datasets"][dataset]["replay_root"]
    frames = {}
    fallbacks = {}
    for split in ("probe_train", "calibration", "heldout"):
        frames[split], fallbacks[split] = load_frame_only(replay_root / split)

    lambdas = np.asarray([float(value) for value in source["dynamic_policy"]["lambda_grid"] for _ in source["dynamic_policy"]["mu_grid"]])
    mus = np.asarray([float(value) for _ in source["dynamic_policy"]["lambda_grid"] for value in source["dynamic_policy"]["mu_grid"]])
    if len(lambdas) != 48:
        raise ValueError("本审计预期48个候选")

    validation_ids = set(map(str, probe["validation_problem_ids"]))
    validation_mask = frames["probe_train"].problem_id.astype(str).isin(validation_ids).to_numpy()
    validation_records, _ = simulate_candidates(
        frames["probe_train"], fallbacks["probe_train"], predictions, "probe_train", mus,
        row_mask=validation_mask, allowed_fallback_ids=validation_ids,
    )
    _, validation_values = candidate_arrays(validation_records)
    dedup = validation_deduplicate(validation_values, config["deduplication"])

    calibration_records, _ = simulate_candidates(frames["calibration"], fallbacks["calibration"], predictions, "calibration", mus)
    heldout_records, _ = simulate_candidates(frames["heldout"], fallbacks["heldout"], predictions, "heldout", mus)
    calibration_ids, calibration_values = candidate_arrays(calibration_records)
    heldout_ids, heldout_values = candidate_arrays(heldout_records)
    if set(calibration_ids) & set(heldout_ids):
        raise ValueError("calibration/heldout问题重叠")
    if set(validation_ids) & (set(calibration_ids) | set(heldout_ids)):
        raise ValueError("内部validation与calibration/heldout问题重叠")

    nominal_summary = summaries_from_arrays(calibration_values)
    budgets = [int(value) for value in config["selection"]["empirical_B"]]
    epsilon = float(config["selection"]["accuracy_epsilon"])
    nominal_full = {budget: nominal_select(nominal_summary, list(range(48)), budget, epsilon) for budget in budgets}
    original_selected = probe["calibration"]["selected"]["empirical_B"]
    original_indices = {int(budget): (-1 if row.get("dense_fallback") else int(row["candidate_index"])) for budget, row in original_selected.items()}
    if nominal_full != original_indices:
        raise ValueError(f"重算未复现原始选择: {nominal_full} != {original_indices}")
    clustered = sorted(int(value) for value in dedup["representatives"])
    retained = [int(value) for value in dedup["retained_candidates"]]
    nominal_clustered = {budget: nominal_select(nominal_summary, clustered, budget, epsilon) for budget in budgets}
    nominal_dedup = {budget: nominal_select(nominal_summary, retained, budget, epsilon) for budget in budgets}

    matrices = bootstrap_matrices(
        calibration_values, int(config["bootstrap"]["replicates"]),
        int(config["bootstrap"]["seed"]) + (0 if dataset == "gsm8k" else 1009),
    )
    full_selected = select_bootstrap(matrices, list(range(48)), budgets, epsilon)
    clustered_selected = select_bootstrap(matrices, clustered, budgets, epsilon)
    dedup_selected = select_bootstrap(matrices, retained, budgets, epsilon)
    cluster_map = {int(key): int(value) for key, value in dedup["candidate_to_cluster"].items()}
    points = [float(value) for value in config["bootstrap"]["interval_quantiles"]]
    full_stability = analyze_selection_family(
        "full48", full_selected, matrices, calibration_values, heldout_values,
        cluster_map, budgets, points, config["stability_gate"],
    )
    clustered_stability = analyze_selection_family(
        "validation_behavior_clustered", clustered_selected, matrices, calibration_values, heldout_values,
        cluster_map, budgets, points, config["stability_gate"],
    )
    dedup_stability = analyze_selection_family(
        "validation_deduplicated", dedup_selected, matrices, calibration_values, heldout_values,
        cluster_map, budgets, points, config["stability_gate"],
    )

    primary = [int(value) for value in config["selection"]["primary_workpoints"][dataset]]
    full_primary_pass = all(full_stability[str(budget)]["stability_gate_passed"] for budget in primary)
    clustered_primary_pass = all(clustered_stability[str(budget)]["stability_gate_passed"] for budget in primary)
    pareto_primary_pass = all(dedup_stability[str(budget)]["stability_gate_passed"] for budget in primary)
    clustered_nominal_preserved = all(nominal_clustered[budget] == nominal_full[budget] for budget in primary)
    pareto_nominal_preserved = all(nominal_dedup[budget] == nominal_full[budget] for budget in primary)
    safe_candidate_reduction = (
        len(clustered) < 48 and clustered_primary_pass and clustered_nominal_preserved
    )
    decision = {
        "primary_workpoints": primary,
        "full48_passed": full_primary_pass,
        "validation_behavior_clustered_passed": clustered_primary_pass,
        "validation_behavior_clustered_nominal_primary_preserved": clustered_nominal_preserved,
        "validation_pareto_passed": pareto_primary_pass,
        "validation_pareto_nominal_primary_preserved": pareto_nominal_preserved,
        "calibration_expansion_recommended": not (clustered_primary_pass and clustered_nominal_preserved),
        "candidate_reduction_accepted": safe_candidate_reduction,
        "validation_pareto_rejected": not pareto_nominal_preserved,
        "reason": (
            "内部validation行为聚类保持原始主工作点且通过稳定性门槛"
            if clustered_primary_pass and clustered_nominal_preserved else
            "安全行为聚类后仍未同时满足主工作点保持与稳定性门槛"
        ),
    }

    ds_root = output_root / dataset
    ds_root.mkdir(parents=True, exist_ok=True)
    cluster_rows = []
    for cluster_index, members in enumerate(dedup["clusters"]):
        for member in members:
            cluster_rows.append({
                "cluster": cluster_index, "candidate": member,
                "lambda": lambdas[member], "mu": mus[member],
                "representative": member == dedup["representatives"][cluster_index],
                "retained": member in retained,
            })
    write_csv(ds_root / "validation_candidate_clusters.csv", cluster_rows)
    frequency_rows = []
    for family, stability in (("full48", full_stability), ("behavior_clustered", clustered_stability), ("pareto", dedup_stability)):
        for budget, values in stability.items():
            for candidate, frequency in values["selection_frequencies"].items():
                frequency_rows.append({"family": family, "B": budget, "candidate": candidate, "frequency": frequency})
    write_csv(ds_root / "bootstrap_selection_frequencies.csv", frequency_rows)

    for budget in primary:
        if plt is None:
            break
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for family, stability, color in (("48 candidates", full_stability, "#d95f02"), ("validation-dedup", dedup_stability, "#1b9e77")):
            row = stability[str(budget)]
            frequencies = sorted(((key, value) for key, value in row["selection_frequencies"].items()), key=lambda item: item[1], reverse=True)[:10]
            axes[0].plot(range(len(frequencies)), [value for _, value in frequencies], marker="o", label=family, color=color)
        axes[0].set_title(f"{dataset} B={budget}: top selection frequencies")
        axes[0].set_xlabel("rank")
        axes[0].set_ylabel("frequency")
        axes[0].legend()
        labels = ["exact checkpoint", "stop/fallback"]
        x = np.arange(2)
        width = 0.35
        a = full_stability[str(budget)]
        b = dedup_stability[str(budget)]
        axes[1].bar(x - width/2, [a["expected_pairwise_exact_checkpoint_agreement"], a["expected_pairwise_stop_fallback_agreement"]], width, label="48")
        axes[1].bar(x + width/2, [b["expected_pairwise_exact_checkpoint_agreement"], b["expected_pairwise_stop_fallback_agreement"]], width, label="dedup")
        axes[1].set_xticks(x, labels, rotation=15)
        axes[1].set_ylim(0, 1)
        axes[1].set_title("expected pairwise action agreement")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(ds_root / f"stability_B{budget}.png", dpi=180)
        plt.close(fig)

    serializable_dedup = {
        key: value for key, value in dedup.items()
        if key not in {"exact_checkpoint_agreement", "stop_fallback_agreement", "validation_summary"}
    }
    serializable_dedup["validation_summary"] = {
        key: [float(value) for value in np.asarray(values)] for key, values in dedup["validation_summary"].items()
    }
    report = {
        "status": "complete",
        "dataset": dataset,
        "source_protocol": source["protocol_id"],
        "constraints": config["constraints"],
        "problem_counts": {"internal_validation": len(validation_ids), "calibration": len(calibration_ids), "heldout": len(heldout_ids)},
        "candidate_count_original": 48,
        "deduplication": serializable_dedup,
        "nominal_selected": {"full48": nominal_full, "behavior_clustered": nominal_clustered, "validation_pareto": nominal_dedup},
        "bootstrap_full48": full_stability,
        "bootstrap_behavior_clustered": clustered_stability,
        "bootstrap_validation_pareto": dedup_stability,
        "decision": decision,
        "figures_generated": plt is not None,
        "figure_skip_reason": None if plt is not None else "matplotlib_not_installed",
        "heldout_used_for_selection": False,
    }
    atomic_json(report, ds_root / "CALIBRATION_STABILITY.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    source = yaml.safe_load((ROOT / config["source_protocol"]).read_text())
    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    resolved = json.dumps(config, ensure_ascii=False, sort_keys=True).encode()
    atomic_json({
        "config": config,
        "config_sha256": hashlib.sha256(resolved).hexdigest(),
        "source_protocol_id": source["protocol_id"],
    }, output_root / "RUN_SPEC.json")
    reports = {dataset: analyze_dataset(dataset, config, source, output_root) for dataset in ("gsm8k", "mmlu_pro")}
    need_expansion = any(row["decision"]["calibration_expansion_recommended"] for row in reports.values())
    retained = {
        dataset: {
            "behavior_cluster_representatives": len(row["deduplication"]["representatives"]),
            "validation_pareto": len(row["deduplication"]["retained_candidates"]),
        }
        for dataset, row in reports.items()
    }
    summary = {
        "status": "complete",
        "bootstrap_replicates": int(config["bootstrap"]["replicates"]),
        "new_llm_generation": 0,
        "new_samples": 0,
        "heldout_used_for_selection": False,
        "retained_candidates": retained,
        "calibration_expansion_recommended": need_expansion,
        "dataset_decisions": {dataset: row["decision"] for dataset, row in reports.items()},
    }
    atomic_json(summary, output_root / "SUMMARY.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

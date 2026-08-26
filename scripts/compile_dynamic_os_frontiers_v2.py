#!/usr/bin/env python3
"""汇总可部署动态方法、OS-Pruner 基线和完整风险—效率前沿。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.dynamic_optimal_stopping_deployable_v2 import summarize_token_records
from src.utils import atomic_json, load_yaml


BUDGETS = (0, 1, 2, 4, 10)
DYNAMIC_TRAINED = ("full", "no_trajectory", "one_step_value", "dense_endpoint_value")
REPLAY_VARIANTS = (
    "no_risk_penalty_mu0",
    "no_compute_cost_lambda0",
    "no_continuation_value_M0",
    "no_stop_correctness_pS0",
)
OS_VARIANTS = ("matched_os_pruner", "constrained_os_pruner")


def align(records: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    mapping = {str(row["problem_id"]): row for row in records}
    if len(mapping) != len(records) or set(mapping) != set(ids):
        raise ValueError(
            f"记录不配对：missing={sorted(set(ids)-set(mapping))[:5]}, "
            f"extra={sorted(set(mapping)-set(ids))[:5]}"
        )
    return [mapping[value] for value in ids]


def dense_records(source: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "problem_id": str(row["problem_id"]),
        "subject": row.get("subject"), "category": row.get("category"),
        "fallback": True, "checkpoint": None, "transition": "fallback",
        "method_prediction": row["dense_prediction"],
        "dense_prediction": row["dense_prediction"], "gold_answer": row["gold_answer"],
        "method_success": bool(row["dense_success"]),
        "dense_success": bool(row["dense_success"]),
        "method_tokens": int(row["dense_tokens"]), "dense_tokens": int(row["dense_tokens"]),
    } for row in source]


def metric_row(dataset: str, method: str, budget: int | str, records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_token_records(records)
    return {
        "dataset": dataset, "method": method, "budget_B": budget,
        "N": summary["problems"], "accuracy": summary["accuracy"],
        "dense_accuracy": summary["dense_accuracy"],
        "delta_dense_pp": summary["delta_dense_pp"],
        "coverage": summary["coverage"], "fallback": summary["fallback"],
        "mean_reasoning_tokens": summary["mean_reasoning_tokens"],
        "mean_dense_reasoning_tokens": summary["mean_dense_reasoning_tokens"],
        "token_reduction": summary["token_reduction"],
        "lost_correct_count": summary["lost_correct_count"],
        "lost_correct_rate": summary["lost_correct_rate"],
        "gained_correct_count": summary["gained_correct_count"],
        **summary["counts"],
    }


def raw_arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "accuracy": np.asarray([row["method_success"] for row in records], dtype=np.float64),
        "lost_correct_rate": np.asarray([
            row.get("transition") == "W_to_C" for row in records
        ], dtype=np.float64),
        "coverage": np.asarray([not row.get("fallback", False) for row in records], dtype=np.float64),
        "tokens": np.asarray([row["method_tokens"] for row in records], dtype=np.float64),
        "dense_tokens": np.asarray([row["dense_tokens"] for row in records], dtype=np.float64),
    }


def bootstrap_indices(
    records: list[dict[str, Any]], dataset: str, replicates: int, seed: int,
) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    if dataset == "gsm8k":
        return rng.integers(0, len(records), size=(replicates, len(records))), "problem"
    categories = np.asarray([str(row.get("category")) for row in records])
    parts = []
    for category in sorted(set(categories)):
        local = np.flatnonzero(categories == category)
        parts.append(rng.choice(local, size=(replicates, len(local)), replace=True))
    return np.concatenate(parts, axis=1), "category_stratified_problem"


def bootstrap_metrics(raw: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "accuracy": raw["accuracy"][indices].mean(axis=1),
        "lost_correct_rate": raw["lost_correct_rate"][indices].mean(axis=1),
        "coverage": raw["coverage"][indices].mean(axis=1),
        "token_reduction": 1.0 - (
            raw["tokens"][indices].mean(axis=1)
            / raw["dense_tokens"][indices].mean(axis=1)
        ),
    }


def point_metrics(raw: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "accuracy": float(raw["accuracy"].mean()),
        "lost_correct_rate": float(raw["lost_correct_rate"].mean()),
        "coverage": float(raw["coverage"].mean()),
        "token_reduction": float(1.0 - raw["tokens"].mean() / raw["dense_tokens"].mean()),
    }


def mark_pareto(frame: pd.DataFrame) -> pd.Series:
    result = []
    for index, row in frame.iterrows():
        dominated = ((frame.lost_correct_count <= row.lost_correct_count)
                     & (frame.token_reduction >= row.token_reduction)
                     & ((frame.lost_correct_count < row.lost_correct_count)
                        | (frame.token_reduction > row.token_reduction))).any()
        result.append(not bool(dominated))
    return pd.Series(result, index=frame.index)


def load_records(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)["records"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    marker = args.output / "pipeline.complete"
    if args.resume and marker.is_file():
        print(json.dumps({"status": "skipped_complete", "output": str(args.output)}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    tables = args.output / "tables"
    figures = args.output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    config = load_yaml(args.config)

    result_rows: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []
    feasibility_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    selected_records_by_dataset: dict[str, dict[tuple[str, int | str], list[dict[str, Any]]]] = {}

    for dataset in ("gsm8k", "mmlu_pro"):
        records_by_method: dict[str, dict[str, list[dict[str, Any]]]] = {}
        curve_sources: dict[str, list[dict[str, Any]]] = {}
        feasibility_sources: dict[str, dict[str, Any]] = {}

        for variant in DYNAMIC_TRAINED:
            root = args.run_root / "dynamic" / variant / dataset
            payload = json.loads((root / "probe.json").read_text(encoding="utf-8"))
            records_by_method[variant] = load_records(root / "policy_records.pt")["empirical_B"]
            curve_sources[variant] = payload["calibration"]["curve"]
            feasibility_sources[variant] = payload["candidate_feasibility"]

        replay_root = args.run_root / "dynamic" / "replay_ablations" / dataset
        replay_json = json.loads((replay_root / "replay_ablations.json").read_text(encoding="utf-8"))
        replay_records = load_records(replay_root / "policy_records.pt")
        for variant in REPLAY_VARIANTS:
            records_by_method[variant] = replay_records[variant]["empirical_B"]
            curve_sources[variant] = replay_json["results"][variant]["calibration_curve"]
            feasibility_sources[variant] = replay_json["results"][variant]["candidate_feasibility"]

        for variant in OS_VARIANTS:
            root = args.run_root / "os_pruner" / variant / dataset
            payload = json.loads((root / "probe.json").read_text(encoding="utf-8"))
            records_by_method[variant] = load_records(root / "policy_records.pt")["empirical_B"]
            curve_sources[variant] = payload["calibration"]["curve"]
            feasibility_sources[variant] = payload["calibration"]["candidate_feasibility"]

        sample = records_by_method["full"]["0"]
        ids = sorted(str(row["problem_id"]) for row in sample)
        aligned_methods: dict[tuple[str, int | str], list[dict[str, Any]]] = {
            ("Dense", "dense"): align(dense_records(sample), ids)
        }
        for method, budget_records in records_by_method.items():
            for budget in BUDGETS:
                aligned_methods[(method, budget)] = align(budget_records[str(budget)], ids)
        selected_records_by_dataset[dataset] = aligned_methods

        for (method, budget), records in aligned_methods.items():
            result_rows.append(metric_row(dataset, method, budget, records))

        dense_accuracy = summarize_token_records(aligned_methods[("Dense", "dense")])["accuracy"]
        epsilon = float(config["dynamic_policy"]["accuracy_epsilon"])
        for method, curve in curve_sources.items():
            local = pd.DataFrame(curve).copy()
            local["dataset"] = dataset
            local["method"] = method
            local["accuracy_feasible"] = local.accuracy >= dense_accuracy - epsilon
            local["pareto_all"] = mark_pareto(local)
            local["pareto_accuracy_feasible"] = False
            feasible = local[local.accuracy_feasible]
            if not feasible.empty:
                local.loc[feasible.index, "pareto_accuracy_feasible"] = mark_pareto(feasible)
            frontier_rows.extend(local.to_dict(orient="records"))
            counts = feasibility_sources[method]
            feasibility_rows.append({
                "dataset": dataset, "method": method,
                "total_candidates": counts["total_candidates"],
                "nonzero_stopping_candidates": counts["nonzero_stopping_candidates"],
                "accuracy_feasible_candidates": counts["accuracy_feasible_candidates"],
                **{
                    f"accuracy_and_B{budget}_feasible": counts["accuracy_and_budget_feasible"][str(budget)]
                    for budget in BUDGETS
                },
            })

        replicates = int(config["statistics"]["bootstrap_replicates"])
        indices, stratification = bootstrap_indices(
            aligned_methods[("Dense", "dense")], dataset, replicates,
            int(config["seed"]["bootstrap"]),
        )
        for budget in BUDGETS:
            main_key = ("full", budget)
            main_raw = raw_arrays(aligned_methods[main_key])
            main_boot = bootstrap_metrics(main_raw, indices)
            main_point = point_metrics(main_raw)
            for comparator in [
                "Dense", "matched_os_pruner", "constrained_os_pruner",
                "no_trajectory", "one_step_value", "dense_endpoint_value",
                "no_risk_penalty_mu0", "no_compute_cost_lambda0",
                "no_continuation_value_M0", "no_stop_correctness_pS0",
            ]:
                comparator_key = ("Dense", "dense") if comparator == "Dense" else (comparator, budget)
                comp_raw = raw_arrays(aligned_methods[comparator_key])
                comp_boot = bootstrap_metrics(comp_raw, indices)
                comp_point = point_metrics(comp_raw)
                for metric in ("accuracy", "token_reduction", "lost_correct_rate", "coverage"):
                    difference = main_boot[metric] - comp_boot[metric]
                    bootstrap_rows.append({
                        "dataset": dataset, "budget_B": budget,
                        "main": "full", "comparator": comparator, "metric": metric,
                        "difference_point": main_point[metric] - comp_point[metric],
                        "bootstrap_difference_mean": float(difference.mean()),
                        "ci_low": float(np.percentile(difference, 2.5)),
                        "ci_high": float(np.percentile(difference, 97.5)),
                        "replicates": replicates, "stratification": stratification,
                    })

    results_frame = pd.DataFrame(result_rows)
    frontier_frame = pd.DataFrame(frontier_rows)
    feasibility_frame = pd.DataFrame(feasibility_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    results_frame.to_csv(tables / "all_B_selected_results.csv", index=False)
    frontier_frame.to_csv(tables / "calibration_candidate_frontiers.csv", index=False)
    feasibility_frame.to_csv(tables / "candidate_feasibility_counts.csv", index=False)
    bootstrap_frame.to_csv(tables / "paired_bootstrap_comparisons.csv", index=False)

    colors = {
        "full": "#1f77b4", "constrained_os_pruner": "#d62728",
        "matched_os_pruner": "#ff7f0e", "no_trajectory": "#2ca02c",
        "one_step_value": "#9467bd", "dense_endpoint_value": "#8c564b",
        "no_risk_penalty_mu0": "#e377c2", "no_compute_cost_lambda0": "#7f7f7f",
        "no_continuation_value_M0": "#bcbd22", "no_stop_correctness_pS0": "#17becf",
    }
    for dataset in ("gsm8k", "mmlu_pro"):
        fig, ax = plt.subplots(figsize=(10, 7))
        subset = frontier_frame[frontier_frame.dataset == dataset]
        for method, group in subset.groupby("method"):
            ax.scatter(
                group.lost_correct_count, 100.0 * group.token_reduction,
                s=np.where(group.accuracy_feasible, 38, 14),
                alpha=np.where(group.accuracy_feasible, 0.75, 0.16),
                color=colors.get(method), label=method,
            )
            pareto = group[group.pareto_accuracy_feasible].sort_values("lost_correct_count")
            if not pareto.empty:
                ax.plot(
                    pareto.lost_correct_count, 100.0 * pareto.token_reduction,
                    color=colors.get(method), linewidth=1.8,
                )
        ax.set_xlabel("Calibration lost-correct count")
        ax.set_ylabel("Token reduction (%)")
        ax.set_title(f"{dataset}: calibration risk-efficiency candidates (opaque=accuracy feasible)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(figures / f"{dataset}_calibration_risk_efficiency_frontier.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 7))
        heldout = results_frame[
            (results_frame.dataset == dataset) & (results_frame.budget_B != "dense")
        ]
        for method, group in heldout.groupby("method"):
            group = group.sort_values("budget_B")
            ax.plot(
                group.lost_correct_count, 100.0 * group.token_reduction,
                marker="o", color=colors.get(method), label=method,
            )
            for _, row in group.iterrows():
                ax.annotate(f"B={row.budget_B}", (row.lost_correct_count, 100.0 * row.token_reduction), fontsize=6)
        ax.set_xlabel("Held-out lost-correct count")
        ax.set_ylabel("Token reduction (%)")
        ax.set_title(f"{dataset}: frozen calibration-selected B workpoints")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(figures / f"{dataset}_heldout_selected_B_frontier.png", dpi=180)
        plt.close(fig)

    report = [
        "# 可部署动态停止、OS-Pruner与完整风险—效率前沿", "",
        "本报告中的动态主方法只使用当前checkpoint可观测的z_t；Q_continue网络输出已经包含下一段期望token成本。所有策略只在calibration上选择，held-out仅评测冻结后的B工作点。", "",
    ]
    for dataset in ("gsm8k", "mmlu_pro"):
        report += [f"## {dataset}", ""]
        local = results_frame[results_frame.dataset == dataset]
        dense = local[local.method == "Dense"].iloc[0]
        report.append(
            f"Dense accuracy={100*dense.accuracy:.2f}%，平均reasoning tokens={dense.mean_reasoning_tokens:.1f}。"
        )
        for budget in BUDGETS:
            report.append(f"\n### B={budget}\n")
            for method in ("full", "matched_os_pruner", "constrained_os_pruner", "no_trajectory", "one_step_value", "dense_endpoint_value"):
                row = local[(local.method == method) & (local.budget_B.astype(str) == str(budget))].iloc[0]
                report.append(
                    f"- {method}: accuracy={100*row.accuracy:.2f}%（ΔDense={row.delta_dense_pp:+.2f}pp），"
                    f"token reduction={100*row.token_reduction:.2f}%，coverage={100*row.coverage:.2f}%，"
                    f"W→C/C→W={int(row.W_to_C)}/{int(row.C_to_W)}。"
                )
        report.append("")
    report += [
        "## 解释边界", "",
        "Matched OS-Pruner是同数据、同2566维特征、同MLP预算下的controlled adaptation；Constrained OS-Pruner是加入期望lost-correct惩罚的controlled extension，二者都不是原论文全部设置的官方复现。", "",
        "完整候选、可行候选计数、10000次配对bootstrap与图形见tables/和figures/。", "",
    ]
    (args.output / "FINAL_REPORT_ZH.md").write_text("\n".join(report), encoding="utf-8")
    atomic_json({
        "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config["protocol_id"], "datasets": ["gsm8k", "mmlu_pro"],
        "budgets": list(BUDGETS), "bootstrap_replicates": int(config["statistics"]["bootstrap_replicates"]),
        "heldout_used_for_selection": False,
        "artifacts": [
            "tables/all_B_selected_results.csv", "tables/calibration_candidate_frontiers.csv",
            "tables/candidate_feasibility_counts.csv", "tables/paired_bootstrap_comparisons.csv",
            "figures/*", "FINAL_REPORT_ZH.md",
        ],
    }, marker)
    print(json.dumps({"status": "complete", "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

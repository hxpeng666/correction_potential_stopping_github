#!/usr/bin/env python3
"""汇总动态最优停止与旧方法的token-only配对结果。"""
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
import pandas as pd
import torch

from src.dynamic_optimal_stopping_v1 import summarize_token_records
from src.utils import atomic_json, load_yaml


def normalize_token_cost(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """所有方法统一忽略短答案成本：stop只计checkpoint，fallback计Dense。"""
    output = []
    for source in records:
        row = dict(source)
        row["method_tokens"] = int(row["dense_tokens"] if row["fallback"] else row["checkpoint"])
        output.append(row)
    return output


def aligned(records: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    mapping = {str(row["problem_id"]): row for row in records}
    if len(mapping) != len(records) or set(mapping) != set(ids):
        raise ValueError("方法间heldout sample ID不一致或重复")
    return [mapping[value] for value in ids]


def dense_records(source: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "problem_id": row["problem_id"],
        "subject": row.get("subject"),
        "category": row.get("category"),
        "fallback": True,
        "checkpoint": None,
        "transition": "fallback",
        "method_prediction": row["dense_prediction"],
        "dense_prediction": row["dense_prediction"],
        "gold_answer": row["gold_answer"],
        "method_success": row["dense_success"],
        "dense_success": row["dense_success"],
        "method_tokens": row["dense_tokens"],
        "dense_tokens": row["dense_tokens"],
    } for row in source]


def result_row(dataset: str, method: str, family: str, key: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    value = summarize_token_records(records)
    return {
        "dataset": dataset,
        "method": method,
        "family": family,
        "key": key,
        "N": value["problems"],
        "accuracy": value["accuracy"],
        "dense_accuracy": value["dense_accuracy"],
        "delta_dense_pp": value["delta_dense_pp"],
        "accuracy_drop_pp": value["accuracy_drop_pp"],
        "coverage": value["coverage"],
        "mean_reasoning_tokens": value["mean_reasoning_tokens"],
        "mean_dense_reasoning_tokens": value["mean_dense_reasoning_tokens"],
        "token_reduction": value["token_reduction"],
        "lost_correct_count": value["lost_correct_count"],
        "lost_correct_rate": value["lost_correct_rate"],
        "gained_correct_count": value["gained_correct_count"],
        "fallback": value["fallback"],
        **value["counts"],
        "short_answer_cost_ignored": True,
    }


def raw_metrics(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "accuracy": np.asarray([row["method_success"] for row in records], dtype=np.float64),
        "dense_accuracy": np.asarray([row["dense_success"] for row in records], dtype=np.float64),
        "lost_correct_rate": np.asarray([row["transition"] == "W_to_C" for row in records], dtype=np.float64),
        "coverage": np.asarray([not row["fallback"] for row in records], dtype=np.float64),
        "tokens": np.asarray([row["method_tokens"] for row in records], dtype=np.float64),
        "dense_tokens": np.asarray([row["dense_tokens"] for row in records], dtype=np.float64),
    }


def point(values: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "accuracy": float(values["accuracy"].mean()),
        "delta_dense": float((values["accuracy"] - values["dense_accuracy"]).mean()),
        "lost_correct_rate": float(values["lost_correct_rate"].mean()),
        "coverage": float(values["coverage"].mean()),
        "token_reduction": float(1.0 - values["tokens"].mean() / values["dense_tokens"].mean()),
    }


def sampled(values: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "accuracy": values["accuracy"][indices].mean(1),
        "delta_dense": (values["accuracy"][indices] - values["dense_accuracy"][indices]).mean(1),
        "lost_correct_rate": values["lost_correct_rate"][indices].mean(1),
        "coverage": values["coverage"][indices].mean(1),
        "token_reduction": 1.0 - values["tokens"][indices].mean(1) / values["dense_tokens"][indices].mean(1),
    }


def bootstrap_indices(records: list[dict[str, Any]], dataset: str, replicates: int, seed: int) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    if dataset == "gsm8k":
        return rng.integers(0, len(records), size=(replicates, len(records))), "problem"
    categories = np.asarray([str(row.get("category")) for row in records])
    parts = []
    for category in sorted(set(categories)):
        local = np.flatnonzero(categories == category)
        parts.append(rng.choice(local, size=(replicates, len(local)), replace=True))
    return np.concatenate(parts, axis=1), "category_stratified_problem"


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    marker = args.output_root / "compile.complete"
    if args.resume and marker.is_file():
        print(json.dumps({"status": "skipped_complete"}))
        return
    config = load_yaml(args.config)
    dataset_config = config["datasets"][args.dataset]
    probe_json = json.loads((args.probe_root / "probe.json").read_text(encoding="utf-8"))
    dynamic_payload = torch.load(args.probe_root / "policy_records.pt", map_location="cpu", weights_only=False)["records"]
    tables = args.output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    methods: dict[str, list[dict[str, Any]]] = {}
    method_meta: dict[str, tuple[str, str]] = {}
    for family, values in dynamic_payload.items():
        for key, records in values.items():
            name = f"Dynamic {family}={key}"
            methods[name] = normalize_token_cost(records)
            method_meta[name] = (family, key)
    any_dynamic = next(iter(methods.values()))
    ids = sorted(str(row["problem_id"]) for row in any_dynamic)
    methods = {name: aligned(records, ids) for name, records in methods.items()}
    methods["Dense"] = aligned(dense_records(any_dynamic), ids)
    method_meta["Dense"] = ("baseline", "dense")

    old_root = ROOT / dataset_config["previous_probe_root"]
    old = torch.load(old_root / "policy_records.pt", map_location="cpu", weights_only=False)["records"]["empirical_B"]
    for budget in (0, 1, 2, 4, 10):
        name = f"Previous correction+trajectory B={budget}"
        methods[name] = aligned(normalize_token_cost(old[str(budget)]), ids)
        method_meta[name] = ("previous_empirical_B", str(budget))

    four_root = ROOT / dataset_config["previous_four_state_root"]
    four_variants = {
        "Previous four-state unweighted zero": four_root / "probe/policy_records.pt",
        "Previous four-state weighted+trajectory zero": four_root / "probe_legacy_weighted_trajectory/policy_records.pt",
    }
    for name, path in four_variants.items():
        records = torch.load(path, map_location="cpu", weights_only=False)["heldout"]
        methods[name] = aligned(normalize_token_cost(records), ids)
        method_meta[name] = ("previous_fixed_zero", "zero")

    rows = [result_row(args.dataset, name, *method_meta[name], records) for name, records in methods.items()]
    frame = pd.DataFrame(rows)
    frame.to_csv(tables / "main_results.csv", index=False)

    calibration_rows = []
    heldout_by_index = {int(row["candidate_index"]): row for row in probe_json["descriptive_heldout_frontier"]}
    for row in probe_json["calibration"]["curve"]:
        combined = {f"calibration_{key}": value for key, value in row.items() if not isinstance(value, dict)}
        heldout = heldout_by_index[int(row["candidate_index"])]
        combined.update({f"heldout_descriptive_{key}": value for key, value in heldout.items() if not isinstance(value, dict)})
        calibration_rows.append(combined)
    pd.DataFrame(calibration_rows).to_csv(tables / "dynamic_candidate_frontier.csv", index=False)

    selection_rows = []
    for family, values in probe_json["frozen_policy_results"].items():
        for key, payload in values.items():
            row = {"dataset": args.dataset, "family": family, "key": key}
            row.update({f"calibration_{name}": value for name, value in payload["calibration"].items() if not isinstance(value, dict)})
            row.update({f"heldout_{name}": value for name, value in payload["heldout"].items() if not isinstance(value, dict)})
            selection_rows.append(row)
    pd.DataFrame(selection_rows).to_csv(tables / "calibration_selected_policies.csv", index=False)

    replicates = int(config["statistics"]["bootstrap_replicates"])
    bootstrap_index, stratification = bootstrap_indices(methods["Dense"], args.dataset, replicates, int(config["seed"]["bootstrap"]))
    distributions = {}
    points = {}
    ci_rows = []
    for name, records in methods.items():
        raw = raw_metrics(records)
        points[name] = point(raw)
        distributions[name] = sampled(raw, bootstrap_index)
        for metric, samples in distributions[name].items():
            ci_rows.append({
                "dataset": args.dataset,
                "method": name,
                "metric": metric,
                "point": points[name][metric],
                "ci_low": float(np.percentile(samples, 2.5)),
                "ci_high": float(np.percentile(samples, 97.5)),
                "replicates": replicates,
                "stratification": stratification,
            })
    pd.DataFrame(ci_rows).to_csv(tables / "bootstrap_confidence_intervals.csv", index=False)

    comparisons = []
    dynamic_names = [name for name in methods if name.startswith("Dynamic")]
    for name in dynamic_names:
        family, key = method_meta[name]
        comparators = ["Dense", "Previous four-state unweighted zero", "Previous four-state weighted+trajectory zero"]
        if family == "empirical_B":
            comparators.append(f"Previous correction+trajectory B={key}")
        else:
            comparators.extend(["Previous correction+trajectory B=0", "Previous correction+trajectory B=4"])
        for comparator in comparators:
            for metric in points[name]:
                difference = distributions[name][metric] - distributions[comparator][metric]
                comparisons.append({
                    "dataset": args.dataset,
                    "main": name,
                    "comparator": comparator,
                    "metric": metric,
                    "difference_point": points[name][metric] - points[comparator][metric],
                    "bootstrap_difference_mean": float(difference.mean()),
                    "ci_low": float(np.percentile(difference, 2.5)),
                    "ci_high": float(np.percentile(difference, 97.5)),
                    "replicates": replicates,
                })
    pd.DataFrame(comparisons).to_csv(tables / "paired_comparisons.csv", index=False)

    if args.dataset == "mmlu_pro":
        category_rows = []
        report_methods = ["Dense", "Dynamic formal_alpha=0.05", "Dynamic empirical_B=0", "Dynamic empirical_B=4", "Previous correction+trajectory B=4"]
        for name in report_methods:
            if name not in methods:
                continue
            categories = sorted({str(row.get("category")) for row in methods[name]})
            for category in categories:
                subset = [row for row in methods[name] if str(row.get("category")) == category]
                summary = summarize_token_records(subset)
                category_rows.append({
                    "category": category,
                    "method": name,
                    "n": len(subset),
                    "accuracy": summary["accuracy"],
                    "coverage": summary["coverage"],
                    "token_reduction": summary["token_reduction"],
                    "lost_correct_count": summary["lost_correct_count"],
                    **summary["counts"],
                })
        pd.DataFrame(category_rows).to_csv(tables / "category_results.csv", index=False)

    dense = frame[frame.method == "Dense"].iloc[0]
    report = [
        f"# {args.dataset.upper()} 风险约束动态推理最优停止", "",
        "成本使用 reasoning token/4096 代替wall time，停止后的短答案成本固定为0。所有lambda、mu和工作点都只在500题calibration上冻结；heldout frontier仅为描述性诊断。", "",
        f"Dense accuracy={pct(dense.accuracy)}，平均reasoning tokens={dense.mean_reasoning_tokens:.1f}。", "",
        "## Formal simultaneous-95%工作点", "",
    ]
    for alpha in ("0.01", "0.02", "0.05"):
        name = f"Dynamic formal_alpha={alpha}"
        row = frame[frame.method == name].iloc[0]
        selected = probe_json["frozen_policy_results"]["formal_alpha"][alpha]["calibration"]
        report.append(
            f"- alpha={float(alpha):.0%}：lambda/mu={selected.get('lambda')}/{selected.get('mu')}，Dense fallback={selected.get('dense_fallback')}；heldout accuracy={pct(row.accuracy)}（ΔDense={row.delta_dense_pp:+.2f}pp），coverage={pct(row.coverage)}，token reduction={pct(row.token_reduction)}，W→C/C→W={int(row.W_to_C)}/{int(row.C_to_W)}。"
        )
    report += ["", "## 经验B补充", ""]
    for budget in (0, 1, 2, 4, 10):
        name = f"Dynamic empirical_B={budget}"
        row = frame[frame.method == name].iloc[0]
        selected = probe_json["frozen_policy_results"]["empirical_B"][str(budget)]["calibration"]
        report.append(
            f"- B={budget}：lambda/mu={selected.get('lambda')}/{selected.get('mu')}；accuracy={pct(row.accuracy)}（ΔDense={row.delta_dense_pp:+.2f}pp），coverage={pct(row.coverage)}，token reduction={pct(row.token_reduction)}，W→C/C→W={int(row.W_to_C)}/{int(row.C_to_W)}。"
        )
    report += [
        "", "完整48候选frontier、旧方法、四状态固定规则、10000次配对bootstrap和W→C→W机制审计见本目录表格及probe.json。", "",
    ]
    (args.output_root / "DYNAMIC_OPTIMAL_STOPPING_REPORT_ZH.md").write_text("\n".join(report), encoding="utf-8")
    atomic_json({
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "heldout": len(ids),
        "candidate_count": len(probe_json["calibration"]["curve"]),
        "bootstrap_replicates": replicates,
        "cost": "reasoning_tokens_only",
        "short_answer_cost": 0,
        "heldout_used_for_selection": False,
    }, marker)
    print(json.dumps({"status": "complete", "dataset": args.dataset, "output": str(args.output_root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

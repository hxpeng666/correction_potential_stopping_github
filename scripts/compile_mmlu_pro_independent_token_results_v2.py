#!/usr/bin/env python3
"""汇总 MMLU-Pro 独立1000/500/1000实验；效率只报告 token，不报告延迟。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.legacy_empirical_probe_v4 import summarize_policy_records
from src.utils import atomic_json, load_yaml

METHOD_LABELS = {
    "correctness": "Correctness (controlled)",
    "consistency": "Consistency (controlled)",
    "last_switch": "Last-switch (controlled)",
    "correction_bce": "Correction BCE",
    "correction_trajectory": "Correction + trajectory",
}


def dense_as_policy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "problem_id": row["problem_id"], "subject": row.get("subject"), "category": row.get("category"),
        "fallback": True, "checkpoint": None, "transition": "fallback",
        "method_prediction": row["prediction"], "dense_prediction": row["prediction"], "gold_answer": row["gold_answer"],
        "method_success": row["success"], "dense_success": row["success"],
        "method_tokens": row["reasoning_tokens"], "dense_tokens": row["reasoning_tokens"],
        "replay_wall_ms": float(row["reasoning_tokens"]), "dense_wall_ms": float(row["reasoning_tokens"]),
    } for row in rows]


def result_row(name: str, family: str, key: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    value = summarize_policy_records(records)
    counts = value["counts"]
    return {
        "dataset": "mmlu_pro", "method": name, "family": family, "key": key,
        "budget_B": int(key) if family == "empirical_B" else np.nan,
        "coverage_target": float(key) / 100 if family == "coverage" else np.nan,
        "N": value["problems"], "accuracy": value["accuracy"], "dense_accuracy": value["dense_accuracy"],
        "delta_dense_pp": 100 * (value["accuracy"] - value["dense_accuracy"]),
        "accuracy_drop_pp": value["accuracy_drop_pp"], "coverage": value["coverage"],
        "mean_generated_tokens": value["mean_reasoning_and_answer_tokens"],
        "mean_dense_tokens": value["mean_dense_reasoning_tokens"], "token_reduction": value["token_reduction"],
        "lost_correct_count": value["lost_correct_count"], "lost_correct_rate": value["lost_correct_rate"],
        "fallback": value["fallback"], **counts,
    }


def aligned(records: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    mapping = {str(row["problem_id"]): row for row in records}
    if set(mapping) != set(ids):
        raise ValueError("方法间 final-test sample ID 不一致")
    return [mapping[value] for value in ids]


def arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "accuracy": np.asarray([row["method_success"] for row in records], dtype=np.float64),
        "dense_accuracy": np.asarray([row["dense_success"] for row in records], dtype=np.float64),
        "lost_correct_rate": np.asarray([row["transition"] == "W_to_C" for row in records], dtype=np.float64),
        "coverage": np.asarray([not row["fallback"] for row in records], dtype=np.float64),
        "used_tokens": np.asarray([row["method_tokens"] for row in records], dtype=np.float64),
        "dense_tokens": np.asarray([row["dense_tokens"] for row in records], dtype=np.float64),
    }


def bootstrap_metrics(values: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "accuracy": values["accuracy"][indices].mean(axis=1),
        "delta_dense": (values["accuracy"][indices] - values["dense_accuracy"][indices]).mean(axis=1),
        "lost_correct_rate": values["lost_correct_rate"][indices].mean(axis=1),
        "coverage": values["coverage"][indices].mean(axis=1),
        "token_reduction": 1 - values["used_tokens"][indices].mean(axis=1) / values["dense_tokens"][indices].mean(axis=1),
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_paper_mmlu_pro_independent_token_v2.yaml")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(ROOT / args.config)
    marker = args.output_root / "compile.complete"
    if args.resume and marker.is_file():
        print(json.dumps({"status": "skipped_complete"})); return
    tables = args.output_root / "tables"; tables.mkdir(parents=True, exist_ok=True)
    baseline = torch.load(args.baseline_root / "baseline_records.pt", map_location="cpu", weights_only=False)["records"]
    records: dict[str, list[dict[str, Any]]] = {"Dense": dense_as_policy(baseline["dense"]), "Direct": baseline["direct"]}
    records.update({f"Fixed {budget}": value for budget, value in baseline["fixed"].items()})
    adaptive: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    calibration_audit = {}
    for method, label in METHOD_LABELS.items():
        root = args.probe_root / method
        probe_json = json.loads((root / "probe.json").read_text(encoding="utf-8"))
        if probe_json["calibration"].get("selection_metric") != "tokens":
            raise ValueError(f"{method} 未使用 token 阈值目标")
        calibration_audit[method] = {
            "selection_metric": "tokens", "best_epoch": probe_json["best_epoch"],
            "calibration_problems": probe_json["split_counts"]["calibration"]["problems"],
            "heldout_problems": probe_json["split_counts"]["heldout"]["problems"],
        }
        payload = torch.load(root / "policy_records.pt", map_location="cpu", weights_only=False)["records"]
        for family, values in payload.items():
            for key, local in values.items():
                adaptive[(label, family, key)] = local

    rows = [result_row(name, "baseline", "", local) for name, local in records.items()]
    rows += [result_row(name, family, key, local) for (name, family, key), local in adaptive.items()]
    frame = pd.DataFrame(rows); frame["report_label"] = config["report_label"]
    frame[frame.family.isin(["baseline", "empirical_B"])].to_csv(tables / "main_results.csv", index=False)
    frame[frame.family == "coverage"].to_csv(tables / "coverage_targeted.csv", index=False)
    frame[(frame.family == "empirical_B") & frame.method.isin(["Correction BCE", "Correction + trajectory"])].to_csv(tables / "loss_ablation.csv", index=False)
    frame[(frame.family == "empirical_B") & frame.method.isin(METHOD_LABELS.values())].to_csv(tables / "target_ablation.csv", index=False)

    ids = sorted(str(row["problem_id"]) for row in records["Dense"])
    dense_ordered = aligned(records["Dense"], ids)
    categories = np.asarray([str(row["category"]) for row in dense_ordered])
    rng = np.random.default_rng(int(config["seed"]["bootstrap"])); reps = int(config["statistics"]["bootstrap_replicates"])
    parts = []
    for category in sorted(set(categories)):
        local = np.flatnonzero(categories == category)
        parts.append(rng.choice(local, size=(reps, len(local)), replace=True))
    indices = np.concatenate(parts, axis=1)
    boot_records: dict[str, list[dict[str, Any]]] = dict(records)
    for (name, family, key), local in adaptive.items():
        if family == "empirical_B": boot_records[f"{name}|B={key}"] = local
    distributions, ci_rows = {}, []
    for name, local in boot_records.items():
        ordered = aligned(local, ids); distribution = bootstrap_metrics(arrays(ordered), indices); distributions[name] = distribution
        point = result_row(name, "bootstrap", "", ordered)
        for metric, samples in distribution.items():
            point_value = point[metric] if metric in point else point["delta_dense_pp"] / 100
            ci_rows.append({"method": name, "metric": metric, "point": point_value, "ci_low": float(np.percentile(samples, 2.5)), "ci_high": float(np.percentile(samples, 97.5)), "replicates": reps, "stratification": "MMLU-Pro category"})
    pd.DataFrame(ci_rows).to_csv(tables / "bootstrap_confidence_intervals.csv", index=False)

    comparisons = []
    for budget in (0, 1, 2, 4, 10):
        main = f"Correction + trajectory|B={budget}"
        if main not in distributions: continue
        comparators = [f"{label}|B={budget}" for label in METHOD_LABELS.values() if label != "Correction + trajectory"] + list(records)
        for comparator in comparators:
            if comparator not in distributions: continue
            for metric in ("accuracy", "token_reduction", "lost_correct_rate", "coverage"):
                delta = distributions[main][metric] - distributions[comparator][metric]
                comparisons.append({"budget_B": budget, "main": main, "comparator": comparator, "metric": metric, "difference_mean": float(delta.mean()), "ci_low": float(np.percentile(delta, 2.5)), "ci_high": float(np.percentile(delta, 97.5)), "replicates": reps})
    pd.DataFrame(comparisons).to_csv(tables / "paired_comparisons.csv", index=False)

    category_rows = []
    chosen = {"Dense": records["Dense"]}
    for budget in (1, 2, 4):
        chosen[f"Correction + trajectory B={budget}"] = adaptive[("Correction + trajectory", "empirical_B", str(budget))]
    for name, local in chosen.items():
        for category in sorted(set(categories)):
            subset = [row for row in local if row["category"] == category]
            category_rows.append({"category": category, "method": name, "n": len(subset), "accuracy": float(np.mean([row["method_success"] for row in subset])), "coverage": float(np.mean([not row["fallback"] for row in subset])), "token_reduction": 1 - np.mean([row["method_tokens"] for row in subset]) / np.mean([row["dense_tokens"] for row in subset]), "lost_correct": sum(row["transition"] == "W_to_C" for row in subset)})
    pd.DataFrame(category_rows).to_csv(tables / "category_results.csv", index=False)

    main = frame[(frame.method == "Correction + trajectory") & (frame.family == "empirical_B") & frame.budget_B.isin([0, 1, 2, 4, 10])].sort_values("budget_B")
    dense = frame[frame.method == "Dense"].iloc[0]
    aliases = {0: "B=0", 1: "Strict", 2: "Balanced", 4: "Aggressive", 10: "B=10"}
    report = [
        "# MMLU-Pro 独立同源划分 token-only 实验报告", "",
        f"协议：`{config['protocol_id']}`。probe-train/calibration/final-test 分别为1000/500/1000题，全部来自官方 test 且互不重叠；官方 validation 仅用于每类5-shot演示。旧800题只进入 probe-train，未进入 calibration 或 final-test。", "",
        "由于同一 GPU 上并发多个模型副本，本轮不测量或推断延迟。阈值在 calibration 上以平均生成 token 最少为目标选择；所有效率结论仅表述为 token reduction。", "",
        f"Dense accuracy={pct(dense.accuracy)}，平均 reasoning tokens={dense.mean_generated_tokens:.1f}。", "", "## 主方法经验 B 工作点", "",
    ]
    for _, row in main.iterrows():
        budget = int(row.budget_B)
        report.append(f"- {aliases[budget]}（B={budget}）：accuracy={pct(row.accuracy)}（相对 Dense {row.delta_dense_pp:+.2f}pp），coverage={pct(row.coverage)}，token reduction={pct(row.token_reduction)}，W→C/C→W={int(row.W_to_C)}/{int(row.C_to_W)}，fallback={int(row.fallback)}。")
    report += ["", "## 解释边界", "", "本报告不包含 latency reduction、wall-time 或 A100 replay latency。token reduction 与延迟的相关性来自此前实验的描述性观察，不能把本表中的 token 数直接改写为本轮实测延迟。", "", "完整受控基线、matched coverage、trajectory loss 消融、类别分解以及10,000次按类别分层的配对 bootstrap 位于 `tables/`。", ""]
    (args.output_root / "MMLU_PRO_TOKEN_REPORT_ZH.md").write_text("\n".join(report), encoding="utf-8")
    atomic_json({"status": "complete", "protocol_id": config["protocol_id"], "heldout": len(ids), "bootstrap_replicates": reps, "latency_reported": False, "primary_efficiency_metric": "token_reduction", "calibration_audit": calibration_audit}, marker)
    print(json.dumps({"status": "complete", "heldout": len(ids), "bootstrap": reps, "latency_reported": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

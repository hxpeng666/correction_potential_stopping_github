#!/usr/bin/env python3
"""汇总四状态固定效用差方法，并与原 Correction+trajectory 配对比较。"""
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

from src.legacy_empirical_probe_v4 import summarize_policy_records
from src.utils import atomic_json, load_yaml


def aligned(records: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    mapping = {str(row["problem_id"]): row for row in records}
    if len(mapping) != len(records):
        raise ValueError("policy records 中 sample ID 重复")
    if set(mapping) != set(ids):
        missing = sorted(set(ids) - set(mapping))[:10]
        extra = sorted(set(mapping) - set(ids))[:10]
        raise ValueError(f"新旧方法 heldout ID 不一致，missing={missing}, extra={extra}")
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
        "replay_wall_ms": row["dense_wall_ms"],
        "dense_wall_ms": row["dense_wall_ms"],
    } for row in source]


def result_row(dataset: str, method: str, workpoint: str, records: list[dict[str, Any]], latency: bool) -> dict[str, Any]:
    summary = summarize_policy_records(records)
    counts = summary["counts"]
    row = {
        "dataset": dataset,
        "method": method,
        "workpoint": workpoint,
        "N": summary["problems"],
        "accuracy": summary["accuracy"],
        "dense_accuracy": summary["dense_accuracy"],
        "delta_dense_pp": 100.0 * (summary["accuracy"] - summary["dense_accuracy"]),
        "accuracy_drop_pp": summary["accuracy_drop_pp"],
        "coverage": summary["coverage"],
        "mean_generated_tokens": summary["mean_reasoning_and_answer_tokens"],
        "mean_dense_tokens": summary["mean_dense_reasoning_tokens"],
        "token_reduction": summary["token_reduction"],
        "lost_correct_count": summary["lost_correct_count"],
        "lost_correct_rate": summary["lost_correct_rate"],
        "fallback": summary["fallback"],
        **counts,
    }
    if latency:
        row.update({
            "mean_replay_wall_ms": summary["mean_replay_wall_ms"],
            "mean_dense_wall_ms": summary["mean_dense_wall_ms"],
            "mean_replay_wall_reduction": summary["replay_wall_reduction"],
            "p95_replay_wall_ms": summary["p95_replay_wall_ms"],
            "p95_replay_wall_reduction": summary["p95_replay_wall_reduction"],
        })
    return row


def arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "accuracy": np.asarray([row["method_success"] for row in records], dtype=np.float64),
        "dense_accuracy": np.asarray([row["dense_success"] for row in records], dtype=np.float64),
        "lost_correct_rate": np.asarray([row["transition"] == "W_to_C" for row in records], dtype=np.float64),
        "coverage": np.asarray([not row["fallback"] for row in records], dtype=np.float64),
        "tokens": np.asarray([row["method_tokens"] for row in records], dtype=np.float64),
        "dense_tokens": np.asarray([row["dense_tokens"] for row in records], dtype=np.float64),
        "wall": np.asarray([row["replay_wall_ms"] for row in records], dtype=np.float64),
        "dense_wall": np.asarray([row["dense_wall_ms"] for row in records], dtype=np.float64),
    }


def bootstrap(values: dict[str, np.ndarray], indices: np.ndarray, latency: bool) -> dict[str, np.ndarray]:
    result = {
        "accuracy": values["accuracy"][indices].mean(axis=1),
        "delta_dense": (values["accuracy"][indices] - values["dense_accuracy"][indices]).mean(axis=1),
        "lost_correct_rate": values["lost_correct_rate"][indices].mean(axis=1),
        "coverage": values["coverage"][indices].mean(axis=1),
        "token_reduction": 1.0 - values["tokens"][indices].mean(axis=1) / values["dense_tokens"][indices].mean(axis=1),
    }
    if latency:
        result["mean_replay_wall_reduction"] = 1.0 - values["wall"][indices].mean(axis=1) / values["dense_wall"][indices].mean(axis=1)
    return result


def point_metrics(values: dict[str, np.ndarray], latency: bool) -> dict[str, float]:
    result = {
        "accuracy": float(values["accuracy"].mean()),
        "delta_dense": float((values["accuracy"] - values["dense_accuracy"]).mean()),
        "lost_correct_rate": float(values["lost_correct_rate"].mean()),
        "coverage": float(values["coverage"].mean()),
        "token_reduction": float(1.0 - values["tokens"].mean() / values["dense_tokens"].mean()),
    }
    if latency:
        result["mean_replay_wall_reduction"] = float(1.0 - values["wall"].mean() / values["dense_wall"].mean())
    return result


def bootstrap_indices(records: list[dict[str, Any]], dataset: str, reps: int, seed: int) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    n = len(records)
    if dataset == "mmlu_pro":
        categories = np.asarray([str(row.get("category")) for row in records])
        parts = []
        for category in sorted(set(categories)):
            local = np.flatnonzero(categories == category)
            parts.append(rng.choice(local, size=(reps, len(local)), replace=True))
        return np.concatenate(parts, axis=1), "MMLU-Pro category-stratified problem"
    return rng.integers(0, n, size=(reps, n)), "GSM8K problem"


def fmt(value: float) -> str:
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
        print(json.dumps({"status": "skipped_complete", "output": str(args.output_root)}))
        return
    config = load_yaml(args.config)
    dataset_config = config["datasets"][args.dataset]
    latency = bool(dataset_config["latency_reported"])
    tables = args.output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    probe_json = json.loads((args.probe_root / "probe.json").read_text(encoding="utf-8"))
    if probe_json.get("calibration_selected_anything") is not False:
        raise ValueError("新方法错误地使用了 calibration 选择停止阈值")
    new_payload = torch.load(args.probe_root / "policy_records.pt", map_location="cpu", weights_only=False)
    new_records = new_payload["heldout"]
    old_root = ROOT / dataset_config["previous_probe_root"]
    old_payload = torch.load(old_root / "policy_records.pt", map_location="cpu", weights_only=False)["records"]
    ids = sorted(str(row["problem_id"]) for row in new_records)
    new_records = aligned(new_records, ids)
    old_records = {
        key: aligned(value, ids)
        for key, value in old_payload["empirical_B"].items()
        if key in {"0", "1", "2", "4", "10"}
    }
    old_coverage_records = {
        key: aligned(value, ids)
        for key, value in old_payload["coverage"].items()
        if key in {"30", "40", "50", "60", "70", "80", "90"}
    }
    dense = aligned(dense_records(new_records), ids)

    methods: dict[str, list[dict[str, Any]]] = {
        "Dense": dense,
        "Four-state utility (zero rule)": new_records,
    }
    for budget in (0, 1, 2, 4, 10):
        methods[f"Previous Correction+trajectory B={budget}"] = old_records[str(budget)]
    for target in (30, 40, 50, 60, 70, 80, 90):
        methods[f"Previous Correction+trajectory coverage target={target}%"] = old_coverage_records[str(target)]
    rows = []
    for name, records in methods.items():
        workpoint = "fixed_zero_utility" if name.startswith("Four-state") else (
            "dense" if name == "Dense" else name.rsplit("=", 1)[-1]
        )
        rows.append(result_row(args.dataset, name, workpoint, records, latency))
    frame = pd.DataFrame(rows)
    frame.to_csv(tables / "main_comparison.csv", index=False)

    calibration = probe_json["calibration_fixed_rule_diagnostic_only"]
    pd.DataFrame([{
        "dataset": args.dataset,
        "selection_usage": "diagnostic_only",
        **{key: value for key, value in calibration.items() if not isinstance(value, dict)},
        **calibration["counts"],
    }]).to_csv(tables / "calibration_diagnostic.csv", index=False)
    atomic_json(probe_json["probability_diagnostics"], args.output_root / "FOUR_STATE_PROBABILITY_DIAGNOSTICS.json")

    reps = int(config["statistics"]["bootstrap_replicates"])
    indices, stratification = bootstrap_indices(dense, args.dataset, reps, int(config["seed"]["bootstrap"]))
    distributions: dict[str, dict[str, np.ndarray]] = {}
    points: dict[str, dict[str, float]] = {}
    ci_rows = []
    for name, records in methods.items():
        raw_values = arrays(records)
        values = bootstrap(raw_values, indices, latency)
        distributions[name] = values
        points[name] = point_metrics(raw_values, latency)
        for metric, samples in values.items():
            point_value = points[name][metric]
            ci_rows.append({
                "dataset": args.dataset,
                "method": name,
                "metric": metric,
                "point": point_value,
                "ci_low": float(np.percentile(samples, 2.5)),
                "ci_high": float(np.percentile(samples, 97.5)),
                "replicates": reps,
                "stratification": stratification,
            })
    pd.DataFrame(ci_rows).to_csv(tables / "bootstrap_confidence_intervals.csv", index=False)

    comparison_rows = []
    new_dist = distributions["Four-state utility (zero rule)"]
    for comparator, comparator_dist in distributions.items():
        if comparator == "Four-state utility (zero rule)":
            continue
        for metric in new_dist:
            difference = new_dist[metric] - comparator_dist[metric]
            comparison_rows.append({
                "dataset": args.dataset,
                "main": "Four-state utility (zero rule)",
                "comparator": comparator,
                "metric": metric,
                "difference_point": float(points["Four-state utility (zero rule)"][metric] - points[comparator][metric]),
                "bootstrap_difference_mean": float(difference.mean()),
                "ci_low": float(np.percentile(difference, 2.5)),
                "ci_high": float(np.percentile(difference, 97.5)),
                "replicates": reps,
            })
    pd.DataFrame(comparison_rows).to_csv(tables / "paired_comparisons.csv", index=False)

    if args.dataset == "mmlu_pro":
        category_rows = []
        categories = sorted({str(row.get("category")) for row in dense})
        for name, records in methods.items():
            for category in categories:
                subset = [row for row in records if str(row.get("category")) == category]
                summary = summarize_policy_records(subset)
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
        pd.DataFrame(category_rows).to_csv(tables / "category_comparison.csv", index=False)

    new_row = frame[frame.method == "Four-state utility (zero rule)"].iloc[0]
    dense_row = frame[frame.method == "Dense"].iloc[0]
    report = [
        f"# {args.dataset.upper()} 四状态概率效用差实验", "",
        f"协议：`{config['protocol_id']}`。新方法输出 softmax 四分类概率，顺序为 W→C、C→W、W→W、C→C；在 sentence checkpoint 上若 `P(W→C)-P(C→W) >= 0` 则继续，否则在首个负值处停止。", "",
        "本方法没有可调停止阈值。policy-calibration 的500题仅报告冻结规则的独立诊断，未选择阈值、工作点或 epoch；epoch 只由 probe-train 内部20%验证集的四分类交叉熵选择。", "",
        "## 核心结果", "",
        f"- Dense：accuracy={fmt(dense_row.accuracy)}，平均 tokens={dense_row.mean_generated_tokens:.1f}。",
        f"- 四状态固定规则：accuracy={fmt(new_row.accuracy)}（ΔDense={new_row.delta_dense_pp:+.2f}pp），coverage={fmt(new_row.coverage)}，token reduction={fmt(new_row.token_reduction)}，W→C/C→W={int(new_row.W_to_C)}/{int(new_row.C_to_W)}，fallback={int(new_row.fallback)}。",
    ]
    if latency:
        report.append(f"- A100 replay-estimated mean latency reduction={fmt(new_row.mean_replay_wall_reduction)}；它复用原公共成本缓存，并非重新在线计时。")
    else:
        report.append("- 本轮 MMLU-Pro 仍只报告 token，不报告 latency。")
    report += ["", "## 与旧方法比较", ""]
    for budget in (0, 1, 2, 4, 10):
        old = frame[frame.method == f"Previous Correction+trajectory B={budget}"].iloc[0]
        report.append(
            f"- 旧 B={budget}：accuracy={fmt(old.accuracy)}，coverage={fmt(old.coverage)}，token reduction={fmt(old.token_reduction)}，W→C/C→W={int(old.W_to_C)}/{int(old.C_to_W)}。"
        )
    coverage_candidates = frame[frame.method.str.contains("coverage target", regex=False)].copy()
    coverage_candidates["coverage_distance"] = (coverage_candidates.coverage - new_row.coverage).abs()
    closest = coverage_candidates.sort_values(["coverage_distance", "token_reduction"], ascending=[True, False]).iloc[0]
    report += [
        "", "## 最接近的旧 matched-coverage 点", "",
        f"- {closest.method}：accuracy={fmt(closest.accuracy)}，coverage={fmt(closest.coverage)}，token reduction={fmt(closest.token_reduction)}，W→C/C→W={int(closest.W_to_C)}/{int(closest.C_to_W)}。",
        f"- 新方法相对该点：accuracy {new_row.accuracy-closest.accuracy:+.4f}，token reduction {new_row.token_reduction-closest.token_reduction:+.4f}，lost-correct rate {new_row.lost_correct_rate-closest.lost_correct_rate:+.4f}。",
    ]
    report += [
        "", "所有点估计、四类状态、10,000次配对 bootstrap 置信区间及逐项差值见 `tables/`。", "",
    ]
    (args.output_root / "FOUR_STATE_UTILITY_REPORT_ZH.md").write_text("\n".join(report), encoding="utf-8")
    atomic_json({
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "heldout": len(ids),
        "bootstrap_replicates": reps,
        "threshold_calibration_used": False,
        "decision_rule": "continue iff p_WC-p_CW >= 0; stop otherwise",
        "latency_reported": latency,
        "artifacts": [
            "FOUR_STATE_UTILITY_REPORT_ZH.md",
            "FOUR_STATE_PROBABILITY_DIAGNOSTICS.json",
            "tables/main_comparison.csv",
            "tables/calibration_diagnostic.csv",
            "tables/bootstrap_confidence_intervals.csv",
            "tables/paired_comparisons.csv",
        ],
    }, marker)
    print(json.dumps({
        "status": "complete",
        "dataset": args.dataset,
        "heldout": len(ids),
        "new_method": new_row.to_dict(),
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

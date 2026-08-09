#!/usr/bin/env python3
"""Compile legacy empirical-v4 tables, paired bootstrap, figures, and reports."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import atomic_json


PROBES = {
    "correctness": "Correctness (controlled)",
    "consistency": "Consistency (controlled)",
    "last_switch": "Last-switch (controlled)",
    "correction_bce": "Correction BCE only",
    "correction_trajectory": "Correction + trajectory",
}
HISTORICAL_B = (0, 1, 2, 4, 10)
ALIASES = {1: "Strict", 2: "Balanced", 4: "Aggressive"}
RATE_MATCHED = {2: "0.5%", 5: "1%", 10: "2%"}
METRIC_KEYS = (
    "accuracy", "delta_dense_pp", "token_reduction", "coverage",
    "lost_correct_rate", "replay_wall_reduction", "p95_replay_wall_reduction",
)


def load_pt(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def normalize_dense(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in records:
        output.append({
            "problem_id": str(row["problem_id"]),
            "subject": row.get("subject"), "category": row.get("category"),
            "fallback": True, "checkpoint": None, "transition": "fallback",
            "method_success": bool(row["success"]), "dense_success": bool(row["success"]),
            "method_tokens": int(row["reasoning_tokens"]), "dense_tokens": int(row["reasoning_tokens"]),
            "replay_wall_ms": float(row["wall_ms"]), "dense_wall_ms": float(row["wall_ms"]),
        })
    return output


def arrays(records: list[dict[str, Any]], ids: list[str]) -> dict[str, np.ndarray]:
    by_id = {str(row["problem_id"]): row for row in records}
    if set(by_id) != set(ids):
        raise ValueError(f"method ID mismatch: expected={len(ids)} observed={len(by_id)}")
    ordered = [by_id[value] for value in ids]
    return {
        "method_success": np.asarray([row["method_success"] for row in ordered], dtype=np.float64),
        "dense_success": np.asarray([row["dense_success"] for row in ordered], dtype=np.float64),
        "method_tokens": np.asarray([row["method_tokens"] for row in ordered], dtype=np.float64),
        "dense_tokens": np.asarray([row["dense_tokens"] for row in ordered], dtype=np.float64),
        "method_wall": np.asarray([row["replay_wall_ms"] for row in ordered], dtype=np.float64),
        "dense_wall": np.asarray([row["dense_wall_ms"] for row in ordered], dtype=np.float64),
        "stopped": np.asarray([not row["fallback"] for row in ordered], dtype=np.float64),
        "lost": np.asarray([row["transition"] == "W_to_C" for row in ordered], dtype=np.float64),
        "gained": np.asarray([row["transition"] == "C_to_W" for row in ordered], dtype=np.float64),
    }


def metrics(record_rows: list[dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    values = arrays(record_rows, ids)
    n = len(ids)
    transitions = {name: 0 for name in ("W_to_C", "C_to_W", "W_to_W", "C_to_C")}
    for row in record_rows:
        if row["transition"] in transitions:
            transitions[row["transition"]] += 1
    accuracy = float(values["method_success"].mean())
    dense_accuracy = float(values["dense_success"].mean())
    return {
        "N": n,
        "dense_accuracy": dense_accuracy,
        "accuracy": accuracy,
        "accuracy_drop_pp": 100.0 * (dense_accuracy - accuracy),
        "delta_dense_pp": 100.0 * (accuracy - dense_accuracy),
        "mean_generated_tokens": float(values["method_tokens"].mean()),
        "mean_dense_tokens": float(values["dense_tokens"].mean()),
        "token_reduction": float(1.0 - values["method_tokens"].mean() / values["dense_tokens"].mean()),
        "coverage": float(values["stopped"].mean()),
        "W_to_C": transitions["W_to_C"], "C_to_W": transitions["C_to_W"],
        "W_to_W": transitions["W_to_W"], "C_to_C": transitions["C_to_C"],
        "lost_correct_rate": float(values["lost"].mean()),
        "gained_correct_rate": float(values["gained"].mean()),
        "fallback": int(n - values["stopped"].sum()),
        "mean_replay_wall_ms": float(values["method_wall"].mean()),
        "median_replay_wall_ms": float(np.median(values["method_wall"])),
        "p95_replay_wall_ms": float(np.percentile(values["method_wall"], 95)),
        "mean_dense_wall_ms": float(values["dense_wall"].mean()),
        "replay_wall_reduction": float(1.0 - values["method_wall"].mean() / values["dense_wall"].mean()),
        "p95_replay_wall_reduction": float(1.0 - np.percentile(values["method_wall"], 95) / np.percentile(values["dense_wall"], 95)),
        "latency_label": "A100 single-request replay-estimated latency",
    }


def bootstrap(
    methods: dict[str, list[dict[str, Any]]], ids: list[str], *, draws: int, seed: int,
    strata: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    values = {name: arrays(rows, ids) for name, rows in methods.items()}
    rng = np.random.default_rng(seed)
    distributions = {name: {key: np.empty(draws, dtype=np.float64) for key in METRIC_KEYS} for name in methods}
    n = len(ids)
    chunk = 100
    for start in range(0, draws, chunk):
        width = min(chunk, draws - start)
        if strata is None:
            index = rng.integers(0, n, size=(width, n), endpoint=False)
        else:
            stratum_array = np.asarray(strata, dtype=object)
            blocks = []
            for value in sorted(set(strata)):
                positions = np.flatnonzero(stratum_array == value)
                blocks.append(positions[rng.integers(0, len(positions), size=(width, len(positions)), endpoint=False)])
            index = np.concatenate(blocks, axis=1)
        for name, value in values.items():
            success = value["method_success"][index].mean(axis=1)
            dense_success = value["dense_success"][index].mean(axis=1)
            mt = value["method_tokens"][index].mean(axis=1)
            dt = value["dense_tokens"][index].mean(axis=1)
            mw = value["method_wall"][index]
            dw = value["dense_wall"][index]
            target = distributions[name]
            target["accuracy"][start:start + width] = success
            target["delta_dense_pp"][start:start + width] = 100.0 * (success - dense_success)
            target["token_reduction"][start:start + width] = 1.0 - mt / dt
            target["coverage"][start:start + width] = value["stopped"][index].mean(axis=1)
            target["lost_correct_rate"][start:start + width] = value["lost"][index].mean(axis=1)
            target["replay_wall_reduction"][start:start + width] = 1.0 - mw.mean(axis=1) / dw.mean(axis=1)
            target["p95_replay_wall_reduction"][start:start + width] = 1.0 - np.percentile(mw, 95, axis=1) / np.percentile(dw, 95, axis=1)
    rows = []
    for name, local in distributions.items():
        for metric_name, samples in local.items():
            low, high = np.percentile(samples, [2.5, 97.5])
            rows.append({"method_key": name, "metric": metric_name, "ci95_low": low, "ci95_high": high, "bootstrap_samples": draws, "resampling": "subject-stratified question-level paired" if strata is not None else "question-level paired"})
    return pd.DataFrame(rows), distributions


def comparisons(distributions: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    metrics_to_compare = ("accuracy", "token_reduction", "replay_wall_reduction", "lost_correct_rate")
    for budget in HISTORICAL_B:
        main = f"Correction + trajectory|B={budget}"
        if main not in distributions:
            continue
        comparators = [
            f"{PROBES[key]}|B={budget}"
            for key in ("correctness", "consistency", "last_switch", "correction_bce")
        ] + ["Direct"] + [f"Fixed-{value}" for value in (64, 96, 128, 192, 256)]
        for comparator in comparators:
            if comparator not in distributions:
                continue
            for metric_name in metrics_to_compare:
                difference = distributions[main][metric_name] - distributions[comparator][metric_name]
                low, high = np.percentile(difference, [2.5, 97.5])
                rows.append({
                    "workpoint": f"B={budget}", "main": main, "comparator": comparator,
                    "metric_difference_main_minus_comparator": metric_name,
                    "mean_bootstrap_difference": float(difference.mean()),
                    "ci95_low": float(low), "ci95_high": float(high),
                })
    return pd.DataFrame(rows)


def save_figures(dataset_rows: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    subset = dataset_rows[dataset_rows.family == "empirical_B"]
    for dataset in sorted(subset.dataset.unique()):
        local = subset[subset.dataset == dataset]
        plt.figure(figsize=(7.2, 5.0))
        for method in local.method.unique():
            group = local[local.method == method].sort_values("lost_correct_rate")
            plt.plot(100 * group.lost_correct_rate, 100 * group.replay_wall_reduction, marker="o", label=method)
        plt.xlabel("Held-out lost-correct rate (%)")
        plt.ylabel("Mean replay latency reduction (%)")
        plt.title(f"{dataset.upper()}: empirical-B frontier")
        plt.grid(alpha=0.25); plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(output / f"{dataset}_risk_latency_frontier.png", dpi=180)
        plt.close()


def outcome(rows: pd.DataFrame) -> tuple[bool, list[str]]:
    reasons = []
    passed = False
    for budget in (1, 2):
        main = rows[(rows.method == "Correction + trajectory") & (rows.family == "empirical_B") & (rows.budget_B == budget)]
        controlled = rows[(rows.method.isin(["Correctness (controlled)", "Consistency (controlled)", "Last-switch (controlled)"])) & (rows.family == "empirical_B") & (rows.budget_B == budget)]
        if len(main) != 1 or len(controlled) != 3:
            continue
        value = main.iloc[0]
        maintains = value.delta_dense_pp >= -1.0
        reduces = value.token_reduction > 0 and value.replay_wall_reduction > 0
        dominates = bool(((value.replay_wall_reduction > controlled.replay_wall_reduction) & (value.lost_correct_rate <= controlled.lost_correct_rate) & (value.accuracy >= controlled.accuracy)).all())
        reasons.append(f"B={budget}: maintains={maintains}, positive_saving={reduces}, dominates_all_controlled={dominates}")
        passed = passed or (maintains and reduces and dominates)
    return passed, reasons


def report_text(dataset: str, rows: pd.DataFrame, source_audit: dict[str, Any], chinese: bool) -> str:
    main = rows[(rows.method == "Correction + trajectory") & (rows.family == "empirical_B")]
    aliases = main[main.budget_B.isin([1, 2, 4])].sort_values("budget_B")
    lines = []
    raw_dtype = str(source_audit.get("dtype_override", "bfloat16")).lower()
    dtype_name = {"float16": "FP16", "bfloat16": "BF16"}.get(raw_dtype, raw_dtype.upper())
    reused = bool(source_audit.get("generation_reused", False))
    if chinese:
        lines += [f"# {dataset.upper()} 旧经验协议实验总结", "", "## 协议", "",
                  f"本结果使用单 seed、{dtype_name} Qwen3-4B、SDPA、pure sentence-step checkpoint，以及 calibration 上的绝对 lost-correct 预算 B。{'Dense/hidden/forced-answer来自用户批准复用的现有公共缓存。' if reused else ''}所有延迟均为冻结成本模型给出的 `A100 single-request replay-estimated latency`，不是完整在线实测。", "",
                  "Strict、Balanced、Aggressive 分别表示 B=1、2、4；它们是经验事件预算，不是总体风险的置信上界。held-out 只应用冻结阈值，不参与阈值或 epoch 选择。", "", "## 主方法", ""]
    else:
        lines += [f"# {dataset.upper()} Legacy Empirical Protocol Summary", "", "## Protocol", "",
                  f"This single-seed experiment uses {dtype_name} Qwen3-4B, SDPA, pure sentence-step checkpoints, and absolute calibration lost-correct budgets B. {'Dense/hidden/forced-answer artifacts are reused from the user-approved existing shared cache. ' if reused else ''}Every latency number is `A100 single-request replay-estimated latency` from the frozen cost model, not measured end-to-end policy latency.", "",
                  "Strict, Balanced, and Aggressive mean B=1, 2, and 4. They are empirical event budgets, not population-level confidence bounds. Held-out data is applied once after threshold freezing.", "", "## Main method", ""]
    for _, row in aliases.iterrows():
        alias = ALIASES[int(row.budget_B)]
        if chinese:
            lines.append(f"- {alias} (B={int(row.budget_B)})：accuracy {100*row.accuracy:.2f}%，相对 Dense {row.delta_dense_pp:+.2f} pp，coverage {100*row.coverage:.2f}%，lost-correct {int(row.W_to_C)}/{int(row.N)}，token 降低 {100*row.token_reduction:.2f}%，平均回放延迟降低 {100*row.replay_wall_reduction:.2f}%。")
        else:
            lines.append(f"- {alias} (B={int(row.budget_B)}): accuracy {100*row.accuracy:.2f}%, delta vs Dense {row.delta_dense_pp:+.2f} pp, coverage {100*row.coverage:.2f}%, lost-correct {int(row.W_to_C)}/{int(row.N)}, token reduction {100*row.token_reduction:.2f}%, mean replay-latency reduction {100*row.replay_wall_reduction:.2f}%.")
    lines += ["", "## " + ("数据来源与完整性" if chinese else "Data source and integrity"), ""]
    if dataset == "mmlu":
        cal_sources = source_audit["checks"]["mmlu"]["calibration"]["source_counts"]
        if chinese:
            lines.append(f"MMLU calibration 来源：{json.dumps(cal_sources, ensure_ascii=False)}。57 学科均有覆盖；test 为57学科分层的1000题。")
            if source_audit.get("mmlu_distribution_shift", {}).get("present"):
                lines.append("该结果存在明确的calibration/test来源偏移，必须解释为distribution-shift结果；test没有用于重新选择阈值。")
        else:
            lines.append(f"MMLU calibration sources: {json.dumps(cal_sources)}. All 57 subjects are covered; test is a 1,000-question stratified sample over 57 subjects.")
            if source_audit.get("mmlu_distribution_shift", {}).get("present"):
                lines.append("This is explicitly a distribution-shift result because calibration and test use different sources; test was not used to retune the threshold.")
    lines.append(("缓存审计状态：" if chinese else "Cache audit status: ") + source_audit["status"] + ".")
    positive, reasons = outcome(rows)
    lines += ["", "## " + ("预设成功判定" if chinese else "Predeclared outcome"), ""]
    if chinese:
        lines.append(("正结果。" if positive else "负结果：未同时满足跨全部受控基线的严格支配条件。") + "判定要求 strict 或 balanced 至少一个点在 Dense 准确率下降不超过1 pp、token和延迟节省均为正，并在相同 B 下同时不增加 held-out lost-correct、不降低准确率且延迟优于 Correctness、Consistency、Last-switch。")
        lines.append("逐点审计：" + "；".join(reasons) + "。")
    else:
        lines.append(("Positive result. " if positive else "Negative result: the method does not satisfy simultaneous strict dominance over every controlled baseline. ") + "The rule requires at least one strict/balanced point with no more than 1 pp Dense accuracy loss, positive token and latency savings, and at the same B no higher held-out lost-correct rate, no lower accuracy, and lower latency than Correctness, Consistency, and Last-switch.")
        lines.append("Per-point audit: " + "; ".join(reasons) + ".")
    lines += ["", "## " + ("解释限制" if chinese else "Interpretation limits"), ""]
    if chinese:
        lines.append("这些结果只支持本次冻结数据划分和经验 calibration 预算下的结论。Coverage-targeted 结果是 calibration 匹配后的 held-out 表现；held-out coverage 不要求与目标完全相同。负结果和 source shift 不做隐藏。")
    else:
        lines.append("Conclusions apply to this frozen split and empirical calibration budget. Coverage-targeted rows report held-out performance after calibration matching; held-out coverage need not equal its target. Negative results and source shifts are retained.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    args = parser.parse_args()
    audit = json.loads(args.cache_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise ValueError("cache audit has not passed")
    tables = args.run_root / "tables"; tables.mkdir(parents=True, exist_ok=True)
    all_main_rows, all_coverage_rows, all_frontier_rows = [], [], []
    all_ci, all_compare, subject_rows, category_rows = [], [], [], []
    for dataset in ("gsm8k", "mmlu"):
        base_dir = args.run_root / dataset / "baselines"
        baseline_payload = load_pt(base_dir / "baseline_records.pt")["records"]["heldout"]
        dense_records = normalize_dense(baseline_payload["dense"])
        ids = sorted(str(row["problem_id"]) for row in dense_records)
        method_records: dict[str, list[dict[str, Any]]] = {"Dense": dense_records, "Direct": baseline_payload["direct"]}
        for budget in (64, 96, 128, 192, 256):
            method_records[f"Fixed-{budget}"] = baseline_payload["fixed"][str(budget)]
        dense_metrics = metrics(dense_records, ids)
        all_main_rows.append({"dataset": dataset, "method": "Dense", "family": "baseline", "workpoint": "Dense", "budget_B": np.nan, **dense_metrics})
        for name in ["Direct"] + [f"Fixed-{value}" for value in (64, 96, 128, 192, 256)]:
            all_main_rows.append({"dataset": dataset, "method": name, "family": "baseline", "workpoint": name, "budget_B": np.nan, **metrics(method_records[name], ids)})
        probe_jsons = {}
        for key, display in PROBES.items():
            directory = args.run_root / dataset / "probes" / key
            payload = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
            records = load_pt(directory / "policy_records.pt")["records"]
            probe_jsons[key] = payload
            for budget in sorted({0, 1, 2, 4, 5, 10}):
                budget_key = str(budget)
                if budget_key not in records["empirical_B"]:
                    continue
                name = f"{display}|B={budget}"
                rows = records["empirical_B"][budget_key]
                method_records[name] = rows
                frozen = payload["frozen_policy_results"]["empirical_B"][budget_key]["calibration"]
                result = {"dataset": dataset, "method": display, "family": "empirical_B", "workpoint": ALIASES.get(budget, f"B={budget}"), "budget_B": budget, "threshold": frozen["threshold"], "calibration_lost_correct": frozen["lost_correct_count"], "calibration_coverage": frozen["coverage"], **metrics(rows, ids)}
                all_main_rows.append(result)
            for coverage_key, rows in records["coverage"].items():
                frozen = payload["frozen_policy_results"]["coverage"][coverage_key]["calibration"]
                all_coverage_rows.append({"dataset": dataset, "method": display, "coverage_target": int(coverage_key), "threshold": frozen["threshold"], "calibration_coverage": frozen["coverage"], "calibration_lost_correct": frozen["lost_correct_count"], **metrics(rows, ids)})
            for point in payload["calibration"]["curve"]:
                local_point = dict(point)
                transition_counts = local_point.pop("counts", {})
                all_frontier_rows.append({
                    "dataset": dataset,
                    "method": display,
                    **local_point,
                    **{name: int(transition_counts.get(name, 0)) for name in ("W_to_C", "C_to_W", "W_to_W", "C_to_C")},
                })
        dense_by_id = {str(row["problem_id"]): row for row in dense_records}
        bootstrap_strata = [str(dense_by_id[value]["subject"]) for value in ids] if dataset == "mmlu" else None
        ci, distributions = bootstrap(method_records, ids, draws=args.bootstrap_samples, seed=args.bootstrap_seed, strata=bootstrap_strata)
        ci.insert(0, "dataset", dataset); all_ci.append(ci)
        compare = comparisons(distributions); compare.insert(0, "dataset", dataset); all_compare.append(compare)
        if dataset == "mmlu":
            selected_names = ["Dense"] + [f"Correction + trajectory|B={value}" for value in (1, 2, 4)]
            for name in selected_names:
                by_id = {str(row["problem_id"]): row for row in method_records[name]}
                subjects = sorted({str(row.get("subject")) for row in by_id.values()})
                for subject in subjects:
                    local = [row for row in by_id.values() if str(row.get("subject")) == subject]
                    subject_rows.append({"subject": subject, "method": name, "N": len(local), "accuracy": float(np.mean([row["method_success"] for row in local])), "dense_accuracy": float(np.mean([row["dense_success"] for row in local]))})
                categories = sorted({str(row.get("category")) for row in by_id.values()})
                for category in categories:
                    local = [row for row in by_id.values() if str(row.get("category")) == category]
                    category_rows.append({"category": category, "method": name, "N": len(local), "accuracy": float(np.mean([row["method_success"] for row in local])), "dense_accuracy": float(np.mean([row["dense_success"] for row in local]))})
    main = pd.DataFrame(all_main_rows)
    coverage = pd.DataFrame(all_coverage_rows)
    frontier = pd.DataFrame(all_frontier_rows)
    main.to_csv(tables / "main_results.csv", index=False)
    main[(main.family == "empirical_B") & main.budget_B.isin(HISTORICAL_B)].to_csv(tables / "historical_empirical_B.csv", index=False)
    main[(main.family == "empirical_B") & main.budget_B.isin(RATE_MATCHED)].assign(rate_matched=lambda x: x.budget_B.map(RATE_MATCHED)).to_csv(tables / "rate_matched_empirical_sensitivity.csv", index=False)
    coverage.to_csv(tables / "coverage_targeted.csv", index=False)
    frontier.to_csv(tables / "risk_frontier.csv", index=False)
    target_methods = list(PROBES.values())
    main[(main.family == "empirical_B") & main.method.isin(target_methods) & main.budget_B.isin(HISTORICAL_B)].to_csv(tables / "target_ablation.csv", index=False)
    main[(main.family == "empirical_B") & main.method.isin(["Correction BCE only", "Correction + trajectory"]) & main.budget_B.isin(HISTORICAL_B)].to_csv(tables / "loss_ablation.csv", index=False)
    pd.concat(all_ci, ignore_index=True).to_csv(tables / "bootstrap_confidence_intervals.csv", index=False)
    pd.concat(all_compare, ignore_index=True).to_csv(tables / "paired_comparisons.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(tables / "mmlu_subject_results.csv", index=False)
    pd.DataFrame(category_rows).to_csv(tables / "mmlu_category_results.csv", index=False)
    save_figures(main, args.run_root / "figures")
    chinese_parts, english_parts = [], []
    for dataset in ("gsm8k", "mmlu"):
        chinese_parts.append(report_text(dataset, main[main.dataset == dataset], audit, True))
        english_parts.append(report_text(dataset, main[main.dataset == dataset], audit, False))
    (args.run_root / "FINAL_EXPERIMENT_SUMMARY_ZH.md").write_text("\n".join(chinese_parts), encoding="utf-8")
    (args.run_root / "FINAL_EXPERIMENT_SUMMARY_EN.md").write_text("\n".join(english_parts), encoding="utf-8")
    atomic_json({"status": "complete", "created_at": datetime.now(timezone.utc).isoformat(), "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.bootstrap_seed, "latency_label": "A100 single-request replay-estimated latency"}, args.run_root / "reporting.complete")
    print(json.dumps({"status": "complete", "tables": len(list(tables.glob("*.csv"))), "bootstrap_samples": args.bootstrap_samples}, indent=2))


if __name__ == "__main__":
    main()

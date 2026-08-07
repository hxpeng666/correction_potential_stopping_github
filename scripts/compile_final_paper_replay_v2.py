#!/usr/bin/env python3
"""将冻结的第二版回放实验编译为论文表格、置信区间、图形和报告。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils import atomic_json


DATASETS = ("gsm8k", "mmlu")
LATENCY_LABEL = "单请求回放估计延迟"
WORKPOINTS = {"strict": "0.005", "balanced": "0.01", "aggressive": "0.02"}
PROBES = (
    "correctness",
    "consistency",
    "last_switch",
    "correction_bce",
    "correction_trajectory",
)
FIXED = (64, 96, 128, 192, 256)
METRIC_KEYS = (
    "accuracy",
    "token_reduction",
    "mean_replay_latency_reduction",
    "p95_replay_latency_reduction",
    "lost_correct_risk",
    "coverage",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dense_as_policy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        success = bool(row["success"])
        output.append({
            "problem_id": str(row["problem_id"]),
            "subject": row.get("subject"),
            "category": row.get("category"),
            "fallback": True,
            "checkpoint": None,
            "transition": "C_to_C" if success else "W_to_W",
            "method_success": success,
            "dense_success": success,
            "method_tokens": int(row["reasoning_tokens"]),
            "dense_tokens": int(row["reasoning_tokens"]),
            "replay_wall_ms": float(row["wall_ms"]),
            "dense_wall_ms": float(row["wall_ms"]),
        })
    return output


def load_methods(root: Path, dataset: str) -> dict[str, list[dict[str, Any]]]:
    baseline_path = root / dataset / "baselines" / "baseline_records.pt"
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)["records"]["heldout"]
    methods: dict[str, list[dict[str, Any]]] = {
        "Dense": dense_as_policy(baseline["dense"]),
        "Direct": baseline["direct"],
    }
    for budget in FIXED:
        methods[f"Fixed-{budget}"] = baseline["fixed"][str(budget)]
    for probe in PROBES:
        source = torch.load(
            root / dataset / "probes" / probe / "policy_records.pt",
            map_location="cpu",
            weights_only=False,
        )["records"]["formal"]
        for workpoint, alpha in WORKPOINTS.items():
            methods[f"{probe}:{workpoint}"] = source[alpha]
    reference = sorted(row["problem_id"] for row in methods["Dense"])
    expected = 1319 if dataset == "gsm8k" else 14042
    if len(reference) != expected or len(set(reference)) != expected:
        raise ValueError(f"{dataset} heldout IDs incomplete/nonunique: {len(reference)}/{expected}")
    for method, rows in methods.items():
        ids = sorted(str(row["problem_id"]) for row in rows)
        if ids != reference:
            raise ValueError(f"sample-ID mismatch for {dataset}/{method}")
        rows.sort(key=lambda row: str(row["problem_id"]))
    return methods


def arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "success": np.asarray([row["method_success"] for row in rows], dtype=np.float64),
        "dense_success": np.asarray([row["dense_success"] for row in rows], dtype=np.float64),
        "tokens": np.asarray([row["method_tokens"] for row in rows], dtype=np.float64),
        "dense_tokens": np.asarray([row["dense_tokens"] for row in rows], dtype=np.float64),
        "latency": np.asarray([row["replay_wall_ms"] for row in rows], dtype=np.float64),
        "dense_latency": np.asarray([row["dense_wall_ms"] for row in rows], dtype=np.float64),
        "lost": np.asarray([row["transition"] == "W_to_C" for row in rows], dtype=np.float64),
        "coverage": np.asarray([not row["fallback"] for row in rows], dtype=np.float64),
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    a = arrays(rows)
    transitions = Counter(row["transition"] for row in rows)
    mean_dense_tokens = float(a["dense_tokens"].mean())
    mean_dense_latency = float(a["dense_latency"].mean())
    p95_dense = float(np.percentile(a["dense_latency"], 95))
    result = {
        "n": len(rows),
        "accuracy": float(a["success"].mean()),
        "accuracy_change_vs_dense": float((a["success"] - a["dense_success"]).mean()),
        "mean_generated_tokens": float(a["tokens"].mean()),
        "token_reduction": float(1.0 - a["tokens"].mean() / mean_dense_tokens),
        "coverage": float(a["coverage"].mean()),
        "W_to_C": int(transitions["W_to_C"]),
        "C_to_W": int(transitions["C_to_W"]),
        "W_to_W": int(transitions["W_to_W"]),
        "C_to_C": int(transitions["C_to_C"]),
        "fallback": int(sum(bool(row["fallback"]) for row in rows)),
        "fallback_rate": float(np.mean([bool(row["fallback"]) for row in rows])),
        "lost_correct_risk": float(a["lost"].mean()),
        "mean_replay_latency_ms": float(a["latency"].mean()),
        "median_replay_latency_ms": float(np.median(a["latency"])),
        "p95_replay_latency_ms": float(np.percentile(a["latency"], 95)),
        "mean_replay_latency_reduction": float(1.0 - a["latency"].mean() / mean_dense_latency),
        "p95_replay_latency_reduction": float(1.0 - np.percentile(a["latency"], 95) / p95_dense),
        "latency_label": LATENCY_LABEL,
    }
    subjects = sorted({str(row.get("subject")) for row in rows if row.get("subject") is not None})
    if subjects:
        result["subject_macro_accuracy"] = float(np.mean([
            np.mean([row["method_success"] for row in rows if str(row.get("subject")) == subject])
            for subject in subjects
        ]))
        result["subjects"] = len(subjects)
    return result


def bootstrap(methods: dict[str, list[dict[str, Any]]], repetitions: int, seed: int) -> tuple[dict, list[dict]]:
    names = list(methods)
    data = {name: arrays(methods[name]) for name in names}
    n = len(next(iter(methods.values())))
    samples = {name: {key: np.empty(repetitions, dtype=np.float64) for key in METRIC_KEYS} for name in names}
    rng = np.random.default_rng(seed)
    chunk = 20
    for start in range(0, repetitions, chunk):
        size = min(chunk, repetitions - start)
        index = rng.integers(0, n, size=(size, n), dtype=np.int32)
        for name in names:
            a = data[name]
            success = a["success"][index]
            tokens = a["tokens"][index]
            dense_tokens = a["dense_tokens"][index]
            latency = a["latency"][index]
            dense_latency = a["dense_latency"][index]
            dest = samples[name]
            sl = slice(start, start + size)
            dest["accuracy"][sl] = success.mean(axis=1)
            dest["token_reduction"][sl] = 1.0 - tokens.mean(axis=1) / dense_tokens.mean(axis=1)
            dest["mean_replay_latency_reduction"][sl] = 1.0 - latency.mean(axis=1) / dense_latency.mean(axis=1)
            dest["p95_replay_latency_reduction"][sl] = 1.0 - np.percentile(latency, 95, axis=1) / np.percentile(dense_latency, 95, axis=1)
            dest["lost_correct_risk"][sl] = a["lost"][index].mean(axis=1)
            dest["coverage"][sl] = a["coverage"][index].mean(axis=1)
    cis: dict[str, dict[str, dict[str, float]]] = {}
    for name in names:
        cis[name] = {}
        for key in METRIC_KEYS:
            low, high = np.percentile(samples[name][key], [2.5, 97.5])
            cis[name][key] = {"low": float(low), "high": float(high)}
    comparisons = []
    baselines = [name for name in names if not name.startswith("correction_trajectory:")]
    for workpoint in WORKPOINTS:
        main = f"correction_trajectory:{workpoint}"
        for baseline in baselines:
            row = {"main": main, "baseline": baseline}
            for key in ("accuracy", "token_reduction", "mean_replay_latency_reduction", "lost_correct_risk"):
                delta = samples[main][key] - samples[baseline][key]
                low, high = np.percentile(delta, [2.5, 97.5])
                row[f"{key}_difference"] = float(delta.mean())
                row[f"{key}_difference_ci_low"] = float(low)
                row[f"{key}_difference_ci_high"] = float(high)
            comparisons.append(row)
    return cis, comparisons


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def frontier_rows(root: Path) -> list[dict[str, Any]]:
    output = []
    for dataset in DATASETS:
        for probe in PROBES:
            payload = read_json(root / dataset / "probes" / probe / "probe.json")
            selected: dict[int, list[str]] = {}
            for name, value in payload["calibration"]["formal"].items():
                selected.setdefault(int(value["grid_index"]), []).append(name)
            for row in payload["calibration"]["curve"]:
                output.append({
                    "dataset": dataset,
                    "method": probe,
                    "split": "calibration",
                    "selection_role": ";".join(selected.get(int(row["grid_index"]), [])),
                    **{key: value for key, value in row.items() if key != "counts"},
                    **{f"count_{key}": value for key, value in row.get("counts", {}).items()},
                    "latency_label": LATENCY_LABEL,
                })
    return output


def subgroup_rows(dataset: str, methods: dict[str, list[dict[str, Any]]], level: str) -> list[dict[str, Any]]:
    output = []
    for method, rows in methods.items():
        groups = sorted({str(row.get(level)) for row in rows if row.get(level) is not None})
        for group in groups:
            local = [row for row in rows if str(row.get(level)) == group]
            output.append({"dataset": dataset, "level": level, "group": group, "method": method, **metrics(local)})
    return output


def plots(root: Path, main_rows: list[dict], frontier: list[dict]) -> None:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for probe in PROBES:
            rows = [row for row in frontier if row["dataset"] == dataset and row["method"] == probe and not row["disabled"]]
            ax.plot([100*r["lost_correct_rate"] for r in rows], [100*r["replay_wall_reduction"] for r in rows], label=probe)
        ax.set_xlabel("校准集正确答案丢失风险（%）")
        ax.set_ylabel("回放延迟降低比例（%）")
        ax.set_title(f"{dataset.upper()} 校准集风险—延迟前沿")
        ax.grid(alpha=.25); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figures / f"{dataset}_risk_latency_frontier.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for dataset, marker in (("gsm8k", "o"), ("mmlu", "s")):
        rows = [row for row in main_rows if row["dataset"] == dataset]
        ax.scatter([100*r["mean_replay_latency_reduction"] for r in rows], [100*r["accuracy"] for r in rows], label=dataset, marker=marker, alpha=.75)
    ax.set_xlabel("回放延迟降低比例（%）"); ax.set_ylabel("准确率（%）")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figures / "accuracy_latency_tradeoff.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for dataset, marker in (("gsm8k", "o"), ("mmlu", "s")):
        rows = [row for row in main_rows if row["dataset"] == dataset]
        ax.scatter([100*r["token_reduction"] for r in rows], [100*r["mean_replay_latency_reduction"] for r in rows], label=dataset, marker=marker, alpha=.75)
    ax.set_xlabel("生成令牌降低比例（%）"); ax.set_ylabel("回放延迟降低比例（%）")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figures / "token_walltime_comparison.png", dpi=180); plt.close(fig)


def verdict(all_metrics: dict[str, dict[str, dict[str, Any]]], root: Path) -> str:
    dataset_positive = {}
    reasons = {}
    for dataset in DATASETS:
        probe_meta = read_json(root / dataset / "probes" / "correction_trajectory" / "probe.json")
        candidates = []
        for workpoint, alpha in WORKPOINTS.items():
            main = all_metrics[dataset][f"correction_trajectory:{workpoint}"]
            calibrated = probe_meta["calibration"]["formal"][alpha]
            controlled = [all_metrics[dataset][f"{name}:{workpoint}"] for name in ("correctness", "consistency", "last_switch")]
            maintained = main["accuracy_change_vs_dense"] >= -0.01
            valid_risk = bool(calibrated["dense_fallback"]) or calibrated["simultaneous_upper_95"] <= float(alpha) + 1e-12
            saving = main["token_reduction"] > 0 and main["mean_replay_latency_reduction"] > 0
            dominates = all(
                main["mean_replay_latency_reduction"] > base["mean_replay_latency_reduction"]
                and main["lost_correct_risk"] <= base["lost_correct_risk"] + 0.001
                and main["accuracy"] >= base["accuracy"] - 0.001
                for base in controlled
            )
            candidates.append((workpoint, maintained, valid_risk, saving, dominates))
        passed = [row for row in candidates if all(row[1:])]
        dataset_positive[dataset] = bool(passed)
        reasons[dataset] = candidates
    if all(dataset_positive.values()):
        zh = "正结果：GSM8K 与 MMLU 均至少有一个预声明工作点满足准确率、校准风险、token/延迟节省及统一受控基线前沿判据。"
    elif any(dataset_positive.values()):
        zh = "跨任务结果不一致：方法只在一个数据集满足预声明正结果判据，不能声称跨任务成功。"
    else:
        zh = "负结果：两个数据集未同时满足预声明正结果判据；报告保留最强基线，不放宽风险或任务标准。"
    atomic_json({"dataset_positive": dataset_positive, "checks": reasons, "结论": zh}, root / "VERDICT.json")
    return zh


def report(root: Path, main_rows: list[dict], zh_verdict: str) -> None:
    def selected(dataset: str, method: str) -> dict:
        return next(row for row in main_rows if row["dataset"] == dataset and row["method"] == method)
    lines = [
        "# 最终单随机种子论文实验报告",
        "",
        f"本报告严格使用随机种子 20260803、冻结的 Qwen3-4B、FP16 与不可变公共缓存。所有延迟均为 **{LATENCY_LABEL}**；不属于完整策略的在线端到端实测耗时，分支采集设备的耗时未进入延迟模型。",
        "",
        "## 实验与数据设置",
        "",
        "GSM8K 从官方训练集中固定选取 5,000/1,000 题作为探针训练集和策略校准集，官方测试集 1,319 题全部作为留出集；MMLU 从非测试数据中分层固定选取 4,000/1,000 题，57 个学科的官方测试集 14,042 题全部作为留出集，5-shot 示例仅来自开发集。特征标准化、训练和阈值选择均未访问留出集。",
        "",
        "完整推理、直接作答、固定预算、三个受控停止目标基线、修正潜力 BCE 消融及轨迹最弱点保护完整方法共享同一套完整轨迹、强制作答和隐藏状态缓存。阈值只在校准集的 101 点有限网格上选择，采用 Bonferroni 校正后的单侧 95% 二项分布同时置信上界；严格、平衡和激进工作点对应 0.5%、1% 和 2%。",
        "",
        "## 主结果摘要",
        "",
        "| 数据集 | 方法 | 准确率 | 相对完整推理变化 | 令牌降低比例 | 平均回放延迟降低比例 | P95 回放延迟降低比例 | 正确答案丢失风险 | 停止覆盖率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    show = ["Dense", "Direct", "Fixed-64", "Fixed-128", "Fixed-256"] + [f"correction_trajectory:{wp}" for wp in WORKPOINTS]
    for dataset in DATASETS:
        for method in show:
            row = selected(dataset, method)
            lines.append(f"| {dataset.upper()} | {method} | {100*row['accuracy']:.2f}% | {100*row['accuracy_change_vs_dense']:+.2f} pp | {100*row['token_reduction']:.2f}% | {100*row['mean_replay_latency_reduction']:.2f}% | {100*row['p95_replay_latency_reduction']:.2f}% | {100*row['lost_correct_risk']:.2f}% | {100*row['coverage']:.2f}% |")
    lines += [
        "",
        "## 统计、分解与审计",
        "",
        "`tables/main_results.csv` 给出全部方法、三个工作点、四类状态转移和 10,000 次样本级配对 Bootstrap 的 95% 置信区间；`tables/paired_bootstrap_differences.csv` 给出完整方法相对每个基线的配对差；`tables/risk_frontier.csv` 是校准集风险前沿，未将留出集描述性结果用于选点。MMLU 的 57 个学科与四大类别分解分别保存于对应表格。",
        "",
        "轨迹保护的主消融是 `correction_bce` 与 `correction_trajectory` 的比较。正确性、一致性和最后切换明确是统一框架下的受控基线，不是其他论文完整训练协议的原样复现。",
        "",
        "缓存完整性与泄漏审计见 `CACHE_AUDIT.json` 和 `CACHE_AND_LEAKAGE_AUDIT.md`。本轮不包含完整策略的在线计时，因此不作端到端实测耗时声明。",
        "",
        "## 结论",
        "",
        zh_verdict,
        "",
    ]
    (root / "FINAL_PAPER_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    (root / "FINAL_PAPER_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    reproducibility = "\n".join([
        "# 可复现性说明", "",
        "- 项目根目录：仓库根目录",
        "- 实验协议：`final_paper_replay_v2`",
        "- 全局随机种子：`20260803`",
        "- 模型：冻结的 Qwen3-4B",
        "- 数据类型：FP16；不量化；每张 GPU 加载一个独立模型副本",
        "- 公共缓存键：`(dataset, split, sample_id, checkpoint)`",
        "- 生成随机种子键：`(global_seed, dataset, split, sample_id, checkpoint)`",
        "- 自适应特征：第 20 层、5,126 维隐藏状态动态特征",
        "- 阈值：仅在校准集上扫描 101 点网格，并使用单侧 95% 二项分布同时置信上界",
        "- Bootstrap：进行 10,000 次样本级配对重采样，所有比较方法使用相同的抽样 ID",
        f"- 延迟标签：{LATENCY_LABEL}；不纳入分支工作进程的计时数据",
        "",
        "不可变数据划分清单、缓存审计、冻结的探针产物、校准曲线、策略记录、成本模型和报告清单均保留在本次结果根目录中。",
        "",
    ])
    (root / "REPRODUCIBILITY.md").write_text(reproducibility, encoding="utf-8")


def main() -> None:
    global LATENCY_LABEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument(
        "--latency-label",
        default="单请求回放估计延迟",
    )
    args = parser.parse_args()
    LATENCY_LABEL = str(args.latency_label)
    root = args.root if args.root.is_absolute() else ROOT / args.root
    tables = root / "tables"; tables.mkdir(parents=True, exist_ok=True)
    all_methods = {dataset: load_methods(root, dataset) for dataset in DATASETS}
    all_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    main_rows, pairwise, subject_rows, category_rows = [], [], [], []
    for offset, dataset in enumerate(DATASETS):
        methods = all_methods[dataset]
        cis, comparisons = bootstrap(methods, args.bootstrap_repetitions, 20260803 + offset)
        all_metrics[dataset] = {}
        for method, rows in methods.items():
            summary = metrics(rows)
            all_metrics[dataset][method] = summary
            flattened = {"dataset": dataset, "method": method, **summary}
            if ":" in method and method.split(":", 1)[0] in PROBES:
                probe, workpoint = method.split(":", 1)
                alpha = WORKPOINTS[workpoint]
                frozen = read_json(root / dataset / "probes" / probe / "probe.json")["calibration"]["formal"][alpha]
                flattened.update({
                    "workpoint": workpoint,
                    "risk_budget_alpha": float(alpha),
                    "calibration_threshold": float(frozen["threshold"]),
                    "calibration_simultaneous_upper_95": float(frozen["simultaneous_upper_95"]),
                    "calibration_dense_fallback": bool(frozen["dense_fallback"]),
                })
            for key, bounds in cis[method].items():
                flattened[f"{key}_ci_low"] = bounds["low"]
                flattened[f"{key}_ci_high"] = bounds["high"]
            main_rows.append(flattened)
        pairwise.extend({"dataset": dataset, **row} for row in comparisons)
        if dataset == "mmlu":
            subject_rows.extend(subgroup_rows(dataset, methods, "subject"))
            category_rows.extend(subgroup_rows(dataset, methods, "category"))
    frontier = frontier_rows(root)
    write_csv(tables / "main_results.csv", main_rows)
    write_csv(tables / "gsm8k_complete_results.csv", [row for row in main_rows if row["dataset"] == "gsm8k"])
    write_csv(tables / "mmlu_complete_results.csv", [row for row in main_rows if row["dataset"] == "mmlu"])
    write_csv(tables / "controlled_target_baselines.csv", [row for row in main_rows if row["method"].split(":", 1)[0] in ("correctness", "consistency", "last_switch")])
    write_csv(tables / "risk_frontier.csv", frontier)
    write_csv(tables / "paired_bootstrap_differences.csv", pairwise)
    write_csv(tables / "trajectory_protection_ablation.csv", [row for row in main_rows if row["method"].startswith("correction_")])
    write_csv(tables / "mmlu_subject_results.csv", subject_rows)
    write_csv(tables / "mmlu_category_results.csv", category_rows)
    plots(root, main_rows, frontier)
    zh_verdict = verdict(all_metrics, root)
    audit = read_json(root / "CACHE_AUDIT.json")
    (root / "CACHE_AND_LEAKAGE_AUDIT.md").write_text(
        "# 缓存与数据泄漏审计\n\n"
        f"状态：**{audit.get('status', '未知')}**。随机种子：20260803。数据类型：FP16。"
        "公共缓存以数据集、数据划分、样本 ID 和检查点为键；所有方法均在该缓存上回放，不重新生成。"
        "审计会验证所有方法使用完全相同的留出集 ID，并以失败即中止的方式检查数据划分重叠和协议指纹。\n",
        encoding="utf-8",
    )
    report(root, main_rows, zh_verdict)
    atomic_json({
        "status": "complete",
        "seed": 20260803,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "latency_label": LATENCY_LABEL,
        "datasets": {dataset: len(all_methods[dataset]["Dense"]) for dataset in DATASETS},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, root / "REPORT_MANIFEST.json")


if __name__ == "__main__":
    main()

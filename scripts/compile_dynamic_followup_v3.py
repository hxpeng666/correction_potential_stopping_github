#!/usr/bin/env python3
"""汇总动态补充实验、统一2566基线、强OS和10000次配对bootstrap。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import atomic_json, load_yaml


BUDGETS = (0, 1, 2, 4, 10)


def load_pt(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def normalize_records(records: list[dict[str, Any]], *, mode: str = "adaptive") -> list[dict[str, Any]]:
    output = []
    for source in records:
        row = dict(source)
        if "method_success" not in row:  # Dense原始记录
            row = {
                "problem_id": str(source["problem_id"]), "subject": source.get("subject"),
                "category": source.get("category"), "fallback": True, "checkpoint": None,
                "method_success": bool(source["success"]), "dense_success": bool(source["success"]),
                "method_tokens": int(source["reasoning_tokens"]),
                "dense_tokens": int(source["reasoning_tokens"]), "transition": "fallback",
            }
        else:
            row["problem_id"] = str(row["problem_id"])
            if mode == "direct":
                row["method_tokens"] = 0
            elif not bool(row.get("fallback", False)) and row.get("checkpoint") is not None:
                row["method_tokens"] = min(int(row["checkpoint"]), int(row["dense_tokens"]))
            else:
                row["method_tokens"] = int(row["dense_tokens"])
        output.append(row)
    ids = [row["problem_id"] for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("policy records存在重复ID")
    return output


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    method_success = np.asarray([bool(row["method_success"]) for row in records], dtype=np.float64)
    dense_success = np.asarray([bool(row["dense_success"]) for row in records], dtype=np.float64)
    method_tokens = np.asarray([int(row["method_tokens"]) for row in records], dtype=np.float64)
    dense_tokens = np.asarray([int(row["dense_tokens"]) for row in records], dtype=np.float64)
    stopped = np.asarray([not bool(row.get("fallback", False)) for row in records], dtype=bool)
    wc = stopped & (~method_success.astype(bool)) & dense_success.astype(bool)
    cw = stopped & method_success.astype(bool) & (~dense_success.astype(bool))
    ww = stopped & (~method_success.astype(bool)) & (~dense_success.astype(bool))
    cc = stopped & method_success.astype(bool) & dense_success.astype(bool)
    return {
        "problems": n, "accuracy": float(method_success.mean()),
        "dense_accuracy": float(dense_success.mean()),
        "delta_dense_pp": float(100 * (method_success.mean() - dense_success.mean())),
        "coverage": float(stopped.mean()), "fallback": int((~stopped).sum()),
        "mean_reasoning_tokens": float(method_tokens.mean()),
        "mean_dense_reasoning_tokens": float(dense_tokens.mean()),
        "token_reduction": float(1.0 - method_tokens.sum() / dense_tokens.sum()),
        "W_to_C": int(wc.sum()), "C_to_W": int(cw.sum()),
        "W_to_W": int(ww.sum()), "C_to_C": int(cc.sum()),
        "lost_correct_rate": float(wc.mean()),
    }


def sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda row: str(row["problem_id"]))


def add_method(
    registry: dict[tuple[str, str, str], list[dict[str, Any]]],
    rows: list[dict[str, Any]], dataset: str, method: str, budget: str | int,
    records: list[dict[str, Any]], notes: str = "",
) -> None:
    normalized = sorted_records(normalize_records(records, mode="direct" if method == "Direct" else "adaptive"))
    metrics = summarize(normalized)
    rows.append({"dataset": dataset, "method": method, "B": str(budget), **metrics, "notes": notes})
    registry[(dataset, method, str(budget))] = normalized


def bootstrap_indices(records: list[dict[str, Any]], dataset: str, repeats: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(records)
    if dataset == "gsm8k":
        return rng.integers(0, n, size=(repeats, n), dtype=np.int32)
    categories = np.asarray([str(row.get("category")) for row in records], dtype=object)
    chunks = []
    for category in sorted(set(categories.tolist())):
        positions = np.flatnonzero(categories == category).astype(np.int32)
        chunks.append(positions[rng.integers(0, len(positions), size=(repeats, len(positions)))])
    return np.concatenate(chunks, axis=1)


def paired_bootstrap(
    reference: list[dict[str, Any]], comparison: list[dict[str, Any]],
    indices: np.ndarray,
) -> dict[str, float]:
    ref = {row["problem_id"]: row for row in reference}
    cmp = {row["problem_id"]: row for row in comparison}
    if set(ref) != set(cmp):
        raise ValueError("bootstrap方法problem ID不一致")
    ids = sorted(ref)
    ref_acc = np.asarray([bool(ref[i]["method_success"]) for i in ids], dtype=np.float64)
    cmp_acc = np.asarray([bool(cmp[i]["method_success"]) for i in ids], dtype=np.float64)
    dense = np.asarray([int(ref[i]["dense_tokens"]) for i in ids], dtype=np.float64)
    ref_tok = np.asarray([int(ref[i]["method_tokens"]) for i in ids], dtype=np.float64)
    cmp_tok = np.asarray([int(cmp[i]["method_tokens"]) for i in ids], dtype=np.float64)
    ref_lost = np.asarray([
        (not bool(ref[i]["method_success"])) and bool(ref[i]["dense_success"])
        and (not bool(ref[i].get("fallback", False))) for i in ids
    ], dtype=np.float64)
    cmp_lost = np.asarray([
        (not bool(cmp[i]["method_success"])) and bool(cmp[i]["dense_success"])
        and (not bool(cmp[i].get("fallback", False))) for i in ids
    ], dtype=np.float64)
    acc_diff = (cmp_acc[indices] - ref_acc[indices]).mean(axis=1) * 100
    dense_sum = dense[indices].sum(axis=1)
    token_diff = (ref_tok[indices].sum(axis=1) - cmp_tok[indices].sum(axis=1)) / dense_sum * 100
    lost_diff = (cmp_lost[indices] - ref_lost[indices]).mean(axis=1) * 100
    result = {}
    for name, values in (("accuracy_diff_pp", acc_diff), ("token_reduction_diff_pp", token_diff), ("lost_correct_diff_pp", lost_diff)):
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_ci_low"] = float(np.quantile(values, 0.025))
        result[f"{name}_ci_high"] = float(np.quantile(values, 0.975))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tables").mkdir(exist_ok=True)
    config = load_yaml(args.config)
    source_v2 = ROOT / "results/final_paper_dynamic_deployable_os_frontier_v2"
    rows: list[dict[str, Any]] = []
    registry: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for dataset in ("gsm8k", "mmlu_pro"):
        # Dense / Direct / Fixed。
        if dataset == "gsm8k":
            baseline = load_pt(ROOT / "results/final_paper_primary_v1/main_float16_seed20260803/gsm8k/baselines/baseline_records.pt")["records"]["heldout"]
        else:
            baseline = load_pt(ROOT / "results/final_paper_mmlu_pro_independent_token_v2_train1000_cal500_test1000/baselines_token/baseline_records.pt")["records"]
        add_method(registry, rows, dataset, "Dense", "-", baseline["dense"], "reasoning token baseline")
        add_method(registry, rows, dataset, "Direct", "-", baseline["direct"], "0 reasoning tokens; answer cost ignored")
        for budget in (64, 96, 128, 192, 256, 384, 512, 768):
            add_method(registry, rows, dataset, f"Fixed-{budget}", "-", baseline["fixed"][str(budget)], "fixed checkpoint")

        # 可部署Full与核心replay消融。
        full_records = load_pt(source_v2 / f"dynamic/full/{dataset}/policy_records.pt")["records"]["empirical_B"]
        replay_records = load_pt(source_v2 / f"dynamic/replay_ablations/{dataset}/policy_records.pt")["records"]
        for budget in BUDGETS:
            key = str(budget)
            add_method(registry, rows, dataset, "Dynamic-Full", budget, full_records[key], "deployable Q_continue")
            add_method(registry, rows, dataset, "Dynamic-M0", budget, replay_records["no_continuation_value_M0"]["empirical_B"][key], "value head removed")

        # validation-only Q偏置小升级。
        q_records = load_pt(args.run_root / f"q_bias/{dataset}/policy_records.pt")["records"]["empirical_B"]
        for budget in BUDGETS:
            add_method(registry, rows, dataset, "Dynamic-QBias", budget, q_records[str(budget)], "internal-validation Huber intercept")

        # 原共享bank OS与本轮更强渐进独立OS。
        for label, path, family in (
            ("OS-Matched-Shared", source_v2 / f"os_pruner/matched_os_pruner/{dataset}/policy_records.pt", "empirical_B"),
            ("OS-Constrained-Shared", source_v2 / f"os_pruner/constrained_os_pruner/{dataset}/policy_records.pt", "empirical_B"),
        ):
            source = load_pt(path)["records"][family]
            for budget in BUDGETS:
                add_method(registry, rows, dataset, label, budget, source[str(budget)], "shared candidate bank")
        progressive = load_pt(args.run_root / f"os_pruner/progressive/{dataset}/policy_records.pt")["records"]
        for family, label in (
            ("progressive_matched_os", "OS-Matched-Progressive"),
            ("progressive_constrained_os", "OS-Constrained-Progressive"),
        ):
            for budget in BUDGETS:
                add_method(registry, rows, dataset, label, budget, progressive[family][str(budget)], "independent mu trunk; descending lambda warm-start")
        for directory, label in (
            ("shared_matched_retry1", "OS-Matched-Shared-Deterministic"),
            ("shared_constrained_retry1", "OS-Constrained-Shared-Deterministic"),
            ("progressive_matched", "OS-Matched-Progressive-Deterministic"),
            ("progressive_constrained", "OS-Constrained-Progressive-Deterministic"),
        ):
            source = load_pt(args.run_root / f"os_deterministic/{directory}/{dataset}/policy_records.pt")["records"]["empirical_B"]
            for budget in BUDGETS:
                add_method(registry, rows, dataset, label, budget, source[str(budget)], "deterministic probability>=0.5 sensitivity")

        # 最终2566维、公平重训的受控标签基线。
        controlled_root = args.run_root / "controlled_labels_2566"
        controlled = {
            "Correctness-2566": "correctness",
            "Consistency-2566": "consistency",
            "LastSwitch-2566": "last_switch",
            "Correction-BCE-2566": "correction_bce",
            "Correction-Trajectory-2566": "correction_bce_traj",
        }
        for label, directory in controlled.items():
            source = load_pt(controlled_root / directory / dataset / "policy_records.pt")["records"]["empirical_B"]
            for budget in BUDGETS:
                add_method(registry, rows, dataset, label, budget, source[str(budget)], "same 2566 features and token-only calibration")

    table_path = args.output / "tables/all_comparisons.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    repeats = int(config["statistics"]["bootstrap_replicates"])
    seed = int(config["seed"]["bootstrap"])
    bootstrap_rows = []
    for dataset in ("gsm8k", "mmlu_pro"):
        for budget in BUDGETS:
            reference = registry[(dataset, "Dynamic-Full", str(budget))]
            indices = bootstrap_indices(reference, dataset, repeats, seed + budget)
            methods = [
                "Dynamic-QBias", "OS-Matched-Progressive", "OS-Constrained-Progressive",
                "OS-Matched-Progressive-Deterministic", "OS-Constrained-Progressive-Deterministic",
            ]
            if budget == 4:
                methods += [
                    "Dynamic-M0", "OS-Matched-Shared", "OS-Constrained-Shared",
                    "Correctness-2566", "Consistency-2566", "LastSwitch-2566",
                    "Correction-BCE-2566", "Correction-Trajectory-2566",
                ]
            for method in methods:
                comparison = registry[(dataset, method, str(budget))]
                bootstrap_rows.append({
                    "dataset": dataset, "B": budget,
                    "reference": "Dynamic-Full", "comparison": method,
                    **paired_bootstrap(reference, comparison, indices),
                })
    bootstrap_path = args.output / "tables/paired_bootstrap.csv"
    with bootstrap_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bootstrap_rows[0]))
        writer.writeheader(); writer.writerows(bootstrap_rows)

    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    selected_methods = [
        "Dynamic-Full", "Dynamic-QBias", "Dynamic-M0",
        "OS-Matched-Progressive", "OS-Constrained-Progressive",
        "OS-Matched-Progressive-Deterministic", "OS-Constrained-Progressive-Deterministic",
        "Correction-Trajectory-2566",
    ]
    for dataset in ("gsm8k", "mmlu_pro"):
        fig, axis = plt.subplots(figsize=(8.2, 5.6))
        for method in selected_methods:
            local = [row for row in rows if row["dataset"] == dataset and row["method"] == method and row["B"] in {str(v) for v in BUDGETS}]
            local.sort(key=lambda row: int(row["B"]))
            axis.plot(
                [row["W_to_C"] for row in local],
                [100 * row["token_reduction"] for row in local],
                marker="o", linewidth=1.4, label=method,
            )
        axis.set_xlabel("Held-out W→C count（工作点仅由calibration选择）")
        axis.set_ylabel("Reasoning token reduction (%)")
        axis.set_title(f"{dataset}: calibration-selected B frontier")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(figures / f"{dataset}_followup_selected_frontier.png", dpi=180)
        plt.close(fig)

    payload = {
        "status": "complete", "single_seed": True, "llm_generation": 0,
        "heldout_used_for_selection": False,
        "tables": [str(table_path), str(bootstrap_path)],
        "methods": sorted(set(row["method"] for row in rows)),
    }
    atomic_json(payload, args.output / "summary.json")

    # 简洁中文报告，数值表由CSV保留全精度。
    lines = [
        "# 动态停止补充实验报告", "",
        "本轮没有新增LLM轨迹、样本或seed；全部结果复用现有公共缓存。成本统一为reasoning token，短答案成本记为0。", "",
        "## 验证问题", "",
        "1. 用逐样本审计解释M=0为何在MMLU-Pro上更省token但准确率更低。",
        "2. 用独立mu主干与lambda渐进warm-start增强OS-Pruner受控基线。",
        "3. 用probe-train内部validation拟合每个Q候选的单一Huber截距，验证最小尺度校正是否足够。",
        "4. 将Correctness、Consistency、Last-switch与旧Correction在最终2566维特征上公平重训。", "",
        "完整点估计见 `tables/all_comparisons.csv`，10000次配对bootstrap见 `tables/paired_bootstrap.csv`。",
    ]
    (args.output / "DYNAMIC_FOLLOWUP_REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output / "pipeline.complete").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""汇总归一化trajectory soft-min实验，并与冻结旧结果做配对比较。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


VARIANTS = {
    "normalized_beta0.25_weight1": "beta025_weight1",
    "normalized_beta0.5_weight0.25": "beta05_weight025",
    "normalized_beta0.5_weight0.5": "beta05_weight05",
    "normalized_beta0.5_weight1": "beta05_weight1",
    "normalized_beta0.5_weight2": "beta05_weight2",
    "normalized_beta1_weight1": "beta1_weight1",
}
BASELINES = {
    "gsm8k": {
        "BCE_only": Path("results/final_paper_greedy_forced_ablation_v1/static_reasoning_only/correction_bce/gsm8k"),
        "unnormalized_beta0.5_weight1": Path("results/final_paper_greedy_forced_ablation_v1/static_reasoning_only/correction_bce_traj/gsm8k"),
    },
    "mmlu_pro": {
        "BCE_only": Path("results/final_paper_greedy_forced_mmlupro_ablation_v1/static_reasoning_only/correction_bce/mmlu_pro"),
        "unnormalized_beta0.5_weight1": Path("results/final_paper_greedy_forced_mmlupro_ablation_v1/static_reasoning_only/correction_bce_traj/mmlu_pro"),
    },
}
BUDGETS = (0, 1, 2, 4, 10)


def load_json(directory: Path) -> dict:
    return json.loads((directory / "probe.json").read_text())


def result_row(dataset: str, method: str, directory: Path, budget: int) -> dict:
    payload = load_json(directory)
    result = payload["frozen_policy_results"]["empirical_B"][str(budget)]
    calibration = result["calibration"]
    heldout = result["heldout"]
    counts = heldout["counts"]
    run = payload["run_spec"]
    return {
        "dataset": dataset,
        "method": method,
        "budget_B": budget,
        "aggregation": run.get("trajectory_aggregation", "bce_only" if run.get("loss") == "bce" else "unnormalized_softmin"),
        "beta": run.get("trajectory_softmin_beta"),
        "trajectory_weight": run.get("trajectory_weight", 0.0 if run.get("loss") == "bce" else 1.0),
        "best_epoch": payload["best_epoch"],
        "calibration_lost_correct": calibration["lost_correct_count"],
        "calibration_accuracy": calibration["accuracy"],
        "threshold": calibration["threshold"],
        "dense_accuracy": heldout["dense_accuracy"],
        "accuracy": heldout["accuracy"],
        "delta_dense_pp": 100.0 * (heldout["accuracy"] - heldout["dense_accuracy"]),
        "token_reduction": heldout["token_reduction"],
        "coverage": heldout["coverage"],
        "lost_correct_count": heldout["lost_correct_count"],
        "lost_correct_rate": heldout["lost_correct_rate"],
        "W_to_C": counts["W_to_C"],
        "C_to_W": counts["C_to_W"],
        "W_to_W": counts["W_to_W"],
        "C_to_C": counts["C_to_C"],
        "fallback": heldout["fallback"],
    }


def records(directory: Path, budget: int = 4) -> list[dict]:
    payload = torch.load(directory / "policy_records.pt", map_location="cpu", weights_only=False)
    return payload["records"]["empirical_B"][str(budget)]


def sample_indices(rows: list[dict], rng: np.random.Generator, stratified: bool) -> np.ndarray:
    if not stratified:
        return rng.integers(0, len(rows), size=len(rows))
    categories = np.asarray([str(row.get("category")) for row in rows])
    chunks = []
    for category in sorted(set(categories.tolist())):
        local = np.flatnonzero(categories == category)
        chunks.append(rng.choice(local, size=len(local), replace=True))
    return np.concatenate(chunks)


def metrics(rows: list[dict], indices: np.ndarray) -> tuple[float, float, float]:
    selected = [rows[int(index)] for index in indices]
    accuracy = np.mean([bool(row["method_success"]) for row in selected])
    method_tokens = np.sum([float(row["method_tokens"]) for row in selected])
    dense_tokens = np.sum([float(row["dense_tokens"]) for row in selected])
    token_reduction = 1.0 - method_tokens / dense_tokens
    lost = np.mean([
        bool(row["dense_success"]) and not bool(row["method_success"])
        for row in selected
    ])
    return float(accuracy), float(token_reduction), float(lost)


def paired_bootstrap(dataset: str, left_dir: Path, right_dir: Path, *, repeats: int = 10000) -> list[dict]:
    left = records(left_dir)
    right = records(right_dir)
    left_map = {str(row["problem_id"]): row for row in left}
    right_map = {str(row["problem_id"]): row for row in right}
    if set(left_map) != set(right_map):
        raise ValueError(f"paired sample IDs differ for {dataset}")
    ids = sorted(left_map)
    left = [left_map[value] for value in ids]
    right = [right_map[value] for value in ids]
    rng = np.random.default_rng(20260803)
    values = np.empty((repeats, 3), dtype=np.float64)
    for repeat in range(repeats):
        index = sample_indices(left, rng, stratified=dataset == "mmlu_pro")
        values[repeat] = np.asarray(metrics(left, index)) - np.asarray(metrics(right, index))
    names = ("accuracy_diff_pp", "token_reduction_diff_pp", "lost_correct_rate_diff_pp")
    output = []
    for column, name in enumerate(names):
        scaled = 100.0 * values[:, column]
        output.append({
            "dataset": dataset,
            "comparison": "normalized_beta0.5_weight1_minus_unnormalized_beta0.5_weight1",
            "metric": name,
            "mean": float(scaled.mean()),
            "ci95_low": float(np.quantile(scaled, 0.025)),
            "ci95_high": float(np.quantile(scaled, 0.975)),
            "bootstrap_repeats": repeats,
            "stratified": dataset == "mmlu_pro",
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    directories: dict[tuple[str, str], Path] = {}
    for dataset in ("gsm8k", "mmlu_pro"):
        for method, relative in BASELINES[dataset].items():
            directory = root / relative
            directories[(dataset, method)] = directory
            for budget in BUDGETS:
                rows.append(result_row(dataset, method, directory, budget))
        for method, slug in VARIANTS.items():
            directory = run_root / dataset / slug
            directories[(dataset, method)] = directory
            for budget in BUDGETS:
                rows.append(result_row(dataset, method, directory, budget))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "all_empirical_B.csv", index=False)
    frame[frame.budget_B == 4].to_csv(output / "B4_comparison.csv", index=False)
    main = frame[frame.method.isin((
        "BCE_only", "unnormalized_beta0.5_weight1", "normalized_beta0.5_weight1"
    ))]
    main.to_csv(output / "main_comparison_all_B.csv", index=False)

    boot = []
    for dataset in ("gsm8k", "mmlu_pro"):
        boot.extend(paired_bootstrap(
            dataset,
            directories[(dataset, "normalized_beta0.5_weight1")],
            directories[(dataset, "unnormalized_beta0.5_weight1")],
        ))
    pd.DataFrame(boot).to_csv(output / "paired_bootstrap_B4.csv", index=False)

    lines = [
        "# 归一化 trajectory soft-min 实验报告",
        "",
        "本实验复用既有greedy forced-answer公共缓存，仅重新训练轻量Correction probe。主修订固定为normalized log-mean-exp、beta=0.5、trajectory weight=1；其他beta和weight只作为预声明敏感性分析。held-out不参与epoch、阈值或超参数选择。",
        "",
        "## B=4主比较",
        "",
    ]
    for dataset in ("gsm8k", "mmlu_pro"):
        lines += [f"### {dataset}", "", "| 方法 | Accuracy | ΔDense | Token↓ | Coverage | W→C | C→W |", "|---|---:|---:|---:|---:|---:|---:|"]
        view = frame[(frame.dataset == dataset) & (frame.budget_B == 4)]
        order = ["BCE_only", "unnormalized_beta0.5_weight1", "normalized_beta0.5_weight1"]
        for method in order:
            row = view[view.method == method].iloc[0]
            lines.append(
                f"| {method} | {100*row.accuracy:.2f}% | {row.delta_dense_pp:+.2f} pp | "
                f"{100*row.token_reduction:.2f}% | {100*row.coverage:.2f}% | {int(row.W_to_C)} | {int(row.C_to_W)} |"
            )
        lines.append("")
    lines += [
        "## 解释边界",
        "",
        "- 当前结果为单seed、token-only replay；不表示真实在线wall-time。",
        "- 旧未归一化结果保持不变；修订版本写入独立目录。",
        "- beta/weight敏感性结果为描述性证据，不能根据held-out重新挑选主版本。",
        "- 配对bootstrap为10,000次；MMLU-Pro按category分层。",
        "",
    ]
    (output / "NORMALIZED_SOFTMIN_REPORT_ZH.md").write_text("\n".join(lines))
    (output / "compile.complete").write_text(json.dumps({
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(frame),
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "complete", "rows": len(frame), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

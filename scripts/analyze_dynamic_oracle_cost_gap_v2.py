#!/usr/bin/env python3
"""配对比较旧oracle-future-cost与修复后的deployable Q_continue。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.dynamic_optimal_stopping_deployable_v2 import summarize_token_records
from src.utils import load_yaml


BUDGETS = (0, 1, 2, 4, 10)


def align(records, ids):
    mapping = {str(row["problem_id"]): row for row in records}
    if len(mapping) != len(records) or set(mapping) != set(ids):
        raise ValueError("oracle/deployable sample ID不配对")
    return [mapping[value] for value in ids]


def arrays(records):
    return {
        "accuracy": np.asarray([row["method_success"] for row in records], dtype=np.float64),
        "lost_correct_rate": np.asarray([
            row.get("transition") == "W_to_C" for row in records
        ], dtype=np.float64),
        "coverage": np.asarray([not row.get("fallback", False) for row in records], dtype=np.float64),
        "tokens": np.asarray([
            row["dense_tokens"] if row.get("fallback", False) else row["checkpoint"]
            for row in records
        ], dtype=np.float64),
        "dense_tokens": np.asarray([row["dense_tokens"] for row in records], dtype=np.float64),
    }


def point(values):
    return {
        "accuracy": float(values["accuracy"].mean()),
        "lost_correct_rate": float(values["lost_correct_rate"].mean()),
        "coverage": float(values["coverage"].mean()),
        "token_reduction": float(1.0 - values["tokens"].mean() / values["dense_tokens"].mean()),
    }


def sampled(values, indices):
    return {
        "accuracy": values["accuracy"][indices].mean(axis=1),
        "lost_correct_rate": values["lost_correct_rate"][indices].mean(axis=1),
        "coverage": values["coverage"][indices].mean(axis=1),
        "token_reduction": 1.0 - (
            values["tokens"][indices].mean(axis=1)
            / values["dense_tokens"][indices].mean(axis=1)
        ),
    }


def indices(records, dataset, replicates, seed):
    rng = np.random.default_rng(seed)
    if dataset == "gsm8k":
        return rng.integers(0, len(records), size=(replicates, len(records)))
    categories = np.asarray([str(row.get("category")) for row in records])
    chunks = []
    for category in sorted(set(categories)):
        local = np.flatnonzero(categories == category)
        chunks.append(rng.choice(local, size=(replicates, len(local)), replace=True))
    return np.concatenate(chunks, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deployable-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_yaml(args.config)
    rows = []
    for dataset in ("gsm8k", "mmlu_pro"):
        deployable = torch.load(
            args.deployable_root / "dynamic/full" / dataset / "policy_records.pt",
            map_location="cpu", weights_only=False,
        )["records"]["empirical_B"]
        oracle = torch.load(
            args.oracle_root / dataset / "probe/policy_records.pt",
            map_location="cpu", weights_only=False,
        )["records"]["empirical_B"]
        ids = sorted(str(row["problem_id"]) for row in deployable["0"])
        bootstrap_indices = indices(
            align(deployable["0"], ids), dataset,
            int(config["statistics"]["bootstrap_replicates"]),
            int(config["seed"]["bootstrap"]),
        )
        for budget in BUDGETS:
            deploy_records = align(deployable[str(budget)], ids)
            oracle_records = align(oracle[str(budget)], ids)
            deploy_summary = summarize_token_records(deploy_records)
            oracle_summary = summarize_token_records(oracle_records)
            deploy_raw = arrays(deploy_records)
            oracle_raw = arrays(oracle_records)
            deploy_point = point(deploy_raw)
            oracle_point = point(oracle_raw)
            deploy_boot = sampled(deploy_raw, bootstrap_indices)
            oracle_boot = sampled(oracle_raw, bootstrap_indices)
            for metric in ("accuracy", "token_reduction", "lost_correct_rate", "coverage"):
                difference = deploy_boot[metric] - oracle_boot[metric]
                rows.append({
                    "dataset": dataset, "budget_B": budget, "metric": metric,
                    "deployable_point": deploy_point[metric],
                    "oracle_future_cost_point": oracle_point[metric],
                    "deployable_minus_oracle": deploy_point[metric] - oracle_point[metric],
                    "ci_low": float(np.percentile(difference, 2.5)),
                    "ci_high": float(np.percentile(difference, 97.5)),
                    "deployable_W_to_C": deploy_summary["lost_correct_count"],
                    "oracle_W_to_C": oracle_summary["lost_correct_count"],
                    "replicates": int(config["statistics"]["bootstrap_replicates"]),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "deployable_vs_oracle_future_cost.csv", index=False)
    report = [
        "# Deployable Q-value 与 oracle future-cost 配对审计", "",
        "正差表示deployable大于oracle。旧版本在动作阶段读取真实下一sentence长度；新版本只由当前z_t预测包含成本的Q_continue。", "",
    ]
    for dataset in ("gsm8k", "mmlu_pro"):
        report += [f"## {dataset}", ""]
        for budget in BUDGETS:
            local = frame[(frame.dataset == dataset) & (frame.budget_B == budget)]
            values = {row.metric: row for _, row in local.iterrows()}
            report.append(
                f"- B={budget}: token reduction差={100*values['token_reduction'].deployable_minus_oracle:+.2f}pp "
                f"[95% CI {100*values['token_reduction'].ci_low:+.2f}, {100*values['token_reduction'].ci_high:+.2f}]；"
                f"accuracy差={100*values['accuracy'].deployable_minus_oracle:+.2f}pp；"
                f"W→C={int(values['accuracy'].deployable_W_to_C)}/{int(values['accuracy'].oracle_W_to_C)}（deployable/oracle）。"
            )
        report.append("")
    (args.output / "DEPLOYABLE_VS_ORACLE_AUDIT_ZH.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

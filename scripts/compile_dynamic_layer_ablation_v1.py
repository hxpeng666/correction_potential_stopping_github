#!/usr/bin/env python3
"""Compile the paired layer 8/20/35 dynamic-stopper ablation."""
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


LAYERS = (8, 20, 35)


def normalized(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in records:
        row = dict(source)
        row["method_tokens"] = int(
            row["dense_tokens"] if row["fallback"] else row["checkpoint"]
        )
        output.append(row)
    return output


def align(records: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    mapping = {str(row["problem_id"]): row for row in records}
    if len(mapping) != len(records) or set(mapping) != set(ids):
        raise ValueError("layer ablation sample IDs are not paired")
    return [mapping[value] for value in ids]


def raw_arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "accuracy": np.asarray([row["method_success"] for row in records], dtype=np.float64),
        "lost_correct_rate": np.asarray(
            [row["transition"] == "W_to_C" for row in records], dtype=np.float64
        ),
        "coverage": np.asarray([not row["fallback"] for row in records], dtype=np.float64),
        "tokens": np.asarray([row["method_tokens"] for row in records], dtype=np.float64),
        "dense_tokens": np.asarray([row["dense_tokens"] for row in records], dtype=np.float64),
    }


def point(values: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "accuracy": float(values["accuracy"].mean()),
        "lost_correct_rate": float(values["lost_correct_rate"].mean()),
        "coverage": float(values["coverage"].mean()),
        "token_reduction": float(
            1.0 - values["tokens"].mean() / values["dense_tokens"].mean()
        ),
    }


def sampled(values: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "accuracy": values["accuracy"][indices].mean(axis=1),
        "lost_correct_rate": values["lost_correct_rate"][indices].mean(axis=1),
        "coverage": values["coverage"][indices].mean(axis=1),
        "token_reduction": 1.0
        - values["tokens"][indices].mean(axis=1)
        / values["dense_tokens"][indices].mean(axis=1),
    }


def bootstrap_indices(
    records: list[dict[str, Any]], dataset: str, replicates: int, seed: int
) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    if dataset == "gsm8k":
        return rng.integers(0, len(records), size=(replicates, len(records))), "problem"
    categories = np.asarray([str(row.get("category")) for row in records])
    pieces = []
    for category in sorted(set(categories)):
        positions = np.flatnonzero(categories == category)
        pieces.append(
            rng.choice(positions, size=(replicates, len(positions)), replace=True)
        )
    return np.concatenate(pieces, axis=1), "category_stratified_problem"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-dir-name", default="probes")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    marker = args.output / "pipeline.complete"
    if args.resume and marker.is_file():
        print(json.dumps({"status": "skipped_complete", "output": str(args.output)}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    tables = args.output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    config = load_yaml(args.config)

    table_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []

    for dataset in ("gsm8k", "mmlu_pro"):
        methods: dict[tuple[int, str], list[dict[str, Any]]] = {}
        ids: list[str] | None = None
        for layer in LAYERS:
            probe_root = (
                args.run_root
                / args.probe_dir_name
                / f"layer_{layer}"
                / dataset
            )
            probe = json.loads((probe_root / "probe.json").read_text(encoding="utf-8"))
            if int(probe["run_spec"]["layer_zero_based"]) != layer:
                raise ValueError(f"run-spec layer mismatch: {probe_root}")
            if probe["run_spec"]["feature_kind"] != "full_no_delta":
                raise ValueError(f"feature mismatch: {probe_root}")
            records_payload = torch.load(
                probe_root / "policy_records.pt", map_location="cpu", weights_only=False
            )["records"]["empirical_B"]
            if ids is None:
                any_records = next(iter(records_payload.values()))
                ids = sorted(str(row["problem_id"]) for row in any_records)
            for budget, records in records_payload.items():
                aligned = align(normalized(records), ids)
                methods[(layer, str(budget))] = aligned
                summary = summarize_token_records(aligned)
                selection = probe["frozen_policy_results"]["empirical_B"][str(budget)]
                table_rows.append(
                    {
                        "dataset": dataset,
                        "layer_zero_based": layer,
                        "transformer_block_one_based": layer + 1,
                        "B": int(budget),
                        "N": summary["problems"],
                        "dense_accuracy": summary["dense_accuracy"],
                        "accuracy": summary["accuracy"],
                        "delta_dense_pp": summary["delta_dense_pp"],
                        "coverage": summary["coverage"],
                        "token_reduction": summary["token_reduction"],
                        "mean_reasoning_tokens": summary["mean_reasoning_tokens"],
                        "lost_correct_count": summary["lost_correct_count"],
                        "lost_correct_rate": summary["lost_correct_rate"],
                        "gained_correct_count": summary["gained_correct_count"],
                        "fallback": summary["fallback"],
                        "W_to_C": summary["counts"]["W_to_C"],
                        "C_to_W": summary["counts"]["C_to_W"],
                        "W_to_W": summary["counts"]["W_to_W"],
                        "C_to_C": summary["counts"]["C_to_C"],
                        "selected_candidate": selection["calibration"].get(
                            "selected_candidate"
                        ),
                        "lambda": selection["calibration"].get("lambda"),
                        "mu": selection["calibration"].get("mu"),
                        "dense_fallback": selection["calibration"].get(
                            "dense_fallback", False
                        ),
                        "calibration_lost_correct_count": selection["calibration"].get(
                            "lost_correct_count"
                        ),
                        "calibration_accuracy": selection["calibration"].get("accuracy"),
                        "calibration_token_reduction": selection["calibration"].get(
                            "token_reduction"
                        ),
                    }
                )
            diagnostic_rows.append(
                {
                    "dataset": dataset,
                    "layer_zero_based": layer,
                    "local_best_epoch": probe["local_best_epoch"],
                    "value_best_epoch": probe["value_best_epoch"],
                    "value_validation_mae": probe["value_history"][
                        probe["value_best_epoch"]
                    ]["validation_mae"],
                    **{
                        f"validation_{key}": value
                        for key, value in probe["local_diagnostics"]["probe_train"].items()
                    },
                }
            )
        assert ids is not None

        reference_records = methods[(20, "4")]
        replicates = int(config["statistics"]["bootstrap_replicates"])
        indices, stratification = bootstrap_indices(
            reference_records,
            dataset,
            replicates,
            int(config["seed"]["bootstrap"]) + 31,
        )
        reference_point = point(raw_arrays(reference_records))
        reference_samples = sampled(raw_arrays(reference_records), indices)
        for layer in LAYERS:
            records = methods[(layer, "4")]
            values = raw_arrays(records)
            layer_point = point(values)
            layer_samples = sampled(values, indices)
            for metric, samples in layer_samples.items():
                bootstrap_rows.append(
                    {
                        "dataset": dataset,
                        "layer_zero_based": layer,
                        "B": 4,
                        "metric": metric,
                        "point": layer_point[metric],
                        "ci_low": float(np.percentile(samples, 2.5)),
                        "ci_high": float(np.percentile(samples, 97.5)),
                        "replicates": replicates,
                        "stratification": stratification,
                    }
                )
                difference = samples - reference_samples[metric]
                paired_rows.append(
                    {
                        "dataset": dataset,
                        "layer_zero_based": layer,
                        "reference_layer": 20,
                        "B": 4,
                        "metric": metric,
                        "difference_point": layer_point[metric]
                        - reference_point[metric],
                        "ci_low": float(np.percentile(difference, 2.5)),
                        "ci_high": float(np.percentile(difference, 97.5)),
                        "replicates": replicates,
                        "stratification": stratification,
                    }
                )

        cache_root = args.run_root / "cache" / dataset
        for summary_path in sorted(cache_root.glob("*/summary_shard*.json")):
            value = json.loads(summary_path.read_text(encoding="utf-8"))
            parity_rows.append(
                {
                    "dataset": dataset,
                    "split": value["split"],
                    "shard_index": value["shard_index"],
                    "completed_now": value["completed_now"],
                    "skipped_complete": value["skipped_complete"],
                    "min_layer20_cosine": value["min_layer20_cosine"],
                    "max_layer20_relative_l2": value["max_layer20_relative_l2"],
                    "max_layer20_absolute_difference": value[
                        "max_layer20_absolute_difference"
                    ],
                }
            )

    frame = pd.DataFrame(table_rows).sort_values(
        ["dataset", "B", "layer_zero_based"]
    )
    frame.to_csv(tables / "layer_all_empirical_B.csv", index=False)
    frame[frame.B.eq(4)].to_csv(tables / "layer_B4.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        tables / "layer_training_diagnostics.csv", index=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        tables / "layer_B4_bootstrap_ci.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(
        tables / "layer_B4_paired_vs_layer20.csv", index=False
    )
    pd.DataFrame(parity_rows).to_csv(
        tables / "layer20_capture_parity.csv", index=False
    )

    report = [
        "# 动态停止器 Layer 8/20/35 消融",
        "",
        "三组实验共享同一 FP16 Qwen3-4B Dense trajectory、forced-answer、sentence checkpoint、2566维特征定义、训练划分、初始化 seed、候选策略网格及 calibration 规则；唯一变化是读取的 zero-based decoder block。Layer 8/20/35 分别对应第9/21/36个 Transformer block，均不是 LM head。",
        "",
        "层选择不使用 held-out；这里报告各层在相同 calibration 规则冻结后的描述性 held-out 结果。",
        "",
    ]
    for dataset in ("gsm8k", "mmlu_pro"):
        report.extend([f"## {dataset}：经验 B=4", ""])
        subset = frame[(frame.dataset == dataset) & frame.B.eq(4)]
        report.extend(
            [
                "| Zero-based layer | 实际 block | Accuracy | ΔDense | Token reduction | Coverage | W→C | C→W |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in subset.itertuples():
            report.append(
                f"| {row.layer_zero_based} | {row.transformer_block_one_based} | "
                f"{100*row.accuracy:.2f}% | {row.delta_dense_pp:+.2f} pp | "
                f"{100*row.token_reduction:.2f}% | {100*row.coverage:.2f}% | "
                f"{row.W_to_C} | {row.C_to_W} |"
            )
        report.append("")
    report.extend(
        [
            "## 解释边界",
            "",
            "完整 B={0,1,2,4,10} 前沿、训练诊断、A100 layer-20 重算一致性以及 B=4 的10,000次配对bootstrap均保存在 tables/。本消融是单 seed；不得据此把 held-out 最好的层追认为新的主配置。",
            "",
        ]
    )
    (args.output / "LAYER_ABLATION_REPORT_ZH.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    completion = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "layers_zero_based": list(LAYERS),
        "datasets": ["gsm8k", "mmlu_pro"],
        "bootstrap_replicates": int(config["statistics"]["bootstrap_replicates"]),
        "heldout_used_for_layer_or_policy_selection": False,
        "probe_dir_name": args.probe_dir_name,
        "cost": "reasoning_tokens_only",
        "short_answer_cost": 0,
    }
    atomic_json(completion, marker)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

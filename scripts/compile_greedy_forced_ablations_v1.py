#!/usr/bin/env python3
"""Compile greedy forced-answer main ablations and paired bootstrap tables."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.utils import atomic_json, load_yaml


def metric_row(dataset: str, family: str, method: str, key: str, item: dict[str, Any]) -> dict[str, Any]:
    calibration = item["calibration"]
    heldout = item["heldout"]
    counts = heldout.get("counts", {})
    return {
        "dataset": dataset,
        "family": family,
        "method": method,
        "key": str(key),
        "accuracy": heldout["accuracy"],
        "dense_accuracy": heldout["dense_accuracy"],
        "delta_dense_pp": heldout.get("delta_dense_pp", -float(heldout["accuracy_drop_pp"])),
        "coverage": heldout["coverage"],
        "token_reduction": heldout["token_reduction"],
        "mean_reasoning_tokens": heldout.get(
            "mean_reasoning_tokens", heldout.get("mean_reasoning_and_answer_tokens")
        ),
        "lost_correct_count": heldout["lost_correct_count"],
        "lost_correct_rate": heldout["lost_correct_rate"],
        "gained_correct_count": heldout.get("gained_correct_count", counts.get("C_to_W", 0)),
        "fallback": heldout["fallback"],
        "W_to_C": counts.get("W_to_C", 0),
        "C_to_W": counts.get("C_to_W", 0),
        "W_to_W": counts.get("W_to_W", 0),
        "C_to_C": counts.get("C_to_C", 0),
        "calibration_accuracy": calibration.get("accuracy"),
        "calibration_coverage": calibration.get("coverage"),
        "calibration_token_reduction": calibration.get("token_reduction"),
        "calibration_lost_correct_count": calibration.get("lost_correct_count"),
        "selected_candidate": calibration.get("selected_candidate"),
        "lambda": calibration.get("lambda"),
        "mu": calibration.get("mu"),
        "dense_fallback": bool(calibration.get("dense_fallback", False)),
    }


def load_probe_rows(dataset: str, method: str, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads((root / "probe.json").read_text(encoding="utf-8"))
    rows = []
    for family, values in payload["frozen_policy_results"].items():
        for key, item in values.items():
            rows.append(metric_row(dataset, family, method, key, item))
    return rows, payload


def sentence_rows(root: Path, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / split).glob("sample_*.pt")):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        for row in artifact["rows"]:
            if "sentence" in row.get("checkpoint_schedules", []):
                rows.append(
                    {
                        "problem_id": str(row["problem_id"]),
                        "checkpoint": int(row["checkpoint"]),
                        "current_prediction": row.get("current_prediction"),
                        "current_success": bool(row["current_success"]),
                        "dense_success": bool(row["dense_success"]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["problem_id", "checkpoint"]).reset_index(drop=True)


def transition_name(current: bool, dense: bool) -> str:
    return ("C" if current else "W") + "_to_" + ("C" if dense else "W")


def label_drift(config: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for dataset, spec in config["datasets"].items():
        greedy_root = ROOT / spec["replay_root"]
        old_root = ROOT / spec["old_sampled_root"]
        for split in ("probe_train", "calibration", "heldout"):
            greedy = sentence_rows(greedy_root, split)
            old = sentence_rows(old_root, split)
            keys = ["problem_id", "checkpoint"]
            merged = greedy.merge(old, on=keys, suffixes=("_greedy", "_sampled"), validate="one_to_one")
            if len(merged) != len(greedy) or len(merged) != len(old):
                raise ValueError(f"row mismatch in label drift {dataset}/{split}")
            greedy_transition = [
                transition_name(c, d)
                for c, d in zip(merged.current_success_greedy, merged.dense_success_greedy)
            ]
            sampled_transition = [
                transition_name(c, d)
                for c, d in zip(merged.current_success_sampled, merged.dense_success_sampled)
            ]
            greedy_counts = Counter(greedy_transition)
            sampled_counts = Counter(sampled_transition)
            output.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "checkpoints": len(merged),
                    "answer_exact_agreement": float(
                        np.mean(
                            merged.current_prediction_greedy.fillna("<MISSING>").astype(str)
                            == merged.current_prediction_sampled.fillna("<MISSING>").astype(str)
                        )
                    ),
                    "correctness_agreement": float(
                        np.mean(merged.current_success_greedy == merged.current_success_sampled)
                    ),
                    "correctness_changed": int(
                        np.sum(merged.current_success_greedy != merged.current_success_sampled)
                    ),
                    **{f"greedy_{name}": greedy_counts[name] for name in ("W_to_C", "C_to_W", "W_to_W", "C_to_C")},
                    **{f"sampled_{name}": sampled_counts[name] for name in ("W_to_C", "C_to_W", "W_to_W", "C_to_C")},
                }
            )
    return output


def align_records(records: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    mapping = {str(row["problem_id"]): row for row in records}
    if len(mapping) != len(records) or set(mapping) != set(ids):
        raise ValueError("policy records are not paired by sample ID")
    return [mapping[value] for value in ids]


def raw_arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "accuracy": np.asarray([row["method_success"] for row in records], dtype=np.float64),
        "lost_correct_rate": np.asarray([row["transition"] == "W_to_C" for row in records], dtype=np.float64),
        "tokens": np.asarray([row["method_tokens"] for row in records], dtype=np.float64),
        "dense_tokens": np.asarray([row["dense_tokens"] for row in records], dtype=np.float64),
    }


def bootstrap_indices(records: list[dict[str, Any]], dataset: str, replicates: int, seed: int) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    if dataset == "gsm8k":
        return rng.integers(0, len(records), size=(replicates, len(records))), "problem"
    categories = np.asarray([str(row.get("category")) for row in records])
    pieces = []
    for category in sorted(set(categories)):
        local = np.flatnonzero(categories == category)
        pieces.append(rng.choice(local, size=(replicates, len(local)), replace=True))
    return np.concatenate(pieces, axis=1), "category_stratified_problem"


def sampled_metrics(raw: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "accuracy": raw["accuracy"][indices].mean(axis=1),
        "lost_correct_rate": raw["lost_correct_rate"][indices].mean(axis=1),
        "token_reduction": 1.0 - raw["tokens"][indices].mean(axis=1) / raw["dense_tokens"][indices].mean(axis=1),
    }


def record_sources(run_root: Path, dataset: str) -> dict[str, Path]:
    values = {
        "Dynamic-Full": run_root / "dynamic" / "full" / dataset / "policy_records.pt",
        "Dynamic-NoTrajectory": run_root / "dynamic" / "no_trajectory" / dataset / "policy_records.pt",
        "Dynamic-OneStep": run_root / "dynamic" / "one_step_value" / dataset / "policy_records.pt",
        "Dynamic-DenseEndpoint": run_root / "dynamic" / "dense_endpoint_value" / dataset / "policy_records.pt",
    }
    for name in ("correction_bce", "correction_bce_traj", "correctness_bce", "consistency_bce", "last_switch_bce"):
        values[f"Static-{name}"] = run_root / "static_reasoning_only" / name / dataset / "policy_records.pt"
    for feature in ("h_only", "full", "main_no_entropy", "main_no_position", "main_no_geometry"):
        values[f"Feature-{feature}"] = run_root / "features" / feature / dataset / "policy_records.pt"
    return values


def paired_bootstrap(run_root: Path, config: dict[str, Any], budget: str = "4") -> list[dict[str, Any]]:
    output = []
    replicates = int(config["statistics"]["bootstrap_replicates"])
    for dataset in config["datasets"]:
        sources = record_sources(run_root, dataset)
        loaded: dict[str, list[dict[str, Any]]] = {}
        for method, path in sources.items():
            payload = torch.load(path, map_location="cpu", weights_only=False)["records"]
            loaded[method] = payload["empirical_B"][budget]
        ids = sorted(str(row["problem_id"]) for row in loaded["Dynamic-Full"])
        loaded = {method: align_records(records, ids) for method, records in loaded.items()}
        indices, stratification = bootstrap_indices(
            loaded["Dynamic-Full"], dataset, replicates, int(config["seed"]["bootstrap"]) + 31
        )
        reference = sampled_metrics(raw_arrays(loaded["Dynamic-Full"]), indices)
        for method, records in loaded.items():
            current = sampled_metrics(raw_arrays(records), indices)
            for metric in ("accuracy", "token_reduction", "lost_correct_rate"):
                difference = current[metric] - reference[metric]
                output.append(
                    {
                        "dataset": dataset,
                        "B": int(budget),
                        "method": method,
                        "reference": "Dynamic-Full",
                        "metric": metric,
                        "difference_mean": float(difference.mean()),
                        "ci_low": float(np.percentile(difference, 2.5)),
                        "ci_high": float(np.percentile(difference, 97.5)),
                        "replicates": replicates,
                        "stratification": stratification,
                    }
                )
    return output


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
    tables.mkdir(parents=True, exist_ok=True)
    config = load_yaml(args.config)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        for variant in config["ablations"]["dynamic_variants"]:
            local, payload = load_probe_rows(
                dataset, f"Dynamic-{variant}", args.run_root / "dynamic" / variant / dataset
            )
            rows.extend(local)
            diagnostics.append(
                {
                    "dataset": dataset,
                    "method": f"Dynamic-{variant}",
                    "feature_width": payload["run_spec"]["feature_width"],
                    "best_local_epoch": payload["local_best_epoch"],
                    "best_value_epoch": payload["value_best_epoch"],
                    "value_validation_mae": payload["value_history"][payload["value_best_epoch"]]["validation_mae"],
                }
            )
        for spec in config["ablations"]["static_targets"]:
            name = f"{spec['method']}_{spec['loss']}"
            local, payload = load_probe_rows(
                dataset, f"Static-{name}", args.run_root / "static_reasoning_only" / name / dataset
            )
            rows.extend(local)
            diagnostics.append(
                {
                    "dataset": dataset,
                    "method": f"Static-{name}",
                    "feature_width": payload["run_spec"]["architecture"][0],
                    "best_local_epoch": payload["best_epoch"],
                    "best_value_epoch": None,
                    "value_validation_mae": None,
                }
            )
        for feature in config["ablations"]["feature_variants"]:
            if feature == "full_no_delta":
                continue
            local, payload = load_probe_rows(
                dataset, f"Feature-{feature}", args.run_root / "features" / feature / dataset
            )
            rows.extend(local)
            diagnostics.append(
                {
                    "dataset": dataset,
                    "method": f"Feature-{feature}",
                    "feature_width": payload["run_spec"]["feature_width"],
                    "best_local_epoch": payload["local_best_epoch"],
                    "best_value_epoch": payload["value_best_epoch"],
                    "value_validation_mae": payload["value_history"][payload["value_best_epoch"]]["validation_mae"],
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(tables / "all_workpoints.csv", index=False)
    frame[(frame.family == "empirical_B")].to_csv(tables / "empirical_B_all.csv", index=False)
    frame[(frame.family == "empirical_B") & (frame.key.astype(str) == "4")].to_csv(
        tables / "main_ablation_B4.csv", index=False
    )
    pd.DataFrame(diagnostics).to_csv(tables / "training_diagnostics.csv", index=False)

    replay_rows = []
    for dataset in config["datasets"]:
        payload = json.loads(
            (args.run_root / "dynamic" / "replay_ablations" / dataset / "replay_ablations.json").read_text(encoding="utf-8")
        )
        for name, result in payload["results"].items():
            for key, item in result.get("selected", {}).get("empirical_B", {}).items():
                replay_rows.append(metric_row(dataset, "empirical_B", f"Replay-{name}", key, item))
    pd.DataFrame(replay_rows).to_csv(tables / "dynamic_replay_ablations.csv", index=False)

    drift = label_drift(config)
    pd.DataFrame(drift).to_csv(tables / "greedy_vs_sampled_label_drift.csv", index=False)
    bootstrap = paired_bootstrap(args.run_root, config)
    pd.DataFrame(bootstrap).to_csv(tables / "paired_bootstrap_B4.csv", index=False)

    report = [
        "# Greedy forced-answer 主要消融结果", "",
        "本实验只改变 checkpoint forced-answer 解码：使用 greedy argmax。Dense trajectory、hidden、sentence checkpoints、Direct、split、parser 与 seed 均保持冻结。效率只报告 reasoning-token reduction，不使用本次并发采集时间。", "",
    ]
    for dataset in config["datasets"]:
        report += [f"## {dataset}：经验 B=4", ""]
        subset = frame[(frame.dataset == dataset) & (frame.family == "empirical_B") & (frame.key.astype(str) == "4")]
        for row in subset.sort_values("method").itertuples():
            report.append(
                f"- {row.method}: Acc={100*row.accuracy:.2f}%，ΔDense={row.delta_dense_pp:+.2f} pp，Token↓={100*row.token_reduction:.2f}%，Coverage={100*row.coverage:.2f}%，W→C/C→W={row.W_to_C}/{row.C_to_W}。"
            )
        report.append("")
    report += [
        "## 解释边界", "",
        "- 所有阈值或动态候选只在 calibration 上选择，held-out 不参与模型、epoch 或策略选择。",
        "- empirical B 是 500 题 calibration 上的绝对 lost-correct 事件预算，不是总体风险保证。",
        "- forced-answer greedy 会改变离线标签；与旧 sampled 分支的逐 checkpoint 漂移见 `tables/greedy_vs_sampled_label_drift.csv`。",
        "- 10,000 次配对 bootstrap 的 B=4 方法差异见 `tables/paired_bootstrap_B4.csv`。", "",
    ]
    (args.output / "GREEDY_FORCED_ABLATION_REPORT_ZH.md").write_text("\n".join(report), encoding="utf-8")
    audit = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config["protocol_id"],
        "datasets": list(config["datasets"]),
        "forced_answer_decoding": "greedy_argmax",
        "dense_hidden_checkpoint_reused": True,
        "heldout_used_for_selection": False,
        "future_fields_used_by_dynamic_action": False,
        "bootstrap_replicates": int(config["statistics"]["bootstrap_replicates"]),
        "cost_metric": "reasoning_tokens_only",
        "short_answer_cost": 0,
    }
    atomic_json(audit, marker)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

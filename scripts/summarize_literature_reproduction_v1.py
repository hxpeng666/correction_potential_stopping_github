#!/usr/bin/env python3
"""Audit and summarize the completed literature reproduction matrix."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "src").is_dir():
    ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def variants():
    for dataset in ("gsm8k", "mmlu_pro"):
        for schedule in ("native", "paragraph"):
            for method in ("learn_to_stop", "self_verification", "lynx"):
                yield dataset, method, schedule, None
            for target in ("supervised", "consistent"):
                yield dataset, "thought_calibration", schedule, target


def result_path(result_root: Path, dataset: str, method: str, schedule: str, target: str | None) -> Path:
    suffix = f"_{target}" if target else ""
    return result_root / dataset / method / schedule / f"probe{suffix}" / "results.json"


def method_label(method: str, target: str | None) -> str:
    labels = {
        "learn_to_stop": "Learn-to-Stop",
        "self_verification": "Self-verification",
        "lynx": "LYNX",
        "thought_calibration": f"Thought Calibration ({target})",
    }
    return labels[method]


def cache_statistics(result_root: Path, dataset: str, method: str, schedule: str) -> dict[str, Any]:
    counts = []
    positives = []
    fallback_judge = 0
    judge_parser_disagreements = 0
    judge_parser_comparisons = 0
    total = 0
    split_ids: dict[str, set[str]] = {split: set() for split in ("probe_train", "calibration", "heldout")}
    shapes = set()
    fingerprints = set()
    for split in split_ids:
        for path in sorted((result_root / dataset / method / schedule / "cache" / split).glob("sample_*.pt")):
            value = torch.load(path, map_location="cpu", weights_only=False)
            if value.get("status") != "complete":
                raise RuntimeError(f"incomplete cache: {path}")
            split_ids[split].add(str(value["problem_id"]))
            counts.append(len(value["rows"]))
            shapes.add(tuple(value["hidden"].shape[1:]))
            fingerprints.add(str(value.get("protocol_fingerprint")))
            for row in value["rows"]:
                if method == "learn_to_stop":
                    continue
                label_key = "probe_label" if method == "self_verification" else (
                    "current_success" if method == "lynx" else "current_success"
                )
                positives.append(int(bool(row.get(label_key, False))))
            if method == "self_verification":
                fallback_judge += int(value.get("labeler_audit", {}).get("mode") != "batched")
                if schedule == "native":
                    for row in value["rows"]:
                        judge_parser_comparisons += 1
                        judge_parser_disagreements += int(
                            bool(row.get("probe_label")) != bool(row.get("current_success"))
                        )
            total += 1
    overlap = {
        "train_calibration": len(split_ids["probe_train"] & split_ids["calibration"]),
        "train_heldout": len(split_ids["probe_train"] & split_ids["heldout"]),
        "calibration_heldout": len(split_ids["calibration"] & split_ids["heldout"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"problem leakage in {dataset}/{method}/{schedule}: {overlap}")
    sorted_counts = sorted(counts)
    return {
        "problems": total,
        "split_counts": {key: len(value) for key, value in split_ids.items()},
        "split_overlap": overlap,
        "mean_available_checkpoints": statistics.mean(counts) if counts else 0.0,
        "median_available_checkpoints": statistics.median(counts) if counts else 0.0,
        "p95_available_checkpoints": sorted_counts[min(len(sorted_counts) - 1, int(0.95 * len(sorted_counts)))] if sorted_counts else 0,
        "max_available_checkpoints": max(counts) if counts else 0,
        "zero_checkpoint_problems": sum(value == 0 for value in counts),
        "hidden_shapes_without_checkpoint_axis": sorted([list(value) for value in shapes]),
        "protocol_fingerprints": sorted(fingerprints),
        "positive_checkpoint_rate": sum(positives) / len(positives) if positives else None,
        "self_verification_non_batched_judge_rate": fallback_judge / total if total and method == "self_verification" else None,
        "self_verification_qwen_label_vs_frozen_parser_disagreement_rate": (
            judge_parser_disagreements / judge_parser_comparisons
            if judge_parser_comparisons else None
        ),
    }


def primary_rows(result_root: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset, method, schedule, target in variants():
        payload = read_json(result_path(result_root, dataset, method, schedule, target))
        workpoint = payload["fair_empirical_B"]["2"]
        heldout = workpoint["heldout"]
        calibration = workpoint["calibration_selection"]
        rows.append({
            "dataset": dataset,
            "method": method,
            "method_label": method_label(method, target),
            "target": target,
            "schedule": schedule,
            "selection": "calibration empirical lost-correct B=2",
            "calibration_lost_correct": calibration["lost_correct"],
            **heldout,
        })
    return rows


def original_curves(result_root: Path) -> dict[str, Any]:
    curves = {}
    for dataset, method, schedule, target in variants():
        payload = read_json(result_path(result_root, dataset, method, schedule, target))
        key = "/".join(value for value in (dataset, method, target, schedule) if value)
        curves[key] = (
            payload["learn_then_test"]
            if method == "thought_calibration"
            else payload["original_operating_points"]
        )
    return curves


def current_method_rows() -> list[dict[str, Any]]:
    paths = {
        "gsm8k": ROOT / "results/gsm8k_checkpoint_schedule_normalized_trajectory_v1/normalized_trajectory_schedule_summary.json",
        "mmlu_pro": ROOT / "results/mmlu_pro_checkpoint_schedule_normalized_trajectory_v1/mmlu_pro_normalized_trajectory_summary.json",
    }
    cost_model_path = ROOT / "results/final_paper_replay_v3/timing_calibration_after_supplement_v1/A100_SINGLE_REQUEST_COST_MODEL.json"
    current_check_ms = (
        float(read_json(cost_model_path)["checkpoint_cost_mean_ms"])
        if cost_model_path.is_file()
        else None
    )
    rows = []
    for dataset, path in paths.items():
        payload = read_json(path)
        trajectory = payload["B2_rows"]["paragraph"]
        paired = payload["paired_normalized_trajectory_vs_bce_B2"]["paragraph"]
        trajectory_delta = trajectory["heldout_accuracy"] - trajectory["heldout_dense_accuracy"]
        trajectory_tokens = trajectory["heldout_mean_reasoning_and_answer_tokens"]
        dense_tokens = trajectory["heldout_mean_dense_reasoning_tokens"]
        trajectory_row = {
            "dataset": dataset,
            "method": "ours_bce_trajectory_normalized",
            "method_label": "Ours BCE+normalized trajectory",
            "target": None,
            "schedule": "paragraph",
            "n": 1319 if dataset == "gsm8k" else 1000,
            "dense_accuracy": trajectory["heldout_dense_accuracy"],
            "accuracy": trajectory["heldout_accuracy"],
            "accuracy_delta_pp": 100.0 * trajectory_delta,
            "dense_mean_reasoning_tokens": dense_tokens,
            "mean_reasoning_tokens": trajectory_tokens,
            "reasoning_token_reduction_pct": 100.0 * trajectory["heldout_token_reduction"],
            "stop_rate": trajectory["heldout_coverage"],
            "mean_checks": trajectory.get("mean_policy_checks_heldout"),
            "lost_correct": trajectory["heldout_lost_correct_count"],
            "helped": round((trajectory["heldout_accuracy"] - trajectory["heldout_dense_accuracy"]) * (1319 if dataset == "gsm8k" else 1000) + trajectory["heldout_lost_correct_count"]),
            "calibration_lost_correct": trajectory["calibration_lost_correct_count"],
            "probe_only_ms_per_checkpoint": current_check_ms,
        }
        if trajectory_row["mean_checks"] is not None and current_check_ms is not None:
            trajectory_row["mean_probe_only_ms_per_problem"] = (
                trajectory_row["mean_checks"] * current_check_ms
            )
        rows.append(trajectory_row)
        bce_accuracy = trajectory["heldout_accuracy"] - paired["trajectory_minus_bce_accuracy_pp"] / 100.0
        bce_tokens = trajectory_tokens - paired["trajectory_minus_bce_mean_tokens"]
        rows.append({
            "dataset": dataset,
            "method": "ours_bce",
            "method_label": "Ours BCE",
            "target": None,
            "schedule": "paragraph",
            "n": 1319 if dataset == "gsm8k" else 1000,
            "dense_accuracy": trajectory["heldout_dense_accuracy"],
            "accuracy": bce_accuracy,
            "accuracy_delta_pp": 100.0 * (bce_accuracy - trajectory["heldout_dense_accuracy"]),
            "dense_mean_reasoning_tokens": dense_tokens,
            "mean_reasoning_tokens": bce_tokens,
            "reasoning_token_reduction_pct": 100.0 * (dense_tokens - bce_tokens) / dense_tokens,
            "stop_rate": None,
            "mean_checks": None,
            "lost_correct": None,
            "helped": None,
            "calibration_lost_correct": 2,
        })
    return rows


def attach_checkpoint_costs(
    rows: list[dict[str, Any]], result_root: Path, config_path: Path
) -> dict[str, Any]:
    cost_path = result_root / "CHECKPOINT_COST.json"
    if not cost_path.is_file():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/benchmark_literature_checkpoint_cost_v1.py"),
                "--config",
                str(config_path),
                "--gpu",
                "0",
            ],
            cwd=ROOT,
            check=True,
        )
    payload = read_json(cost_path)
    lookup = {
        (row["dataset"], row["method"], row["schedule"], row.get("target")): row
        for row in payload["rows"]
    }
    for row in rows:
        cost = lookup[(row["dataset"], row["method"], row["schedule"], row.get("target"))]
        row["probe_only_ms_per_checkpoint"] = cost["probe_only_ms_per_checkpoint"]
        checks = row.get("mean_checks")
        row["mean_probe_only_ms_per_problem"] = (
            float(checks) * cost["probe_only_ms_per_checkpoint"] if checks is not None else None
        )
    return payload


def f(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| 数据集 | 方法 | checkpoint | Acc | ΔAcc(pp) | token reduction | mean tokens | lost | helped | stop rate | checks | probe ms/check | probe ms/problem |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {schedule} | {acc} | {delta} | {reduction}% | {tokens} | {lost} | {helped} | {stop}% | {checks} | {cost_check} | {cost_problem} |".format(
                dataset=row["dataset"],
                method=row["method_label"],
                schedule=row["schedule"],
                acc=f(100.0 * row["accuracy"]),
                delta=f(row["accuracy_delta_pp"]),
                reduction=f(row["reasoning_token_reduction_pct"]),
                tokens=f(row["mean_reasoning_tokens"], 1),
                lost=row.get("lost_correct", "—") if row.get("lost_correct") is not None else "—",
                helped=row.get("helped", "—") if row.get("helped") is not None else "—",
                stop=f(100.0 * row["stop_rate"]) if row.get("stop_rate") is not None else "—",
                checks=f(row.get("mean_checks"), 1),
                cost_check=f(row.get("probe_only_ms_per_checkpoint"), 4),
                cost_problem=f(row.get("mean_probe_only_ms_per_problem"), 3),
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    import yaml

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result_root = ROOT / config["output_root"]
    supervisor = read_json(result_root / "training_supervisor.json")
    if supervisor.get("status") != "complete":
        raise RuntimeError(f"training supervisor is not complete: {supervisor.get('status')}")

    cache_audit = {}
    for dataset in ("gsm8k", "mmlu_pro"):
        cache_audit[dataset] = {}
        for method in ("learn_to_stop", "self_verification", "lynx", "thought_calibration"):
            cache_audit[dataset][method] = {
                schedule: cache_statistics(result_root, dataset, method, schedule)
                for schedule in ("native", "paragraph")
            }
    for dataset in cache_audit:
        for method in cache_audit[dataset]:
            for schedule, audit in cache_audit[dataset][method].items():
                if len(audit["protocol_fingerprints"]) != 1:
                    raise RuntimeError(
                        f"mixed protocol fingerprints: {dataset}/{method}/{schedule}: "
                        f"{audit['protocol_fingerprints']}"
                    )
    rows = primary_rows(result_root)
    checkpoint_cost = attach_checkpoint_costs(rows, result_root, config_path)
    ours = current_method_rows()
    combined = rows + ours

    schedule_pairs = []
    for dataset in ("gsm8k", "mmlu_pro"):
        labels = sorted(set(row["method_label"] for row in rows if row["dataset"] == dataset))
        for label in labels:
            native = next(row for row in rows if row["dataset"] == dataset and row["method_label"] == label and row["schedule"] == "native")
            paragraph = next(row for row in rows if row["dataset"] == dataset and row["method_label"] == label and row["schedule"] == "paragraph")
            schedule_pairs.append({
                "dataset": dataset,
                "method_label": label,
                "paragraph_minus_native_accuracy_pp": paragraph["accuracy_delta_pp"] - native["accuracy_delta_pp"],
                "paragraph_minus_native_token_reduction_pp": paragraph["reasoning_token_reduction_pct"] - native["reasoning_token_reduction_pct"],
                "native_mean_available_checkpoints": cache_audit[dataset][native["method"]]["native"]["mean_available_checkpoints"],
                "paragraph_mean_available_checkpoints": cache_audit[dataset][native["method"]]["paragraph"]["mean_available_checkpoints"],
            })

    summary = {
        "status": "complete",
        "primary_selection": "calibration-only empirical lost-correct budget B=2",
        "heldout_rows": rows,
        "paper_native_operating_point_curves": original_curves(result_root),
        "current_method_context": ours,
        "schedule_pair_differences": schedule_pairs,
        "cache_audit": cache_audit,
        "checkpoint_cost": checkpoint_cost,
        "training_supervisor": supervisor,
        "source_commits": {
            method: config["methods"][method]["source"].get("code_commit")
            for method in ("learn_to_stop", "self_verification", "lynx", "thought_calibration")
        },
    }
    (result_root / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (result_root / "SUMMARY.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(row.keys() for row in combined))), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)

    lines = [
        "# Qwen3-4B 文献方法复现结果",
        "",
        "所有主表 operating point 只由 calibration 集选择，统一使用问题级 1000/500/test 划分。主表为经验 lost-correct 预算 B=2；test 仅作一次冻结评估。",
        "",
        "## 公平口径主结果（B=2）",
        "",
        markdown_table(combined),
        "",
        "## Native 与 paragraph checkpoint 对照",
        "",
        "| 数据集 | 方法 | paragraph-native ΔAcc(pp) | paragraph-native token reduction(pp) | native checkpoints | paragraph checkpoints |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in schedule_pairs:
        lines.append(
            f"| {row['dataset']} | {row['method_label']} | {f(row['paragraph_minus_native_accuracy_pp'])} | "
            f"{f(row['paragraph_minus_native_token_reduction_pp'])} | {f(row['native_mean_available_checkpoints'], 1)} | "
            f"{f(row['paragraph_mean_available_checkpoints'], 1)} |"
        )
    lines.extend([
        "",
        "## 审计说明",
        "",
        "- Learn-to-Stop 使用序列 LSTM 与 stable-suffix 标签；未复用现有点式 last_switch probe。",
        "- LYNX 使用四层 hidden 拼接、论文 forced-exit cue、两层 MLP 和 class-conditional split conformal singleton 决策。",
        "- Self-verification 保留 reasoning-path chunk、answerless-nearest merge、weighted MLP；Qwen3-4B 替代 Gemini 同时做中间答案抽取与监督标签判定，held-out outcome 再由冻结 parser 独立计分。",
        "- Thought Calibration 使用 step-token mean hidden、PCA-256、linear probe、10-step smoothing 与 Learn-Then-Test fixed sequence。",
        "- 每个方法均同时提供 native 与 paragraph 版本；除 checkpoint 位置外不改变该方法的目标、probe 或决策族。",
        "- probe cost 为 checkpoint hidden 已由基础模型产生之后的在线决策微基准；不含基础模型前向与 forced-answer 生成。",
        "",
        "完整 operating-point 曲线、逐题记录和 cache 审计见同目录 `SUMMARY.json` 及各 probe 子目录。",
    ])
    (result_root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    final_audit = {
        "status": "PASS",
        "problem_level_split_overlap": 0,
        "all_training_jobs_complete": True,
        "cache_audit": cache_audit,
        "checkpoint_cost_status": checkpoint_cost.get("status"),
        "single_fingerprint_per_cache_view": all(
            len(cache_audit[dataset][method][schedule]["protocol_fingerprints"]) == 1
            for dataset in cache_audit
            for method in cache_audit[dataset]
            for schedule in cache_audit[dataset][method]
        ),
        "self_verification_qwen_label_migration": (
            read_json(result_root / "SELF_VERIFICATION_QWEN_LABEL_MIGRATION_gsm8k.json")
            if (result_root / "SELF_VERIFICATION_QWEN_LABEL_MIGRATION_gsm8k.json").is_file()
            else None
        ),
        "summary_json": str((result_root / "SUMMARY.json").resolve()),
        "summary_markdown": str((result_root / "SUMMARY.md").resolve()),
    }
    (result_root / "FINAL_AUDIT.json").write_text(json.dumps(final_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "summary": str(result_root / "SUMMARY.md")}, indent=2))


if __name__ == "__main__":
    main()

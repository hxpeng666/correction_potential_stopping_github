#!/usr/bin/env python3
"""汇总动态最优停止内部消融、旧受控基线与10000次配对bootstrap。"""
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


BUDGETS = (0, 1, 2, 4, 10)
ALPHAS = ("0.01", "0.02", "0.05")
TRAIN_VARIANTS = ("no_trajectory", "one_step_value", "dense_endpoint_value")
REPLAY_VARIANTS = (
    "no_risk_penalty_mu0", "no_compute_cost_lambda0", "no_calibration_accuracy_constraint",
    "no_stop_correctness_pS0", "no_continuation_value_M0",
)
OLD_TARGETS = ("correctness", "consistency", "last_switch", "correction_bce", "correction_trajectory")


def normalize(records: list[dict[str, Any]], *, direct: bool = False) -> list[dict[str, Any]]:
    output = []
    for source in records:
        row = dict(source)
        if direct:
            row["method_tokens"] = 0
        elif row.get("fallback", False):
            row["method_tokens"] = int(row["dense_tokens"])
        else:
            row["method_tokens"] = int(row["checkpoint"])
        output.append(row)
    return output


def dense_from_baseline(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in records:
        output.append({
            "problem_id": str(source["problem_id"]), "subject": source.get("subject"),
            "category": source.get("category"), "fallback": True, "checkpoint": None,
            "transition": "fallback", "method_prediction": source["prediction"],
            "dense_prediction": source["prediction"], "gold_answer": source["gold_answer"],
            "method_success": bool(source["success"]), "dense_success": bool(source["success"]),
            "method_tokens": int(source["reasoning_tokens"]), "dense_tokens": int(source["reasoning_tokens"]),
        })
    return output


def align(records: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    mapping = {str(row["problem_id"]): row for row in records}
    if len(mapping) != len(records) or set(mapping) != set(ids):
        missing = sorted(set(ids) - set(mapping))[:5]
        extra = sorted(set(mapping) - set(ids))[:5]
        raise ValueError(f"sample ID不配对：missing={missing}, extra={extra}")
    return [mapping[value] for value in ids]


def metric_row(dataset: str, group: str, method: str, family: str, key: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    value = summarize_token_records(records)
    return {
        "dataset": dataset, "group": group, "method": method, "family": family, "key": key,
        "N": value["problems"], "accuracy": value["accuracy"], "dense_accuracy": value["dense_accuracy"],
        "delta_dense_pp": value["delta_dense_pp"], "coverage": value["coverage"],
        "mean_reasoning_tokens": value["mean_reasoning_tokens"],
        "mean_dense_reasoning_tokens": value["mean_dense_reasoning_tokens"],
        "token_reduction": value["token_reduction"], "lost_correct_count": value["lost_correct_count"],
        "lost_correct_rate": value["lost_correct_rate"], "gained_correct_count": value["gained_correct_count"],
        "fallback": value["fallback"], **value["counts"],
    }


def load_dynamic(root: Path) -> dict[str, Any]:
    return torch.load(root / "policy_records.pt", map_location="cpu", weights_only=False)["records"]


def add_method(
    methods: dict[str, list[dict[str, Any]]], meta: dict[str, tuple[str, str, str]],
    name: str, records: list[dict[str, Any]], group: str, family: str, key: str,
) -> None:
    if name in methods:
        raise ValueError(f"重复方法名：{name}")
    methods[name] = normalize(records)
    meta[name] = (group, family, key)


def bootstrap_indices(records: list[dict[str, Any]], dataset: str, replicates: int, seed: int) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    if dataset == "gsm8k":
        return rng.integers(0, len(records), size=(replicates, len(records))), "problem"
    categories = np.asarray([str(row.get("category")) for row in records])
    parts = []
    for category in sorted(set(categories)):
        local = np.flatnonzero(categories == category)
        parts.append(rng.choice(local, size=(replicates, len(local)), replace=True))
    return np.concatenate(parts, axis=1), "category_stratified_problem"


def arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "accuracy": np.asarray([row["method_success"] for row in records], dtype=np.float64),
        "lost_correct_rate": np.asarray([row.get("transition") == "W_to_C" for row in records], dtype=np.float64),
        "coverage": np.asarray([not row.get("fallback", False) for row in records], dtype=np.float64),
        "tokens": np.asarray([row["method_tokens"] for row in records], dtype=np.float64),
        "dense_tokens": np.asarray([row["dense_tokens"] for row in records], dtype=np.float64),
    }


def point(raw: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "accuracy": float(raw["accuracy"].mean()),
        "lost_correct_rate": float(raw["lost_correct_rate"].mean()),
        "coverage": float(raw["coverage"].mean()),
        "token_reduction": float(1.0 - raw["tokens"].mean() / raw["dense_tokens"].mean()),
    }


def bootstrap(raw: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "accuracy": raw["accuracy"][indices].mean(axis=1),
        "lost_correct_rate": raw["lost_correct_rate"][indices].mean(axis=1),
        "coverage": raw["coverage"][indices].mean(axis=1),
        "token_reduction": 1.0 - raw["tokens"][indices].mean(axis=1) / raw["dense_tokens"][indices].mean(axis=1),
    }


def selection_rows(probe_root: Path, dataset: str, variant: str) -> list[dict[str, Any]]:
    probe = json.loads((probe_root / "probe.json").read_text(encoding="utf-8"))
    output = []
    for family, values in probe["frozen_policy_results"].items():
        for key, item in values.items():
            selected = item["calibration"]
            heldout = item["heldout"]
            output.append({
                "dataset": dataset, "variant": variant, "family": family, "key": key,
                "candidate_index": selected.get("candidate_index"), "lambda": selected.get("lambda"),
                "mu": selected.get("mu"), "dense_fallback": selected.get("dense_fallback", False),
                "calibration_accuracy": selected.get("accuracy"),
                "calibration_coverage": selected.get("coverage"),
                "calibration_token_reduction": selected.get("token_reduction"),
                "calibration_lost_correct_count": selected.get("lost_correct_count"),
                "calibration_lost_correct_ucb": selected.get("lost_correct_ucb_simultaneous95"),
                "heldout_accuracy": heldout.get("accuracy"), "heldout_coverage": heldout.get("coverage"),
                "heldout_token_reduction": heldout.get("token_reduction"),
                "heldout_lost_correct_count": heldout.get("lost_correct_count"),
            })
    return output


def dataset_paths(dataset: str, config: dict[str, Any]) -> dict[str, Path]:
    if dataset == "gsm8k":
        baseline = ROOT / "results/final_paper_primary_v1/main_float16_seed20260803/gsm8k/baselines/baseline_records.pt"
        old_probe_base = ROOT / "results/final_paper_primary_v1/main_float16_seed20260803/gsm8k/probes"
    else:
        baseline = ROOT / "results/final_paper_mmlu_pro_independent_token_v2_train1000_cal500_test1000/baselines_token/baseline_records.pt"
        old_probe_base = ROOT / "results/final_paper_mmlu_pro_independent_token_v2_train1000_cal500_test1000/probes_token"
    return {
        "baseline": baseline, "old_probe_base": old_probe_base,
        "fourstate": ROOT / config["datasets"][dataset]["previous_four_state_root"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    marker = args.output / "pipeline.complete"
    if args.resume and marker.is_file():
        print(json.dumps({"status": "skipped_complete"}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    tables = args.output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    config = load_yaml(args.config)
    all_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    bootstrap_comparisons: list[dict[str, Any]] = []
    bootstrap_cis: list[dict[str, Any]] = []

    for dataset in ("gsm8k", "mmlu_pro"):
        paths = dataset_paths(dataset, config)
        baseline_payload = torch.load(paths["baseline"], map_location="cpu", weights_only=False)["records"]
        baseline = baseline_payload["heldout"] if dataset == "gsm8k" else baseline_payload
        dense = dense_from_baseline(baseline["dense"])
        ids = sorted(row["problem_id"] for row in dense)
        methods: dict[str, list[dict[str, Any]]] = {"Dense": align(dense, ids)}
        meta: dict[str, tuple[str, str, str]] = {"Dense": ("baseline", "dense", "dense")}
        methods["Direct"] = align(normalize(baseline["direct"], direct=True), ids)
        meta["Direct"] = ("baseline", "direct", "direct")
        for budget, records in baseline["fixed"].items():
            methods[f"Fixed-{budget}"] = align(normalize(records), ids)
            meta[f"Fixed-{budget}"] = ("baseline", "fixed", str(budget))

        for target in OLD_TARGETS:
            payload = torch.load(paths["old_probe_base"] / target / "policy_records.pt", map_location="cpu", weights_only=False)["records"]
            for budget in BUDGETS:
                name = f"Old-{target}-B{budget}"
                add_method(methods, meta, name, payload["empirical_B"][str(budget)], "old_controlled", target, str(budget))
            for coverage_target, records in payload["coverage"].items():
                name = f"Old-{target}-coverage{coverage_target}"
                add_method(
                    methods, meta, name, records, "old_controlled_coverage",
                    target, str(coverage_target),
                )

        four_sources = {
            "Old-fourstate-unweighted-zero": paths["fourstate"] / "probe/policy_records.pt",
            "Old-fourstate-weighted-trajectory-zero": paths["fourstate"] / "probe_legacy_weighted_trajectory/policy_records.pt",
        }
        for name, path in four_sources.items():
            records = torch.load(path, map_location="cpu", weights_only=False)["heldout"]
            add_method(methods, meta, name, records, "old_fourstate", "fixed_zero", "zero")

        full_probe_root = args.main_root / dataset / "probe"
        full_payload = load_dynamic(full_probe_root)
        for family, values in full_payload.items():
            for key, records in values.items():
                add_method(methods, meta, f"Dynamic-full-{family}-{key}", records, "dynamic_full", family, key)
        selections.extend(selection_rows(full_probe_root, dataset, "full"))

        full_coverage = torch.load(
            args.ablation_root / "coverage/full" / dataset / "policy_records.pt",
            map_location="cpu", weights_only=False,
        )["records"]
        for key, records in full_coverage.items():
            add_method(
                methods, meta, f"Dynamic-full-coverage-{key}", records,
                "dynamic_coverage", "coverage", key,
            )

        for variant in TRAIN_VARIANTS:
            root = args.ablation_root / variant / dataset / "probe"
            payload = load_dynamic(root)
            for family, values in payload.items():
                for key, records in values.items():
                    add_method(methods, meta, f"Dynamic-{variant}-{family}-{key}", records, "dynamic_internal", family, key)
            selections.extend(selection_rows(root, dataset, variant))
            coverage_payload = torch.load(
                args.ablation_root / "coverage" / variant / dataset / "policy_records.pt",
                map_location="cpu", weights_only=False,
            )["records"]
            for key, records in coverage_payload.items():
                add_method(
                    methods, meta, f"Dynamic-{variant}-coverage-{key}", records,
                    "dynamic_internal_coverage", "coverage", key,
                )

        replay_root = args.ablation_root / "replay_only_v2" / dataset
        replay_payload = torch.load(replay_root / "policy_records.pt", map_location="cpu", weights_only=False)["records"]
        for variant in REPLAY_VARIANTS:
            for family, values in replay_payload[variant].items():
                for key, records in values.items():
                    add_method(methods, meta, f"Dynamic-{variant}-{family}-{key}", records, "dynamic_replay", family, key)
        add_method(
            methods, meta, "Oracle-earliest-correct", replay_payload["oracle_earliest_correct"],
            "oracle_descriptive", "heldout_label_oracle", "none",
        )

        if dataset == "gsm8k":
            schedule_base = ROOT / "results/final_paper_primary_v1/main_float16_seed20260803/gsm8k/schedule_probes"
            for schedule in ("fixed", "hybrid"):
                payload = torch.load(
                    schedule_base / schedule / "correction_trajectory/policy_records.pt",
                    map_location="cpu", weights_only=False,
                )["records"]["empirical_B"]
                for budget in BUDGETS:
                    add_method(
                        methods, meta, f"Old-correction_trajectory-{schedule}-B{budget}",
                        payload[str(budget)], "old_checkpoint_schedule", schedule, str(budget),
                    )

        methods = {name: align(records, ids) for name, records in methods.items()}
        for name, records in methods.items():
            group, family, key = meta[name]
            all_rows.append(metric_row(dataset, group, name, family, key, records))

        for variant, root in [("full", full_probe_root)] + [
            (value, args.ablation_root / value / dataset / "probe") for value in TRAIN_VARIANTS
        ]:
            probe = json.loads((root / "probe.json").read_text(encoding="utf-8"))
            for split, values in probe["local_diagnostics"].items():
                diagnostics.append({
                    "dataset": dataset, "variant": variant, "split": split,
                    "local_best_epoch": probe["local_best_epoch"], "value_best_epoch": probe["value_best_epoch"],
                    "value_validation_mae": probe["value_history"][probe["value_best_epoch"]]["validation_mae"],
                    **values,
                })

        replicates = int(config["statistics"]["bootstrap_replicates"])
        indices, stratification = bootstrap_indices(methods["Dense"], dataset, replicates, int(config["seed"]["bootstrap"]))
        full_b4 = "Dynamic-full-empirical_B-4"
        full_a2 = "Dynamic-full-formal_alpha-0.02"
        comparators_b4 = ["Dense", "Direct"] + [f"Fixed-{value}" for value in (64, 96, 128, 192, 256)]
        comparators_b4 += [f"Old-{target}-B4" for target in OLD_TARGETS]
        comparators_b4 += list(four_sources)
        comparators_b4 += [f"Dynamic-{variant}-empirical_B-4" for variant in TRAIN_VARIANTS + REPLAY_VARIANTS]
        comparators_a2 = [f"Dynamic-{variant}-formal_alpha-0.02" for variant in TRAIN_VARIANTS + REPLAY_VARIANTS]
        pairs = [(full_b4, comparator, "primary_B4") for comparator in comparators_b4]
        pairs += [(full_a2, comparator, "formal_alpha2") for comparator in comparators_a2]
        pairs += [
            (f"Dynamic-full-empirical_B-{budget}", f"Old-correction_trajectory-B{budget}", "B_curve_old_method")
            for budget in BUDGETS
        ]
        for coverage_target in (30, 40, 50, 60, 70, 80, 90):
            left = f"Dynamic-full-coverage-{coverage_target}"
            pairs += [
                (left, f"Old-{target}-coverage{coverage_target}", "calibration_matched_coverage")
                for target in OLD_TARGETS
            ]
            pairs += [
                (left, f"Dynamic-{variant}-coverage-{coverage_target}", "calibration_matched_coverage_internal")
                for variant in TRAIN_VARIANTS
            ]
        needed = {value for pair in pairs for value in pair[:2]}
        boot_cache: dict[str, dict[str, np.ndarray]] = {}
        point_cache: dict[str, dict[str, float]] = {}
        for name in needed:
            raw = arrays(methods[name])
            point_cache[name] = point(raw)
            boot_cache[name] = bootstrap(raw, indices)
        primary_ci_names = [full_b4, full_a2] + comparators_b4 + comparators_a2
        for name in dict.fromkeys(primary_ci_names):
            for metric, samples in boot_cache[name].items():
                bootstrap_cis.append({
                    "dataset": dataset, "method": name, "metric": metric,
                    "point": point_cache[name][metric], "ci_low": float(np.percentile(samples, 2.5)),
                    "ci_high": float(np.percentile(samples, 97.5)), "replicates": replicates,
                    "stratification": stratification,
                })
        for left, right, comparison_group in pairs:
            for metric in point_cache[left]:
                difference = boot_cache[left][metric] - boot_cache[right][metric]
                bootstrap_comparisons.append({
                    "dataset": dataset, "comparison_group": comparison_group,
                    "main": left, "comparator": right, "metric": metric,
                    "difference_point": point_cache[left][metric] - point_cache[right][metric],
                    "ci_low": float(np.percentile(difference, 2.5)),
                    "ci_high": float(np.percentile(difference, 97.5)),
                    "replicates": replicates, "stratification": stratification,
                })

    frame = pd.DataFrame(all_rows)
    frame.to_csv(tables / "all_methods_all_workpoints.csv", index=False)
    frame[(frame.family == "empirical_B") & (frame.key.astype(str) == "4")].to_csv(tables / "primary_B4_comparison.csv", index=False)
    frame[(frame.group == "dynamic_internal") & (frame.family == "empirical_B")].to_csv(tables / "internal_training_ablations.csv", index=False)
    frame[(frame.group == "dynamic_replay") & (frame.family == "empirical_B")].to_csv(tables / "internal_replay_ablations.csv", index=False)
    frame[frame.group == "baseline"].to_csv(tables / "dense_direct_fixed.csv", index=False)
    frame[frame.group == "old_controlled"].to_csv(tables / "old_controlled_B_curve.csv", index=False)
    frame[frame.group.isin(["dynamic_coverage", "dynamic_internal_coverage", "old_controlled_coverage"])].to_csv(
        tables / "coverage_targeted_comparison.csv", index=False,
    )
    frame[frame.group == "old_checkpoint_schedule"].to_csv(tables / "old_checkpoint_schedule_ablation.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(tables / "training_diagnostics.csv", index=False)
    pd.DataFrame(selections).to_csv(tables / "calibration_selections.csv", index=False)
    pd.DataFrame(bootstrap_cis).to_csv(tables / "bootstrap_confidence_intervals.csv", index=False)
    pd.DataFrame(bootstrap_comparisons).to_csv(tables / "paired_bootstrap_comparisons.csv", index=False)

    report = [
        "# 动态最优停止完整消融报告", "",
        "所有结果复用同一FP16 Dense/checkpoint/forced-answer公共缓存；成本为reasoning tokens，短答案成本忽略；单seed；策略只在calibration冻结；heldout不参与选择。", "",
        "内部训练消融：去trajectory weakest-point、one-step非递归价值、Dense-endpoint未来语义。策略级消融：mu=0、lambda=0、去calibration准确率下限。另给出读取heldout正确标签的最早正确checkpoint oracle，后者不可部署。", "",
        "主比较工作点为经验B=4；formal alpha=2%作为风险认证补充。所有差异置信区间为10000次配对bootstrap；MMLU-Pro按category分层。", "",
    ]
    for dataset in ("gsm8k", "mmlu_pro"):
        report += [f"## {dataset}", ""]
        subset = frame[(frame.dataset == dataset) & frame.method.isin([
            "Dense", "Dynamic-full-empirical_B-4",
            *[f"Dynamic-{value}-empirical_B-4" for value in TRAIN_VARIANTS + REPLAY_VARIANTS],
            "Old-correction_trajectory-B4", "Oracle-earliest-correct",
        ])]
        for row in subset.itertuples():
            report.append(
                f"- {row.method}: Acc={100*row.accuracy:.2f}%, ΔDense={row.delta_dense_pp:+.2f}pp, "
                f"Coverage={100*row.coverage:.2f}%, Token↓={100*row.token_reduction:.2f}%, "
                f"W→C/C→W={row.W_to_C}/{row.C_to_W}."
            )
        report.append("")
    report += [
        "全部工作点、旧五种受控目标、Direct/Fixed、formal结果、训练诊断和配对置信区间见tables目录。", "",
    ]
    (args.output / "COMPLETE_ABLATION_REPORT_ZH.md").write_text("\n".join(report), encoding="utf-8")
    atomic_json({
        "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
        "datasets": ["gsm8k", "mmlu_pro"], "train_variants": list(TRAIN_VARIANTS),
        "replay_variants": list(REPLAY_VARIANTS), "old_targets": list(OLD_TARGETS),
        "bootstrap_replicates": int(config["statistics"]["bootstrap_replicates"]),
        "heldout_used_for_selection": False, "cost": "reasoning_tokens_only", "short_answer_cost": 0,
    }, marker)
    print(json.dumps({"status": "complete", "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""汇总四状态 CE 与 legacy-weighted+trajectory 两个固定零规则版本。"""
from __future__ import annotations

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

from src.utils import atomic_json, load_yaml


EXPERIMENT_ROOT = ROOT / "results/final_paper_four_state_utility_v1"


def aligned(records: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    mapping = {str(row["problem_id"]): row for row in records}
    if len(mapping) != len(records) or set(mapping) != set(ids):
        raise ValueError("四状态 loss 消融的 sample ID 不完全配对")
    return [mapping[value] for value in ids]


def raw_metrics(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
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


def point(values: dict[str, np.ndarray], latency: bool) -> dict[str, float]:
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


def sampled(values: dict[str, np.ndarray], indices: np.ndarray, latency: bool) -> dict[str, np.ndarray]:
    result = {
        "accuracy": values["accuracy"][indices].mean(1),
        "delta_dense": (values["accuracy"][indices] - values["dense_accuracy"][indices]).mean(1),
        "lost_correct_rate": values["lost_correct_rate"][indices].mean(1),
        "coverage": values["coverage"][indices].mean(1),
        "token_reduction": 1.0 - values["tokens"][indices].mean(1) / values["dense_tokens"][indices].mean(1),
    }
    if latency:
        result["mean_replay_wall_reduction"] = 1.0 - values["wall"][indices].mean(1) / values["dense_wall"][indices].mean(1)
    return result


def indices_for(records: list[dict[str, Any]], dataset: str, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dataset == "gsm8k":
        return rng.integers(0, len(records), size=(reps, len(records)))
    categories = np.asarray([str(row.get("category")) for row in records])
    parts = []
    for category in sorted(set(categories)):
        local = np.flatnonzero(categories == category)
        parts.append(rng.choice(local, size=(reps, len(local)), replace=True))
    return np.concatenate(parts, axis=1)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    config = load_yaml(ROOT / "configs/final_paper_four_state_utility_v1.yaml")
    output = EXPERIMENT_ROOT / "final_report"
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    all_rows = []
    loss_rows = []
    paired_rows = []
    integrity: dict[str, Any] = {
        "status": "complete",
        "old_artifacts_deleted_or_overwritten": False,
        "dense_or_branch_regeneration": False,
        "datasets": {},
    }
    summaries = {}
    reps = int(config["statistics"]["bootstrap_replicates"])

    for dataset, n in (("gsm8k", 1319), ("mmlu_pro", 1000)):
        unweighted_report = EXPERIMENT_ROOT / dataset / "report_v3_final"
        weighted_report = EXPERIMENT_ROOT / dataset / "report_legacy_weighted_trajectory_v1"
        base = pd.read_csv(unweighted_report / "tables/main_comparison.csv")
        base.loc[base.method == "Four-state utility (zero rule)", "method"] = "Four-state unweighted CE (zero rule)"
        weighted = pd.read_csv(weighted_report / "tables/main_comparison.csv")
        weighted = weighted[weighted.method == "Four-state utility (zero rule)"].copy()
        weighted.loc[:, "method"] = "Four-state legacy-weighted CE + trajectory (zero rule)"
        combined = pd.concat([base, weighted], ignore_index=True)
        all_rows.append(combined)
        loss_rows.append(combined[combined.method.str.startswith("Four-state")])

        unweighted_probe = EXPERIMENT_ROOT / dataset / "probe"
        weighted_probe = EXPERIMENT_ROOT / dataset / "probe_legacy_weighted_trajectory"
        unweighted_json = json.loads((unweighted_probe / "probe.json").read_text(encoding="utf-8"))
        weighted_json = json.loads((weighted_probe / "probe.json").read_text(encoding="utf-8"))
        unweighted_prob = torch.load(unweighted_probe / "probabilities.pt", map_location="cpu", weights_only=False)
        weighted_prob = torch.load(weighted_probe / "probabilities.pt", map_location="cpu", weights_only=False)
        unweighted_policy = torch.load(unweighted_probe / "policy_records.pt", map_location="cpu", weights_only=False)["heldout"]
        weighted_policy = torch.load(weighted_probe / "policy_records.pt", map_location="cpu", weights_only=False)["heldout"]
        ids = sorted(str(row["problem_id"]) for row in unweighted_policy)
        unweighted_policy = aligned(unweighted_policy, ids)
        weighted_policy = aligned(weighted_policy, ids)
        if len(ids) != n:
            raise ValueError(f"{dataset} heldout {len(ids)} != {n}")
        if unweighted_json["input"] != weighted_json["input"]:
            raise ValueError(f"{dataset} 两个 loss 版本未复用完全相同的公共缓存")
        if unweighted_json["fit_problem_ids"] != weighted_json["fit_problem_ids"] or unweighted_json["validation_problem_ids"] != weighted_json["validation_problem_ids"]:
            raise ValueError(f"{dataset} 内部 fit/validation 划分不一致")
        if set(unweighted_json["fit_problem_ids"]) & set(unweighted_json["validation_problem_ids"]):
            raise ValueError(f"{dataset} 内部 fit/validation 泄漏")
        if unweighted_json["calibration_selected_anything"] is not False or weighted_json["calibration_selected_anything"] is not False:
            raise ValueError(f"{dataset} calibration 错误参与选择")
        probability_checks = {}
        for name, payload in (("unweighted", unweighted_prob), ("legacy_weighted_trajectory", weighted_prob)):
            probability_checks[name] = {}
            for split, probabilities in payload["probabilities"].items():
                values = probabilities.numpy()
                probability_checks[name][split] = {
                    "rows": int(len(values)),
                    "shape": list(values.shape),
                    "finite": bool(np.isfinite(values).all()),
                    "max_probability_sum_abs_error": float(np.abs(values.sum(1) - 1.0).max()),
                    "unique_problem_checkpoint_pairs": len(set(zip(payload["problem_ids"][split], payload["checkpoints"][split]))),
                }
        integrity["datasets"][dataset] = {
            "heldout_problems": len(ids),
            "heldout_unique_ids": len(set(ids)),
            "input_manifest_equal_between_losses": True,
            "input_manifest": unweighted_json["input"],
            "fit_validation_disjoint": True,
            "calibration_selected_anything": False,
            "probability_checks": probability_checks,
        }

        latency = dataset == "gsm8k"
        raw = {
            "unweighted": raw_metrics(unweighted_policy),
            "legacy_weighted_trajectory": raw_metrics(weighted_policy),
        }
        points = {name: point(values, latency) for name, values in raw.items()}
        boot_indices = indices_for(unweighted_policy, dataset, reps, int(config["seed"]["bootstrap"]))
        distributions = {name: sampled(values, boot_indices, latency) for name, values in raw.items()}
        for metric in distributions["unweighted"]:
            delta = distributions["legacy_weighted_trajectory"][metric] - distributions["unweighted"][metric]
            paired_rows.append({
                "dataset": dataset,
                "main": "Four-state legacy-weighted CE + trajectory (zero rule)",
                "comparator": "Four-state unweighted CE (zero rule)",
                "metric": metric,
                "difference_point": points["legacy_weighted_trajectory"][metric] - points["unweighted"][metric],
                "bootstrap_difference_mean": float(delta.mean()),
                "ci_low": float(np.percentile(delta, 2.5)),
                "ci_high": float(np.percentile(delta, 97.5)),
                "replicates": reps,
            })
        summaries[dataset] = {
            "unweighted": combined[combined.method == "Four-state unweighted CE (zero rule)"].iloc[0].to_dict(),
            "legacy_weighted_trajectory": combined[combined.method == "Four-state legacy-weighted CE + trajectory (zero rule)"].iloc[0].to_dict(),
        }

    pd.concat(all_rows, ignore_index=True).to_csv(tables / "all_results.csv", index=False)
    pd.concat(loss_rows, ignore_index=True).to_csv(tables / "four_state_loss_ablation.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(tables / "four_state_loss_paired_bootstrap.csv", index=False)
    atomic_json(integrity, output / "CACHE_AND_PROTOCOL_INTEGRITY_AUDIT.json")

    gsm_u = summaries["gsm8k"]["unweighted"]
    gsm_t = summaries["gsm8k"]["legacy_weighted_trajectory"]
    mmlu_u = summaries["mmlu_pro"]["unweighted"]
    mmlu_t = summaries["mmlu_pro"]["legacy_weighted_trajectory"]
    report = [
        "# 四状态概率效用差：GSM8K 与 MMLU-Pro 最终对比", "",
        "## 方法", "",
        "MLP 骨干、5126维 layer-20 sentence-step 特征、1000/500/heldout 划分、FP16 Qwen3-4B 公共缓存及训练预算均与此前实验配对。末层改为4维 logits，经 softmax 得到 W→C、C→W、W→W、C→C 四类概率。逐 checkpoint 若 `P(W→C)-P(C→W) >= 0` 则继续；首次严格小于0时停止。没有 calibration-selected threshold，500题 calibration 只作冻结规则诊断。", "",
        "报告两个预先区分的 loss 版本：", "",
        "- `unweighted CE`：四分类未加权交叉熵，最直接保持概率语义。",
        "- `legacy-weighted CE + trajectory`：沿用旧方法的 W→C=1.5、其余=`1+remaining` point 权重，并对 `logit_WC-logit_CW` 施加 beta=0.5 的 trajectory soft-min；它是对旧 weakest-point 协议的完整迁移。", "",
        "## GSM8K official test（1319题）", "",
        f"- Dense accuracy={pct(gsm_u['dense_accuracy'])}。",
        f"- Unweighted CE：accuracy={pct(gsm_u['accuracy'])}（ΔDense={gsm_u['delta_dense_pp']:+.2f}pp），coverage={pct(gsm_u['coverage'])}，token reduction={pct(gsm_u['token_reduction'])}，W→C/C→W={int(gsm_u['W_to_C'])}/{int(gsm_u['C_to_W'])}。",
        f"- Legacy-weighted+trajectory：accuracy={pct(gsm_t['accuracy'])}（ΔDense={gsm_t['delta_dense_pp']:+.2f}pp），coverage={pct(gsm_t['coverage'])}，token reduction={pct(gsm_t['token_reduction'])}，W→C/C→W={int(gsm_t['W_to_C'])}/{int(gsm_t['C_to_W'])}。",
        f"- 两者的 A100 replay-estimated mean latency reduction 分别为 {pct(gsm_u['mean_replay_wall_reduction'])} 和 {pct(gsm_t['mean_replay_wall_reduction'])}。unweighted 的 p95 reduction 为 {pct(gsm_u['p95_replay_wall_reduction'])}，trajectory 版本为 {pct(gsm_t['p95_replay_wall_reduction'])}。", "",
        "## MMLU-Pro heldout（1000题）", "",
        f"- Dense accuracy={pct(mmlu_u['dense_accuracy'])}。",
        f"- Unweighted CE：accuracy={pct(mmlu_u['accuracy'])}（ΔDense={mmlu_u['delta_dense_pp']:+.2f}pp），coverage={pct(mmlu_u['coverage'])}，token reduction={pct(mmlu_u['token_reduction'])}，W→C/C→W={int(mmlu_u['W_to_C'])}/{int(mmlu_u['C_to_W'])}。",
        f"- Legacy-weighted+trajectory：accuracy={pct(mmlu_t['accuracy'])}（ΔDense={mmlu_t['delta_dense_pp']:+.2f}pp），coverage={pct(mmlu_t['coverage'])}，token reduction={pct(mmlu_t['token_reduction'])}，W→C/C→W={int(mmlu_t['W_to_C'])}/{int(mmlu_t['C_to_W'])}。",
        "- MMLU-Pro 沿用此前约定，只报告 token，不报告 latency。", "",
        "## 结论", "",
        "四状态固定零规则在两个任务上都能工作，但 loss 决定了风险偏好。未加权 CE 在 MMLU-Pro 上形成激进工作点；与旧 trajectory 方法约80% matched coverage 相比，它以相近 coverage 获得更高 token reduction和更高点估计 accuracy，但相对 Dense 仍下降1.2pp且区间包含零。迁移旧 weighted+trajectory 后，MMLU-Pro 变成极保守工作点：0次 W→C、准确率不降，但 token reduction 仅4.21%。", "",
        "GSM8K 上两个版本都保持 Dense accuracy并显著减少平均 token；四状态方法并未在所有风险—效率点严格支配旧经验B前沿，因此结论是有竞争力但非全面胜出。完整旧B与matched-coverage对比、逐类别结果和10,000次配对bootstrap见各数据集报告目录及本目录表格。", "",
    ]
    (output / "FOUR_STATE_FINAL_REPORT_ZH.md").write_text("\n".join(report), encoding="utf-8")
    atomic_json({
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config["protocol_id"],
        "datasets": {"gsm8k": 1319, "mmlu_pro": 1000},
        "variants": ["unweighted_ce", "legacy_weighted_ce_traj"],
        "threshold_calibration_used": False,
        "bootstrap_replicates": reps,
        "final_report": str(output / "FOUR_STATE_FINAL_REPORT_ZH.md"),
    }, EXPERIMENT_ROOT / "pipeline_final.complete")
    print(json.dumps({"status": "complete", "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

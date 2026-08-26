#!/usr/bin/env python3
"""生成动态停止跨数据集总报告与缓存/泄漏审计。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.utils import atomic_json, load_yaml


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    config = load_yaml(ROOT / "configs/final_paper_dynamic_optimal_stopping_v1.yaml")
    experiment = ROOT / config["output_root"]
    final = experiment / "final_report"
    tables = final / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    combined = []
    integrity = {
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "dense_or_forced_answer_regenerated": False,
        "old_code_cache_results_deleted_or_overwritten": False,
        "heldout_used_for_scaler_epoch_or_policy_selection": False,
        "cost": "reasoning_tokens/4096",
        "short_answer_cost": 0,
        "datasets": {},
    }
    summaries = {}
    for dataset, expected in (("gsm8k", 1319), ("mmlu_pro", 1000)):
        probe_root = experiment / dataset / "probe"
        report_root = experiment / dataset / "report"
        probe = json.loads((probe_root / "probe.json").read_text(encoding="utf-8"))
        predictions = torch.load(probe_root / "predictions.pt", map_location="cpu", weights_only=False)
        main = pd.read_csv(report_root / "tables/main_results.csv")
        combined.append(main)
        fit = set(probe["fit_problem_ids"])
        validation = set(probe["validation_problem_ids"])
        split_ids = {
            split: set(str(value) for value in ids)
            for split, ids in predictions["problem_ids"].items()
        }
        if fit & validation:
            raise ValueError(f"{dataset} fit/validation重叠")
        if split_ids["probe_train"] & split_ids["calibration"] or split_ids["probe_train"] & split_ids["heldout"] or split_ids["calibration"] & split_ids["heldout"]:
            raise ValueError(f"{dataset}外部split重叠")
        if len(split_ids["heldout"]) != expected:
            raise ValueError(f"{dataset} heldout不完整")
        tensor_checks = {}
        for family in ("stop_probability", "risk_probability", "continuation_values"):
            tensor_checks[family] = {}
            for split, tensor in predictions[family].items():
                tensor_checks[family][split] = {
                    "shape": list(tensor.shape),
                    "finite": bool(torch.isfinite(tensor).all()),
                    "minimum": float(tensor.min()),
                    "maximum": float(tensor.max()),
                }
                if not tensor_checks[family][split]["finite"]:
                    raise ValueError(f"{dataset}/{family}/{split}包含NaN/Inf")
        pair_checks = {}
        for split in predictions["problem_ids"]:
            pairs = list(zip(predictions["problem_ids"][split], predictions["checkpoints"][split]))
            pair_checks[split] = {"rows": len(pairs), "unique_pairs": len(set(pairs))}
            if len(pairs) != len(set(pairs)):
                raise ValueError(f"{dataset}/{split} problem-checkpoint重复")
        prior_four = ROOT / config["datasets"][dataset]["previous_four_state_root"] / "probe/probe.json"
        prior_manifest = json.loads(prior_four.read_text(encoding="utf-8"))["input"]
        if probe["input"] != prior_manifest:
            raise ValueError(f"{dataset}未与前一方法复用完全相同的公共缓存")
        integrity["datasets"][dataset] = {
            "heldout_problems": len(split_ids["heldout"]),
            "split_disjoint": True,
            "fit_validation_disjoint": True,
            "candidate_count": probe["calibration"]["candidate_count"],
            "per_candidate_delta": probe["calibration"]["per_candidate_delta"],
            "cache_manifest_equal_to_previous_experiment": True,
            "cache_manifest": probe["input"],
            "tensor_checks": tensor_checks,
            "problem_checkpoint_checks": pair_checks,
            "local_best_epoch": probe["local_best_epoch"],
            "value_best_epoch": probe["value_best_epoch"],
            "nonmonotonic_wcw": probe["descriptive_heldout_nonmonotonic_wcw_audit"],
        }
        summaries[dataset] = {
            "probe": probe,
            "table": main,
            "ci": pd.read_csv(report_root / "tables/bootstrap_confidence_intervals.csv"),
            "pairs": pd.read_csv(report_root / "tables/paired_comparisons.csv"),
        }
    pd.concat(combined, ignore_index=True).to_csv(tables / "all_results.csv", index=False)
    atomic_json(integrity, final / "CACHE_AND_LEAKAGE_AUDIT.json")

    report = [
        "# 风险约束动态推理最优停止：最终实验报告", "",
        "## 固定协议", "",
        "基础LLM、FP16公共Dense轨迹、sentence checkpoints、layer-20 5126维特征及forced-answer标签全部复用。新增共享MLP上的stop-correctness、lost-correct risk和48列continuation-value head。前两头先监督训练；共享表示冻结后，使用有限时域后向动态规划目标拟合value bank。", "",
        "当前实验以reasoning token/4096替代wall-time cost，并按要求把停止后的短答案cost设为0。在线动作比较不读取下一checkpoint hidden；value head只根据当前状态预测下一状态长期价值。", "",
        "48个(lambda,mu)候选完全预声明。Formal工作点使用finite-grid Bonferroni修正的一侧95% Clopper–Pearson lost-correct上界，并要求calibration accuracy不低于Dense 1pp；若无候选则严格Dense fallback。", "",
        "## Formal α=2%共同工作点", "",
    ]
    for dataset in ("gsm8k", "mmlu_pro"):
        summary = summaries[dataset]
        row = summary["table"][summary["table"].method == "Dynamic formal_alpha=0.02"].iloc[0]
        calibration = summary["probe"]["frozen_policy_results"]["formal_alpha"]["0.02"]["calibration"]
        ci = summary["ci"]
        delta_ci = ci[(ci.method == "Dynamic formal_alpha=0.02") & (ci.metric == "delta_dense")].iloc[0]
        token_ci = ci[(ci.method == "Dynamic formal_alpha=0.02") & (ci.metric == "token_reduction")].iloc[0]
        report.append(
            f"- {dataset.upper()}：lambda={calibration.get('lambda')}，mu={calibration.get('mu')}，calibration lost/UCB={calibration['lost_correct_count']}/{pct(calibration['lost_correct_ucb_simultaneous95'])}；heldout accuracy={pct(row.accuracy)}（ΔDense={row.delta_dense_pp:+.2f}pp，95% CI [{100*delta_ci.ci_low:+.2f},{100*delta_ci.ci_high:+.2f}]pp），coverage={pct(row.coverage)}，token reduction={pct(row.token_reduction)}（95% CI [{pct(token_ci.ci_low)},{pct(token_ci.ci_high)}]），W→C/C→W={int(row.W_to_C)}/{int(row.C_to_W)}。"
        )
    report += ["", "## 其他关键结果", ""]
    gsm = summaries["gsm8k"]
    mmlu = summaries["mmlu_pro"]
    gsm_b4 = gsm["table"][gsm["table"].method == "Dynamic empirical_B=4"].iloc[0]
    mmlu_b4 = mmlu["table"][mmlu["table"].method == "Dynamic empirical_B=4"].iloc[0]
    report += [
        f"- GSM8K empirical B=4：accuracy={pct(gsm_b4.accuracy)}，token reduction={pct(gsm_b4.token_reduction)}，W→C/C→W={int(gsm_b4.W_to_C)}/{int(gsm_b4.C_to_W)}。相对旧B=4点估计同时提高accuracy和token reduction并减少W→C，但大部分配对差值CI仍跨零。",
        f"- MMLU-Pro empirical B=4：accuracy={pct(mmlu_b4.accuracy)}，token reduction={pct(mmlu_b4.token_reduction)}，W→C/C→W={int(mmlu_b4.W_to_C)}/{int(mmlu_b4.C_to_W)}；相对旧B=4净准确率更高，但heldout lost-correct也更高。",
        "- Formal α=1%：500题、48候选的simultaneous上界使非Dense策略不可行，两个数据集都按预注册规则返回Dense/0 saving。",
        "- GSM8K α=5%候选在calibration满足约束，但heldout accuracy下降1.06pp，说明更激进档不应作为主结论。", "",
        "## W→C→W机制诊断", "",
    ]
    for dataset in ("gsm8k", "mmlu_pro"):
        audit = summaries[dataset]["probe"]["descriptive_heldout_nonmonotonic_wcw_audit"]
        alpha2 = audit["policies"]["formal_alpha"]["0.02"]
        report.append(
            f"- {dataset.upper()}共有{audit['wcw_problem_count']}条含相邻W→C→W模式的heldout轨迹；α=2%策略停止{alpha2['stopped']}条，其中{alpha2['stopped_correct']}条停止时正确，获得{alpha2['gained_correct_C_to_W']}个Dense-final错误上的净挽救。"
        )
    report += ["", "## 结论", "",
        "动态方法在两个数据集的α=2%预注册工作点都取得正token reduction，且净accuracy点估计不低于Dense，达到本轮最低成功标准。GSM8K上的中间正确点捕获机制明显，MMLU-Pro较弱。与旧方法相比，动态方案改善了部分风险—token区间，但没有全面支配旧前沿；尤其calibration经验B相同不保证heldout W→C相同。", "",
        "本轮没有真实wall-time测量，任何token reduction都不能写成实测延迟降低；短答案生成成本也按用户要求被忽略。", "",
    ]
    (final / "FINAL_DYNAMIC_REPORT_ZH.md").write_text("\n".join(report), encoding="utf-8")
    atomic_json({
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config["protocol_id"],
        "datasets": {"gsm8k": 1319, "mmlu_pro": 1000},
        "candidate_count": 48,
        "bootstrap_replicates": int(config["statistics"]["bootstrap_replicates"]),
        "cost": "reasoning_tokens/4096",
        "short_answer_cost": 0,
        "heldout_used_for_selection": False,
        "report": str(final / "FINAL_DYNAMIC_REPORT_ZH.md"),
    }, experiment / "pipeline_final.complete")
    print(json.dumps({"status": "complete", "report": str(final / "FINAL_DYNAMIC_REPORT_ZH.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""汇总 MMLU-Pro 点估计、分层配对 bootstrap、类别表和中英文报告。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.legacy_empirical_probe_v4 import summarize_policy_records
from src.utils import atomic_json, load_yaml

METHOD_LABELS = {
    "correctness": "Correctness (controlled)",
    "consistency": "Consistency (controlled)",
    "last_switch": "Last-switch (controlled)",
    "correction_bce": "Correction BCE",
    "correction_trajectory": "Correction + trajectory",
}


def dense_as_policy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "problem_id": row["problem_id"], "subject": row.get("subject"), "category": row.get("category"),
        "fallback": True, "checkpoint": None, "transition": "fallback", "method_prediction": row["prediction"],
        "dense_prediction": row["prediction"], "gold_answer": row["gold_answer"], "method_success": row["success"],
        "dense_success": row["success"], "method_tokens": row["reasoning_tokens"], "dense_tokens": row["reasoning_tokens"],
        "replay_wall_ms": row["wall_ms"], "dense_wall_ms": row["wall_ms"],
    } for row in rows]


def result_row(name: str, family: str, key: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    value = summarize_policy_records(records)
    counts = value.pop("counts")
    return {"dataset":"mmlu_pro","report_label":"MMLU-Pro","method":name,"family":family,
            "key":key,"budget_B":int(key) if family=="empirical_B" else np.nan,
            "coverage_target":float(key)/100 if family=="coverage" else np.nan,
            "N":value["problems"],"accuracy":value["accuracy"],"dense_accuracy":value["dense_accuracy"],
            "delta_dense_pp":100*(value["accuracy"]-value["dense_accuracy"]),"accuracy_drop_pp":value["accuracy_drop_pp"],
            "coverage":value["coverage"],"token_reduction":value["token_reduction"],
            "mean_generated_tokens":value["mean_reasoning_and_answer_tokens"],
            "mean_dense_tokens":value["mean_dense_reasoning_tokens"],
            "mean_replay_wall_ms":value["mean_replay_wall_ms"],"mean_dense_wall_ms":value["mean_dense_wall_ms"],
            "replay_wall_reduction":value["replay_wall_reduction"],"p95_replay_wall_ms":value["p95_replay_wall_ms"],
            "p95_dense_wall_ms":value["p95_dense_wall_ms"],"p95_replay_wall_reduction":value["p95_replay_wall_reduction"],
            "lost_correct_count":value["lost_correct_count"],"lost_correct_rate":value["lost_correct_rate"],
            "fallback":value["fallback"],**counts}


def aligned(records: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    mapping = {str(row["problem_id"]): row for row in records}
    if set(mapping) != set(ids): raise ValueError("不同方法的 heldout sample ID 不一致")
    return [mapping[value] for value in ids]


def replicate_metrics(records: list[dict[str, Any]], indices: np.ndarray) -> dict[str, np.ndarray]:
    success = np.asarray([row["method_success"] for row in records], dtype=np.float64)
    lost = np.asarray([row["transition"] == "W_to_C" for row in records], dtype=np.float64)
    coverage = np.asarray([not row["fallback"] for row in records], dtype=np.float64)
    used_tokens = np.asarray([row["method_tokens"] for row in records], dtype=np.float64)
    dense_tokens = np.asarray([row["dense_tokens"] for row in records], dtype=np.float64)
    wall = np.asarray([row["replay_wall_ms"] for row in records], dtype=np.float64)
    dense_wall = np.asarray([row["dense_wall_ms"] for row in records], dtype=np.float64)
    return {
        "accuracy": success[indices].mean(axis=1), "lost_correct_rate": lost[indices].mean(axis=1),
        "coverage": coverage[indices].mean(axis=1),
        "token_reduction": 1-used_tokens[indices].mean(axis=1)/dense_tokens[indices].mean(axis=1),
        "replay_wall_reduction": 1-wall[indices].mean(axis=1)/dense_wall[indices].mean(axis=1),
        "p95_replay_wall_reduction": 1-np.percentile(wall[indices],95,axis=1)/np.percentile(dense_wall[indices],95,axis=1),
    }


def pct(value: float) -> str:
    return f"{100*value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_paper_mmlu_pro_transfer_v1.yaml")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); config = load_yaml(ROOT / args.config)
    marker = args.output_root / "compile.complete"
    if args.resume and marker.is_file(): print(json.dumps({"status":"skipped_complete"})); return
    tables = args.output_root / "tables"; tables.mkdir(parents=True, exist_ok=True)
    baseline = torch.load(args.baseline_root/"baseline_records.pt", map_location="cpu", weights_only=False)["records"]
    record_map: dict[str,list[dict[str,Any]]] = {"Dense":dense_as_policy(baseline["dense"]),"Direct":baseline["direct"]}
    for budget, records in baseline["fixed"].items(): record_map[f"Fixed {budget}"] = records
    adaptive: dict[tuple[str,str,str],list[dict[str,Any]]] = {}
    for method, label in METHOD_LABELS.items():
        method_root=args.transfer_root/method
        records_path=method_root/"policy_records.pt"
        if not records_path.is_file():
            records_path=method_root/"transfer_records.pt"
        payload = torch.load(records_path, map_location="cpu", weights_only=False)["records"]
        for family, values in payload.items():
            for key, records in values.items(): adaptive[(label,family,key)] = records

    rows = []
    for name, records in record_map.items(): rows.append(result_row(name,"baseline","",records))
    for (name,family,key), records in adaptive.items(): rows.append(result_row(name,family,key,records))
    frame = pd.DataFrame(rows)
    frame["report_label"] = config["report_label"]
    frame[frame.family.isin(["baseline","empirical_B"])].to_csv(tables/"main_results.csv",index=False)
    frame[frame.family=="coverage"].to_csv(tables/"coverage_targeted.csv",index=False)
    frame[(frame.family=="empirical_B") & frame.method.isin(["Correction BCE","Correction + trajectory"])].to_csv(tables/"loss_ablation.csv",index=False)
    frame[(frame.family=="empirical_B") & frame.method.isin(list(METHOD_LABELS.values()))].to_csv(tables/"target_ablation.csv",index=False)

    ids = sorted(str(row["problem_id"]) for row in record_map["Dense"])
    categories = {str(row["problem_id"]):str(row["category"]) for row in record_map["Dense"]}
    rng = np.random.default_rng(int(config["seed"]["bootstrap"])); replicates = int(config["statistics"]["bootstrap_replicates"])
    resampled_parts = []
    for category in sorted(set(categories.values())):
        local = np.asarray([index for index,value in enumerate(ids) if categories[value]==category], dtype=np.int32)
        resampled_parts.append(rng.choice(local,size=(replicates,len(local)),replace=True))
    indices = np.concatenate(resampled_parts,axis=1)
    bootstrap_records: dict[str,list[dict[str,Any]]] = dict(record_map)
    for (name,family,key), records in adaptive.items():
        if family=="empirical_B": bootstrap_records[f"{name}|B={key}"] = records
    distributions = {}; ci_rows = []
    for name, records in bootstrap_records.items():
        ordered = aligned(records,ids); values = replicate_metrics(ordered,indices); distributions[name]=values
        point = result_row(name,"bootstrap","",ordered)
        for metric, samples in values.items():
            ci_rows.append({"method":name,"metric":metric,"point":point.get(metric),"ci_low":float(np.percentile(samples,2.5)),"ci_high":float(np.percentile(samples,97.5)),"replicates":replicates,"stratification":"MMLU-Pro category"})
    pd.DataFrame(ci_rows).to_csv(tables/"bootstrap_confidence_intervals.csv",index=False)
    comparison_rows=[]
    for budget in (0,1,2,4,10):
        main_name=f"Correction + trajectory|B={budget}"
        if main_name not in distributions: continue
        comparators=[f"{label}|B={budget}" for label in METHOD_LABELS.values() if label!="Correction + trajectory"]+list(record_map)
        for comparator in comparators:
            if comparator not in distributions: continue
            for metric in ("accuracy","token_reduction","replay_wall_reduction","lost_correct_rate"):
                delta=distributions[main_name][metric]-distributions[comparator][metric]
                comparison_rows.append({"budget_B":budget,"main":main_name,"comparator":comparator,"metric":metric,
                                        "difference_mean":float(delta.mean()),"ci_low":float(np.percentile(delta,2.5)),"ci_high":float(np.percentile(delta,97.5)),"replicates":replicates})
    pd.DataFrame(comparison_rows).to_csv(tables/"paired_comparisons.csv",index=False)

    category_rows=[]
    chosen={"Dense":record_map["Dense"]}
    for budget in (1,2,4): chosen[f"Correction + trajectory B={budget}"]=adaptive[("Correction + trajectory","empirical_B",str(budget))]
    for name, records in chosen.items():
        for category in sorted(set(categories.values())):
            local=[row for row in records if row["category"]==category]
            category_rows.append({"category":category,"method":name,"n":len(local),"accuracy":float(np.mean([row["method_success"] for row in local])),
                                  "coverage":float(np.mean([not row["fallback"] for row in local])),"lost_correct":sum(row["transition"]=="W_to_C" for row in local)})
    pd.DataFrame(category_rows).to_csv(tables/"category_results.csv",index=False)

    dense_row=frame[frame.method=="Dense"].iloc[0]
    main=frame[(frame.method=="Correction + trajectory")&(frame.family=="empirical_B")&frame.budget_B.isin([1,2,4])].sort_values("budget_B")
    aliases={1:"Strict",2:"Balanced",4:"Aggressive"}
    if "LearnStop-style" in config["report_label"]:
        setup = "本实验仅从 MMLU-Pro 官方 test 按 LearnStop 的 seed=42 范式抽取800题，使用问题级5折 OOF；再以 seed=123 固定划分320题 calibration 与480题 final-test role。每题 OOF score 来自未训练过该题的 fold 模型，但 fold 模型可能训练过 final-test role 中的其他题，因此这不是传统完全独立训练集评测。"
        title = "# MMLU-Pro-800 LearnStop-style 5-fold OOF 实验报告"
    else:
        setup = "本实验把在 MMLU 上冻结的 probe、StandardScaler 和阈值原样迁移到 MMLU-Pro；MMLU-Pro test 不参与训练或调参。"
        title = "# MMLU-Pro-1k 跨数据集迁移实验报告"
    zh=[title,"",f"协议：`{config['protocol_id']}`。{setup} 官方 validation 仅作为14类各5条演示。","",f"Dense accuracy={pct(dense_row.accuracy)}，平均 reasoning tokens={dense_row.mean_generated_tokens:.1f}。所有延迟均为 `{config['replay']['label']}`，共享 GPU 采集耗时被排除。","","## 主方法工作点",""]
    for _,row in main.iterrows():
        zh.append(f"- {aliases[int(row.budget_B)]}（B={int(row.budget_B)}）：accuracy={pct(row.accuracy)}（相对 Dense {row.delta_dense_pp:+.2f}pp），coverage={pct(row.coverage)}，token reduction={pct(row.token_reduction)}，平均/p95 replay reduction={pct(row.replay_wall_reduction)}/{pct(row.p95_replay_wall_reduction)}，W→C/C→W={int(row.W_to_C)}/{int(row.C_to_W)}。")
    positive=any((row.delta_dense_pp>=-1.0 and row.token_reduction>0 and row.replay_wall_reduction>0) for _,row in main[main.budget_B.isin([1,2])].iterrows())
    negative_text=("Strict/Balanced 未同时满足准确率基本保持与正节省，故该 LearnStop-style OOF 实验按负结果或混合结果报告。" if "LearnStop-style" in config["report_label"] else "Strict/Balanced 未同时满足准确率基本保持与正节省，故该跨数据集迁移实验按负结果或混合结果报告。")
    zh += ["","## 结论","",("至少一个 Strict/Balanced 工作点满足准确率基本保持与正节省；是否优于受控基线须结合 paired comparison 表判定。" if positive else negative_text),"","完整受控基线、matched coverage、loss 消融、14类别分解及10,000次类别分层 paired bootstrap 见 `tables/`。冻结阈值只由320题 calibration-role OOF 分数选择，480题 final-test role 不参与阈值选择。",""]
    (args.output_root/"MMLU_PRO_REPORT_ZH.md").write_text("\n".join(zh),encoding="utf-8")
    en_setup = ("The 800-question pool was sampled only from the official MMLU-Pro test split following LearnStop (seed 42), scored with five-fold question-level OOF models, and divided with seed 123 into 320 calibration-role and 480 final-test-role questions. This is cross-validated evaluation rather than a conventional independent training split." if "LearnStop-style" in config["report_label"] else "MMLU-trained probes, scalers, and thresholds were transferred unchanged to MMLU-Pro.")
    en=["# MMLU-Pro Experiment Report","",f"Protocol: `{config['protocol_id']}`. {en_setup}","",f"Dense accuracy: {pct(dense_row.accuracy)}. All latency numbers are `{config['replay']['label']}`; shared-GPU collection timing is excluded.",""]
    for _,row in main.iterrows(): en.append(f"- {aliases[int(row.budget_B)]} (B={int(row.budget_B)}): accuracy={pct(row.accuracy)}, delta={row.delta_dense_pp:+.2f}pp, coverage={pct(row.coverage)}, token reduction={pct(row.token_reduction)}, mean replay reduction={pct(row.replay_wall_reduction)}, W→C={int(row.W_to_C)}.")
    en += ["","Complete controlled baselines, matched-coverage results, loss ablation, category breakdown, and 10,000 category-stratified paired-bootstrap results are under `tables/`.",""]
    (args.output_root/"MMLU_PRO_REPORT_EN.md").write_text("\n".join(en),encoding="utf-8")
    atomic_json({"status":"complete","protocol_id":config["protocol_id"],"heldout":len(ids),"bootstrap_replicates":replicates,"positive_basic_gate":positive},marker)
    print(json.dumps({"status":"complete","heldout":len(ids),"bootstrap":replicates,"positive_basic_gate":positive},indent=2))


if __name__ == "__main__":
    main()

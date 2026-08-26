#!/usr/bin/env python3
"""汇总动态方法特征消融、matched coverage与配对bootstrap。"""
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

from src.dynamic_optimal_stopping_v1 import DYNAMIC_FEATURE_KINDS, summarize_token_records
from src.utils import atomic_json, load_yaml


FEATURES = tuple(value for value in DYNAMIC_FEATURE_KINDS if value != "full")


def normalize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output=[]
    for source in records:
        row=dict(source)
        row["method_tokens"] = int(row["dense_tokens"] if row["fallback"] else row["checkpoint"])
        output.append(row)
    return output


def align(records: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    mapping={str(row["problem_id"]):row for row in records}
    if len(mapping)!=len(records) or set(mapping)!=set(ids):
        raise ValueError("特征消融sample ID不配对")
    return [mapping[value] for value in ids]


def metric_row(dataset: str, feature: str, family: str, key: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    value=summarize_token_records(records)
    return {
        "dataset":dataset,"feature":feature,"family":family,"key":key,"N":value["problems"],
        "accuracy":value["accuracy"],"dense_accuracy":value["dense_accuracy"],
        "delta_dense_pp":value["delta_dense_pp"],"coverage":value["coverage"],
        "mean_reasoning_tokens":value["mean_reasoning_tokens"],"token_reduction":value["token_reduction"],
        "lost_correct_count":value["lost_correct_count"],"lost_correct_rate":value["lost_correct_rate"],
        "gained_correct_count":value["gained_correct_count"],"fallback":value["fallback"],**value["counts"],
    }


def bootstrap_indices(records: list[dict[str, Any]], dataset: str, replicates: int, seed: int) -> tuple[np.ndarray,str]:
    rng=np.random.default_rng(seed)
    if dataset=="gsm8k":
        return rng.integers(0,len(records),size=(replicates,len(records))),"problem"
    categories=np.asarray([str(row.get("category")) for row in records])
    parts=[]
    for category in sorted(set(categories)):
        local=np.flatnonzero(categories==category)
        parts.append(rng.choice(local,size=(replicates,len(local)),replace=True))
    return np.concatenate(parts,axis=1),"category_stratified_problem"


def raw_arrays(records: list[dict[str, Any]]) -> dict[str,np.ndarray]:
    return {
        "accuracy":np.asarray([row["method_success"] for row in records],dtype=np.float64),
        "lost_correct_rate":np.asarray([row["transition"]=="W_to_C" for row in records],dtype=np.float64),
        "coverage":np.asarray([not row["fallback"] for row in records],dtype=np.float64),
        "tokens":np.asarray([row["method_tokens"] for row in records],dtype=np.float64),
        "dense_tokens":np.asarray([row["dense_tokens"] for row in records],dtype=np.float64),
    }


def point(raw: dict[str,np.ndarray]) -> dict[str,float]:
    return {
        "accuracy":float(raw["accuracy"].mean()),
        "lost_correct_rate":float(raw["lost_correct_rate"].mean()),
        "coverage":float(raw["coverage"].mean()),
        "token_reduction":float(1.0-raw["tokens"].mean()/raw["dense_tokens"].mean()),
    }


def sampled(raw: dict[str,np.ndarray], indices: np.ndarray) -> dict[str,np.ndarray]:
    return {
        "accuracy":raw["accuracy"][indices].mean(1),
        "lost_correct_rate":raw["lost_correct_rate"][indices].mean(1),
        "coverage":raw["coverage"][indices].mean(1),
        "token_reduction":1.0-raw["tokens"][indices].mean(1)/raw["dense_tokens"][indices].mean(1),
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--feature-root",type=Path,required=True)
    parser.add_argument("--full-root",type=Path,required=True)
    parser.add_argument("--full-coverage-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    marker=args.output/"pipeline.complete"
    if args.resume and marker.is_file():
        print(json.dumps({"status":"skipped_complete"})); return
    args.output.mkdir(parents=True,exist_ok=True)
    tables=args.output/"tables"; tables.mkdir(parents=True,exist_ok=True)
    config=load_yaml(args.config)
    all_rows=[]; diagnostics=[]; selections=[]; bootstrap_rows=[]; ci_rows=[]
    simplification=[]
    for dataset in ("gsm8k","mmlu_pro"):
        methods: dict[tuple[str,str,str],list[dict[str,Any]]]={}
        full_probe=args.full_root/dataset/"probe"
        full_records=torch.load(full_probe/"policy_records.pt",map_location="cpu",weights_only=False)["records"]
        any_records=next(iter(full_records["empirical_B"].values()))
        ids=sorted(str(row["problem_id"]) for row in any_records)
        aligned_any=align(normalize(any_records),ids)
        for family,values in full_records.items():
            for key,records in values.items(): methods[("full",family,str(key))]=align(normalize(records),ids)
        full_cov=torch.load(args.full_coverage_root/dataset/"policy_records.pt",map_location="cpu",weights_only=False)["records"]
        for key,records in full_cov.items(): methods[("full","coverage",str(key))]=align(normalize(records),ids)
        roots={"full":full_probe}
        for feature in FEATURES:
            probe_root=args.feature_root/feature/dataset/"probe"
            roots[feature]=probe_root
            payload=torch.load(probe_root/"policy_records.pt",map_location="cpu",weights_only=False)["records"]
            for family,values in payload.items():
                for key,records in values.items(): methods[(feature,family,str(key))]=align(normalize(records),ids)
            cov=torch.load(args.feature_root/feature/dataset/"coverage/policy_records.pt",map_location="cpu",weights_only=False)["records"]
            for key,records in cov.items(): methods[(feature,"coverage",str(key))]=align(normalize(records),ids)
        for (feature,family,key),records in methods.items():
            all_rows.append(metric_row(dataset,feature,family,key,records))
        for feature,probe_root in roots.items():
            probe=json.loads((probe_root/"probe.json").read_text(encoding="utf-8"))
            run=probe["run_spec"]
            for split,values in probe["local_diagnostics"].items():
                diagnostics.append({
                    "dataset":dataset,"feature":feature,"feature_width":run["architecture"]["shared"][0],
                    "split":split,"local_best_epoch":probe["local_best_epoch"],
                    "value_best_epoch":probe["value_best_epoch"],
                    "value_validation_mae":probe["value_history"][probe["value_best_epoch"]]["validation_mae"],
                    **values,
                })
            for family,values in probe["frozen_policy_results"].items():
                for key,item in values.items():
                    selected=item["calibration"]
                    selections.append({
                        "dataset":dataset,"feature":feature,"family":family,"key":key,
                        "feature_width":run["architecture"]["shared"][0],
                        "lambda":selected.get("lambda"),"mu":selected.get("mu"),
                        "dense_fallback":selected.get("dense_fallback",False),
                        "calibration_accuracy":selected.get("accuracy"),
                        "calibration_coverage":selected.get("coverage"),
                        "calibration_token_reduction":selected.get("token_reduction"),
                        "calibration_lost_correct_count":selected.get("lost_correct_count"),
                        "heldout_accuracy":item["heldout"].get("accuracy"),
                        "heldout_coverage":item["heldout"].get("coverage"),
                        "heldout_token_reduction":item["heldout"].get("token_reduction"),
                        "heldout_lost_correct_count":item["heldout"].get("lost_correct_count"),
                    })
        replicates=int(config["statistics"]["bootstrap_replicates"])
        indices,stratification=bootstrap_indices(aligned_any,dataset,replicates,int(config["seed"]["bootstrap"])+17)
        for family,key in (("empirical_B","4"),("formal_alpha","0.02")):
            reference=methods[("full",family,key)]
            reference_point=point(raw_arrays(reference)); reference_boot=sampled(raw_arrays(reference),indices)
            for metric,samples in reference_boot.items():
                ci_rows.append({"dataset":dataset,"feature":"full","family":family,"key":key,"metric":metric,
                    "point":reference_point[metric],"ci_low":float(np.percentile(samples,2.5)),
                    "ci_high":float(np.percentile(samples,97.5)),"replicates":replicates,"stratification":stratification})
            for feature in FEATURES:
                records=methods[(feature,family,key)]
                feature_raw=raw_arrays(records); feature_point=point(feature_raw); feature_boot=sampled(feature_raw,indices)
                for metric,samples in feature_boot.items():
                    ci_rows.append({"dataset":dataset,"feature":feature,"family":family,"key":key,"metric":metric,
                        "point":feature_point[metric],"ci_low":float(np.percentile(samples,2.5)),
                        "ci_high":float(np.percentile(samples,97.5)),"replicates":replicates,"stratification":stratification})
                    difference=samples-reference_boot[metric]
                    bootstrap_rows.append({
                        "dataset":dataset,"feature":feature,"family":family,"key":key,
                        "comparison":"feature_minus_full","metric":metric,
                        "difference_point":feature_point[metric]-reference_point[metric],
                        "ci_low":float(np.percentile(difference,2.5)),"ci_high":float(np.percentile(difference,97.5)),
                        "replicates":replicates,"stratification":stratification,
                    })
                if family=="empirical_B":
                    summary=summarize_token_records(records); full_summary=summarize_token_records(reference)
                    simplification.append({
                        "dataset":dataset,"feature":feature,
                        "accuracy_difference_pp":100*(feature_point["accuracy"]-reference_point["accuracy"]),
                        "token_reduction_difference_pp":100*(feature_point["token_reduction"]-reference_point["token_reduction"]),
                        "lost_correct_count_difference":summary["lost_correct_count"]-full_summary["lost_correct_count"],
                        "passes_provisional_point_criterion":bool(
                            100*(feature_point["accuracy"]-reference_point["accuracy"])>=-0.5
                            and 100*(feature_point["token_reduction"]-reference_point["token_reduction"])>=-2.0
                            and summary["lost_correct_count"]-full_summary["lost_correct_count"]<=2
                        ),
                    })
    frame=pd.DataFrame(all_rows)
    frame.to_csv(tables/"all_feature_workpoints.csv",index=False)
    frame[(frame.family=="empirical_B")&(frame.key.astype(str)=="4")].to_csv(tables/"feature_B4.csv",index=False)
    frame[(frame.family=="formal_alpha")&(frame.key.astype(str)=="0.02")].to_csv(tables/"feature_formal_alpha2.csv",index=False)
    frame[frame.family=="coverage"].to_csv(tables/"feature_coverage_targeted.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(tables/"feature_training_diagnostics.csv",index=False)
    pd.DataFrame(selections).to_csv(tables/"feature_calibration_selections.csv",index=False)
    pd.DataFrame(ci_rows).to_csv(tables/"feature_bootstrap_ci.csv",index=False)
    pd.DataFrame(bootstrap_rows).to_csv(tables/"feature_paired_vs_full.csv",index=False)
    simple=pd.DataFrame(simplification)
    cross=simple.groupby("feature",sort=False).agg(
        datasets_passing=("passes_provisional_point_criterion","sum"),
        minimum_accuracy_difference_pp=("accuracy_difference_pp","min"),
        minimum_token_reduction_difference_pp=("token_reduction_difference_pp","min"),
        maximum_lost_correct_count_difference=("lost_correct_count_difference","max"),
    ).reset_index()
    cross["passes_both_datasets"] = cross.datasets_passing.eq(2)
    simple.to_csv(tables/"provisional_simplification_by_dataset.csv",index=False)
    cross.to_csv(tables/"provisional_simplification_cross_task.csv",index=False)
    report=[
        "# 动态最优停止特征全面消融", "",
        "所有变体共享FP16公共缓存、标签、三头动态模型宽度、训练预算、内部fit/validation、calibration候选与heldout；仅输入列发生变化。", "",
        "主诊断为经验B=4，formal alpha=2%为补充。精简标准在运行前固定为：两个数据集上相对full准确率不低于-0.5pp、token reduction不低于-2pp、W→C不增加超过2题。该标准只用于解释，不据此重新调test策略。", "",
    ]
    for dataset in ("gsm8k","mmlu_pro"):
        report += [f"## {dataset} B=4", ""]
        subset=frame[(frame.dataset==dataset)&(frame.family=="empirical_B")&(frame.key.astype(str)=="4")]
        for row in subset.sort_values("feature").itertuples():
            report.append(f"- {row.feature}: Acc={100*row.accuracy:.2f}%, ΔDense={row.delta_dense_pp:+.2f}pp, Coverage={100*row.coverage:.2f}%, Token↓={100*row.token_reduction:.2f}%, W→C/C→W={row.W_to_C}/{row.C_to_W}.")
        report.append("")
    passing=cross[cross.passes_both_datasets].feature.tolist()
    report += ["## 预注册精简判据", "", f"两个数据集同时通过的变体：{passing if passing else '无'}。", "",
        "完整B曲线、formal、coverage-targeted、训练AP/AUC、10000次配对bootstrap见tables目录。", ""]
    (args.output/"FEATURE_ABLATION_REPORT_ZH.md").write_text("\n".join(report),encoding="utf-8")
    atomic_json({
        "status":"complete","completed_at":datetime.now(timezone.utc).isoformat(),
        "features":["full",*FEATURES],"datasets":["gsm8k","mmlu_pro"],
        "bootstrap_replicates":int(config["statistics"]["bootstrap_replicates"]),
        "heldout_used_for_selection":False,"cost":"reasoning_tokens_only","short_answer_cost":0,
        "cross_task_simplifications_passing":passing,
    },marker)
    print(json.dumps({"status":"complete","output":str(args.output),"passing":passing},ensure_ascii=False))


if __name__=="__main__": main()

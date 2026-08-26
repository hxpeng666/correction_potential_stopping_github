#!/usr/bin/env python3
"""精确复刻 LearnStop 的 MMLU-Pro test 抽样与 5-fold OOF / 40:60 划分范式。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.final_paper_protocol import canonical_fingerprint, normalize_question
from src.mmlu_pro_protocol import answer_letter, valid_letters
from src.utils import atomic_json, load_yaml


def record(raw:dict[str,Any],source_index:int,split:str) -> dict[str,Any]:
    choices=[str(value) for value in raw["options"] if value is not None]; valid_letters(len(choices))
    gold=answer_letter(raw["answer_index"],len(choices))
    if str(raw["answer"]).strip().upper()!=gold: raise ValueError(f"answer mismatch: {raw['question_id']}")
    value={"problem_id":f"mmlu_pro_test_{raw['question_id']}","question_id":str(raw["question_id"]),"source_index":source_index,
           "source_split":split,"source":str(raw.get("src","")),"category":str(raw["category"]),"subject":str(raw["category"]),
           "question":str(raw["question"]),"choices":choices,"answer":gold,"answer_index":int(raw["answer_index"]),"option_count":len(choices)}
    value["record_fingerprint"]=canonical_fingerprint(value); return value


def write_jsonl(path:Path,rows:list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w",encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    os.replace(temporary,path)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/final_paper_mmlu_pro_learnstop_style_v1.yaml")
    parser.add_argument("--resume",action="store_true"); args=parser.parse_args(); config=load_yaml(ROOT/args.config)
    output=ROOT/config["dataset"]["prepared_root"]; results=ROOT/config["output_root"]; manifest_path=results/"splits"/"mmlu_pro_learnstop_style_split.json"
    if args.resume and manifest_path.is_file() and (output/"heldout.jsonl").is_file():
        old=json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("status")=="frozen" and old.get("protocol_id")==config["protocol_id"]:
            print(json.dumps({"status":"skipped_frozen","manifest":str(manifest_path)})); return
        raise RuntimeError("已有划分不兼容")
    for path in (output/"heldout.jsonl",output/"demonstrations.jsonl",manifest_path):
        if path.exists(): raise RuntimeError(f"拒绝覆盖：{path}")
    dataset=load_dataset(config["dataset"]["name"]); validation=dataset["validation"]; test=dataset["test"]
    demos=[record(dict(raw),index,"validation") for index,raw in enumerate(validation)]
    demo_counts=Counter(row["category"] for row in demos)
    if len(demo_counts)!=14 or set(demo_counts.values())!={5}: raise ValueError("validation 不是14类各5条")
    valid=[]
    for index,raw in enumerate(test):
        try: valid.append(record(dict(raw),index,"test"))
        except (ValueError,TypeError,KeyError): continue
    n=int(config["dataset"]["pool_count"]); sample_seed=int(config["seed"]["dataset_sampling"])
    rng=np.random.default_rng(sample_seed); selected_indices=rng.choice(len(valid),size=n,replace=False)
    selected=[valid[int(index)] for index in selected_indices]
    if len({row["problem_id"] for row in selected})!=n: raise ValueError("抽样 ID 重复")
    demo_questions={normalize_question(row["question"]) for row in demos}
    overlaps=[row["problem_id"] for row in selected if normalize_question(row["question"]) in demo_questions]
    if overlaps: raise ValueError(f"validation/test 规范化问题重复：{overlaps[:5]}")
    folds=np.full(n,-1,dtype=int); groups=np.arange(n)
    for fold,(_train,test_positions) in enumerate(GroupKFold(n_splits=int(config["dataset"]["oof_folds"])).split(np.zeros((n,1)),groups=groups)):
        folds[test_positions]=fold
    split_rng=np.random.default_rng(int(config["seed"]["calibration_test_split"])); permutation=split_rng.permutation(n)
    n_cal=int(round(n*float(config["dataset"]["policy_calibration_fraction"])))
    calibration_positions=set(map(int,permutation[:n_cal])); final_positions=set(map(int,permutation[n_cal:]))
    for index,row in enumerate(selected):
        row["pool_position"]=index; row["oof_fold"]=int(folds[index]); row["policy_role"]="calibration" if index in calibration_positions else "final_test"
    write_jsonl(output/"demonstrations.jsonl",demos); write_jsonl(output/"heldout.jsonl",selected)
    smoke=[]
    for category in sorted(set(row["category"] for row in selected)):
        smoke.append(next(row for row in selected if row["category"]==category))
    write_jsonl(output/"smoke_heldout.jsonl",smoke)
    payload={"protocol_id":config["protocol_id"],"sampling_seed":sample_seed,"selected_source_indices":[row["source_index"] for row in selected],
             "selected_ids":[row["problem_id"] for row in selected],"folds":folds.tolist(),"calibration_positions":sorted(calibration_positions),"final_test_positions":sorted(final_positions)}
    manifest={"status":"frozen","protocol_id":config["protocol_id"],"report_label":config["report_label"],
              "dataset":{"name":config["dataset"]["name"],"validation_rows":len(validation),"validation_fingerprint":getattr(validation,"_fingerprint",None),
                         "test_rows":len(test),"test_fingerprint":getattr(test,"_fingerprint",None)},
              "learnstop_sampling":{"source_split":"test","eligible_rows":len(valid),"n":n,"seed":sample_seed,"algorithm":"numpy.default_rng(seed).choice(len(rows),size=n,replace=False)",
                                    "selected_source_indices":[row["source_index"] for row in selected]},
              "oof":{"folds":5,"algorithm":"sklearn GroupKFold(no shuffle) over pool positions","fold_counts":dict(sorted(Counter(map(str,folds)).items()))},
              "policy_split":{"seed":int(config["seed"]["calibration_test_split"]),"algorithm":"numpy.default_rng(seed).permutation(N)",
                              "calibration_count":len(calibration_positions),"final_test_count":len(final_positions),
                              "calibration_ids":[selected[i]["problem_id"] for i in sorted(calibration_positions)],
                              "final_test_ids":[selected[i]["problem_id"] for i in sorted(final_positions)]},
              "category_counts":{"pool":dict(sorted(Counter(row["category"] for row in selected).items())),
                                 "calibration":dict(sorted(Counter(selected[i]["category"] for i in calibration_positions).items())),
                                 "final_test":dict(sorted(Counter(selected[i]["category"] for i in final_positions).items()))},
              "demonstrations":{"source":"official_validation","count":len(demos),"per_category":dict(sorted(demo_counts.items())),"used_for_probe_training":False},
              "fingerprint":canonical_fingerprint(payload),
              "evaluation_warning":"Each question receives an out-of-fold score, but an OOF fold model can train on other questions assigned to the 480-question final-test role; this matches LearnStop and is not a conventional fully independent train/test split."}
    atomic_json(manifest,manifest_path); print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=="__main__": main()

#!/usr/bin/env python3
"""审计 MMLU-Pro 公共缓存完整性、指纹与 held-out 隔离。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.final_paper_inference import read_jsonl
from src.mmlu_pro_protocol import valid_letters
from src.utils import atomic_json, load_yaml


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/final_paper_mmlu_pro_transfer_v1.yaml")
    parser.add_argument("--replay-root",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args(); config=load_yaml(ROOT/args.config)
    prepared=ROOT/config["dataset"]["prepared_root"]
    expected={row["problem_id"]:row for row in read_jsonl(prepared/"heldout.jsonl")}
    demos=read_jsonl(prepared/"demonstrations.jsonl"); demo_ids={row["problem_id"] for row in demos}
    paths=sorted((args.replay_root/"heldout").glob("sample_*.pt")); seen=set(); fingerprints=set(); errors=[]; category=Counter(); checkpoints=0; missing_answers=0
    if args.smoke:
        observed_ids={path.stem.removeprefix("sample_") for path in paths}
        expected={pid:row for pid,row in expected.items() if pid in observed_ids}
    for path in paths:
        value=torch.load(path,map_location="cpu",weights_only=False); pid=str(value.get("problem_id"))
        if pid in seen: errors.append(f"duplicate:{pid}")
        seen.add(pid); fingerprints.add(str(value.get("protocol_fingerprint"))); category[str(value["record"].get("category"))]+=1
        if value.get("status")!="complete" or value.get("dtype")!="float16" or value.get("attention_backend")!="sdpa": errors.append(f"identity:{pid}")
        if value["dense"]["reasoning_tokens"]!=len(value["dense"]["tokens"]): errors.append(f"dense_length:{pid}")
        if len(value["rows"])!=int(value["hidden"].shape[0]): errors.append(f"row_hidden:{pid}")
        if not torch.isfinite(value["hidden"]).all(): errors.append(f"hidden_nonfinite:{pid}")
        positions=[int(row["checkpoint"]) for row in value["rows"]]; checkpoints+=len(positions)
        if len(positions)!=len(set(positions)) or positions!=sorted(positions): errors.append(f"checkpoint_duplicate_or_order:{pid}")
        sentence=[int(x) for x in value["schedules"]["sentence"]]
        if any(x<64 or x>768 for x in sentence) or any(b-a<8 for a,b in zip(sentence,sentence[1:])): errors.append(f"sentence_schedule:{pid}")
        allowed=set(valid_letters(int(value["record"]["option_count"])))
        for answer in [value["gold_answer"],value["dense"]["prediction"],value["direct"]["prediction"]]+[row["current_prediction"] for row in value["rows"]]:
            if answer is not None and answer not in allowed: errors.append(f"invalid_answer:{pid}:{answer}")
            missing_answers+=int(answer is None)
    missing=sorted(set(expected)-seen); unexpected=sorted(seen-set(expected))
    if missing: errors.append(f"missing_samples:{len(missing)}")
    if unexpected: errors.append(f"unexpected_samples:{len(unexpected)}")
    learnstop_style="learnstop" in str(config.get("status","")).lower() or int(config["dataset"].get("oof_folds",0))>1
    calibration_count=int(config["dataset"].get("policy_calibration_count",500))
    final_test_count=int(config["dataset"].get("final_test_count",len(expected)-calibration_count))
    report={"status":"passed" if not errors else "failed","protocol_id":config["protocol_id"],"expected":len(expected),"observed":len(paths),
            "unique_sample_ids":len(seen),"missing_sample_ids":missing,"unexpected_sample_ids":unexpected,"errors":errors,"protocol_fingerprints":sorted(fingerprints),
            "category_counts":dict(sorted(category.items())),"checkpoints":checkpoints,"missing_answer_fields":missing_answers,
            "heldout_disjoint_from_demonstration_ids":not bool(set(expected)&demo_ids),
            "source_split":config["dataset"].get("source_split","test"),
            "evaluation_design":"LearnStop-style grouped 5-fold OOF" if learnstop_style else "independent train/calibration/heldout",
            "official_test_pool_used_for_cross_fold_probe_training":learnstop_style,
            "each_problem_scored_by_model_not_trained_on_that_problem":learnstop_style,
            "policy_threshold_uses_final_test_role":False,
            "policy_calibration_count":calibration_count,
            "final_report_count":final_test_count,
            "threshold_source":f"frozen OOF policy calibration {calibration_count}" if learnstop_style else f"frozen MMLU calibration {calibration_count}",
            "latency_source":"frozen A100 single-request cost model"}
    atomic_json(report,args.output); print(json.dumps(report,indent=2))
    if errors: raise SystemExit(3)


if __name__=="__main__": main()

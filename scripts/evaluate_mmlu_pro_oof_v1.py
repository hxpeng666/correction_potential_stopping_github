#!/usr/bin/env python3
"""组装五折 OOF 分数，在320 calibration选择阈值并在480 final-role题上评测。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.final_paper_inference import atomic_torch_save, read_jsonl
from src.legacy_empirical_probe_v4 import calibrate_policies, load_checkpoint_split, method_direction, safe_ap_auc, simulate_policy, target_values
from src.utils import atomic_json, load_yaml

METHODS={"correctness":("correctness","bce"),"consistency":("consistency","bce"),"last_switch":("last_switch","bce"),"correction_bce":("correction","bce"),"correction_trajectory":("correction","bce_traj")}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/final_paper_mmlu_pro_learnstop_style_v1.yaml")
    parser.add_argument("--replay-root",type=Path,required=True); parser.add_argument("--probe-root",type=Path,required=True); parser.add_argument("--output-root",type=Path,required=True); parser.add_argument("--resume",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args(); config=load_yaml(ROOT/args.config); marker=args.output_root/"evaluation.complete"
    if args.resume and marker.is_file(): print(json.dumps({"status":"skipped_complete"})); return
    if args.output_root.exists() and any(args.output_root.iterdir()): raise RuntimeError(f"拒绝覆盖：{args.output_root}")
    args.output_root.mkdir(parents=True,exist_ok=True)
    frame,_hidden,_layers,fallbacks=load_checkpoint_split(args.replay_root/"heldout","sentence"); del _hidden
    records={row["problem_id"]:row for row in read_jsonl(ROOT/config["dataset"]["prepared_root"]/"heldout.jsonl")}
    observed=set(frame.problem_id.astype(str))|{row["problem_id"] for row in fallbacks}
    if args.smoke:
        records={pid:records[pid] for pid in observed}
    elif set(records)!=observed:
        raise ValueError("正式 pool ID 不完整")
    calibration_ids={pid for pid,row in records.items() if row["policy_role"]=="calibration"}; final_ids=set(records)-calibration_ids
    cal_mask=frame.problem_id.astype(str).isin(calibration_ids).to_numpy(); test_mask=frame.problem_id.astype(str).isin(final_ids).to_numpy()
    cal_frame=frame.loc[cal_mask].reset_index(drop=True); test_frame=frame.loc[test_mask].reset_index(drop=True)
    cal_fallbacks=[row for row in fallbacks if row["problem_id"] in calibration_ids]; test_fallbacks=[row for row in fallbacks if row["problem_id"] in final_ids]
    summary={"status":"complete","created_at":datetime.now(timezone.utc).isoformat(),"protocol_id":config["protocol_id"],"pool_problems":len(records),
             "calibration_problems":int(cal_frame.problem_id.nunique())+len(cal_fallbacks),"final_test_problems":int(test_frame.problem_id.nunique())+len(test_fallbacks),"methods":{}}
    for output_name,(method,loss) in METHODS.items():
        score_map={}
        fold_fallbacks=set()
        for fold in range(5):
            path=args.probe_root/output_name/f"fold_{fold}"/"oof_scores.pt"; value=torch.load(path,map_location="cpu",weights_only=False)
            for pid,checkpoint,score in zip(value["problem_ids"],value["checkpoints"],value["scores"].tolist()):
                key=(str(pid),int(checkpoint))
                if key in score_map: raise ValueError(f"重复 OOF row：{key}")
                score_map[key]=float(score)
            fold_fallbacks.update(map(str,value["fallback_problem_ids"]))
        keys=list(zip(frame.problem_id.astype(str),frame.checkpoint.astype(int)))
        if set(keys)!=set(score_map): raise ValueError(f"{output_name} OOF row不完整：frame={len(keys)} score={len(score_map)}")
        if fold_fallbacks!={row["problem_id"] for row in fallbacks}: raise ValueError(f"{output_name} fallback OOF 不完整")
        scores=np.asarray([score_map[key] for key in keys],dtype=np.float64); cal_scores=scores[cal_mask]; test_scores=scores[test_mask]; direction=method_direction(method)
        calibrated=calibrate_policies(cal_frame,cal_scores,direction,grid_size=int(config["calibration"]["threshold_quantiles"]),
                                      empirical_budgets=[int(x) for x in config["calibration"]["empirical_lost_correct_B"]],
                                      coverage_targets=[float(x) for x in config["calibration"]["coverage_targets"]],fallback_records=cal_fallbacks)
        results={}; policy_records={}
        for family in ("empirical_B","coverage"):
            results[family]={}; policy_records[family]={}
            for key,frozen in calibrated[family].items():
                evaluated=simulate_policy(test_frame,test_scores,direction,float(frozen["threshold"]),include_records=True,fallback_records=test_fallbacks,force_dense=bool(frozen.get("is_no_stop_sentinel",False)))
                local=evaluated.pop("records"); results[family][key]={"calibration":frozen,"final_test":evaluated}; policy_records[family][key]=local
        cal_labels=target_values(cal_frame,method); test_labels=target_values(test_frame,method); cal_ap,cal_auc=safe_ap_auc(cal_labels,cal_scores); test_ap,test_auc=safe_ap_auc(test_labels,test_scores)
        payload={"status":"complete","method":output_name,"target":method,"loss":loss,"direction":direction,"oof_folds":5,"calibration_count":len(calibration_ids),"final_test_count":len(final_ids),
                 "calibration_label_AP":cal_ap,"calibration_label_AUC":cal_auc,"final_test_label_AP_descriptive":test_ap,"final_test_label_AUC_descriptive":test_auc,
                 "threshold_selected_only_on_calibration_role":True,"results":results}
        destination=args.output_root/output_name; destination.mkdir(parents=True,exist_ok=True); atomic_json(payload,destination/"evaluation.json")
        atomic_torch_save({"status":"complete","records":policy_records,"calibration":calibrated},destination/"policy_records.pt")
        summary["methods"][output_name]=payload
    atomic_json(summary,marker); print(json.dumps({"status":"complete","calibration":summary["calibration_problems"],"final_test":summary["final_test_problems"],"methods":list(summary["methods"])},indent=2))


if __name__=="__main__": main()

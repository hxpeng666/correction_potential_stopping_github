#!/usr/bin/env python3
"""训练一个 MMLU-Pro LearnStop-style 外层 OOF fold 的 stopper。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.final_paper_inference import atomic_torch_save, read_jsonl
from src.final_paper_protocol import canonical_fingerprint
from src.legacy_empirical_probe_v4 import (
    FinalPaperProbe, binary_point_loss, build_features, correction_loss, fit_validation_masks,
    fit_validation_problem_ids, load_checkpoint_split, method_direction, predict_scores,
    problem_batches, safe_ap_auc, select_empirical_budget, simulate_policy, target_values, threshold_grid,
)
from src.utils import atomic_json, load_yaml, seed_everything

METHODS=("correctness","consistency","last_switch","correction")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/final_paper_mmlu_pro_learnstop_style_v1.yaml")
    parser.add_argument("--replay-root",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--method",choices=METHODS,required=True); parser.add_argument("--loss",choices=("bce","bce_traj"),default="bce")
    parser.add_argument("--fold",type=int,choices=range(5),required=True); parser.add_argument("--gpu",type=int,required=True); parser.add_argument("--resume",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args(); config=load_yaml(ROOT/args.config)
    if args.method!="correction" and args.loss!="bce": raise ValueError("受控目标只使用 BCE")
    destination=args.output; marker=destination/"phase.complete"
    invocation={"protocol_id":config["protocol_id"],"method":args.method,"loss":args.loss,"fold":args.fold,"replay_root":str(args.replay_root.resolve()),"smoke":args.smoke}
    fingerprint=canonical_fingerprint(invocation)
    if args.resume and marker.is_file():
        old=json.loads(marker.read_text(encoding="utf-8"))
        if old.get("invocation_fingerprint")==fingerprint and (destination/"oof_scores.pt").is_file(): print(json.dumps({"status":"skipped_complete","output":str(destination)})); return
        raise RuntimeError("拒绝 resume 不同指纹")
    if destination.exists() and any(destination.iterdir()): raise RuntimeError(f"拒绝覆盖：{destination}")
    destination.mkdir(parents=True,exist_ok=True); seed_everything(0); torch.cuda.set_device(args.gpu); device=torch.device(f"cuda:{args.gpu}")
    torch.set_num_threads(min(8,torch.get_num_threads()))
    frame,hidden,layers,fallbacks=load_checkpoint_split(args.replay_root/"heldout","sentence")
    records={row["problem_id"]:row for row in read_jsonl(ROOT/config["dataset"]["prepared_root"]/"heldout.jsonl")}
    observed=set(frame.problem_id.astype(str))|{row["problem_id"] for row in fallbacks}
    if args.smoke:
        records={pid:records[pid] for pid in observed}
    expected=set(records)
    if expected!=observed: raise ValueError(f"pool ID mismatch: expected={len(expected)} observed={len(observed)}")
    outer_test_ids={pid for pid,row in records.items() if int(row["oof_fold"])==args.fold}; outer_train_ids=expected-outer_test_ids
    if args.smoke:
        if not outer_test_ids or len(outer_train_ids)<4: raise ValueError("smoke outer fold 过小")
    elif len(outer_test_ids)!=160 or len(outer_train_ids)!=640: raise ValueError("outer fold 必须是160/640")
    train_mask=frame.problem_id.astype(str).isin(outer_train_ids).to_numpy(); test_mask=frame.problem_id.astype(str).isin(outer_test_ids).to_numpy()
    train_frame=frame.loc[train_mask].reset_index(drop=True); test_frame=frame.loc[test_mask].reset_index(drop=True)
    train_hidden=hidden[train_mask]; test_hidden=hidden[test_mask]; del hidden
    train_fallbacks=[row for row in fallbacks if row["problem_id"] in outer_train_ids]; test_fallbacks=[row for row in fallbacks if row["problem_id"] in outer_test_ids]
    raw_train=build_features(train_frame,train_hidden,layers,layer=20,feature_kind="full"); raw_test=build_features(test_frame,test_hidden,layers,layer=20,feature_kind="full")
    del train_hidden,test_hidden
    fit_ids,validation_ids=fit_validation_problem_ids(train_frame,"mmlu_pro",seed=0,additional_problem_ids=[row["problem_id"] for row in train_fallbacks])
    fit_mask,validation_mask=fit_validation_masks(train_frame,"mmlu_pro",seed=0,additional_problem_ids=[row["problem_id"] for row in train_fallbacks])
    validation_fallbacks=[row for row in train_fallbacks if row["problem_id"] in validation_ids]
    scaler=StandardScaler(copy=False); scaler.fit(raw_train[fit_mask]); train_features=scaler.transform(raw_train).astype(np.float32,copy=False); test_features=scaler.transform(raw_test).astype(np.float32,copy=False)
    del raw_train,raw_test
    train_labels=target_values(train_frame,args.method); test_labels=target_values(test_frame,args.method)
    positives=float(train_labels[fit_mask].sum()); negatives=float(fit_mask.sum())-positives
    positive_weight=torch.tensor(negatives/positives if positives>0 else 1.0,dtype=torch.float32,device=device)
    remaining=np.clip((train_frame.dense_tokens.to_numpy(np.float32)-train_frame.checkpoint.to_numpy(np.float32))/np.maximum(train_frame.dense_tokens.to_numpy(np.float32),1),0,1).astype(np.float32)
    model=FinalPaperProbe(5126).to(device); probe=config["probe"]
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(probe["learning_rate"]),weight_decay=float(probe["weight_decay"]))
    rng=random.Random(0); best=None; patience=0; history=[]; direction=method_direction(args.method)
    for epoch in range(int(probe["max_epochs"])):
        model.train(); losses=[]; points=[]; trajectories=[]
        for positions,offsets in problem_batches(train_frame,fit_mask,int(probe["trajectory_batch_size"]),rng):
            values=torch.from_numpy(train_features[positions]).to(device); target=torch.from_numpy(train_labels[positions]).to(device); logits=model(values)
            if args.method=="correction":
                loss,point,trajectory=correction_loss(logits,target,torch.from_numpy(remaining[positions]).to(device),offsets,beta=float(probe["trajectory_softmin_beta"]),trajectory=args.loss=="bce_traj")
            else:
                loss=binary_point_loss(logits,target,positive_weight); point=loss; trajectory=loss.detach()*0
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(probe["gradient_clip"])); optimizer.step()
            losses.append(float(loss.detach().cpu())); points.append(float(point.detach().cpu())); trajectories.append(float(trajectory.detach().cpu()))
        scores=predict_scores(model,train_features[validation_mask],device); truth=train_labels[validation_mask]; ap,auc=safe_ap_auc(truth,scores)
        validation_frame=train_frame.loc[validation_mask].reset_index(drop=True); strict=None
        if args.method=="correction":
            curve=[]
            for threshold,no_stop in threshold_grid(scores,direction,grid_size=int(config["calibration"]["threshold_quantiles"])):
                row=simulate_policy(validation_frame,scores,direction,threshold,force_dense=no_stop,fallback_records=validation_fallbacks); row["is_no_stop_sentinel"]=no_stop; curve.append(row)
            strict=select_empirical_budget(curve,0); key=(strict["replay_wall_reduction"],ap,auc)
        else: key=(ap,auc,-float(np.mean(losses)))
        record_epoch={"epoch":epoch,"loss":float(np.mean(losses)),"point_loss":float(np.mean(points)),"trajectory_loss":float(np.mean(trajectories)),"validation_ap":ap,"validation_auc":auc,"validation_B0":strict}
        history.append(record_epoch); print(json.dumps({"fold":args.fold,"method":args.method,**record_epoch}),flush=True)
        if best is None or key>best[0]:
            best=(key,epoch,{name:value.detach().cpu().clone() for name,value in model.state_dict().items()}); patience=0
        else: patience+=1
        if patience>=int(probe["patience"]): break
    if best is None: raise RuntimeError("未产生模型")
    model.load_state_dict(best[2]); oof_scores=predict_scores(model,test_features,device); ap,auc=safe_ap_auc(test_labels,oof_scores)
    payload={"status":"complete","created_at":datetime.now(timezone.utc).isoformat(),"protocol_id":config["protocol_id"],"method":args.method,"loss":args.loss,"fold":args.fold,
             "outer_train_problems":len(outer_train_ids),"outer_test_problems":len(outer_test_ids),"outer_test_scorable":int(test_frame.problem_id.nunique()),"outer_test_fallback_only":len(test_fallbacks),
             "inner_fit_problems":len(fit_ids),"inner_validation_problems":len(validation_ids),"best_epoch":int(best[1]),"history":history,"oof_label_AP_descriptive":ap,"oof_label_AUC_descriptive":auc,
             "scaler_scope":"outer_train/internal_fit_only","epoch_scope":"outer_train/internal_validation_only","outer_test_used_for_training_or_epoch":False}
    atomic_torch_save({"status":"complete","state_dict":best[2],"scaler_mean":torch.from_numpy(scaler.mean_.astype(np.float32)),"scaler_scale":torch.from_numpy(scaler.scale_.astype(np.float32)),"input_width":5126,"capture_layers":layers},destination/"probe.pt")
    atomic_torch_save({"status":"complete","scores":torch.from_numpy(oof_scores.astype(np.float32)),"problem_ids":test_frame.problem_id.astype(str).tolist(),"checkpoints":test_frame.checkpoint.astype(int).tolist(),"fallback_problem_ids":[row["problem_id"] for row in test_fallbacks]},destination/"oof_scores.pt")
    atomic_json(payload,destination/"training.json"); atomic_json({"status":"complete","invocation_fingerprint":fingerprint,"best_epoch":int(best[1])},marker)
    print(json.dumps({"status":"complete","fold":args.fold,"method":args.method,"loss":args.loss,"best_epoch":int(best[1])},indent=2))


if __name__=="__main__": main()

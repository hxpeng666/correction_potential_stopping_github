#!/usr/bin/env python3
"""MMLU-Pro LearnStop-style 数据与停止语义回归测试。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.final_paper_inference import read_jsonl
from src.legacy_empirical_probe_v4 import _last_switch_flags, simulate_policy, threshold_grid
from src.utils import load_yaml

ROOT=Path(__file__).resolve().parents[1]


def main() -> None:
    config=load_yaml(ROOT/"configs/final_paper_mmlu_pro_learnstop_style_v1.yaml")
    assert "MMLU" not in config["dataset"].get("probe_train_source","")
    assert config["dataset"]["source_split"]=="test" and config["dataset"]["pool_count"]==800
    rows=read_jsonl(ROOT/config["dataset"]["prepared_root"]/"heldout.jsonl")
    assert len(rows)==800 and len({row["problem_id"] for row in rows})==800
    manifest=json.loads((ROOT/config["output_root"]/config["dataset"]["split_manifest"]).read_text(encoding="utf-8"))
    expected=np.random.default_rng(42).choice(12032,size=800,replace=False).tolist()
    assert manifest["learnstop_sampling"]["selected_source_indices"]==expected
    assert sum(row["policy_role"]=="calibration" for row in rows)==320
    assert sum(row["policy_role"]=="final_test" for row in rows)==480
    folds=np.full(800,-1); groups=np.arange(800)
    for fold,(_train,test) in enumerate(GroupKFold(5).split(np.zeros((800,1)),groups=groups)): folds[test]=fold
    assert [row["oof_fold"] for row in rows]==folds.astype(int).tolist()
    assert _last_switch_flags(["A","A","B"],"B")==[False,False,True]
    assert _last_switch_flags(["A","A","A"],"B")==[False,False,False]
    frame=pd.DataFrame([
        {"problem_id":"x","checkpoint":64,"current_success":False,"dense_success":True,"current_prediction":"A","dense_prediction":"B","gold_answer":"B","branch_tokens":1,"dense_tokens":200,"dense_wall_ms":10.,"dense_prefill_cuda_ms":1.,"prefix_decode_cuda_ms":1.,"branch_wall_ms":1.,"replay_stop_wall_ms":3.},
        {"problem_id":"x","checkpoint":96,"current_success":True,"dense_success":True,"current_prediction":"B","dense_prediction":"B","gold_answer":"B","branch_tokens":1,"dense_tokens":200,"dense_wall_ms":10.,"dense_prefill_cuda_ms":1.,"prefix_decode_cuda_ms":2.,"branch_wall_ms":1.,"replay_stop_wall_ms":4.},
    ])
    high=simulate_policy(frame,np.asarray([0.9,0.8]),"high",0.85,include_records=True)
    low=simulate_policy(frame,np.asarray([0.9,0.1]),"low",0.2,include_records=True)
    assert high["records"][0]["checkpoint"]==64 and low["records"][0]["checkpoint"]==96
    sentinel=threshold_grid(np.asarray([0.2,0.8]),"low",101)[0]
    dense=simulate_policy(frame,np.asarray([0.2,0.8]),"low",sentinel[0],force_dense=sentinel[1])
    assert dense["coverage"]==0 and dense["token_reduction"]==0 and dense["accuracy"]==dense["dense_accuracy"]
    print("MMLU-Pro LearnStop-style 协议测试：全部通过")


if __name__=="__main__": main()

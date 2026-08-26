#!/usr/bin/env python3
"""验证特征消融的完整性、指标恒等式、bootstrap与日志。"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--logs",type=Path,required=True)
    args=parser.parse_args()
    tables=args.root/"tables"
    rows=pd.read_csv(tables/"all_feature_workpoints.csv")
    ci=pd.read_csv(tables/"feature_bootstrap_ci.csv")
    paired=pd.read_csv(tables/"feature_paired_vs_full.csv")
    diagnostics=pd.read_csv(tables/"feature_training_diagnostics.csv")
    errors=[]
    expected_n={"gsm8k":1319,"mmlu_pro":1000}
    expected_features=21
    for dataset,count in expected_n.items():
        subset=rows[rows.dataset==dataset]
        if subset.feature.nunique()!=expected_features:
            errors.append(f"{dataset} feature数量不是{expected_features}")
        if not (subset.N==count).all(): errors.append(f"{dataset} N不是{count}")
    identity=100.0*(rows.C_to_W-rows.W_to_C)/rows.N
    if not np.allclose(identity,rows.delta_dense_pp,atol=1e-10,rtol=0):
        errors.append("transition/accuracy恒等式失败")
    for name,frame in (("results",rows),("ci",ci),("paired",paired),("diagnostics",diagnostics)):
        numeric=frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all(): errors.append(f"{name}含NaN/Inf")
    if not (ci.replicates==10000).all() or not (paired.replicates==10000).all():
        errors.append("bootstrap次数不是10000")
    widths=diagnostics.groupby("feature").feature_width.nunique()
    if not widths.eq(1).all(): errors.append("同一feature跨split/dataset宽度不唯一")
    if int(diagnostics[diagnostics.feature=="full"].feature_width.iloc[0])!=5126:
        errors.append("full宽度不是5126")
    log_hits=[]
    for path in sorted(args.logs.glob("*.log")):
        text=path.read_text(encoding="utf-8",errors="replace")
        for pattern in ("Traceback","CUDA out of memory","RuntimeError","NaN","missing sample","row/vector mismatch","phase failed"):
            if pattern in text: log_hits.append({"path":str(path),"pattern":pattern})
    if log_hits: errors.append("日志命中错误模式")
    payload={
        "status":"failed" if errors else "complete",
        "verified_at":datetime.now(timezone.utc).isoformat(),
        "datasets":expected_n,"features":expected_features,
        "result_rows":len(rows),"bootstrap_ci_rows":len(ci),"paired_rows":len(paired),
        "bootstrap_replicates":10000,"transition_identity":"passed" if "transition/accuracy恒等式失败" not in errors else "failed",
        "finite_values":"passed" if not any("NaN/Inf" in value for value in errors) else "failed",
        "feature_widths":"passed" if not any("宽度" in value for value in errors) else "failed",
        "log_hits":log_hits,"errors":errors,
    }
    destination=args.root/"FINAL_AUDIT.json"
    temporary=destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(payload,ensure_ascii=False,indent=2))
    if errors: raise SystemExit(1)


if __name__=="__main__": main()

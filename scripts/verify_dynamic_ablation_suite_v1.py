#!/usr/bin/env python3
"""最终消融套件的指标恒等式、样本数、bootstrap与日志验收。"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    tables = args.root / "tables"
    methods = pd.read_csv(tables / "all_methods_all_workpoints.csv")
    expected = {"gsm8k": 1319, "mmlu_pro": 1000}
    errors: list[str] = []
    for dataset, count in expected.items():
        subset = methods[methods.dataset == dataset]
        if subset.empty or not (subset.N == count).all():
            errors.append(f"{dataset}方法样本数不全为{count}")
    identity = 100.0 * (methods.C_to_W - methods.W_to_C) / methods.N
    if not np.allclose(identity, methods.delta_dense_pp, atol=1e-10, rtol=0):
        errors.append("transition与DeltaDense恒等式失败")
    dense = methods[methods.method == "Dense"]
    for column in ("coverage", "token_reduction", "lost_correct_count", "gained_correct_count"):
        if not np.allclose(dense[column], 0.0, atol=0, rtol=0):
            errors.append(f"Dense sentinel的{column}非零")
    numeric = methods.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        errors.append("主结果包含NaN/Inf")
    ci = pd.read_csv(tables / "bootstrap_confidence_intervals.csv")
    comparisons = pd.read_csv(tables / "paired_bootstrap_comparisons.csv")
    if not (ci.replicates == 10000).all() or not (comparisons.replicates == 10000).all():
        errors.append("bootstrap次数不是10000")
    patterns = (
        "Traceback", "CUDA out of memory", "RuntimeError", "missing sample",
        "row/vector mismatch", "phase failed",
    )
    log_hits = []
    for path in sorted(args.logs.glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in patterns:
            if pattern in text:
                log_hits.append({"path": str(path), "pattern": pattern})
    if log_hits:
        errors.append("日志存在错误模式")
    payload = {
        "status": "failed" if errors else "complete",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "datasets": expected,
        "method_rows": len(methods),
        "unique_methods": {dataset: int(methods[methods.dataset == dataset].method.nunique()) for dataset in expected},
        "bootstrap_ci_rows": len(ci),
        "paired_comparison_rows": len(comparisons),
        "bootstrap_replicates": 10000,
        "transition_identity": "passed" if not errors or "transition与DeltaDense恒等式失败" not in errors else "failed",
        "dense_sentinel": "passed" if not any("Dense sentinel" in value for value in errors) else "failed",
        "finite_metrics": "passed" if "主结果包含NaN/Inf" not in errors else "failed",
        "log_hits": log_hits,
        "errors": errors,
    }
    destination = args.root / "FINAL_AUDIT.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

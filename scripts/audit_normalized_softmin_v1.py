#!/usr/bin/env python3
"""最终完整性审计：归一化trajectory soft-min实验。"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/final_paper_normalized_softmin_greedy_v1"
VARIANTS = (
    "beta025_weight1", "beta05_weight025", "beta05_weight05",
    "beta05_weight1", "beta05_weight2", "beta1_weight1",
)
EXPECTED = {"gsm8k": 1319, "mmlu_pro": 1000}
OLD = {
    "gsm8k": ROOT / "results/final_paper_greedy_forced_ablation_v1/static_reasoning_only/correction_bce_traj/gsm8k/probe.json",
    "mmlu_pro": ROOT / "results/final_paper_greedy_forced_mmlupro_ablation_v1/static_reasoning_only/correction_bce_traj/mmlu_pro/probe.json",
}


errors: list[str] = []
checks: dict[str, object] = {}
probe_checks = []
for dataset in EXPECTED:
    old = json.loads(OLD[dataset].read_text())
    old_input = old["input"]["artifact_identity_fingerprint"]
    for variant in VARIANTS:
        directory = RUN / dataset / variant
        missing = [name for name in ("probe.json", "probe.pt", "scores.pt", "policy_records.pt", "phase.complete") if not (directory / name).is_file()]
        if missing:
            errors.append(f"missing {dataset}/{variant}: {missing}")
            continue
        payload = json.loads((directory / "probe.json").read_text())
        run = payload["run_spec"]
        valid = (
            payload.get("status") == "complete"
            and run.get("trajectory_aggregation") == "normalized_softmin"
            and run.get("trajectory_normalize_by_count") is True
            and payload["input"]["artifact_identity_fingerprint"] == old_input
            and len(payload["frozen_policy_results"]["empirical_B"]) >= 5
        )
        if not valid:
            errors.append(f"probe invariant failed: {dataset}/{variant}")
        probe_checks.append({"dataset": dataset, "variant": variant, "valid": valid})
checks["probes"] = probe_checks

table_path = RUN / "final_report/all_empirical_B.csv"
if not table_path.is_file():
    errors.append("missing all_empirical_B.csv")
    frame = pd.DataFrame()
else:
    frame = pd.read_csv(table_path)
    if len(frame) != 80:
        errors.append(f"expected 80 result rows, got {len(frame)}")
metric_violations = []
for index, row in frame.iterrows():
    n = EXPECTED[str(row.dataset)]
    state_total = int(row.W_to_C + row.C_to_W + row.W_to_W + row.C_to_C + row.fallback)
    expected_delta = 100.0 * (row.C_to_W - row.W_to_C) / n
    if state_total != n:
        metric_violations.append({"row": int(index), "reason": "denominator"})
    if not math.isclose(float(row.delta_dense_pp), expected_delta, abs_tol=2e-6):
        metric_violations.append({"row": int(index), "reason": "transition_identity"})
if metric_violations:
    errors.append("metric invariant violations")
checks["result_rows"] = len(frame)
checks["metric_violations"] = metric_violations

bootstrap_path = RUN / "final_report/paired_bootstrap_B4.csv"
if not bootstrap_path.is_file():
    errors.append("missing paired bootstrap")
    bootstrap_rows = 0
else:
    bootstrap_rows = len(pd.read_csv(bootstrap_path))
    if bootstrap_rows != 6:
        errors.append(f"expected 6 bootstrap rows, got {bootstrap_rows}")
checks["bootstrap_rows"] = bootstrap_rows

unit_text = (RUN / "logs/unit.log").read_text(errors="replace")
checks["unit_passed"] = "'status': 'passed'" in unit_text
if not checks["unit_passed"]:
    errors.append("unit test did not pass")

pattern = re.compile(r"Traceback|CUDA out of memory|RuntimeError|\bNaN\b|phase failed|Killed", re.I)
hits = []
for path in sorted((RUN / "logs").glob("*.log")):
    found = pattern.findall(path.read_text(errors="replace"))
    if found:
        hits.append({"path": str(path), "hits": found[:10]})
checks["log_error_hits"] = hits
if hits:
    errors.append("unhandled log errors")

report = {
    "status": "complete" if not errors else "failed",
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "protocol": "GSM8K/MMLU-Pro greedy forced-answer normalized trajectory soft-min v1",
    "checks": checks,
    "errors": errors,
}
(RUN / "FINAL_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
if errors:
    raise SystemExit("; ".join(errors))
(RUN / "pipeline.final.complete").write_text(json.dumps({
    "status": "complete",
    "created_at": report["created_at"],
    "audit": "FINAL_AUDIT.json",
    "report": "final_report/NORMALIZED_SOFTMIN_REPORT_ZH.md",
}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({
    "status": "complete",
    "probes": len(probe_checks),
    "result_rows": len(frame),
    "bootstrap_rows": bootstrap_rows,
    "metric_violations": len(metric_violations),
    "log_error_hits": len(hits),
}, ensure_ascii=False, indent=2))

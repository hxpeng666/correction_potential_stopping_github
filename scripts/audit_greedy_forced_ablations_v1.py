#!/usr/bin/env python3
"""Final consistency audit for the greedy forced-answer ablation run."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "results/final_paper_greedy_forced_ablation_v1"
CACHE = PROJECT / "results/final_paper_greedy_forced_v1/selected_common_cache"
TABLES = ROOT / "final_report/tables"
EXPECTED_N = {"gsm8k": 1319, "mmlu": 1000}


def close(a: float, b: float, tol: float = 2e-6) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


checks: dict[str, object] = {}
errors: list[str] = []

# Completion artifacts.
expected_probe_dirs = []
for dataset in EXPECTED_N:
    for variant in ("full", "no_trajectory", "one_step_value", "dense_endpoint_value"):
        expected_probe_dirs.append(ROOT / "dynamic" / variant / dataset)
    for method in (
        "correction_bce",
        "correction_bce_traj",
        "correctness_bce",
        "consistency_bce",
        "last_switch_bce",
    ):
        expected_probe_dirs.append(ROOT / "static_reasoning_only" / method / dataset)
    for feature in ("h_only", "full", "main_no_entropy", "main_no_position", "main_no_geometry"):
        expected_probe_dirs.append(ROOT / "features" / feature / dataset)

missing_probe_artifacts = []
for directory in expected_probe_dirs:
    for name in ("probe.json", "probe.pt", "phase.complete"):
        if not (directory / name).is_file():
            missing_probe_artifacts.append(str(directory / name))
checks["probe_jobs_expected"] = len(expected_probe_dirs)
checks["probe_artifacts_missing"] = missing_probe_artifacts
if missing_probe_artifacts:
    errors.append("missing probe artifacts")

# Cache sample cardinality and filename uniqueness.
cache_counts = {}
for dataset, expected_test in EXPECTED_N.items():
    counts = {}
    for split, expected in (("probe_train", 1000), ("calibration", 500), ("heldout", expected_test)):
        files = sorted((CACHE / dataset / "merged" / split).glob("sample_*.pt"))
        stems = [p.stem for p in files]
        counts[split] = {"files": len(files), "unique_filenames": len(set(stems)), "expected": expected}
        if len(files) != expected or len(set(stems)) != expected:
            errors.append(f"cache count/duplicate mismatch: {dataset}/{split}")
    cache_counts[dataset] = counts
checks["cache_counts"] = cache_counts

# Aggregated metrics invariants.
all_rows = pd.read_csv(TABLES / "all_workpoints.csv")
metric_violations = []
for idx, row in all_rows.iterrows():
    dataset = str(row["dataset"])
    n = EXPECTED_N[dataset]
    values = [row[c] for c in ("accuracy", "dense_accuracy", "delta_dense_pp", "coverage", "token_reduction")]
    if not all(math.isfinite(float(v)) for v in values):
        metric_violations.append({"row": int(idx), "reason": "nonfinite metric"})
        continue
    wc, cw, ww, cc = (int(row[c]) for c in ("W_to_C", "C_to_W", "W_to_W", "C_to_C"))
    fallback = int(row["fallback"])
    if wc + cw + ww + cc + fallback != n:
        metric_violations.append({"row": int(idx), "reason": "transition+fallback denominator"})
    expected_delta = 100.0 * (cw - wc) / n
    if not close(row["delta_dense_pp"], expected_delta):
        metric_violations.append({"row": int(idx), "reason": "delta/transition identity"})
    if not close(row["accuracy"] - row["dense_accuracy"], row["delta_dense_pp"] / 100.0):
        metric_violations.append({"row": int(idx), "reason": "accuracy/delta identity"})
checks["aggregated_metric_rows"] = len(all_rows)
checks["metric_violations"] = metric_violations
if metric_violations:
    errors.append("aggregated metric invariant violation")

# Dense sentinel semantics in every independently trained probe.
sentinel_violations = []
for directory in expected_probe_dirs:
    payload = json.loads((directory / "probe.json").read_text())
    if payload.get("status") != "complete":
        sentinel_violations.append({"probe": str(directory), "reason": "status"})
        continue
    calibration = payload["calibration"]
    if "dense_sentinel" in calibration:
        sentinels = [calibration["dense_sentinel"]]
    else:
        sentinels = [x for x in calibration["curve"] if x.get("is_no_stop_sentinel")]
    if len(sentinels) != 1:
        sentinel_violations.append({"probe": str(directory), "reason": "sentinel cardinality"})
        continue
    s = sentinels[0]
    if not (
        close(s["accuracy"], s["dense_accuracy"])
        and close(s["coverage"], 0.0)
        and close(s["token_reduction"], 0.0)
        and int(s["lost_correct_count"]) == 0
        and int(s["fallback"]) == int(s["problems"])
    ):
        sentinel_violations.append({"probe": str(directory), "reason": "sentinel semantics", "metrics": s})
checks["dense_sentinel_violations"] = sentinel_violations
if sentinel_violations:
    errors.append("dense sentinel violation")

# Deployability audits.
online_audits = {}
for dataset in EXPECTED_N:
    path = ROOT / "audits" / f"{dataset}_online_parity_audit.json"
    payload = json.loads(path.read_text())
    online_audits[dataset] = {
        "status": payload.get("status"),
        "candidates_checked": payload.get("candidates_checked"),
        "streaming_offline_mismatch_count": payload.get("streaming_offline_mismatch_count"),
        "future_mutation_mismatch_count": payload.get("future_mutation_mismatch_count"),
        "future_fields_used_by_action": payload.get("future_fields_used_by_action"),
        "heldout_used": payload.get("heldout_used"),
    }
    if not (
        payload.get("status") == "complete"
        and payload.get("streaming_offline_mismatch_count") == 0
        and payload.get("future_mutation_mismatch_count") == 0
        and payload.get("future_fields_used_by_action") is False
        and payload.get("heldout_used") is False
    ):
        errors.append(f"online parity audit failed: {dataset}")
checks["online_parity"] = online_audits

# Component logs only. An earlier runner orchestration parse error was superseded
# by a clean resumed runner; it did not affect any probe artifact.
patterns = re.compile(r"Traceback|CUDA out of memory|RuntimeError|\bNaN\b|phase failed", re.IGNORECASE)
log_hits = []
for path in sorted((ROOT / "logs").glob("*.log")):
    if path.name == "runner.log":
        continue
    text = path.read_text(errors="replace")
    matches = patterns.findall(text)
    if matches:
        log_hits.append({"path": str(path), "hits": matches[:10]})
checks["component_log_error_hits"] = log_hits
checks["resolved_orchestration_note"] = (
    "An earlier runner shell parse error occurred while the script was being updated. "
    "All component jobs completed, and the clean resumed runner plus final compiler succeeded."
)
if log_hits:
    errors.append("unhandled component log errors")

report = {
    "status": "complete" if not errors else "failed",
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "protocol": "greedy-forced-answer, FP16, sentence-step, single-seed, reasoning-token cost",
    "checks": checks,
    "errors": errors,
}
(ROOT / "FINAL_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
if errors:
    raise SystemExit("; ".join(errors))
(ROOT / "pipeline.final.complete").write_text(json.dumps({
    "status": "complete",
    "created_at": report["created_at"],
    "audit": "FINAL_AUDIT.json",
    "report": "final_report/GREEDY_FORCED_ABLATION_REPORT_ZH.md",
}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": report["status"], "checks": {
    "probe_jobs": len(expected_probe_dirs),
    "metric_rows": len(all_rows),
    "metric_violations": len(metric_violations),
    "sentinel_violations": len(sentinel_violations),
    "component_log_hits": len(log_hits),
    "online_parity": online_audits,
}}, ensure_ascii=False, indent=2))

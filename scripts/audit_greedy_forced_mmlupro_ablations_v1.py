#!/usr/bin/env python3
"""Final integrity audit for the MMLU-Pro greedy forced-answer ablations."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/final_paper_greedy_forced_mmlupro_ablation_v1"
COLLECTION = ROOT / "results/final_paper_greedy_forced_mmlupro_v1"
TABLES = RUN / "final_report/tables"
N = 1000


def close(a: float, b: float, tolerance: float = 2e-6) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


errors: list[str] = []
checks: dict[str, object] = {}

collection = json.loads((COLLECTION / "COLLECTION_AUDIT.json").read_text())
checks["collection"] = {
    "status": collection.get("status"),
    "validated_samples": collection.get("validated_samples"),
    "branches": collection.get("branch_counts", {}).get("mmlu_pro"),
    "error_count": collection.get("error_count"),
    "queue": collection.get("queue"),
}
if not (
    collection.get("status") == "passed"
    and collection.get("validated_samples") == 2500
    and collection.get("error_count") == 0
    and collection.get("queue", {}).get("done") == 2500
):
    errors.append("collection audit failed")

probes: dict[str, Path] = {}
for variant in ("full", "no_trajectory", "one_step_value", "dense_endpoint_value"):
    probes[f"Dynamic-{variant}"] = RUN / "dynamic" / variant / "mmlu_pro"
for name in (
    "correction_bce",
    "correction_bce_traj",
    "correctness_bce",
    "consistency_bce",
    "last_switch_bce",
):
    probes[f"Static-{name}"] = RUN / "static_reasoning_only" / name / "mmlu_pro"
for feature in ("h_only", "full", "main_no_entropy", "main_no_position", "main_no_geometry"):
    probes[f"Feature-{feature}"] = RUN / "features" / feature / "mmlu_pro"

missing = []
sentinel_violations = []
id_sets: dict[str, set[str]] = {}
for name, directory in probes.items():
    for filename in ("probe.json", "probe.pt", "policy_records.pt", "phase.complete"):
        if not (directory / filename).is_file():
            missing.append(str(directory / filename))
    if missing and not (directory / "probe.json").is_file():
        continue
    payload = json.loads((directory / "probe.json").read_text())
    if payload.get("status") != "complete":
        errors.append(f"incomplete probe: {name}")
    calibration = payload["calibration"]
    if "dense_sentinel" in calibration:
        candidates = [calibration["dense_sentinel"]]
    else:
        candidates = [row for row in calibration["curve"] if row.get("is_no_stop_sentinel")]
    if len(candidates) != 1:
        sentinel_violations.append({"method": name, "reason": "cardinality"})
    else:
        row = candidates[0]
        if not (
            close(row["accuracy"], row["dense_accuracy"])
            and close(row["coverage"], 0)
            and close(row["token_reduction"], 0)
            and int(row["lost_correct_count"]) == 0
            and int(row["fallback"]) == int(row["problems"])
        ):
            sentinel_violations.append({"method": name, "reason": "semantics"})
    records = torch.load(directory / "policy_records.pt", map_location="cpu", weights_only=False)["records"]
    b4 = records["empirical_B"]["4"]
    ids = [str(row["problem_id"]) for row in b4]
    if len(ids) != N or len(set(ids)) != N:
        errors.append(f"unpaired/duplicate records: {name}")
    id_sets[name] = set(ids)

checks["probe_count"] = len(probes)
checks["missing_probe_artifacts"] = missing
checks["dense_sentinel_violations"] = sentinel_violations
if missing:
    errors.append("missing probe artifacts")
if sentinel_violations:
    errors.append("dense sentinel violation")
if len({frozenset(values) for values in id_sets.values()}) != 1:
    errors.append("method sample-ID sets differ")
checks["all_methods_share_heldout_ids"] = len({frozenset(values) for values in id_sets.values()}) == 1

frame = pd.read_csv(TABLES / "all_workpoints.csv")
violations = []
for index, row in frame.iterrows():
    wc, cw, ww, cc, fallback = (
        int(row["W_to_C"]), int(row["C_to_W"]), int(row["W_to_W"]),
        int(row["C_to_C"]), int(row["fallback"]),
    )
    if wc + cw + ww + cc + fallback != N:
        violations.append({"row": int(index), "reason": "denominator"})
    expected_delta = 100.0 * (cw - wc) / N
    if not close(row["delta_dense_pp"], expected_delta):
        violations.append({"row": int(index), "reason": "transition identity"})
    if not close(row["accuracy"] - row["dense_accuracy"], row["delta_dense_pp"] / 100.0):
        violations.append({"row": int(index), "reason": "accuracy identity"})
checks["metric_rows"] = len(frame)
checks["metric_violations"] = violations
if violations:
    errors.append("metric invariant violation")

parity = json.loads((RUN / "audits/mmlu_pro_online_parity_audit.json").read_text())
checks["online_parity"] = {
    "status": parity.get("status"),
    "candidates_checked": parity.get("candidates_checked"),
    "streaming_offline_mismatch_count": parity.get("streaming_offline_mismatch_count"),
    "future_mutation_mismatch_count": parity.get("future_mutation_mismatch_count"),
    "future_fields_used_by_action": parity.get("future_fields_used_by_action"),
    "heldout_used": parity.get("heldout_used"),
}
if not (
    parity.get("status") == "complete"
    and parity.get("candidates_checked") == 48
    and parity.get("streaming_offline_mismatch_count") == 0
    and parity.get("future_mutation_mismatch_count") == 0
    and parity.get("future_fields_used_by_action") is False
    and parity.get("heldout_used") is False
):
    errors.append("online parity failed")

pattern = re.compile(r"Traceback|CUDA out of memory|RuntimeError|\bNaN\b|phase failed|Killed", re.I)
log_hits = []
for path in sorted((RUN / "logs").glob("*.log")):
    hits = pattern.findall(path.read_text(errors="replace"))
    if hits:
        log_hits.append({"path": str(path), "hits": hits[:10]})
checks["log_error_hits"] = log_hits
if log_hits:
    errors.append("unhandled log errors")

report = {
    "status": "complete" if not errors else "failed",
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "protocol": "MMLU-Pro-1k greedy forced-answer FP16 sentence-step single seed token-only",
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
    "report": "final_report/GREEDY_FORCED_ABLATION_REPORT_ZH.md",
}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({
    "status": report["status"],
    "probe_count": len(probes),
    "metric_rows": len(frame),
    "metric_violations": len(violations),
    "sentinel_violations": len(sentinel_violations),
    "log_error_hits": len(log_hits),
    "all_methods_share_heldout_ids": checks["all_methods_share_heldout_ids"],
}, ensure_ascii=False, indent=2))

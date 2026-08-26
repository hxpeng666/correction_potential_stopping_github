#!/usr/bin/env python3
"""Fail-closed integrity/leakage audit for the immutable legacy-v4 cache."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_paper_protocol import canonical_fingerprint
from src.utils import atomic_json, load_yaml


EXPECTED = {
    "gsm8k": {"probe_train": 1000, "calibration": 500, "heldout": 1319},
    "mmlu": {"probe_train": 1000, "calibration": 500, "heldout": 1000},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fail(errors: list[str], message: str) -> None:
    if len(errors) < 200:
        errors.append(message)


def audit_dense(
    path: Path,
    expected_id: str,
    expected_fingerprint: str,
    expected_example_seed: int,
    errors: list[str],
) -> dict[str, Any] | None:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        fail(errors, f"cannot load Dense {path}: {error!r}")
        return None
    if value.get("status") != "complete":
        fail(errors, f"incomplete Dense: {path}")
    if str(value.get("problem_id")) != expected_id:
        fail(errors, f"Dense problem_id mismatch: {path}")
    if value.get("config_fingerprint") != expected_fingerprint:
        fail(errors, f"Dense config fingerprint mismatch: {path}")
    if value.get("dtype") != "bfloat16" or value.get("attention_backend") != "sdpa":
        fail(errors, f"Dense backend mismatch: {path}")
    if int(value.get("example_seed", -1)) != expected_example_seed:
        fail(errors, f"Dense example seed mismatch: {path}")
    dense = value.get("dense", {})
    tokens = dense.get("tokens", [])
    declared = int(dense.get("reasoning_tokens", -1))
    if declared != len(tokens):
        fail(errors, f"Dense token length mismatch: {path}")
    for key in ("logps", "margins", "entropies_top20"):
        if len(dense.get(key, [])) != declared:
            fail(errors, f"Dense {key} length mismatch: {path}")
    for key in ("wall_ms", "prefill_cuda_ms", "decode_cuda_ms"):
        value_ms = dense.get(key)
        if value_ms is None or not math.isfinite(float(value_ms)):
            fail(errors, f"Dense invalid {key}: {path}")
    return value


def audit_checkpoint(
    path: Path,
    expected_id: str,
    expected_fingerprint: str,
    expected_example_seed: int,
    dense: dict[str, Any] | None,
    errors: list[str],
) -> dict[str, Any] | None:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        fail(errors, f"cannot load checkpoint {path}: {error!r}")
        return None
    if value.get("status") != "complete":
        fail(errors, f"incomplete checkpoint: {path}")
    if str(value.get("problem_id")) != expected_id:
        fail(errors, f"checkpoint problem_id mismatch: {path}")
    if value.get("config_fingerprint") != expected_fingerprint:
        fail(errors, f"checkpoint config fingerprint mismatch: {path}")
    if value.get("dtype") != "bfloat16" or value.get("attention_backend") != "sdpa":
        fail(errors, f"checkpoint backend mismatch: {path}")
    if list(value.get("capture_layers", [])) != [8, 20, 35]:
        fail(errors, f"checkpoint capture layers mismatch: {path}")
    rows = value.get("rows", [])
    hidden = value.get("hidden")
    if not torch.is_tensor(hidden) or tuple(hidden.shape) != (len(rows), 3, 2560):
        fail(errors, f"checkpoint row/vector shape mismatch: {path}")
    elif not bool(torch.isfinite(hidden).all()):
        fail(errors, f"checkpoint hidden NaN/Inf: {path}")
    checkpoints = [int(row.get("checkpoint", -1)) for row in rows]
    if len(checkpoints) != len(set(checkpoints)):
        fail(errors, f"duplicate checkpoint positions: {path}")
    for row in rows:
        checkpoint = int(row.get("checkpoint", -1))
        if str(row.get("problem_id")) != expected_id:
            fail(errors, f"checkpoint row ID mismatch: {path}")
        if int(row.get("branch_seed", -1)) != expected_example_seed + checkpoint * 7919:
            fail(errors, f"branch seed mismatch at {expected_id}/{checkpoint}")
        if not math.isfinite(float(row.get("prefix_mean_entropy_tail8", float("nan")))):
            fail(errors, f"invalid entropy at {expected_id}/{checkpoint}")
        schedules = set(row.get("checkpoint_schedules", []))
        if not schedules or not schedules <= {"fixed", "sentence", "hybrid"}:
            fail(errors, f"invalid checkpoint schedule at {expected_id}/{checkpoint}")
    declared = value.get("schedules", {})
    for schedule in ("fixed", "sentence", "hybrid"):
        observed = [
            int(row["checkpoint"])
            for row in rows
            if schedule in row.get("checkpoint_schedules", [])
        ]
        if observed != list(declared.get(schedule, [])):
            fail(errors, f"schedule/row mismatch {schedule}: {path}")
    if dense is not None:
        if int(value.get("dense_content_tokens", -1)) > int(dense["dense"]["reasoning_tokens"]):
            fail(errors, f"checkpoint content length exceeds Dense: {path}")
        for row in rows:
            if int(row.get("dense_tokens", -1)) != int(dense["dense"]["reasoning_tokens"]):
                fail(errors, f"checkpoint Dense length mismatch: {path}")
                break
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gsm8k-config", required=True)
    parser.add_argument("--mmlu-config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configs = {
        "gsm8k": load_yaml(args.gsm8k_config),
        "mmlu": load_yaml(args.mmlu_config),
    }
    fingerprints = {name: canonical_fingerprint(config) for name, config in configs.items()}
    errors: list[str] = []
    report: dict[str, Any] = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "legacy_empirical_v4",
        "checks": {},
    }
    split_id_sets: dict[str, dict[str, set[str]]] = {}
    for dataset in ("gsm8k", "mmlu"):
        split_id_sets[dataset] = {}
        report["checks"][dataset] = {}
        for split, expected_count in EXPECTED[dataset].items():
            records = read_jsonl(args.data_root / dataset / f"{split}.jsonl")
            ids = [str(row["problem_id"]) for row in records]
            split_id_sets[dataset][split] = set(ids)
            if len(records) != expected_count or len(set(ids)) != expected_count:
                fail(errors, f"prepared count/duplicate mismatch {dataset}/{split}")
            dense_root = args.run_root / "raw" / dataset / "dense" / split
            checkpoint_root = args.run_root / "raw" / dataset / "checkpoints" / split
            dense_paths = {path.stem.removeprefix("sample_"): path for path in dense_root.glob("sample_*.pt")}
            checkpoint_paths = {path.stem.removeprefix("sample_"): path for path in checkpoint_root.glob("sample_*.pt")}
            if set(dense_paths) != set(ids):
                fail(errors, f"Dense missing/extra IDs {dataset}/{split}: expected={len(ids)} observed={len(dense_paths)}")
            if set(checkpoint_paths) != set(ids):
                fail(errors, f"checkpoint missing/extra IDs {dataset}/{split}: expected={len(ids)} observed={len(checkpoint_paths)}")
            source_counter = Counter(str(row.get("source_split", "official_train" if split != "heldout" else "official_test")) for row in records)
            subject_counter = Counter(str(row.get("subject")) for row in records if row.get("subject") is not None)
            checked = 0
            for row in records:
                problem_id = str(row["problem_id"])
                if dataset == "gsm8k":
                    example_seed = int(configs[dataset]["seed"]) + int(row["source_index"]) * 1009
                else:
                    example_seed = int(configs[dataset]["seed"]) + int(row["legacy_seed_index"]) * 1009
                dense = None
                if problem_id in dense_paths:
                    dense = audit_dense(dense_paths[problem_id], problem_id, fingerprints[dataset], example_seed, errors)
                if problem_id in checkpoint_paths:
                    audit_checkpoint(checkpoint_paths[problem_id], problem_id, fingerprints[dataset], example_seed, dense, errors)
                checked += 1
            report["checks"][dataset][split] = {
                "expected": expected_count,
                "prepared": len(records),
                "dense_files": len(dense_paths),
                "checkpoint_files": len(checkpoint_paths),
                "audited_records": checked,
                "source_counts": dict(sorted(source_counter.items())),
                "subject_counts": dict(sorted(subject_counter.items())),
            }
        names = tuple(split_id_sets[dataset])
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                overlap = split_id_sets[dataset][left] & split_id_sets[dataset][right]
                if overlap:
                    fail(errors, f"data leakage {dataset}: {left}/{right} overlap={len(overlap)}")
    report["errors"] = errors
    report["error_count"] = len(errors)
    report["status"] = "passed" if not errors else "failed"
    atomic_json(report, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

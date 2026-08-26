#!/usr/bin/env python3
"""Deterministically repair GSM8K/AIME success-derived cache fields.

The expensive Dense and forced-answer generations are immutable inputs here.
Only labels derived from the already stored gold/current/Dense predictions are
recomputed with the audited numeric answer comparator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from deepseek7b_protocol_v1 import atomic_torch_save, canonical_fingerprint, success


NUMERIC_DATASETS = ("gsm8k", "aime")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_paths(config: dict[str, Any], dataset: str) -> list[Path]:
    prepared = Path(config["data"]["prepared_root"]) / dataset
    cache = Path(config["output_root"]) / "cache" / dataset
    paths: list[Path] = []
    for manifest in sorted(prepared.glob("*.jsonl")):
        split = manifest.stem
        records = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]
        local = [cache / split / f"sample_{record['problem_id']}.pt" for record in records]
        missing = [str(path) for path in local if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{dataset}/{split} is incomplete; first missing artifacts: {missing[:5]}"
            )
        paths.extend(local)
    return paths


def counters(artifact: dict[str, Any]) -> Counter:
    result = Counter()
    dense_success = bool(artifact["dense"]["success"])
    result["problems"] = 1
    result["dense_correct"] = int(dense_success)
    for row in artifact["rows"]:
        current_success = bool(row["current_success"])
        result["checkpoints"] += 1
        result["current_correct"] += int(current_success)
        result["consistency"] += int(bool(row["consistency"]))
        result["W_to_C"] += int((not current_success) and dense_success)
        result["C_to_W"] += int(current_success and (not dense_success))
    return result


def repair_artifact(
    artifact: dict[str, Any], *, dataset: str, protocol_fingerprint: str
) -> tuple[dict[str, Any], bool, Counter, Counter]:
    if artifact.get("status") != "complete":
        raise ValueError("artifact status is not complete")
    if artifact.get("dataset") != dataset:
        raise ValueError(f"dataset mismatch: {artifact.get('dataset')} != {dataset}")
    if artifact.get("protocol_fingerprint") != protocol_fingerprint:
        raise ValueError("protocol fingerprint mismatch")
    gold = artifact.get("gold_answer")
    dense = dict(artifact["dense"])
    dense_prediction = dense.get("prediction")
    dense_success = success(dataset, gold, dense_prediction)
    before = counters(artifact)
    changed = bool(dense.get("success")) != dense_success
    dense["success"] = bool(dense_success)
    repaired_rows = []
    for original in artifact["rows"]:
        row = dict(original)
        if row.get("dense_prediction") != dense_prediction:
            raise ValueError("row/Dense prediction mismatch")
        current_prediction = row.get("current_prediction")
        current_success = success(dataset, gold, current_prediction)
        consistency = bool(
            current_prediction is not None
            and dense_prediction is not None
            and current_prediction == dense_prediction
        )
        expected = {
            "dense_success": bool(dense_success),
            "current_success": bool(current_success),
            "consistency": consistency,
            "correction": bool((not current_success) and dense_success),
            "damage": bool(current_success and (not dense_success)),
        }
        changed = changed or any(row.get(key) != value for key, value in expected.items())
        row.update(expected)
        repaired_rows.append(row)
    repaired = dict(artifact)
    repaired["dense"] = dense
    repaired["rows"] = repaired_rows
    collection = dict(repaired.get("collection", {}))
    collection["numeric_label_repair_v2"] = {
        "reason": "restore unreachable numeric_value body; recompute labels from stored predictions",
        "datasets": list(NUMERIC_DATASETS),
        "dense_or_checkpoint_fields_changed": bool(changed),
        "repaired_at": datetime.now(timezone.utc).isoformat(),
    }
    repaired["collection"] = collection
    after = counters(repaired)
    return repaired, changed, before, after


def verify_artifact(artifact: dict[str, Any], dataset: str) -> None:
    gold = artifact.get("gold_answer")
    dense_prediction = artifact["dense"].get("prediction")
    dense_success = success(dataset, gold, dense_prediction)
    if bool(artifact["dense"].get("success")) != dense_success:
        raise ValueError("Dense success repair verification failed")
    for row in artifact["rows"]:
        current_success = success(dataset, gold, row.get("current_prediction"))
        expected = {
            "dense_success": dense_success,
            "current_success": current_success,
            "correction": (not current_success) and dense_success,
            "damage": current_success and (not dense_success),
        }
        for key, value in expected.items():
            if bool(row.get(key)) != bool(value):
                raise ValueError(f"checkpoint {row.get('checkpoint')} invalid {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fingerprint = canonical_fingerprint(config)
    report: dict[str, Any] = {
        "status": "complete" if args.apply else "dry_run_complete",
        "protocol_id": config["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "mode": "apply" if args.apply else "dry_run",
        "datasets": {},
    }
    for dataset in NUMERIC_DATASETS:
        paths = expected_paths(config, dataset)
        before_total = Counter()
        after_total = Counter()
        changed_files = 0
        rewritten_files = 0
        for path in paths:
            original = torch.load(path, map_location="cpu", weights_only=False)
            repaired, changed, before, after = repair_artifact(
                original, dataset=dataset, protocol_fingerprint=fingerprint
            )
            before_total.update(before)
            after_total.update(after)
            changed_files += int(changed)
            if args.apply:
                atomic_torch_save(repaired, path)
                persisted = torch.load(path, map_location="cpu", weights_only=False)
                verify_artifact(persisted, dataset)
                rewritten_files += 1
        report["datasets"][dataset] = {
            "files": len(paths),
            "changed_files": changed_files,
            "rewritten_files": rewritten_files,
            "before": dict(before_total),
            "after": dict(after_total),
        }
    report["verified_after_repair"] = bool(args.apply)
    report["repair_script_sha256"] = file_sha256(Path(__file__).resolve())
    if args.apply:
        target = Path(config["output_root"]) / "NUMERIC_LABEL_REPAIR.json"
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

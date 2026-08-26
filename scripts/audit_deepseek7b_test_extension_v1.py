#!/usr/bin/env python3
"""Strict audit for held-out-only exact 13K-to-32K DeepSeek extensions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from deepseek7b_protocol_v1 import canonical_fingerprint, prediction, success  # noqa: E402


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tensor_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)


def token_hash(tokens: list[int]) -> str:
    raw = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    source_root = Path(config["cache_migration"]["source_output_root"])
    expected_fingerprint = canonical_fingerprint(config)
    manifest = json.loads(args.test_manifest.read_text(encoding="utf-8"))
    eligible = list(manifest["eligible"])
    if len(eligible) != int(manifest["eligible_count"]):
        raise RuntimeError("test manifest eligible_count mismatch")
    if any(str(item["split"]) != "heldout" for item in eligible):
        raise RuntimeError("test manifest contains a non-heldout item")

    errors: list[str] = []
    audited = []
    counts = Counter()
    dense_success_recomputed = 0
    reused_rows_checked = 0
    reused_hidden_checked = 0
    prefix_tokens_checked = 0
    prefix_entropies_checked = 0
    for item in eligible:
        dataset = str(item["dataset"])
        split = str(item["split"])
        problem_id = str(item["problem_id"])
        relative = Path("cache") / dataset / split / f"sample_{problem_id}.pt"
        target_path = output_root / relative
        source_path = source_root / relative
        if not target_path.is_file():
            errors.append(f"missing target: {relative}")
            continue
        if not source_path.is_file():
            errors.append(f"missing source: {relative}")
            continue
        target = torch.load(target_path, map_location="cpu", weights_only=False, mmap=True)
        source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
        collection = target.get("collection", {})
        dense_generation = target.get("dense_generation", {})
        if (
            target.get("status") != "complete"
            or target.get("protocol_fingerprint") != expected_fingerprint
            or target.get("dataset") != dataset
            or target.get("split") != split
            or str(target.get("problem_id")) != problem_id
            or target.get("actual_checkpoint_schedule") != "paragraph"
        ):
            errors.append(f"invalid target metadata: {relative}")
            continue
        if (
            collection.get("execution_mode")
            != "incremental_exact_resume_from_capped_13k_source"
            or int(collection.get("reused_checkpoints", 0)) <= 0
            or int(collection.get("new_checkpoints", -1)) < 0
            or dense_generation.get("incremental_exact_resume") is not True
        ):
            errors.append(f"invalid incremental metadata: {relative}")
            continue

        source_tokens = [int(value) for value in source["dense"]["tokens"]]
        target_tokens = [int(value) for value in target["dense"]["tokens"]]
        source_entropies = [float(value) for value in source["dense"]["entropies_top20"]]
        target_entropies = [float(value) for value in target["dense"]["entropies_top20"]]
        if len(source_tokens) != 13000 or not bool(source["dense"].get("reached_max_tokens")):
            errors.append(f"invalid capped source: {relative}")
            continue
        if target_tokens[:13000] != source_tokens:
            errors.append(f"dense prefix token mismatch: {relative}")
            continue
        if target_entropies[:13000] != source_entropies:
            errors.append(f"dense prefix entropy mismatch: {relative}")
            continue
        if token_hash(source_tokens) != str(item["source_dense_token_sha256"]):
            errors.append(f"source token hash mismatch: {relative}")
            continue
        prefix_tokens_checked += 13000
        prefix_entropies_checked += 13000

        source_rows = list(source["rows"])
        target_rows = list(target["rows"])
        reused = int(collection["reused_checkpoints"])
        if reused != len(source_rows) or len(target_rows) < reused:
            errors.append(f"reused row count mismatch: {relative}")
            continue
        row_keys = (
            "checkpoint",
            "current_prediction",
            "current_success",
            "branch_tokens",
            "branch_token_ids",
            "branch_text",
            "branch_generated_text",
            "forced_answer_decoding",
            "forced_answer_do_sample",
        )
        mismatch = False
        for index, source_row in enumerate(source_rows):
            target_row = target_rows[index]
            if any(source_row.get(key) != target_row.get(key) for key in row_keys):
                errors.append(f"reused row mismatch: {relative} row={index}")
                mismatch = True
                break
        if mismatch:
            continue
        reused_rows_checked += reused
        if not tensor_equal(target["hidden"][:reused], source["hidden"]):
            errors.append(f"reused hidden mismatch: {relative}")
            continue
        reused_hidden_checked += reused

        dense_prediction = prediction(dataset, target["dense"]["text"])
        dense_success = success(dataset, target["gold_answer"], dense_prediction)
        if (
            dense_prediction != target["dense"]["prediction"]
            or bool(dense_success) != bool(target["dense"]["success"])
        ):
            errors.append(f"dense label mismatch: {relative}")
            continue
        dense_success_recomputed += 1
        counts[dataset] += 1
        audited.append(
            {
                "dataset": dataset,
                "problem_id": problem_id,
                "source_tokens": len(source_tokens),
                "target_tokens": len(target_tokens),
                "reused_checkpoints": reused,
                "new_checkpoints": int(collection["new_checkpoints"]),
                "target_reached_max_tokens": bool(target["dense"]["reached_max_tokens"]),
            }
        )

    expected_counts = Counter(str(item["dataset"]) for item in eligible)
    if counts != expected_counts:
        errors.append(f"dataset counts mismatch: observed={dict(counts)} expected={dict(expected_counts)}")
    report = {
        "status": "complete" if not errors else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config["protocol_id"],
        "protocol_fingerprint": expected_fingerprint,
        "scope": "heldout_extension_targets_only",
        "eligible_count": len(eligible),
        "counts": dict(counts),
        "expected_counts": dict(expected_counts),
        "dense_success_recomputed": dense_success_recomputed,
        "prefix_tokens_checked": prefix_tokens_checked,
        "prefix_entropies_checked": prefix_entropies_checked,
        "reused_rows_checked": reused_rows_checked,
        "reused_hidden_checked": reused_hidden_checked,
        "artifacts": audited,
        "errors": errors,
    }
    atomic_json(report, output_root / "TEST_ONLY_COLLECTION_AUDIT.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)
    atomic_json(
        {
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "heldout_extension_targets_only",
            "eligible_count": len(eligible),
            "counts": dict(counts),
            "audit": "TEST_ONLY_COLLECTION_AUDIT.json",
            "non_test_extension_policy": "not_collected_or_used",
        },
        output_root / "TEST_ONLY_COLLECTION_COMPLETE.json",
    )


if __name__ == "__main__":
    main()

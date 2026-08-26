#!/usr/bin/env python3
"""Reuse natural 13K completions and select only capped trajectories for 32K collection."""
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


DATASETS = ("gsm8k", "math", "math500", "aime")
SPLITS = ("probe_train", "calibration", "heldout")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_sha256(tokens: list[int]) -> str:
    raw = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def expected_gold(dataset: str, record: dict[str, Any]) -> str:
    if "gold_answer" in record:
        return str(record["gold_answer"])
    if dataset == "gsm8k":
        return str(record["answer"]).rsplit("####", 1)[-1].strip().replace(",", "")
    raise KeyError(f"no gold-answer rule for {dataset}")


def validate_common(
    value: dict[str, Any],
    *,
    dataset: str,
    split: str,
    record: dict[str, Any],
    gold: str,
    fingerprint: str,
) -> None:
    failures = []
    if value.get("status") != "complete":
        failures.append("status")
    if value.get("protocol_fingerprint") != fingerprint:
        failures.append("protocol fingerprint")
    if value.get("actual_checkpoint_schedule") != "paragraph":
        failures.append("checkpoint schedule")
    if str(value.get("problem_id")) != str(record["problem_id"]):
        failures.append("problem_id")
    if str(value.get("record", {}).get("question")) != str(record["question"]):
        failures.append("question")
    if str(value.get("gold_answer")) != gold:
        failures.append("gold")
    if value.get("dataset") != dataset or value.get("split") != split:
        failures.append("dataset/split")
    if failures:
        raise ValueError(f"artifact validation failed: {failures}")


def refresh_labels(value: dict[str, Any], dataset: str, gold: str) -> dict[str, Any]:
    migrated = dict(value)
    dense = dict(migrated["dense"])
    dense_prediction = dense.get("prediction")
    dense_success = success(dataset, gold, dense_prediction)
    dense["success"] = bool(dense_success)
    migrated["dense"] = dense
    rows = []
    for original in migrated.get("rows", []):
        row = dict(original)
        current_prediction = row.get("current_prediction")
        current_success = success(dataset, gold, current_prediction)
        row.update(
            {
                "dense_success": bool(dense_success),
                "current_success": bool(current_success),
                "consistency": bool(
                    current_prediction is not None
                    and dense_prediction is not None
                    and current_prediction == dense_prediction
                ),
                "correction": bool((not current_success) and dense_success),
                "damage": bool(current_success and (not dense_success)),
            }
        )
        rows.append(row)
    migrated["rows"] = rows
    return migrated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    args = parser.parse_args()
    source_config = yaml.safe_load(args.source_config.read_text(encoding="utf-8"))
    target_config = yaml.safe_load(args.target_config.read_text(encoding="utf-8"))
    source_fingerprint = canonical_fingerprint(source_config)
    target_fingerprint = canonical_fingerprint(target_config)
    extension = target_config.get("selective_dense_extension", {})
    source_budget = int(extension.get("source_dense_max_new_tokens", -1))
    target_budget = int(extension.get("target_dense_max_new_tokens", -1))
    if not extension.get("enabled"):
        raise ValueError("selective_dense_extension.enabled must be true")
    if source_budget != int(source_config["generation"]["dense_max_new_tokens"]):
        raise ValueError("source Dense budget mismatch")
    if target_budget != int(target_config["generation"]["dense_max_new_tokens"]):
        raise ValueError("target Dense budget mismatch")
    if target_budget <= source_budget:
        raise ValueError("target Dense budget must exceed source budget")

    source_cache = Path(source_config["output_root"]) / "cache"
    target_output = Path(target_config["output_root"])
    target_cache = target_output / "cache"
    prepared = Path(target_config["data"]["prepared_root"])
    target_output.mkdir(parents=True, exist_ok=True)

    eligible: list[dict[str, Any]] = []
    migrated = already_migrated = already_extended = 0
    by_split: dict[str, Counter] = {}
    for dataset in DATASETS:
        for split in SPLITS:
            records_path = prepared / dataset / f"{split}.jsonl"
            if not records_path.is_file():
                continue
            local = Counter()
            for record in read_jsonl(records_path):
                problem_id = str(record["problem_id"])
                gold = expected_gold(dataset, record)
                source = source_cache / dataset / split / f"sample_{problem_id}.pt"
                destination = target_cache / dataset / split / f"sample_{problem_id}.pt"
                if not source.is_file():
                    raise FileNotFoundError(source)
                source_value = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
                validate_common(
                    source_value,
                    dataset=dataset,
                    split=split,
                    record=record,
                    gold=gold,
                    fingerprint=source_fingerprint,
                )
                source_tokens = list(source_value["dense"]["tokens"])
                reached_max = bool(source_value["dense"]["reached_max_tokens"])
                if reached_max:
                    if len(source_tokens) != source_budget:
                        raise ValueError(f"capped source has {len(source_tokens)} != {source_budget}: {source}")
                    eligible.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "problem_id": problem_id,
                            "source_artifact": str(source.resolve()),
                            "source_artifact_sha256": sha256(source),
                            "source_dense_tokens": len(source_tokens),
                            "source_dense_token_sha256": token_sha256(source_tokens),
                            "source_dense_prediction": source_value["dense"].get("prediction"),
                            "source_dense_success": bool(source_value["dense"].get("success")),
                        }
                    )
                    local["eligible_for_32k_extension"] += 1
                    if destination.is_file():
                        target_value = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
                        validate_common(
                            target_value,
                            dataset=dataset,
                            split=split,
                            record=record,
                            gold=gold,
                            fingerprint=target_fingerprint,
                        )
                        if target_value.get("dense_generation", {}).get("execution_mode") != "generated_at_configured_budget":
                            raise ValueError(f"invalid pre-existing extended artifact: {destination}")
                        already_extended += 1
                        local["already_extended"] += 1
                    continue

                if len(source_tokens) >= source_budget:
                    raise ValueError(f"non-capped source length is not below source budget: {source}")
                if destination.is_file():
                    target_value = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
                    validate_common(
                        target_value,
                        dataset=dataset,
                        split=split,
                        record=record,
                        gold=gold,
                        fingerprint=target_fingerprint,
                    )
                    mode = target_value.get("dense_generation", {}).get("execution_mode")
                    if mode != "reused_noncapped_source_trajectory":
                        raise ValueError(f"invalid pre-existing migrated artifact: {destination}")
                    already_migrated += 1
                    local["already_migrated"] += 1
                    continue

                migrated_value = dict(source_value)
                migrated_value.update(
                    {
                        "protocol_id": target_config["protocol_id"],
                        "protocol_fingerprint": target_fingerprint,
                        "primary_replay_view_fingerprint": target_fingerprint + ":paragraph",
                        "checkpoint_protocol": target_config["checkpoint"],
                        "source_dense_artifact": str(destination.resolve()),
                        "source_common_cache_artifact": str(destination.resolve()),
                        "dense_generation": {
                            "requested_max_new_tokens": target_budget,
                            "execution_mode": "reused_noncapped_source_trajectory",
                            "source_requested_max_new_tokens": source_budget,
                            "equivalence_reason": "source trajectory naturally terminated before the old cap",
                        },
                    }
                )
                migrated_value = refresh_labels(migrated_value, dataset, gold)
                collection = dict(migrated_value.get("collection", {}))
                collection["selective_dense_budget_migration_v3"] = {
                    "source": str(source.resolve()),
                    "source_protocol_id": source_config["protocol_id"],
                    "source_protocol_fingerprint": source_fingerprint,
                    "source_dense_max_new_tokens": source_budget,
                    "target_dense_max_new_tokens": target_budget,
                    "source_reached_max_tokens": False,
                    "generation_content_preserved_exactly": True,
                    "migrated_at": datetime.now(timezone.utc).isoformat(),
                }
                migrated_value["collection"] = collection
                atomic_torch_save(migrated_value, destination)
                migrated += 1
                local["migrated_noncapped"] += 1
            by_split[f"{dataset}/{split}"] = local

    eligible.sort(key=lambda row: (row["dataset"], row["split"], row["problem_id"]))
    manifest_payload = {
        "status": "ready_for_collection",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_protocol_id": source_config["protocol_id"],
        "source_protocol_fingerprint": source_fingerprint,
        "target_protocol_id": target_config["protocol_id"],
        "target_protocol_fingerprint": target_fingerprint,
        "source_dense_max_new_tokens": source_budget,
        "target_dense_max_new_tokens": target_budget,
        "selection_rule": "source dense.reached_max_tokens == true",
        "prefix_identity_tokens": source_budget,
        "eligible_count": len(eligible),
        "eligible": eligible,
        "counts_by_split": {
            split: int(counts["eligible_for_32k_extension"])
            for split, counts in by_split.items()
        },
    }
    manifest_path = Path(extension["manifest"])
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    migration_payload = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_protocol_id": source_config["protocol_id"],
        "source_protocol_fingerprint": source_fingerprint,
        "target_protocol_id": target_config["protocol_id"],
        "target_protocol_fingerprint": target_fingerprint,
        "migrated_noncapped": migrated,
        "already_migrated_noncapped": already_migrated,
        "already_extended": already_extended,
        "unavailable_for_collection": len(eligible) - already_extended,
        "selective_extension_manifest": str(manifest_path.resolve()),
        "splits": {name: dict(counts) for name, counts in by_split.items()},
    }
    (target_output / "CACHE_MIGRATION.json").write_text(
        json.dumps(migration_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(migration_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

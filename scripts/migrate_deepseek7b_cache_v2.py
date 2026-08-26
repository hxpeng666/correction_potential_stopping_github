#!/usr/bin/env python3
"""Audit and migrate unchanged v1 cache artifacts into the v2 protocol."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from deepseek7b_protocol_v1 import atomic_torch_save, canonical_fingerprint, success


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_identity(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record["problem_id"]),
        str(record["question"]),
    )


def expected_gold(dataset: str, record: dict[str, Any]) -> str:
    if "gold_answer" in record:
        return str(record["gold_answer"])
    if dataset == "gsm8k":
        return str(record["answer"]).rsplit("####", 1)[-1].strip().replace(",", "")
    raise KeyError(f"no gold-answer rule for {dataset}")


def numeric_labels_valid(value: dict[str, Any], dataset: str, gold: str) -> bool:
    if dataset not in {"gsm8k", "aime"}:
        return True
    dense_prediction = value.get("dense", {}).get("prediction")
    dense_success = success(dataset, gold, dense_prediction)
    if bool(value.get("dense", {}).get("success")) != dense_success:
        return False
    for row in value.get("rows", []):
        current_success = success(dataset, gold, row.get("current_prediction"))
        expected = {
            "dense_success": dense_success,
            "current_success": current_success,
            "correction": (not current_success) and dense_success,
            "damage": current_success and (not dense_success),
        }
        if any(bool(row.get(key)) != bool(expected_value) for key, expected_value in expected.items()):
            return False
    return True


def refresh_numeric_labels(
    value: dict[str, Any], dataset: str, gold: str
) -> dict[str, Any]:
    if dataset not in {"gsm8k", "aime"}:
        return value
    repaired = dict(value)
    dense = dict(repaired["dense"])
    dense_prediction = dense.get("prediction")
    dense_success = success(dataset, gold, dense_prediction)
    dense["success"] = bool(dense_success)
    repaired["dense"] = dense
    rows = []
    for original in repaired.get("rows", []):
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
    repaired["rows"] = rows
    return repaired


def valid_target(
    path: Path,
    fingerprint: str,
    identity: tuple[str, str],
    gold: str,
    dataset: str,
) -> bool:
    if not path.is_file():
        return False
    value = torch.load(path, map_location="cpu", weights_only=False)
    return (
        value.get("status") == "complete"
        and value.get("protocol_fingerprint") == fingerprint
        and record_identity(value["record"]) == identity
        and str(value.get("gold_answer")) == gold
        and value.get("actual_checkpoint_schedule") == "paragraph"
        and numeric_labels_valid(value, dataset, gold)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    args = parser.parse_args()
    source_config = yaml.safe_load(args.source_config.read_text(encoding="utf-8"))
    target_config = yaml.safe_load(args.target_config.read_text(encoding="utf-8"))
    source_fingerprint = canonical_fingerprint(source_config)
    target_fingerprint = canonical_fingerprint(target_config)
    source_cache = Path(source_config["output_root"]) / "cache"
    target_cache = Path(target_config["output_root"]) / "cache"
    prepared = Path(target_config["data"]["prepared_root"])
    migrated = already = unavailable = 0
    by_split: dict[str, dict[str, int]] = {}

    for dataset in ("gsm8k", "math", "math500", "aime"):
        for split in ("probe_train", "calibration", "heldout"):
            manifest = prepared / dataset / f"{split}.jsonl"
            if not manifest.is_file():
                continue
            local = {"migrated": 0, "already": 0, "unavailable": 0}
            for record in read_jsonl(manifest):
                problem_id = str(record["problem_id"])
                identity = record_identity(record)
                gold = expected_gold(dataset, record)
                destination = target_cache / dataset / split / f"sample_{problem_id}.pt"
                if valid_target(destination, target_fingerprint, identity, gold, dataset):
                    already += 1
                    local["already"] += 1
                    continue
                source = source_cache / dataset / split / f"sample_{problem_id}.pt"
                if not source.is_file():
                    unavailable += 1
                    local["unavailable"] += 1
                    continue
                value = torch.load(source, map_location="cpu", weights_only=False)
                failures = []
                if value.get("status") != "complete":
                    failures.append("status")
                if value.get("protocol_fingerprint") != source_fingerprint:
                    failures.append("source fingerprint")
                if value.get("actual_checkpoint_schedule") != "paragraph":
                    failures.append("checkpoint schedule")
                if record_identity(value["record"]) != identity:
                    failures.append("record identity")
                if str(value.get("gold_answer")) != gold:
                    failures.append("gold answer")
                if str(value.get("dataset")) != dataset or str(value.get("split")) != split:
                    failures.append("dataset/split")
                decoding = value.get("forced_answer_decoding", {})
                if decoding.get("strategy") != "greedy_argmax" or int(
                    decoding.get("max_new_tokens", -1)
                ) != int(target_config["generation"]["force_answer_max_new_tokens"]):
                    failures.append("forced-answer decoding")
                if int(value["model_audit"]["hidden_size"]) != int(
                    target_config["model"]["hidden_size"]
                ):
                    failures.append("model hidden size")
                if failures:
                    raise ValueError(f"unsafe migration {source}: {failures}")
                migrated_value = dict(value)
                migrated_value.update(
                    {
                        "protocol_id": target_config["protocol_id"],
                        "protocol_fingerprint": target_fingerprint,
                        "primary_replay_view_fingerprint": target_fingerprint + ":paragraph",
                        "checkpoint_protocol": target_config["checkpoint"],
                        "source_dense_artifact": str(destination.resolve()),
                        "source_common_cache_artifact": str(destination.resolve()),
                    }
                )
                migrated_value = refresh_numeric_labels(migrated_value, dataset, gold)
                collection = dict(migrated_value.get("collection", {}))
                collection["migration"] = {
                    "source": str(source.resolve()),
                    "source_protocol_id": source_config["protocol_id"],
                    "source_protocol_fingerprint": source_fingerprint,
                    "migrated_at": datetime.now(timezone.utc).isoformat(),
                    "validation": (
                        "exact problem_id/question/gold/dataset/split plus decoding/model audit "
                        "and recomputed numeric success-derived labels"
                    ),
                }
                migrated_value["collection"] = collection
                atomic_torch_save(migrated_value, destination)
                migrated += 1
                local["migrated"] += 1
            by_split[f"{dataset}/{split}"] = local
    output = Path(target_config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "complete",
        "source_protocol_id": source_config["protocol_id"],
        "source_protocol_fingerprint": source_fingerprint,
        "target_protocol_id": target_config["protocol_id"],
        "target_protocol_fingerprint": target_fingerprint,
        "migrated": migrated,
        "already": already,
        "unavailable_for_collection": unavailable,
        "splits": by_split,
    }
    (output / "CACHE_MIGRATION.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

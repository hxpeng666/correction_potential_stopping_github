#!/usr/bin/env python3
"""Full sample/checkpoint/parser/integrity audit for greedy forced-answer v1."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from greedy_forced_common_v1 import (
    atomic_json,
    load_config,
    output_path,
    protocol_fingerprint,
    queue_counts,
    resolve,
    source_split_path,
)
from src.final_paper_inference import prediction_for, success_for
from src.mmlu_pro_protocol import parse_answer as parse_mmlu_pro_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_paper_greedy_forced_v1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    fingerprint = protocol_fingerprint(config)
    errors: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    branch_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    missing_predictions: Counter[str] = Counter()
    device_counts: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()
    for dataset, dataset_config in config["datasets"].items():
        counts[dataset] = {}
        for split, expected in dataset_config["expected_counts"].items():
            sources = sorted(source_split_path(dataset_config, split).glob("sample_*.pt"))
            counts[dataset][split] = 0
            if len(sources) != int(expected):
                errors.append(f"source count mismatch {dataset}/{split}")
            for source_path in sources:
                problem_id = source_path.stem.removeprefix("sample_")
                key = (dataset, split, problem_id)
                if key in seen:
                    errors.append(f"duplicate problem key {key}")
                    continue
                seen.add(key)
                destination = output_path(config, dataset, split, problem_id)
                if not destination.is_file():
                    errors.append(f"missing artifact {destination}")
                    continue
                try:
                    source = torch.load(source_path, map_location="cpu", weights_only=False)
                    value = torch.load(destination, map_location="cpu", weights_only=False)
                except Exception as error:
                    errors.append(f"unreadable {destination}: {error!r}")
                    continue
                if value.get("status") != "complete" or value.get("protocol_fingerprint") != fingerprint:
                    errors.append(f"protocol/status mismatch {destination}")
                    continue
                if value.get("source_protocol_fingerprint") != dataset_config["source_protocol_fingerprint"]:
                    errors.append(f"source fingerprint mismatch {destination}")
                decoding = value.get("forced_answer_decoding", {})
                if decoding.get("strategy") != "greedy_argmax" or decoding.get("do_sample") is not False:
                    errors.append(f"not greedy {destination}")
                source_checkpoints = [int(row["checkpoint"]) for row in source["rows"]]
                output_checkpoints = [int(row["checkpoint"]) for row in value["rows"]]
                if source_checkpoints != output_checkpoints:
                    errors.append(f"checkpoint mismatch {destination}")
                if len(value["rows"]) != int(value["hidden"].shape[0]):
                    errors.append(f"row/vector mismatch {destination}")
                if not torch.equal(value["hidden"], source["hidden"]):
                    errors.append(f"hidden changed {destination}")
                if value["dense"]["tokens"] != source["dense"]["tokens"]:
                    errors.append(f"dense tokens changed {destination}")
                if value.get("direct") != source.get("direct"):
                    errors.append(f"direct baseline changed {destination}")
                device_counts[value["forced_answer_collection"]["device"]] += 1
                for row in value["rows"]:
                    branch_counts[dataset] += 1
                    if row.get("forced_answer_do_sample") is not False or row.get("forced_answer_decoding") != "greedy_argmax":
                        errors.append(f"row decoding mismatch {destination}:{row.get('checkpoint')}")
                        break
                    if len(row.get("branch_token_ids", [])) != int(row["branch_tokens"]):
                        errors.append(f"branch token mismatch {destination}:{row.get('checkpoint')}")
                        break
                    if dataset == "mmlu_pro":
                        prediction = parse_mmlu_pro_answer(
                            row["branch_text"], int(value["record"]["option_count"])
                        )
                    else:
                        prediction = prediction_for(dataset, row["branch_text"])
                    success = success_for(dataset, value["gold_answer"], prediction)
                    if prediction != row["current_prediction"] or bool(success) != bool(row["current_success"]):
                        errors.append(f"parser mismatch {destination}:{row.get('checkpoint')}")
                        break
                    if prediction is None:
                        missing_predictions[dataset] += 1
                    current = "C" if success else "W"
                    final = "C" if value["dense"]["success"] else "W"
                    transition_counts[f"{dataset}:{current}->{final}"] += 1
                counts[dataset][split] += 1
    queue = queue_counts(config)
    expected_total = sum(
        int(count)
        for dataset in config["datasets"].values()
        for count in dataset["expected_counts"].values()
    )
    if any(queue[state] for state in ("pending", "claimed", "failed", "requires_a100")):
        errors.append(f"queue incomplete: {queue}")
    if queue["done"] != expected_total:
        errors.append(f"done count mismatch: {queue['done']} != {expected_total}")
    report = {
        "status": "passed" if not errors else "failed",
        "protocol_id": config["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "expected_samples": expected_total,
        "validated_samples": sum(sum(x.values()) for x in counts.values()),
        "sample_counts": counts,
        "branch_counts": dict(branch_counts),
        "transition_counts": dict(transition_counts),
        "missing_predictions": dict(missing_predictions),
        "collection_devices": dict(device_counts),
        "queue": queue,
        "errors": errors[:200],
        "error_count": len(errors),
    }
    root = resolve(config["output_root"])
    atomic_json(report, root / "COLLECTION_AUDIT.json")
    if not errors:
        atomic_json(
            {
                "status": "complete",
                "protocol_id": config["protocol_id"],
                "protocol_fingerprint": fingerprint,
                "samples": expected_total,
                "branches": sum(branch_counts.values()),
            },
            root / "collection.complete",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

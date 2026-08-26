#!/usr/bin/env python3
"""Strict completion audit for the BLADE reproduction."""
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

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "src").is_dir():
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from src.utils import load_yaml


def fingerprint(config: dict[str, Any]) -> str:
    payload = {
        "protocol_id": config["protocol_id"],
        "model": config["model"],
        "datasets": config["datasets"],
        "checkpoints": config["checkpoints"],
        "strict_clean_supervision": config["strict_clean_supervision"],
        "apls": config["apls"],
        "training": config["training"],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def collection_fingerprint(config: dict[str, Any], dataset: str) -> str:
    payload = {
        "protocol_id": config["protocol_id"],
        "source": config["source"],
        "model": config["model"],
        "common_scope": config["common_scope"],
        "dataset": dataset,
        "dataset_config": config["datasets"][dataset],
        "checkpoints": config["checkpoints"],
        "strict_clean_supervision": config["strict_clean_supervision"],
        "apls_capture": {
            "decoder_layers": config["model"]["decoder_layers"],
            "hidden_size": config["model"]["hidden_size"],
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    output = ROOT / config["output_root"]
    protocol_fingerprint = fingerprint(config)
    failures = []
    collection = {}
    for dataset, dataset_config in config["datasets"].items():
        collection[dataset] = {}
        expected_collection_fingerprint = collection_fingerprint(config, dataset)
        for split in ("probe_train", "calibration", "heldout"):
            paths = sorted((output / dataset / "cache" / split).glob("sample_*.pt"))
            expected = int(dataset_config[split])
            if len(paths) != expected:
                failures.append(f"{dataset}/{split}: expected {expected} artifacts, found {len(paths)}")
            checkpoints = clean = ambiguous = paragraphs = sentences = doubts = 0
            ids = set()
            for path in paths:
                value = torch.load(path, map_location="cpu", weights_only=False)
                problem_id = str(value.get("problem_id"))
                if problem_id in ids:
                    failures.append(f"{dataset}/{split}: duplicate problem id {problem_id}")
                ids.add(problem_id)
                if value.get("status") != "complete" or value.get("protocol_fingerprint") != expected_collection_fingerprint:
                    failures.append(f"{path}: invalid status/fingerprint")
                    continue
                hidden = value.get("hidden", torch.empty(0))
                expected_shape = (int(config["model"]["decoder_layers"]), int(config["model"]["hidden_size"]))
                if hidden.ndim != 3 or tuple(hidden.shape[1:]) != expected_shape or hidden.shape[0] != len(value.get("rows", [])):
                    failures.append(f"{path}: hidden/row shape mismatch {tuple(hidden.shape)}")
                previous = -1
                for row in value.get("rows", []):
                    checkpoint = int(row["checkpoint"])
                    if checkpoint <= previous:
                        failures.append(f"{path}: checkpoints not strictly increasing")
                    previous = checkpoint
                    checkpoints += 1
                    paragraphs += int(bool(row.get("is_paragraph")))
                    sentences += int(bool(row.get("is_sentence")))
                    doubts += int(bool(row.get("is_self_doubt")))
                    label = row.get("strict_clean_label")
                    completions = row.get("strict_clean_completions", [])
                    if split in ("probe_train", "calibration"):
                        if len(completions) != int(config["strict_clean_supervision"]["completions_per_checkpoint"]):
                            failures.append(f"{path}: checkpoint {checkpoint} lacks K16 completions")
                        correct = sum(bool(item["success"]) for item in completions)
                        expected_label = 1 if correct == len(completions) else 0 if correct == 0 else None
                        if label != expected_label:
                            failures.append(f"{path}: checkpoint {checkpoint} strict-clean label mismatch")
                        clean += int(label in (0, 1))
                        ambiguous += int(label is None)
                    elif completions or label is not None:
                        failures.append(f"{path}: heldout must not contain K16 supervision")
                    if split != "probe_train" and row.get("is_paragraph") and not (
                        row.get("is_sentence") or row.get("is_self_doubt")
                    ):
                        failures.append(f"{path}: paragraph-only checkpoint leaked into inference split")
            collection[dataset][split] = {
                "artifacts": len(paths), "unique_problem_ids": len(ids), "checkpoints": checkpoints,
                "strict_clean": clean, "ambiguous": ambiguous, "sentence": sentences,
                "self_doubt": doubts, "paragraph": paragraphs,
            }

    required_json = [
        output / "packed" / "PACK_COMPLETE.json",
        output / "projected" / "PROJECT_COMPLETE.json",
        output / "models" / "selected_layers.json",
        output / "RESULTS_ALL_DELTAS.json",
    ]
    for path in required_json:
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "complete" or value.get("protocol_fingerprint") != protocol_fingerprint:
            failures.append(f"invalid {path}")
    required_torch = [output / "models" / "dense_teacher.pt", output / "models" / "compact_probe.pt"]
    required_torch.extend(output / "selectors" / f"selector_{index:02d}.pt" for index in range(len(config["apls"]["selection_seeds"])))
    for path in required_torch:
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("status") != "complete" or value.get("protocol_fingerprint") != protocol_fingerprint:
            failures.append(f"invalid {path}")

    selected_path = output / "models" / "selected_layers.json"
    selected = None
    if selected_path.is_file():
        selected_value = json.loads(selected_path.read_text(encoding="utf-8"))
        selected = selected_value.get("selected_layers_zero_based")
        if not isinstance(selected, list) or len(selected) != int(config["apls"]["selected_layers"]) or len(set(selected)) != len(selected):
            failures.append(f"selected layer set is not unique K={config['apls']['selected_layers']}")

    results_path = output / "RESULTS_ALL_DELTAS.json"
    result_rows = []
    if results_path.is_file():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        result_rows = results.get("rows", [])
        expected_rows = len(config["datasets"]) * len(config["calibration"]["deltas"])
        if len(result_rows) != expected_rows:
            failures.append(f"expected {expected_rows} result rows, found {len(result_rows)}")
        keys = Counter((row.get("dataset"), row.get("delta")) for row in result_rows)
        if any(count != 1 for count in keys.values()) or len(keys) != expected_rows:
            failures.append("result dataset/delta keys are incomplete or duplicated")
        for row in result_rows:
            dataset = row.get("dataset")
            if int(row.get("heldout_n", -1)) != int(config["datasets"][dataset]["heldout"]):
                failures.append(f"{dataset}/{row.get('delta')}: wrong heldout n")
            if int(row.get("calibration_n", -1)) != int(config["datasets"][dataset]["calibration"]):
                failures.append(f"{dataset}/{row.get('delta')}: wrong calibration n")
            if not 0 <= float(row.get("heldout_accuracy", -1)) <= 1:
                failures.append(f"{dataset}/{row.get('delta')}: invalid accuracy")
            if not math_isfinite(row.get("heldout_aes")):
                failures.append(f"{dataset}/{row.get('delta')}: invalid AES")

    record = {
        "status": "complete" if not failures else "failed",
        "protocol_id": config["protocol_id"],
        "protocol_fingerprint": protocol_fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection": collection,
        "selected_layers_zero_based": selected,
        "result_rows": len(result_rows),
        "failures": failures,
    }
    atomic_json(record, output / "COMPLETION_AUDIT.json")
    print(json.dumps(record, indent=2))
    if failures:
        raise SystemExit(2)
    atomic_json({
        "status": "complete", "protocol_id": config["protocol_id"],
        "protocol_fingerprint": protocol_fingerprint, "created_at": datetime.now(timezone.utc).isoformat(),
        "completion_audit": "COMPLETION_AUDIT.json",
    }, output / "EXPERIMENT_COMPLETE.json")


def math_isfinite(value: Any) -> bool:
    try:
        return bool(torch.isfinite(torch.tensor(float(value))).item())
    except Exception:
        return False


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed audit for the paired checkpoint-schedule cache."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from collect_full_checkpoint_schedule_v1 import SCHEDULES, canonical_fingerprint
from src.legacy_empirical_probe_v4 import load_checkpoint_split
from src.utils import load_yaml


def quantiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    return {key: float(value) for key, value in zip(("min", "p25", "median", "p75", "p90", "max", "mean"), (*np.quantile(data, [0, .25, .5, .75, .9, 1]), data.mean()))}


def source_ids(root: Path, split: str) -> set[str]:
    return {path.stem.removeprefix("sample_") for path in (root / split).glob("sample_*.pt")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), default="gsm8k")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    fingerprint = canonical_fingerprint(config)
    output_root = ROOT / config["output_root"]
    cache_root = output_root / "cache"
    dataset_config = config["datasets"][args.dataset]
    source_root = ROOT / dataset_config["source_root"]
    report: dict[str, Any] = {"status": "complete", "protocol_fingerprint": fingerprint, "schedules": {}, "errors": []}
    expected_by_split = {split: source_ids(source_root, split) for split in ("probe_train", "calibration", "heldout")}
    declared_counts = {split: int(dataset_config[split]) for split in ("probe_train", "calibration", "heldout")}
    if {key: len(value) for key, value in expected_by_split.items()} != declared_counts:
        report["errors"].append(f"source split counts do not match frozen counts: {declared_counts}")

    active_schedules = [str(value) for value in config["checkpoint"]["schedules"]]
    for schedule in active_schedules:
        schedule_report: dict[str, Any] = {"splits": {}}
        all_counts: list[float] = []
        all_first: list[float] = []
        all_last: list[float] = []
        all_ends: list[float] = []
        fallbacks = 0
        label_totals = Counter()
        row_total = 0
        for split, expected in expected_by_split.items():
            directory = cache_root / schedule / split
            paths = sorted(directory.glob("sample_*.pt"))
            actual = {path.stem.removeprefix("sample_") for path in paths}
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            if missing or extra:
                report["errors"].append(f"{schedule}/{split}: missing={len(missing)} extra={len(extra)}")
            local_counts: list[float] = []
            for path in paths:
                artifact = torch.load(path, map_location="cpu", weights_only=False)
                rows = artifact.get("rows", [])
                hidden = artifact.get("hidden")
                problem_id = str(artifact.get("problem_id"))
                if artifact.get("status") != "complete" or artifact.get("protocol_fingerprint") != fingerprint or artifact.get("actual_checkpoint_schedule") != schedule:
                    report["errors"].append(f"identity mismatch: {path}")
                    continue
                if not torch.is_tensor(hidden) or hidden.ndim != 3 or tuple(hidden.shape[1:]) != (1, 2560) or len(rows) != int(hidden.shape[0]):
                    report["errors"].append(f"row/hidden mismatch: {path}")
                    continue
                checkpoints = [int(row["checkpoint"]) for row in rows]
                reasoning_end = int(artifact["trajectory"]["reasoning_end"])
                if checkpoints != sorted(set(checkpoints)) or any(value <= 0 or value > reasoning_end for value in checkpoints):
                    report["errors"].append(f"checkpoint legality mismatch: {path}")
                if any(row.get("forced_answer_do_sample") is not False or row.get("forced_answer_decoding") != "greedy_argmax" for row in rows):
                    report["errors"].append(f"non-greedy row: {path}")
                if problem_id not in expected:
                    report["errors"].append(f"problem id outside split: {path}")
                local_counts.append(float(len(rows)))
                all_counts.append(float(len(rows)))
                all_ends.append(float(reasoning_end))
                if rows:
                    all_first.append(float(checkpoints[0]))
                    all_last.append(float(checkpoints[-1]))
                else:
                    fallbacks += 1
            schedule_report["splits"][split] = {"files": len(paths), "checkpoint_count": quantiles(local_counts) if local_counts else None}
            if len(paths) == len(expected):
                try:
                    frame, _hidden, _layers, local_fallbacks = load_checkpoint_split(directory, "sentence")
                    row_total += len(frame)
                    fallbacks = max(fallbacks, len(local_fallbacks)) if split == "probe_train" else fallbacks + len(local_fallbacks)
                    for target in ("correctness", "consistency", "last_switch", "correction"):
                        label_totals[target + "_positive"] += int(frame[f"target_{target}"].sum())
                        label_totals[target + "_total"] += len(frame)
                except ValueError as error:
                    report["errors"].append(f"loader failed {schedule}/{split}: {error}")
        schedule_report.update({
            "checkpoint_count_all": quantiles(all_counts) if all_counts else None,
            "first_checkpoint": quantiles(all_first) if all_first else None,
            "last_checkpoint": quantiles(all_last) if all_last else None,
            "reasoning_end": quantiles(all_ends) if all_ends else None,
            "zero_checkpoint_problems": int(sum(value == 0 for value in all_counts)),
            "scorable_rows": row_total,
            "target_positive_rates": {target: label_totals[target + "_positive"] / max(1, label_totals[target + "_total"]) for target in ("correctness", "consistency", "last_switch", "correction")},
        })
        report["schedules"][schedule] = schedule_report
    temporary_files = list(cache_root.rglob(".*.tmp.*"))
    if temporary_files:
        report["errors"].append(f"temporary files remain: {len(temporary_files)}")
    if report["errors"]:
        report["status"] = "failed"
    destination = args.output or (output_root / "cache_audit.json")
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

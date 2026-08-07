#!/usr/bin/env python3
"""将完整轨迹及特征与直接作答及强制作答分支合并为不可变缓存。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.final_paper_inference import atomic_torch_save
from src.final_paper_cache import (
    BRANCH_DIRECT,
    artifact_matches,
    branch_path,
    cache_paths,
)
from src.utils import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("probe_train", "calibration", "heldout"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cache_root = args.cache_root if args.cache_root.is_absolute() else ROOT / args.cache_root
    dense_paths = sorted((cache_root / "dense" / args.split).glob("sample_*.pt"))
    completed = skipped = missing = 0
    for dense_path in dense_paths:
        dense = torch.load(dense_path, map_location="cpu", weights_only=False)
        problem_id = str(dense["problem_id"])
        fingerprint = str(dense["protocol_fingerprint"])
        destination = cache_paths(cache_root, args.split, problem_id)["merged"]
        if args.resume and artifact_matches(destination, problem_id=problem_id, fingerprint=fingerprint):
            skipped += 1
            continue
        if destination.exists():
            raise RuntimeError(f"incompatible merged artifact preserved: {destination}")
        direct_path = branch_path(cache_root, args.split, problem_id, BRANCH_DIRECT)
        if not artifact_matches(direct_path, problem_id=problem_id, fingerprint=fingerprint):
            missing += 1
            continue
        direct = torch.load(direct_path, map_location="cpu", weights_only=False)
        rows = []
        local_missing = False
        for base in dense["rows"]:
            checkpoint = int(base["checkpoint"])
            path = branch_path(cache_root, args.split, problem_id, checkpoint)
            if not artifact_matches(path, problem_id=problem_id, fingerprint=fingerprint):
                local_missing = True
                break
            branch = torch.load(path, map_location="cpu", weights_only=False)
            prediction = branch["prediction"]
            current_success = bool(branch["success"])
            dense_prediction = dense["dense"]["prediction"]
            row = dict(base)
            row.update(
                {
                    "current_prediction": prediction,
                    "current_success": current_success,
                    "consistency": bool(
                        prediction is not None
                        and dense_prediction is not None
                        and prediction == dense_prediction
                    ),
                    "correction": bool(
                        (not current_success) and dense["dense"]["success"]
                    ),
                    "damage": bool(
                        current_success and (not dense["dense"]["success"])
                    ),
                    "branch_tokens": int(branch["generated_tokens"]),
                    "forced_context_tokens": int(branch["context_tokens"]),
                    "branch_text": branch["text"],
                    "branch_generated_text": branch["generated_text"],
                    "branch_generation_seed": int(branch["generation_seed"]),
                    "branch_timing_source": "excluded_from_replay_cost_model",
                }
            )
            rows.append(row)
        if local_missing:
            missing += 1
            continue
        merged: dict[str, Any] = {
            "schema_version": 3,
            "status": "complete",
            "protocol_id": "final_paper_replay_v2",
            "protocol_fingerprint": fingerprint,
            "dataset": dense["dataset"],
            "split": dense["split"],
            "seed": dense["seed"],
            "problem_id": problem_id,
            "dense_generation_seed": int(dense["generation_seed"]),
            "record": dense["record"],
            "dense_content_tokens": len(dense["dense"]["content_tokens"]),
            "gold_answer": dense["gold_answer"],
            "model_audit": dense["model_audit"],
            "dtype": dense["dtype"],
            "attention_backend": dense["attention_backend"],
            "collection_device": dense["collection_device"],
            "collection_device_index": dense["collection_device_index"],
            "timing_valid": bool(dense.get("timing_valid", False)),
            "timing_mode": dense.get("timing_mode", "not_collected"),
            "prompt_text": dense["prompt_text"],
            "prompt_tokens": dense["prompt_tokens"],
            "dense": dense["dense"],
            "direct": direct,
            "capture_layers": dense["capture_layers"],
            "checkpoint_protocol": dense["checkpoint_protocol"],
            "schedules": dense["schedules"],
            "rows": rows,
            "hidden": dense["hidden"],
            "source_dense_artifact": str(destination),
            "latency_label": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if len(rows) != int(merged["hidden"].shape[0]):
            raise RuntimeError(f"row/vector mismatch for {problem_id}")
        atomic_torch_save(merged, destination)
        completed += 1
    summary = {
        "status": "complete" if missing == 0 else "incomplete",
        "cache_root": str(cache_root),
        "split": args.split,
        "dense_samples": len(dense_paths),
        "completed_now": completed,
        "skipped": skipped,
        "samples_missing_branches": missing,
    }
    atomic_json(summary, cache_root / "merged" / args.split / "merge_summary.json")
    print(json.dumps(summary, indent=2))
    if missing:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

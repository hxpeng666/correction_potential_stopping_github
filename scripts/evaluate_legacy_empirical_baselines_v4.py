#!/usr/bin/env python3
"""Evaluate legacy-v4 Dense, Direct, and fixed-budget baselines from one shared cache."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.final_paper_inference import atomic_torch_save
from src.legacy_empirical_probe_v4 import summarize_policy_records, transition_name
from src.utils import atomic_json, load_yaml


def dense_direct_records(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dense_records = []
    direct_records = []
    for path in paths:
        source = torch.load(path, map_location="cpu", weights_only=False)
        if source.get("status") != "complete":
            raise ValueError(f"incomplete Dense artifact: {path}")
        dense = source["dense"]
        direct = source["direct"]
        common = {
            "problem_id": str(source["problem_id"]),
            "subject": source["record"].get("subject"),
            "category": source["record"].get("category"),
            "gold_answer": source["gold_answer"],
        }
        dense_records.append(
            {
                **common,
                "prediction": dense["prediction"],
                "success": bool(dense["success"]),
                "reasoning_tokens": int(dense["reasoning_tokens"]),
                "wall_ms": float(dense["wall_ms"]),
                "reached_max_tokens": bool(dense["reached_max_tokens"]),
            }
        )
        direct_records.append(
            {
                **common,
                "fallback": False,
                "checkpoint": 0,
                "transition": transition_name(
                    bool(direct["success"]), bool(dense["success"])
                ),
                "method_prediction": direct["prediction"],
                "dense_prediction": dense["prediction"],
                "method_success": bool(direct["success"]),
                "dense_success": bool(dense["success"]),
                "method_tokens": int(direct["generated_tokens"]),
                "dense_tokens": int(dense["reasoning_tokens"]),
                "replay_wall_ms": float(direct["wall_ms"]),
                "dense_wall_ms": float(dense["wall_ms"]),
            }
        )
    return dense_records, direct_records


def summarize_dense(records: list[dict[str, Any]]) -> dict[str, Any]:
    success = np.asarray([row["success"] for row in records], dtype=float)
    tokens = np.asarray([row["reasoning_tokens"] for row in records], dtype=float)
    wall = np.asarray([row["wall_ms"] for row in records], dtype=float)
    return {
        "problems": len(records),
        "accuracy": float(success.mean()),
        "mean_reasoning_tokens": float(tokens.mean()),
        "median_reasoning_tokens": float(np.median(tokens)),
        "p95_reasoning_tokens": float(np.percentile(tokens, 95)),
        "mean_wall_ms": float(wall.mean()),
        "median_wall_ms": float(np.median(wall)),
        "p95_wall_ms": float(np.percentile(wall, 95)),
        "reached_max_count": int(sum(row["reached_max_tokens"] for row in records)),
        "reached_max_rate": float(np.mean([
            row["reached_max_tokens"] for row in records
        ])),
    }


def fixed_records(
    dense_paths: list[Path],
    checkpoint_root: Path,
    budget: int,
) -> list[dict[str, Any]]:
    records = []
    for dense_path in dense_paths:
        source = torch.load(dense_path, map_location="cpu", weights_only=False)
        problem_id = str(source["problem_id"])
        checkpoint_path = checkpoint_root / f"sample_{problem_id}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if artifact.get("status") != "complete":
            raise ValueError(f"incomplete checkpoint artifact: {checkpoint_path}")
        exact = [
            row for row in artifact["rows"]
            if int(row["checkpoint"]) == budget
            and "fixed" in row.get("checkpoint_schedules", [])
        ]
        dense = source["dense"]
        fallback = not exact
        if fallback:
            method_success = bool(dense["success"])
            method_prediction = dense["prediction"]
            method_tokens = int(dense["reasoning_tokens"])
            method_wall = float(dense["wall_ms"])
            checkpoint = None
            transition = "fallback"
        else:
            row = exact[0]
            method_success = bool(row["current_success"])
            method_prediction = row["current_prediction"]
            method_tokens = min(
                int(dense["reasoning_tokens"]),
                int(row["checkpoint"]) + int(row["branch_tokens"]),
            )
            method_wall = float(
                row["dense_prefill_cuda_ms"]
                + row["prefix_decode_cuda_ms"]
                + row["branch_wall_ms"]
            )
            checkpoint = int(row["checkpoint"])
            transition = transition_name(
                method_success, bool(dense["success"])
            )
        records.append(
            {
                "problem_id": problem_id,
                "subject": source["record"].get("subject"),
                "category": source["record"].get("category"),
                "fallback": fallback,
                "checkpoint": checkpoint,
                "transition": transition,
                "method_prediction": method_prediction,
                "dense_prediction": dense["prediction"],
                "gold_answer": source["gold_answer"],
                "method_success": method_success,
                "dense_success": bool(dense["success"]),
                "method_tokens": method_tokens,
                "dense_tokens": int(dense["reasoning_tokens"]),
                "replay_wall_ms": method_wall,
                "dense_wall_ms": float(dense["wall_ms"]),
            }
        )
    return records


def checkpoint_diagnostics(paths: list[Path]) -> dict[str, Any]:
    sentence_counts = []
    positions = []
    no_legal = 0
    shorter = 0
    for path in paths:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        sentence = [int(value) for value in artifact["schedules"]["sentence"]]
        sentence_counts.append(len(sentence))
        positions.extend(sentence)
        no_legal += int(not sentence)
        shorter += int(int(artifact["dense_content_tokens"]) < 64)
    position_array = np.asarray(positions, dtype=float)
    return {
        "problems": len(paths),
        "mean_sentence_checkpoints": float(np.mean(sentence_counts)),
        "no_legal_checkpoint_count": no_legal,
        "no_legal_checkpoint_rate": float(no_legal / len(paths)),
        "dense_shorter_than_64_count": shorter,
        "dense_shorter_than_64_rate": float(shorter / len(paths)),
        "sentence_checkpoint_positions": {
            "count": len(positions),
            "mean": float(position_array.mean()) if len(position_array) else None,
            "median": float(np.median(position_array)) if len(position_array) else None,
            "p05": float(np.percentile(position_array, 5)) if len(position_array) else None,
            "p95": float(np.percentile(position_array, 95)) if len(position_array) else None,
            "minimum": int(position_array.min()) if len(position_array) else None,
            "maximum": int(position_array.max()) if len(position_array) else None,
        },
    }


def subject_accuracy(records: list[dict[str, Any]], success_key: str) -> dict[str, Any]:
    output = {}
    subjects = sorted({
        str(row["subject"]) for row in records if row.get("subject") is not None
    })
    for subject in subjects:
        local = [row for row in records if row.get("subject") == subject]
        output[subject] = {
            "n": len(local),
            "accuracy": float(np.mean([row[success_key] for row in local])),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    destination = args.output if args.output.is_absolute() else ROOT / args.output
    marker = destination / "phase.complete"
    if args.resume and marker.is_file() and (destination / "baselines.json").is_file():
        print(json.dumps({"status": "skipped_complete", "output": str(destination)}))
        return
    destination.mkdir(parents=True, exist_ok=True)
    config = load_yaml(args.config)
    summaries: dict[str, Any] = {}
    all_records: dict[str, Any] = {}
    for split in ("calibration", "heldout"):
        dense_paths = sorted((args.dense_root / split).glob("sample_*.pt"))
        checkpoint_paths = sorted((args.checkpoint_root / split).glob("sample_*.pt"))
        if not dense_paths or len(dense_paths) != len(checkpoint_paths):
            raise ValueError(
                f"artifact count mismatch for {split}: "
                f"dense={len(dense_paths)} checkpoint={len(checkpoint_paths)}"
            )
        dense, direct = dense_direct_records(dense_paths)
        fixed = {}
        fixed_record_map = {}
        for budget in config["generation"]["fixed_budgets"]:
            records = fixed_records(
                dense_paths, args.checkpoint_root / split, int(budget)
            )
            fixed[str(budget)] = summarize_policy_records(records)
            fixed_record_map[str(budget)] = records
        summary = {
            "dense": summarize_dense(dense),
            "direct": summarize_policy_records(direct),
            "fixed": fixed,
            "checkpoint_diagnostics": checkpoint_diagnostics(checkpoint_paths),
        }
        if args.dataset == "mmlu":
            summary["dense_subjects"] = subject_accuracy(dense, "success")
            summary["direct_subjects"] = subject_accuracy(direct, "method_success")
        summaries[split] = summary
        all_records[split] = {
            "dense": dense,
            "direct": direct,
            "fixed": fixed_record_map,
        }
    heldout_paths = sorted((args.dense_root / "heldout").glob("sample_*.pt"))
    replay_v2 = bool(
        heldout_paths
        and torch.load(heldout_paths[0], map_location="cpu", weights_only=False).get("latency_label") == "A100 single-request replay-estimated latency"
    )
    payload = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dense_root": str(args.dense_root),
        "checkpoint_root": str(args.checkpoint_root),
        "summaries": summaries,
        "timing_note": (
            "All latency values are A100 single-request replay-estimated latency under the frozen cost model; collection timings and probe-check overhead are excluded."
            if replay_v2 else "Dense/Direct are actual isolated per-sample generation timings from collection; fixed-budget values are cached trajectory replay estimates."
        ),
    }
    payload["latency_label"] = "A100 single-request replay-estimated latency"
    atomic_json(payload, destination / "baselines.json")
    atomic_torch_save(
        {"status": "complete", "records": all_records},
        destination / "baseline_records.pt",
    )
    atomic_json(
        {
            "status": "complete",
            "artifacts": ["baselines.json", "baseline_records.pt"],
        },
        marker,
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate online checkpoint/score parity against cached replay without tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.utils import atomic_json


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--probe-run", type=Path, required=True)
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    scores_artifact = torch.load(
        args.probe_run / "scores.pt", map_location="cpu", weights_only=False
    )
    scores = scores_artifact["scores"][args.split].numpy()
    ids = scores_artifact["problem_ids"][args.split]
    checkpoints = scores_artifact["checkpoints"][args.split]
    score_map = {
        (str(problem_id), int(checkpoint)): float(score)
        for problem_id, checkpoint, score in zip(ids, checkpoints, scores)
    }
    rows = []
    failures = []
    for online_path in sorted((args.online_root / "raw").glob("sample_*.pt")):
        online = torch.load(online_path, map_location="cpu", weights_only=False)
        problem_id = str(online["problem_id"])
        checkpoint_path = args.checkpoint_root / args.split / f"sample_{problem_id}.pt"
        dense_path = args.dense_root / args.split / f"sample_{problem_id}.pt"
        checkpoint_artifact = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        dense_artifact = torch.load(dense_path, map_location="cpu", weights_only=False)
        offline_schedule = [
            int(value) for value in checkpoint_artifact["schedules"]["sentence"]
        ]
        dense_online = online["runs"]["dense"][0]
        dense_match = (
            dense_online["reasoning_tokens"]
            == dense_artifact["dense"]["reasoning_tokens"]
            and text_hash(dense_online["text"])
            == text_hash(dense_artifact["dense"]["text"])
            and dense_online["prediction"] == dense_artifact["dense"]["prediction"]
        )
        if not dense_match:
            failures.append(f"{problem_id}: dense trajectory mismatch")
        for method, repetitions in online["runs"].items():
            if method == "dense":
                continue
            for repetition in repetitions:
                observed = repetition["checkpoints_evaluated"]
                observed_positions = [int(row["checkpoint"]) for row in observed]
                final_position = (
                    int(repetition["stop_checkpoint"])
                    if repetition["stopped"]
                    else (offline_schedule[-1] if offline_schedule else 0)
                )
                expected_positions = [
                    value for value in offline_schedule if value <= final_position
                ]
                schedule_match = observed_positions == expected_positions
                differences = [
                    abs(float(row["score"]) - score_map[(problem_id, int(row["checkpoint"]))])
                    for row in observed
                ]
                maximum_difference = max(differences, default=0.0)
                if not schedule_match:
                    failures.append(
                        f"{problem_id}/{method}: online {observed_positions} != offline {expected_positions}"
                    )
                if maximum_difference > args.score_tolerance:
                    failures.append(
                        f"{problem_id}/{method}: score difference {maximum_difference}"
                    )
                rows.append(
                    {
                        "problem_id": problem_id,
                        "method": method,
                        "dense_match": dense_match,
                        "online_checkpoints": observed_positions,
                        "offline_prefix_checkpoints": expected_positions,
                        "schedule_match": schedule_match,
                        "maximum_score_difference": maximum_difference,
                        "intermediate_answer_branches": 0,
                    }
                )
    if not rows:
        raise FileNotFoundError(f"no online artifacts in {args.online_root}")
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "score_tolerance": args.score_tolerance,
        "comparisons": len(rows),
        "maximum_score_difference": max(
            row["maximum_score_difference"] for row in rows
        ),
        "all_dense_trajectories_match": all(row["dense_match"] for row in rows),
        "all_checkpoint_schedules_match": all(row["schedule_match"] for row in rows),
        "intermediate_answer_branches": 0,
        "failures": failures,
        "rows": rows,
    }
    atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

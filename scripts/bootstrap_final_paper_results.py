#!/usr/bin/env python3
"""Paired problem/subject-stratified bootstrap for final-paper replay and online results."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.utils import atomic_json, load_yaml


METRICS = (
    "accuracy_difference",
    "token_reduction",
    "mean_wall_reduction",
    "p95_wall_reduction",
    "lost_correct_rate",
    "coverage",
)


def aligned_arrays(
    records_by_seed: list[list[dict[str, Any]]],
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    maps = [
        {str(row["problem_id"]): row for row in records}
        for records in records_by_seed
    ]
    ids = sorted(maps[0])
    if any(sorted(values) != ids for values in maps[1:]):
        raise ValueError("seed policy records are not problem-aligned")
    fields = {
        "method_success": bool,
        "dense_success": bool,
        "method_tokens": float,
        "dense_tokens": float,
        "replay_wall_ms": float,
        "dense_wall_ms": float,
        "fallback": bool,
    }
    arrays = {
        field: np.asarray(
            [[mapping[problem_id][field] for problem_id in ids] for mapping in maps],
            dtype=dtype,
        )
        for field, dtype in fields.items()
    }
    subjects = [
        str(maps[0][problem_id].get("subject") or "<NONE>")
        for problem_id in ids
    ]
    return arrays, ids, subjects


def metrics(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, float]:
    method_success = arrays["method_success"][:, indices].astype(float).reshape(-1)
    dense_success = arrays["dense_success"][:, indices].astype(float).reshape(-1)
    method_tokens = arrays["method_tokens"][:, indices].reshape(-1)
    dense_tokens = arrays["dense_tokens"][:, indices].reshape(-1)
    method_wall = arrays["replay_wall_ms"][:, indices].reshape(-1)
    dense_wall = arrays["dense_wall_ms"][:, indices].reshape(-1)
    fallback = arrays["fallback"][:, indices].reshape(-1)
    return {
        "accuracy_difference": float(np.mean(method_success - dense_success)),
        "token_reduction": float(1.0 - method_tokens.mean() / dense_tokens.mean()),
        "mean_wall_reduction": float(1.0 - method_wall.mean() / dense_wall.mean()),
        "p95_wall_reduction": float(
            1.0 - np.percentile(method_wall, 95) / np.percentile(dense_wall, 95)
        ),
        "lost_correct_rate": float(
            np.mean((~method_success.astype(bool)) & dense_success.astype(bool))
        ),
        "coverage": float(np.mean(~fallback.astype(bool))),
    }


def resample_indices(
    rng: np.random.Generator,
    dataset: str,
    subjects: list[str],
) -> np.ndarray:
    n = len(subjects)
    if dataset == "gsm8k":
        return rng.integers(0, n, size=n, dtype=np.int64)
    subject_array = np.asarray(subjects)
    pieces = []
    for subject in sorted(set(subjects)):
        local = np.flatnonzero(subject_array == subject)
        pieces.append(rng.choice(local, size=len(local), replace=True))
    return np.concatenate(pieces).astype(np.int64, copy=False)


def bootstrap(
    dataset: str,
    records_by_seed: list[list[dict[str, Any]]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    arrays, ids, subjects = aligned_arrays(records_by_seed)
    point = metrics(arrays, np.arange(len(ids), dtype=np.int64))
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(replicates, dtype=np.float64) for name in METRICS}
    for replicate in range(replicates):
        sampled = resample_indices(rng, dataset, subjects)
        value = metrics(arrays, sampled)
        for name in METRICS:
            draws[name][replicate] = value[name]
    return {
        "problems": len(ids),
        "seeds_pooled": len(records_by_seed),
        "replicates": replicates,
        "design": (
            "problem-level paired bootstrap"
            if dataset == "gsm8k"
            else "within-subject stratified paired bootstrap"
        ),
        "metrics": {
            name: {
                "point": point[name],
                "ci95_low": float(np.quantile(draws[name], 0.025)),
                "ci95_high": float(np.quantile(draws[name], 0.975)),
            }
            for name in METRICS
        },
    }


def point_summary(
    dataset: str,
    records_by_seed: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Return raw seed estimates; uncertainty is computed only for pooled results."""
    arrays, ids, _ = aligned_arrays(records_by_seed)
    point = metrics(arrays, np.arange(len(ids), dtype=np.int64))
    return {
        "problems": len(ids),
        "seeds_pooled": len(records_by_seed),
        "replicates": 0,
        "design": (
            "raw problem-paired point estimate"
            if dataset == "gsm8k"
            else "raw subject-stratified point estimate"
        ),
        "uncertainty_note": (
            "Per-seed raw result only; 10,000-replicate confidence intervals "
            "are reported for the single-seed replay and actual-online results."
        ),
        "metrics": {
            name: {"point": point[name]}
            for name in METRICS
        },
    }


def salted_seed(base: int, value: str) -> int:
    digest = hashlib.sha256(f"{base}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def load_policy_records(run: Path, family: str, key: str) -> list[dict[str, Any]]:
    artifact = torch.load(
        run / "policy_records.pt", map_location="cpu", weights_only=False
    )
    if artifact.get("status") != "complete":
        raise ValueError(f"incomplete records: {run}")
    return artifact["records"][family][key]


def online_records(
    online_root: Path,
    workpoint: str,
) -> list[dict[str, Any]]:
    records = []
    for path in sorted((online_root / "raw").glob("sample_*.pt")):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        dense_runs = {
            int(row["repeat"]): row for row in artifact["runs"]["dense"]
        }
        for row in artifact["runs"][workpoint]:
            dense = dense_runs[int(row["repeat"])]
            records.append(
                {
                    "problem_id": f"{artifact['problem_id']}::repeat{row['repeat']}",
                    "subject": artifact["record"].get("subject"),
                    "method_success": bool(row["success"]),
                    "dense_success": bool(dense["success"]),
                    "method_tokens": int(row["total_generated_tokens"]),
                    "dense_tokens": int(dense["total_generated_tokens"]),
                    "replay_wall_ms": float(row["wall_ms"]),
                    "dense_wall_ms": float(dense["wall_ms"]),
                    "fallback": bool(row["fallback"]),
                }
            )
    if not records:
        raise FileNotFoundError(f"no online records for {workpoint} in {online_root}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--online-root", type=Path)
    parser.add_argument("--replicates", type=int)
    args = parser.parse_args()
    config = load_yaml(args.config)
    replicates = int(args.replicates or config["statistics"]["bootstrap_replicates"])
    base_seed = int(config["statistics"]["bootstrap_seed"])
    results_root = args.results_root if args.results_root.is_absolute() else ROOT / args.results_root
    workpoint_config = config["calibration"]["online_workpoints"]
    seeds = [int(value) for value in config["probe"]["seeds"]]
    payload: dict[str, Any] = {
        "status": "complete",
        "dataset": args.dataset,
        "replicates": replicates,
        "replay": {},
    }
    for method in ("correctness", "consistency", "last_switch", "correction"):
        payload["replay"][method] = {}
        runs = {
            seed: results_root / "seeds" / f"stopper_seed_{seed}" / f"target_{method}"
            for seed in seeds
        }
        for seed, run in runs.items():
            if not (run / "phase.complete").is_file():
                raise FileNotFoundError(f"missing run {run}")
        for workpoint, selection in workpoint_config.items():
            family = str(selection["family"])
            key = str(selection["key"])
            per_seed = {
                str(seed): point_summary(
                    args.dataset,
                    [load_policy_records(run, family, key)],
                )
                for seed, run in runs.items()
            }
            pooled_records = [
                load_policy_records(runs[seed], family, key)
                for seed in seeds
            ]
            payload["replay"][method][workpoint] = {
                "per_seed": per_seed,
                "pooled": bootstrap(
                    args.dataset,
                    pooled_records,
                    replicates,
                    salted_seed(base_seed, f"{method}:{workpoint}:pooled"),
                ),
            }
    if args.online_root is not None:
        online_root = (
            args.online_root
            if args.online_root.is_absolute()
            else ROOT / args.online_root
        )
        payload["actual_online"] = {
            workpoint: bootstrap(
                args.dataset,
                [online_records(online_root, workpoint)],
                replicates,
                salted_seed(base_seed, f"online:{workpoint}"),
            )
            for workpoint in workpoint_config
        }
    destination = args.output if args.output.is_absolute() else ROOT / args.output
    atomic_json(payload, destination)
    print(json.dumps({
        "status": "complete",
        "dataset": args.dataset,
        "output": str(destination),
        "replicates": replicates,
    }, indent=2))


if __name__ == "__main__":
    main()

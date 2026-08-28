#!/usr/bin/env python3
"""Audit scientific bitwise identity of two collection artifacts.

Operational metadata (wall time, timestamp, host, worker, logical GPU index and
artifact path) is deliberately excluded.  Every value used by training,
calibration or evaluation remains in the compared payload.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reproducibility import code_provenance, sha256_json


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def without(mapping: dict[str, Any], names: set[str]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in names}


def scientific_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    dense = without(dict(artifact["dense"]), {"wall_ms"})
    rows = [
        without(
            dict(row),
            {"dense_wall_ms", "branch_collection_wall_ms", "producer_gpu"},
        )
        for row in artifact["rows"]
    ]
    model_audit = without(dict(artifact["model_audit"]), {"path"})
    reproducibility = artifact["reproducibility"]
    runtime_environment = dict(reproducibility["environment"])
    gpu = without(
        dict(runtime_environment["gpu"]),
        {"logical_index", "uuid"},
    )
    runtime_environment["gpu"] = gpu
    runtime_lock = without(dict(reproducibility["runtime_lock"]), {"path"})
    return {
        "schema_version": artifact["schema_version"],
        "status": artifact["status"],
        "protocol_id": artifact["protocol_id"],
        "protocol_fingerprint": artifact["protocol_fingerprint"],
        "primary_replay_view_fingerprint": artifact[
            "primary_replay_view_fingerprint"
        ],
        "dataset": artifact["dataset"],
        "split": artifact["split"],
        "problem_id": artifact["problem_id"],
        "dtype": artifact["dtype"],
        "seed": artifact["seed"],
        "problem_seed": artifact["problem_seed"],
        "actual_checkpoint_schedule": artifact["actual_checkpoint_schedule"],
        "checkpoint_protocol": artifact["checkpoint_protocol"],
        "capture_layers": artifact["capture_layers"],
        "rows": rows,
        "record": artifact["record"],
        "gold_answer": artifact["gold_answer"],
        "prompt_text": artifact["prompt_text"],
        "prompt_tokens": artifact["prompt_tokens"],
        "dense": dense,
        "dense_generation": artifact["dense_generation"],
        "forced_answer_decoding": artifact["forced_answer_decoding"],
        "trajectory": artifact["trajectory"],
        "schedule_checkpoints": artifact["schedule_checkpoints"],
        "model_audit": model_audit,
        "reproducibility": {
            "settings": reproducibility["settings"],
            "runtime_lock": runtime_lock,
            "environment": runtime_environment,
            "code": reproducibility["code"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    left_payload = scientific_payload(left)
    right_payload = scientific_payload(right)
    checks = {
        "same_scientific_payload": left_payload == right_payload,
        "hidden_tensor_exact": bool(torch.equal(left["hidden"], right["hidden"])),
        "distinct_certified_gpu_uuid": (
            left["reproducibility"]["environment"]["gpu"]["uuid"]
            != right["reproducibility"]["environment"]["gpu"]["uuid"]
        ),
    }
    payload = {
        "status": "complete" if all(checks.values()) else "failed",
        "all_exact": bool(all(checks.values())),
        "checks": checks,
        "scientific_sha256": {
            "left": sha256_json(left_payload),
            "right": sha256_json(right_payload),
        },
        "hidden": {
            "shape": list(left["hidden"].shape),
            "max_abs_difference": float(
                (left["hidden"].float() - right["hidden"].float()).abs().max()
            )
            if left["hidden"].numel()
            else 0.0,
        },
        "gpu_uuid": {
            "left": left["reproducibility"]["environment"]["gpu"]["uuid"],
            "right": right["reproducibility"]["environment"]["gpu"]["uuid"],
        },
        "artifacts": {"left": str(args.left.resolve()), "right": str(args.right.resolve())},
        "audit_code_identity": code_provenance(
            ROOT,
            (
                "scripts/audit_deterministic_collection_pair_v1.py",
                "src/reproducibility.py",
            ),
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2))
    if not payload["all_exact"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

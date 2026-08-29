#!/usr/bin/env python3
"""Compare two vLLM artifacts across engine/cache/batch profiles."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def without(value: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in keys}


def normalized_dense(artifact: dict[str, Any]) -> dict[str, Any]:
    dense = without(dict(artifact["dense"]), {"wall_ms"})
    if isinstance(dense.get("cap_forced_answer"), dict):
        dense["cap_forced_answer"] = without(
            dense["cap_forced_answer"], {"wall_ms", "num_cached_tokens"}
        )
    return dense


def normalized_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    operational = {
        "dense_wall_ms",
        "branch_collection_wall_ms",
        "branch_num_cached_tokens",
        "producer_gpu",
    }
    return [without(dict(row), operational) for row in artifact["rows"]]


def labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = (
        "checkpoint",
        "dense_prediction",
        "dense_success",
        "current_prediction",
        "current_success",
        "consistency",
        "correction",
        "damage",
    )
    return [{name: row.get(name) for name in names} for row in rows]


def metrics(artifact: dict[str, Any]) -> dict[str, Any]:
    rows = artifact["rows"]
    cached = sum(int(row.get("branch_num_cached_tokens", 0)) for row in rows)
    context = sum(int(row.get("prefix_context_tokens", 0)) for row in rows)
    collection = artifact.get("collection", {})
    return {
        "profile": artifact.get("vllm_engine", {}).get("profile"),
        "task_order": artifact.get("task_order"),
        "dense_tokens": len(artifact["dense"]["tokens"]),
        "checkpoints": len(rows),
        "dense_wall_ms": float(artifact["dense"]["wall_ms"]),
        "hidden_wall_ms": float(collection.get("hidden_replay_wall_ms", 0.0)),
        "branch_wall_ms": float(collection.get("branch_wall_ms", 0.0)),
        "branch_cached_tokens": cached,
        "branch_context_tokens": context,
        "branch_cache_fraction": cached / context if context else 0.0,
    }


def speedup(left: dict[str, Any], right: dict[str, Any], name: str) -> float | None:
    denominator = float(right[name])
    return float(left[name]) / denominator if denominator > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left = torch.load(args.left, map_location="cpu", weights_only=False, mmap=True)
    right = torch.load(args.right, map_location="cpu", weights_only=False, mmap=True)
    left_rows = normalized_rows(left)
    right_rows = normalized_rows(right)
    left_hidden_shape = tuple(left["hidden"].shape)
    right_hidden_shape = tuple(right["hidden"].shape)
    hidden_shape_exact = left_hidden_shape == right_hidden_shape
    hidden_tensor_bitwise_exact = bool(
        hidden_shape_exact and torch.equal(left["hidden"], right["hidden"])
    )
    checks = {
        "problem_identity_exact": (
            left.get("dataset"), left.get("split"), left.get("problem_id")
        )
        == (right.get("dataset"), right.get("split"), right.get("problem_id")),
        "prompt_token_ids_exact": left.get("prompt_token_ids")
        == right.get("prompt_token_ids"),
        "problem_seed_exact": left.get("problem_seed") == right.get("problem_seed"),
        "dense_token_ids_exact": left["dense"]["tokens"]
        == right["dense"]["tokens"],
        "dense_payload_exact": normalized_dense(left) == normalized_dense(right),
        "checkpoint_schedule_exact": left.get("schedule_checkpoints")
        == right.get("schedule_checkpoints"),
        "branch_and_row_payload_exact": left_rows == right_rows,
        "labels_exact": labels(left_rows) == labels(right_rows),
        "hidden_shape_exact": hidden_shape_exact,
        "hidden_tensor_bitwise_exact": hidden_tensor_bitwise_exact,
    }
    left_metrics = metrics(left)
    right_metrics = metrics(right)
    difference = (
        (left["hidden"].float() - right["hidden"].float()).abs()
        if hidden_shape_exact
        else None
    )
    payload = {
        "status": "complete",
        "all_scientific_exact": bool(all(checks.values())),
        "checks": checks,
        "hidden": {
            "left_shape": list(left_hidden_shape),
            "right_shape": list(right_hidden_shape),
            "max_abs_difference": (
                (
                    float(difference.max())
                    if difference is not None and difference.numel()
                    else 0.0
                )
                if hidden_shape_exact
                else None
            ),
            "changed_elements": (
                int(torch.count_nonzero(difference)) if difference is not None else None
            ),
            "left_elements": int(left["hidden"].numel()),
            "right_elements": int(right["hidden"].numel()),
        },
        "metrics": {"left": left_metrics, "right": right_metrics},
        "right_vs_left_speedup": {
            name.removesuffix("_wall_ms"): speedup(left_metrics, right_metrics, name)
            for name in ("dense_wall_ms", "hidden_wall_ms", "branch_wall_ms")
        },
        "artifacts": {
            "left": str(args.left.resolve()),
            "right": str(args.right.resolve()),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

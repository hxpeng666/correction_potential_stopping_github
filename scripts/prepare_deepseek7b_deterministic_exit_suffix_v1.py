#!/usr/bin/env python3
"""Freeze a label-blind paired trajectory/checkpoint sample for suffix analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reproducibility import code_provenance, sha256_json


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_rank(seed: int, dataset: str, split: str, name: str) -> str:
    return hashlib.sha256(f"{seed}:{dataset}:{split}:{name}".encode()).hexdigest()


def select_checkpoints(rows: list[dict[str, Any]], count: int) -> list[int]:
    ordered = sorted({int(row["checkpoint"]) for row in rows})
    if len(ordered) <= count:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, count)
    return sorted({ordered[int(round(value))] for value in indices})


def directory(data_root: Path, dataset: str, split: str) -> Path:
    return data_root / dataset / split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    provenance = code_provenance(
        ROOT,
        (
            "configs/deepseek7b_deterministic_exit_suffix_v1.yaml",
            "scripts/prepare_deepseek7b_deterministic_exit_suffix_v1.py",
            "src/reproducibility.py",
        ),
    )
    data_root = Path(config["source"]["data_root"]).resolve()
    seed = int(config["sampling"]["stable_rank_seed"])
    checkpoint_count = int(config["sampling"]["checkpoints_per_trajectory"])
    entries: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    for spec in config["sampling"]["strata"]:
        dataset, split, requested = str(spec["dataset"]), str(spec["split"]), int(spec["count"])
        source_dir = directory(data_root, dataset, split)
        paths = sorted(
            source_dir.glob("sample_*.pt"),
            key=lambda path: stable_rank(seed, dataset, split, path.name),
        )
        if len(paths) < requested:
            raise ValueError(f"{dataset}/{split}: requested {requested}, found {len(paths)}")
        selected = paths[:requested]
        names: list[str] = []
        for source_path in selected:
            artifact = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
            if artifact.get("status") != "complete":
                raise ValueError(f"incomplete source artifact: {source_path}")
            rows = [
                row for row in artifact.get("rows", [])
                if config["source"]["checkpoint_stored_compatibility_tag"]
                in row.get("checkpoint_schedules", [])
            ]
            checkpoints = select_checkpoints(rows, checkpoint_count)
            if not checkpoints:
                raise ValueError(f"selected trajectory has no checkpoints: {source_path}")
            names.append(source_path.name)
            entries.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "problem_id": str(artifact["problem_id"]),
                    "source_path": str(source_path.resolve()),
                    "source_protocol_fingerprint": str(artifact.get("protocol_fingerprint")),
                    "dense_reasoning_tokens": int(artifact["dense"]["reasoning_tokens"]),
                    "dense_success": bool(artifact["dense"]["success"]),
                    "checkpoints": checkpoints,
                }
            )
        strata.append(
            {
                "dataset": dataset,
                "split": split,
                "requested": requested,
                "selected": len(selected),
                "sample_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
            }
        )
    entries.sort(key=lambda row: (row["dataset"], row["split"], row["problem_id"]))
    payload = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config["protocol_id"],
        "code_identity": provenance,
        "selection_uses_labels": False,
        "seed": seed,
        "checkpoints_per_trajectory": checkpoint_count,
        "trajectory_count": len(entries),
        "checkpoint_count": sum(len(row["checkpoints"]) for row in entries),
        "strata": strata,
        "entries": entries,
    }
    payload["manifest_fingerprint"] = sha256_json(
        {key: payload[key] for key in ("protocol_id", "seed", "checkpoints_per_trajectory", "strata", "entries")}
    )
    atomic_json(payload, args.output)
    print(json.dumps({key: payload[key] for key in ("status", "trajectory_count", "checkpoint_count", "manifest_fingerprint")}, indent=2))


if __name__ == "__main__":
    main()


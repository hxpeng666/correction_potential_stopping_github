#!/usr/bin/env python3
"""Fail-closed completeness audit for the Qwen3-14B deterministic collection."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from collect_qwen3_14b_deterministic_ood_v1 import DATA_LAYOUT, all_tasks, gold_for
from deepseek7b_protocol_v1 import success
from src.reproducibility import code_provenance, sha256_file


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gate-audit", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tasks = all_tasks(args.prepared_root)
    expected_total = (
        sum(int(value) for value in config["data"]["gsm8k"].values())
        + int(config["data"]["math"]["probe_train"])
        + int(config["data"]["math"]["calibration"])
        + int(config["data"]["math500"]["heldout"])
        + int(config["data"]["aime"]["heldout"])
    )
    if len(tasks) != expected_total:
        raise RuntimeError(
            f"prepared data/config count mismatch: {len(tasks)} != {expected_total}"
        )
    expected = {
        (dataset, split, str(record["problem_id"])): record
        for dataset, split, record in tasks
    }
    errors: list[str] = []
    fingerprint: str | None = None
    counts: dict[str, dict[str, int]] = {}
    capped = zero_checkpoint = checkpoints = 0
    for dataset, split in DATA_LAYOUT:
        local_expected = {
            key: record for key, record in expected.items()
            if key[0] == dataset and key[1] == split
        }
        paths = sorted((args.output_root / "cache" / dataset / split).glob("sample_*.pt"))
        seen: set[tuple[str, str, str]] = set()
        for path in paths:
            value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            key = (dataset, split, str(value.get("problem_id")))
            if key not in local_expected or key in seen:
                errors.append(f"unexpected/duplicate artifact: {path}")
                continue
            seen.add(key)
            if value.get("status") != "complete" or value.get("protocol_id") != config["protocol_id"]:
                errors.append(f"status/protocol mismatch: {path}")
            local_fingerprint = str(value.get("protocol_fingerprint"))
            if fingerprint is None:
                fingerprint = local_fingerprint
            elif fingerprint != local_fingerprint:
                errors.append(f"fingerprint mismatch: {path}")
            rows = list(value.get("rows", []))
            hidden = value.get("hidden")
            checkpoints += len(rows)
            if not isinstance(hidden, torch.Tensor) or list(hidden.shape) != [len(rows), 1, 5120]:
                errors.append(f"hidden shape mismatch: {path}")
            if value.get("capture_layers") != [20] or value.get("actual_checkpoint_schedule") != "paragraph":
                errors.append(f"checkpoint/layer mismatch: {path}")
            dense = value.get("dense", {})
            reached = bool(dense.get("reached_max_tokens"))
            if reached:
                capped += 1
                if dense.get("grader") != "forced_answer_at_exact_13k_prefix" or dense.get("cap_forced_answer") is None:
                    errors.append(f"cap grader mismatch: {path}")
            elif dense.get("grader") != "natural_dense_completion":
                errors.append(f"natural grader mismatch: {path}")
            gold = gold_for(dataset, local_expected[key])
            if bool(dense.get("success")) != success(dataset, gold, dense.get("prediction")):
                errors.append(f"dense label mismatch: {path}")
            if not rows:
                zero_checkpoint += 1
            for row in rows:
                current = bool(row.get("current_success"))
                final = bool(dense.get("success"))
                if bool(row.get("correction")) != ((not current) and final):
                    errors.append(f"correction label mismatch: {path}")
                    break
                if bool(row.get("damage")) != (current and (not final)):
                    errors.append(f"damage label mismatch: {path}")
                    break
        missing = set(local_expected) - seen
        if missing:
            errors.append(f"missing {dataset}/{split}: {len(missing)}")
        counts.setdefault(dataset, {})[split] = len(seen)

    gate = json.loads(args.gate_audit.read_text(encoding="utf-8"))
    if gate.get("status") != "complete" or gate.get("all_exact") is not True:
        errors.append("cross-GPU determinism gate not exact")
    workers = []
    for name in (
        "formal_gpu0_replica0",
        "formal_gpu1_replica0",
        "formal_gpu1_replica1",
    ):
        path = args.output_root / "workers" / f"{name}.json"
        if not path.is_file():
            errors.append(f"missing worker summary: {name}")
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        workers.append(value)
        if value.get("status") != "complete" or value.get("failures") != 0:
            errors.append(f"failed worker summary: {name}")
    if workers:
        worker_shards = {
            (int(value.get("gpu", -1)), int(value.get("shard_index", -1)), int(value.get("num_shards", -1)))
            for value in workers
        }
        expected_shards = {(0, 0, 3), (1, 1, 3), (1, 2, 3)}
        if worker_shards != expected_shards:
            errors.append(f"formal worker shard mismatch: {sorted(worker_shards)}")
        if sum(int(value.get("assigned", 0)) for value in workers) != len(expected):
            errors.append("formal worker assigned counts do not cover the dataset exactly once")

    payload = {
        "status": "complete" if not errors else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "expected_trajectories": len(expected),
        "actual_trajectories": sum(sum(value.values()) for value in counts.values()),
        "counts": counts,
        "checkpoint_count": checkpoints,
        "zero_checkpoint_dense_fallback": zero_checkpoint,
        "cap_hit_trajectories": capped,
        "workers": [
            {key: value.get(key) for key in (
                "worker", "gpu", "shard_index", "num_shards", "assigned",
                "completed", "skipped", "failures",
            )}
            for value in workers
        ],
        "gate_audit_sha256": sha256_file(args.gate_audit),
        "errors": errors,
        "code_identity": code_provenance(
            ROOT,
            (
                "scripts/audit_qwen3_14b_deterministic_ood_v1.py",
                "scripts/collect_qwen3_14b_deterministic_ood_v1.py",
            ),
        ),
    }
    atomic_json(payload, args.output_root / "COLLECTION_AUDIT.json")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed completeness audit for the full-vLLM Qwen3-14B collection."""
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

from collect_qwen3_14b_vllm_full_v1 import DATA_LAYOUT, all_tasks, gold_for
from deepseek7b_protocol_v1 import stable_seed, success
from src.reproducibility import code_provenance, sha256_file, sha256_json


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
    parser.add_argument("--profile", required=True)
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
            expected_seed = stable_seed(int(config["seed"]), str(value.get("problem_id")))
            if value.get("seed") != int(config["seed"]) or value.get("problem_seed") != expected_seed:
                errors.append(f"Dense rollout seed mismatch: {path}")
            prompt_token_ids = value.get("prompt_token_ids")
            if (
                not isinstance(prompt_token_ids, list)
                or len(prompt_token_ids) != int(value.get("prompt_tokens", -1))
                or not all(isinstance(token, int) for token in prompt_token_ids)
            ):
                errors.append(f"prompt token audit mismatch: {path}")
            local_fingerprint = str(value.get("protocol_fingerprint"))
            if fingerprint is None:
                fingerprint = local_fingerprint
            elif fingerprint != local_fingerprint:
                errors.append(f"fingerprint mismatch: {path}")
            rows = list(value.get("rows", []))
            hidden = value.get("hidden")
            schedule = [int(item) for item in value.get("schedule_checkpoints", [])]
            checkpoints += len(rows)
            if (
                not isinstance(hidden, torch.Tensor)
                or hidden.dtype != torch.float16
                or list(hidden.shape) != [len(rows), 1, 5120]
            ):
                errors.append(f"hidden shape mismatch: {path}")
            if schedule != [int(row.get("checkpoint", -1)) for row in rows]:
                errors.append(f"checkpoint sequence mismatch: {path}")
            if value.get("capture_layers") != [20] or value.get("actual_checkpoint_schedule") != "paragraph":
                errors.append(f"checkpoint/layer mismatch: {path}")
            hidden_replay = value.get("hidden_replay_audit", {})
            expected_replay_tokens = len(prompt_token_ids or []) + len(
                value.get("dense", {}).get("tokens", [])
            )
            expected_selection = [
                len(prompt_token_ids or []) + checkpoint - 1
                for checkpoint in schedule
            ]
            if (
                hidden_replay.get("token_ids_exact") is not True
                or hidden_replay.get("replay_token_count") != expected_replay_tokens
                or hidden_replay.get("replay_token_ids_sha256")
                != sha256_json(
                    list(prompt_token_ids or [])
                    + list(value.get("dense", {}).get("tokens", []))
                )
                or hidden_replay.get("selection_indices") != expected_selection
            ):
                errors.append(f"hidden replay token/selection mismatch: {path}")
            engine = value.get("vllm_engine", {})
            expected_phases = {
                phase: {
                    "enable_prefix_caching": bool(settings["enable_prefix_caching"]),
                    "max_num_seqs": int(settings["max_num_seqs"]),
                    "request_batch_size": int(settings["request_batch_size"]),
                }
                for phase, settings in config["vllm"]["profiles"][args.profile].items()
            }
            if (
                engine.get("version") != str(config["vllm"]["version"])
                or engine.get("multiprocessing") is not False
                or engine.get("async_scheduling") is not False
                or engine.get("enforce_eager") is not True
                or engine.get("profile") != args.profile
                or engine.get("phases") != expected_phases
                or engine.get("requested_zero_based_decoder_layer") != 20
                or engine.get("vllm_aux_hidden_state_layer_ids") != [21]
                or engine.get("forbidden_optional_packages_absent")
                != ["flash-attn", "xformers"]
            ):
                errors.append(f"vLLM deterministic engine mismatch: {path}")
            environment_lock = value.get("reproducibility", {}).get(
                "full_environment_lock", {}
            )
            if (
                environment_lock.get("exact") is not True
                or environment_lock.get("sha256")
                != sha256_file(
                    ROOT / config["reproducibility"]["full_environment_lock"]
                )
            ):
                errors.append(f"full environment lock mismatch: {path}")
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
    scheduler = config["formal_scheduler"]
    num_shards = int(scheduler["num_shards"])
    workers = []
    for shard in range(num_shards):
        name = f"formal_shard{shard}"
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
            (int(value.get("shard_index", -1)), int(value.get("num_shards", -1)))
            for value in workers
        }
        expected_shards = {(shard, num_shards) for shard in range(num_shards)}
        if worker_shards != expected_shards:
            errors.append(f"formal logical shard mismatch: {sorted(worker_shards)}")
        allowed_gpus = {int(value) for value in scheduler["physical_gpus"]}
        if any(int(value.get("physical_gpu", -1)) not in allowed_gpus for value in workers):
            errors.append("formal worker used a GPU outside the scheduler allowlist")
        if sum(int(value.get("assigned", 0)) for value in workers) != len(expected):
            errors.append("formal worker assigned counts do not cover the dataset exactly once")

    payload = {
        "status": "complete" if not errors else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config["protocol_id"],
        "engine_profile": args.profile,
        "protocol_fingerprint": fingerprint,
        "expected_trajectories": len(expected),
        "actual_trajectories": sum(sum(value.values()) for value in counts.values()),
        "counts": counts,
        "checkpoint_count": checkpoints,
        "zero_checkpoint_dense_fallback": zero_checkpoint,
        "cap_hit_trajectories": capped,
        "workers": [
            {key: value.get(key) for key in (
                "worker", "physical_gpu", "shard_index", "num_shards", "assigned",
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
                "scripts/collect_qwen3_14b_vllm_full_v1.py",
            ),
        ),
    }
    atomic_json(payload, args.output_root / "COLLECTION_AUDIT.json")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

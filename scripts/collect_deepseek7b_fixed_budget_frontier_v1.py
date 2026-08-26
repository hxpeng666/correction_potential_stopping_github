#!/usr/bin/env python3
"""Collect a held-out fixed relative-budget frontier from frozen DeepSeek traces."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from deepseek7b_protocol_v1 import (  # noqa: E402
    atomic_torch_save,
    canonical_fingerprint as source_fingerprint,
    greedy_branch,
    load_model,
    prediction,
    reasoning_end,
    success,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def fixed_fingerprint(config: dict[str, Any], dataset: str) -> str:
    payload = {
        "protocol_id": config["protocol_id"],
        "model": config["model"],
        "source": config["source"],
        "generation": config["generation"],
        "budget": config["budget"],
        "dataset": dataset,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def dense_token_fingerprint(tokens: list[int]) -> str:
    return hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()


def read_records(source_config: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    path = Path(source_config["data"]["prepared_root"]) / dataset / "heldout.jsonl"
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    expected = int(source_config["data"][dataset]["heldout"])
    if len(rows) != expected:
        raise RuntimeError(f"{dataset}: expected {expected} prepared rows, found {len(rows)}")
    ids = [str(row["problem_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{dataset}: duplicate prepared problem IDs")
    return sorted(rows, key=lambda row: str(row["problem_id"]))


def extension_targets(config: dict[str, Any]) -> set[tuple[str, str]]:
    manifest_path = config["source"].get("selective_extension_manifest")
    if not manifest_path:
        return set()
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    targets = {
        (str(item["dataset"]), str(item["problem_id"]))
        for item in manifest.get("eligible", [])
        if str(item.get("split")) == "heldout"
    }
    return targets


def source_is_valid(
    value: dict[str, Any],
    *,
    dataset: str,
    problem_id: str,
    expected_fingerprint: str,
    requires_extension: bool,
    allow_missing_execution_mode: bool = False,
) -> bool:
    if (
        value.get("status") != "complete"
        or value.get("dataset") != dataset
        or value.get("split") != "heldout"
        or str(value.get("problem_id")) != problem_id
        or value.get("protocol_fingerprint") != expected_fingerprint
    ):
        return False
    dense = value.get("dense", {})
    tokens = dense.get("tokens", [])
    if not tokens or int(dense.get("reasoning_tokens", -1)) != len(tokens):
        return False
    if requires_extension:
        collection = value.get("collection", {})
        generation = value.get("dense_generation", {})
        return (
            collection.get("execution_mode")
            == "incremental_exact_resume_from_capped_13k_source"
            and int(collection.get("reused_checkpoints", 0)) > 0
            and int(collection.get("new_checkpoints", -1)) >= 0
            and generation.get("incremental_exact_resume") is True
        )
    execution_mode = value.get("dense_generation", {}).get("execution_mode")
    return execution_mode in {
        "reused_noncapped_source_trajectory",
        "generated_at_configured_budget",
    } or (allow_missing_execution_mode and execution_mode is None)


def output_is_valid(path: Path, fingerprint: str, problem_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        return (
            value.get("status") == "complete"
            and value.get("protocol_fingerprint") == fingerprint
            and str(value.get("problem_id")) == problem_id
        )
    except Exception:
        return False


def collect_one(
    source_path: Path,
    destination: Path,
    *,
    config: dict[str, Any],
    dataset: str,
    fingerprint: str,
    model,
    tokenizer,
    model_audit: dict[str, Any],
    device: torch.device,
    gpu: int,
    worker_id: str,
) -> dict[str, Any]:
    source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    problem_id = str(source["problem_id"])
    tokens = [int(token) for token in source["dense"]["tokens"]]
    dense_tokens = len(tokens)
    prompt_ids = tokenizer(source["prompt_text"], return_tensors="pt").input_ids.to(device)
    prompt_tokens = int(prompt_ids.shape[1])
    if prompt_tokens != int(source["prompt_tokens"]):
        raise ValueError(f"prompt retokenization mismatch: {source_path}")

    fractions = [float(value) for value in config["budget"]["retained_fractions"]]
    minimum = int(config["budget"]["minimum_retained_dense_tokens"])
    fraction_to_point: dict[float, int | None] = {}
    for fraction in fractions:
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"retained fraction must be in (0,1): {fraction}")
        fraction_to_point[fraction] = (
            None
            if dense_tokens <= 0
            else min(dense_tokens, max(minimum, int(math.floor(fraction * dense_tokens))))
        )
    points = sorted({point for point in fraction_to_point.values() if point is not None})

    generation = config["generation"]
    suffix = str(generation["force_answer_suffix"])
    maximum = int(generation["force_answer_max_new_tokens"])
    suffix_ids = tokenizer(
        suffix, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    suffix_tokens = int(suffix_ids.shape[1])
    branches: dict[int, dict[str, Any]] = {}
    with torch.inference_mode():
        prefill = model.model(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            use_cache=True,
            return_dict=True,
        )
        base_cache = prefill.past_key_values
        del prefill
        previous = 0
        for point in points:
            if point > previous:
                delta = torch.tensor([tokens[previous:point]], dtype=torch.long, device=device)
                mask = torch.ones(
                    (1, prompt_tokens + point), dtype=torch.long, device=device
                )
                teacher = model.model(
                    input_ids=delta,
                    attention_mask=mask,
                    past_key_values=base_cache,
                    use_cache=True,
                    return_dict=True,
                )
                base_cache = teacher.past_key_values
                del teacher
            branches[point] = greedy_branch(
                model,
                tokenizer,
                base_cache,
                prefix_context=prompt_tokens + point,
                suffix_ids=suffix_ids,
                maximum=maximum,
            )
            previous = point

    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    dense_success = bool(source["dense"]["success"])
    dense_prediction = source["dense"]["prediction"]
    gold = source["gold_answer"]
    think_end, think_end_source = reasoning_end(tokenizer, tokens)
    rows = []
    for fraction in fractions:
        point = fraction_to_point[fraction]
        if point is None:
            rows.append(
                {
                    "retained_fraction": fraction,
                    "target_reasoning_saving_fraction": 1.0 - fraction,
                    "dense_fallback": True,
                    "current_prediction": dense_prediction,
                    "current_success": dense_success,
                    "dense_prediction": dense_prediction,
                    "dense_success": dense_success,
                    "dense_tokens": dense_tokens,
                    "stop_reasoning_tokens": dense_tokens,
                    "stop_total_tokens": dense_tokens,
                    "lost_correct": False,
                    "helped": False,
                }
            )
            continue
        branch = branches[point]
        current_prediction = prediction(dataset, branch["text"])
        current_success = success(dataset, gold, current_prediction)
        branch_tokens = [int(token) for token in branch["tokens"]]
        terminated_by_eos = bool(branch_tokens and branch_tokens[-1] in eos)
        rows.append(
            {
                "retained_fraction": fraction,
                "target_reasoning_saving_fraction": 1.0 - fraction,
                "dense_fallback": False,
                "checkpoint": point,
                "retained_dense_tokens": point,
                "actual_retained_dense_fraction": point / dense_tokens,
                "checkpoint_after_think_close": bool(point > think_end),
                "current_prediction": current_prediction,
                "current_success": bool(current_success),
                "dense_prediction": dense_prediction,
                "dense_success": dense_success,
                "dense_tokens": dense_tokens,
                "branch_tokens": len(branch_tokens),
                "branch_token_ids": branch_tokens,
                "branch_text": branch["text"],
                "branch_generated_text": branch["generated_text"],
                "forced_answer_truncated": not terminated_by_eos,
                "stop_reasoning_tokens": point,
                "stop_total_tokens": point + suffix_tokens + len(branch_tokens),
                "lost_correct": bool(dense_success and not current_success),
                "helped": bool((not dense_success) and current_success),
            }
        )

    artifact = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "dataset": dataset,
        "split": "heldout",
        "problem_id": problem_id,
        "record": source["record"],
        "gold_answer": gold,
        "prompt_tokens": prompt_tokens,
        "source_artifact": str(source_path.resolve()),
        "source_protocol_id": source["protocol_id"],
        "source_protocol_fingerprint": source["protocol_fingerprint"],
        "source_dense_token_fingerprint": dense_token_fingerprint(tokens),
        "dense": {
            "tokens": dense_tokens,
            "prediction": dense_prediction,
            "success": dense_success,
            "reached_max_tokens": bool(source["dense"]["reached_max_tokens"]),
            "requested_max_new_tokens": int(
                source.get("dense_generation", {}).get(
                    "requested_max_new_tokens",
                    config["source"]["dense_max_new_tokens"],
                )
            ),
        },
        "reasoning_end": {
            "token": think_end,
            "source": think_end_source,
        },
        "rows": rows,
        "forced_answer_decoding": {
            "strategy": "greedy_argmax",
            "do_sample": False,
            "max_new_tokens": maximum,
            "suffix": suffix,
            "suffix_tokens": suffix_tokens,
        },
        "model_audit": model_audit,
        "collection": {
            "worker": worker_id,
            "host": socket.gethostname(),
            "gpu": gpu,
            "device": torch.cuda.get_device_name(device),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(artifact, destination)
    return {
        "problem_id": problem_id,
        "dense_tokens": dense_tokens,
        "dense_success": dense_success,
        "rows": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "math500", "aime"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--maximum-source-tokens", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    source_config = load_yaml(Path(config["source"]["config"]))
    expected_source_fingerprint = source_fingerprint(source_config)
    fingerprint = fixed_fingerprint(config, args.dataset)
    records = read_records(source_config, args.dataset)
    targets = extension_targets(config)
    source_root = Path(config["source"]["output_root"]) / config["source"]["cache_subdirectory"]
    output_root = Path(config["output_root"])

    tasks = []
    pending_sources = deferred = skipped = 0
    for index, record in enumerate(records):
        if index % args.num_shards != args.shard_index:
            continue
        problem_id = str(record["problem_id"])
        destination = output_root / args.dataset / "heldout" / f"sample_{problem_id}.pt"
        if args.resume and output_is_valid(destination, fingerprint, problem_id):
            skipped += 1
            continue
        if destination.exists():
            raise RuntimeError(f"refusing to overwrite incompatible artifact: {destination}")
        source_path = source_root / args.dataset / "heldout" / f"sample_{problem_id}.pt"
        if not source_path.is_file():
            pending_sources += 1
            continue
        source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
        if not source_is_valid(
            source,
            dataset=args.dataset,
            problem_id=problem_id,
            expected_fingerprint=expected_source_fingerprint,
            requires_extension=(args.dataset, problem_id) in targets,
            allow_missing_execution_mode=bool(
                config["source"].get("allow_missing_execution_mode", False)
            ),
        ):
            pending_sources += 1
            continue
        dense_tokens = len(source["dense"]["tokens"])
        if args.maximum_source_tokens is not None and dense_tokens > args.maximum_source_tokens:
            deferred += 1
            continue
        tasks.append((dense_tokens, problem_id, source_path, destination))
    tasks.sort(key=lambda value: (value[0], value[1]))
    if args.limit is not None:
        tasks = tasks[: args.limit]

    summary_base = {
        "worker": args.worker_id,
        "dataset": args.dataset,
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "protocol_fingerprint": fingerprint,
        "pending_sources": pending_sources,
        "deferred": deferred,
        "skipped": skipped,
    }
    if not tasks:
        summary = {"status": "complete", "completed": 0, **summary_base}
        summary_path = output_root / "workers" / f"{args.worker_id}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    free, total = torch.cuda.mem_get_info(device)
    print(
        json.dumps(
            {
                "status": "loading",
                "tasks": len(tasks),
                "free_gib": free / 2**30,
                "total_gib": total / 2**30,
                **summary_base,
            }
        ),
        flush=True,
    )
    model, tokenizer, model_audit = load_model(Path(config["model"]["local_path"]), device)
    completed = failures = 0
    started = time.time()
    for _dense_tokens, problem_id, source_path, destination in tasks:
        try:
            result = collect_one(
                source_path,
                destination,
                config=config,
                dataset=args.dataset,
                fingerprint=fingerprint,
                model=model,
                tokenizer=tokenizer,
                model_audit=model_audit,
                device=device,
                gpu=args.gpu,
                worker_id=args.worker_id,
            )
            completed += 1
            print(json.dumps({"status": "completed", "completed": completed, **result}), flush=True)
        except Exception as error:
            failures += 1
            print(
                json.dumps(
                    {
                        "status": "error",
                        "problem_id": problem_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                ),
                flush=True,
            )
            if isinstance(error, torch.cuda.OutOfMemoryError):
                raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    summary = {
        "status": "complete" if failures == 0 else "failed",
        "completed": completed,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
        **summary_base,
    }
    summary_path = output_root / "workers" / f"{args.worker_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

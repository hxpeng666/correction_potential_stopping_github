#!/usr/bin/env python3
"""Resume-safe paragraph checkpoint collection for DeepSeek-R1-Distill-Qwen-7B."""
from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepseek7b_protocol_v1 import (
    CheckpointHiddenCapture,
    atomic_torch_save,
    canonical_fingerprint,
    generate_dense,
    generate_dense_from_prefix,
    greedy_branch,
    load_model,
    paragraph_checkpoints,
    prediction,
    render_prompt,
    stable_seed,
    success,
    tail_mean,
)
from src.reproducibility import (
    code_provenance,
    enforce_runtime_lock,
    environment_provenance,
    sha256_file,
    strict_reproducibility,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def all_tasks(prepared_root: Path) -> list[tuple[str, str, dict[str, Any]]]:
    tasks = []
    for dataset in ("gsm8k", "math", "math500", "aime"):
        for split in ("probe_train", "calibration", "heldout"):
            path = prepared_root / dataset / f"{split}.jsonl"
            if path.is_file():
                tasks.extend((dataset, split, row) for row in read_jsonl(path))
    return sorted(tasks, key=lambda item: (item[0], item[1], str(item[2]["problem_id"])))


def gold_for(dataset: str, record: dict[str, Any]) -> str | None:
    if "gold_answer" in record:
        return str(record["gold_answer"])
    if dataset == "gsm8k":
        marker = str(record["answer"]).rsplit("####", 1)
        return marker[-1].strip().replace(",", "") if marker else None
    raise KeyError(f"no gold answer for {dataset}:{record.get('problem_id')}")


def valid(
    path: Path,
    fingerprint: str,
    problem_id: str,
    *,
    require_incremental_exact_resume: bool = False,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        base_valid = (
            value.get("status") == "complete"
            and value.get("protocol_fingerprint") == fingerprint
            and str(value.get("problem_id")) == problem_id
            and value.get("actual_checkpoint_schedule") == "paragraph"
        )
        if not base_valid:
            return False
        if not require_incremental_exact_resume:
            return True
        collection = value.get("collection", {})
        dense_generation = value.get("dense_generation", {})
        return (
            collection.get("execution_mode")
            == "incremental_exact_resume_from_capped_13k_source"
            and int(collection.get("reused_checkpoints", 0)) > 0
            and int(collection.get("new_checkpoints", -1)) >= 0
            and dense_generation.get("incremental_exact_resume") is True
        )
    except Exception:
        return False


def incremental_extension_targets(config: dict[str, Any]) -> set[tuple[str, str, str]]:
    extension = config.get("selective_dense_extension", {})
    if extension.get("enabled") is not True:
        return set()
    manifest_path = Path(extension["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = {
        (str(item["dataset"]), str(item["split"]), str(item["problem_id"]))
        for item in manifest.get("eligible", [])
    }
    expected = int(manifest.get("eligible_count", len(targets)))
    if len(targets) != expected:
        raise ValueError(
            f"selective extension manifest target mismatch: {len(targets)} != {expected}"
        )
    return targets


def load_incremental_source(
    destination: Path,
    config: dict[str, Any],
    problem_id: str,
) -> tuple[Path | None, dict[str, Any] | None]:
    extension = config.get("selective_dense_extension", {})
    if extension.get("enabled") is not True:
        return None, None
    target_cache = (Path(config["output_root"]) / "cache").resolve()
    source_cache = (Path(config["cache_migration"]["source_output_root"]) / "cache").resolve()
    relative = destination.resolve().relative_to(target_cache)
    source_path = source_cache / relative
    if not source_path.is_file():
        raise FileNotFoundError(f"missing incremental source artifact: {source_path}")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_budget = int(extension["source_dense_max_new_tokens"])
    if (
        source.get("status") != "complete"
        or str(source.get("problem_id")) != problem_id
        or not bool(source["dense"].get("reached_max_tokens"))
        or len(source["dense"]["tokens"]) != source_budget
    ):
        raise ValueError(f"invalid incremental source artifact: {source_path}")
    return source_path, source


def collect_one(
    dataset: str,
    split: str,
    record: dict[str, Any],
    destination: Path,
    config: dict[str, Any],
    fingerprint: str,
    model,
    tokenizer,
    model_audit: dict[str, Any],
    device: torch.device,
    worker_id: str,
    gpu: int,
    reproducibility_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    problem_id = str(record["problem_id"])
    problem_seed = stable_seed(int(config["seed"]), problem_id)
    prompt_text = render_prompt(tokenizer, str(record["question"]))
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    generation = config["generation"]
    source_path, source = load_incremental_source(destination, config, problem_id)
    if source is None:
        dense = generate_dense(
            model,
            tokenizer,
            input_ids,
            seed=problem_seed,
            max_new_tokens=int(generation["dense_max_new_tokens"]),
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            top_k=int(generation["top_k"]),
        )
    else:
        if source.get("prompt_text") != prompt_text:
            raise ValueError(f"incremental source prompt mismatch: {source_path}")
        dense = generate_dense_from_prefix(
            model,
            tokenizer,
            input_ids,
            source_dense=source["dense"],
            seed=problem_seed,
            max_new_tokens=int(generation["dense_max_new_tokens"]),
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            top_k=int(generation["top_k"]),
        )
    gold = gold_for(dataset, record)
    dense_prediction = prediction(dataset, dense["text"])
    dense_success = success(dataset, gold, dense_prediction)
    checkpoints, trajectory = paragraph_checkpoints(tokenizer, dense["tokens"])
    suffix_ids = tokenizer(
        generation["force_answer_suffix"],
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    capture_layer = int(config["features"]["layer_zero_based"])
    capture = CheckpointHiddenCapture(model, capture_layer)
    rows: list[dict[str, Any]] = []
    vectors: list[torch.Tensor] = []
    branch_wall_ms = 0.0
    new_branch_wall_ms = 0.0
    reused_checkpoints = 0
    try:
        with torch.inference_mode():
            if source is None:
                prefill = model.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=True,
                    return_dict=True,
                )
                base_cache = prefill.past_key_values
                del prefill
                previous = 0
                pending_checkpoints = checkpoints
            else:
                source_rows = list(source["rows"])
                source_hidden = source["hidden"]
                source_budget = len(source["dense"]["tokens"])
                expected_reused = [int(value) for value in checkpoints if int(value) <= source_budget]
                actual_reused = [int(row["checkpoint"]) for row in source_rows]
                if expected_reused != actual_reused:
                    raise ValueError(
                        f"incremental checkpoint prefix mismatch: {expected_reused[-5:]} != {actual_reused[-5:]}"
                    )
                if int(source_hidden.shape[0]) != len(source_rows):
                    raise ValueError("incremental source hidden/row mismatch")
                for index, source_row in enumerate(source_rows):
                    reused = dict(source_row)
                    current_prediction = reused.get("current_prediction")
                    current_success = bool(reused.get("current_success"))
                    reused.update(
                        {
                            "dense_prediction": dense_prediction,
                            "dense_success": bool(dense_success),
                            "dense_tokens": len(dense["tokens"]),
                            "dense_wall_ms": float(dense["wall_ms"]),
                            "consistency": bool(
                                current_prediction is not None
                                and dense_prediction is not None
                                and current_prediction == dense_prediction
                            ),
                            "correction": bool((not current_success) and dense_success),
                            "damage": bool(current_success and (not dense_success)),
                            "reused_from_13k_source": True,
                        }
                    )
                    rows.append(reused)
                    vectors.append(source_hidden[index, 0].detach().cpu().clone().to(torch.float16))
                    branch_wall_ms += float(reused.get("branch_collection_wall_ms", 0.0))
                reused_checkpoints = len(source_rows)
                # Dense continuation must replay the sampled prefix token by token to
                # preserve its exact RNG/numerical path.  Checkpoint extraction has a
                # different reference path: the original collector advances the KV
                # cache paragraph by paragraph.  Rebuild that cache cheaply here,
                # without hidden-state capture or forced-answer branches, so every new
                # checkpoint is bitwise comparable with a full 32K collection.
                prefill = model.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=True,
                    return_dict=True,
                )
                base_cache = prefill.past_key_values
                del prefill
                previous = 0
                for source_row in source_rows:
                    checkpoint = int(source_row["checkpoint"])
                    delta = torch.tensor(
                        [dense["tokens"][previous:checkpoint]],
                        dtype=torch.long,
                        device=device,
                    )
                    mask = torch.ones(
                        (1, input_ids.shape[1] + checkpoint),
                        dtype=torch.long,
                        device=device,
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
                    previous = checkpoint
                pending_checkpoints = [
                    int(value) for value in checkpoints if int(value) > source_budget
                ]
            for checkpoint in pending_checkpoints:
                delta = torch.tensor(
                    [dense["tokens"][previous:checkpoint]], dtype=torch.long, device=device
                )
                mask = torch.ones(
                    (1, input_ids.shape[1] + checkpoint), dtype=torch.long, device=device
                )
                capture.begin()
                teacher = model.model(
                    input_ids=delta,
                    attention_mask=mask,
                    past_key_values=base_cache,
                    use_cache=True,
                    return_dict=True,
                )
                vectors.append(capture.finish_cpu().to(torch.float16))
                base_cache = teacher.past_key_values
                del teacher
                branch = greedy_branch(
                    model,
                    tokenizer,
                    base_cache,
                    prefix_context=int(input_ids.shape[1]) + checkpoint,
                    suffix_ids=suffix_ids,
                    maximum=int(generation["force_answer_max_new_tokens"]),
                )
                current_prediction = prediction(dataset, branch["text"])
                current_success = success(dataset, gold, current_prediction)
                branch_wall_ms += float(branch["wall_ms"])
                new_branch_wall_ms += float(branch["wall_ms"])
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "problem_id": problem_id,
                        "checkpoint": checkpoint,
                        "checkpoint_schedules": ["sentence"],
                        "actual_checkpoint_schedule": "paragraph",
                        "gold_answer": gold,
                        "dense_prediction": dense_prediction,
                        "dense_success": bool(dense_success),
                        "dense_tokens": len(dense["tokens"]),
                        "dense_wall_ms": float(dense["wall_ms"]),
                        "current_prediction": current_prediction,
                        "current_success": bool(current_success),
                        "consistency": bool(
                            current_prediction is not None
                            and dense_prediction is not None
                            and current_prediction == dense_prediction
                        ),
                        "correction": bool((not current_success) and dense_success),
                        "damage": bool(current_success and (not dense_success)),
                        "branch_tokens": len(branch["tokens"]),
                        "branch_token_ids": branch["tokens"],
                        "branch_text": branch["text"],
                        "branch_generated_text": branch["generated_text"],
                        "branch_collection_wall_ms": float(branch["wall_ms"]),
                        "forced_answer_decoding": "greedy_argmax",
                        "forced_answer_do_sample": False,
                        "prompt_tokens": int(input_ids.shape[1]),
                        "prefix_context_tokens": int(input_ids.shape[1]) + checkpoint,
                        "prefix_mean_entropy_tail8": tail_mean(
                            dense["entropies_top20"], checkpoint
                        ),
                        "producer_gpu": gpu,
                        "reused_from_13k_source": False,
                    }
                )
                previous = checkpoint
    finally:
        capture.close()

    hidden_size = int(model_audit["hidden_size"])
    hidden = (
        torch.stack(vectors)[:, None, :]
        if vectors
        else torch.empty((0, 1, hidden_size), dtype=torch.float16)
    )
    absolute_destination = destination.resolve()
    artifact = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "primary_replay_view_fingerprint": fingerprint + ":paragraph",
        "dataset": dataset,
        "split": split,
        "problem_id": problem_id,
        "dtype": "bfloat16",
        "seed": int(config["seed"]),
        "problem_seed": problem_seed,
        "reproducibility": reproducibility_audit,
        "actual_checkpoint_schedule": "paragraph",
        "checkpoint_protocol": config["checkpoint"],
        "capture_layers": [capture_layer],
        "rows": rows,
        "hidden": hidden,
        "source_dense_artifact": str(absolute_destination),
        "source_common_cache_artifact": str(absolute_destination),
        "incremental_source_artifact": str(source_path) if source_path is not None else None,
        "record": record,
        "gold_answer": gold,
        "prompt_text": prompt_text,
        "prompt_tokens": int(input_ids.shape[1]),
        "dense": {
            **dense,
            "content_tokens": dense["tokens"],
            "prediction": dense_prediction,
            "success": bool(dense_success),
            "reasoning_tokens": len(dense["tokens"]),
        },
        "dense_generation": {
            "requested_max_new_tokens": int(generation["dense_max_new_tokens"]),
            "execution_mode": "generated_at_configured_budget",
            "incremental_exact_resume": source is not None,
            "source_requested_max_new_tokens": (
                len(source["dense"]["tokens"]) if source is not None else None
            ),
            "temperature": float(generation["temperature"]),
            "top_p": float(generation["top_p"]),
            "top_k": int(generation["top_k"]),
            "do_sample": bool(generation["do_sample"]),
        },
        "forced_answer_decoding": {
            "strategy": "greedy_argmax",
            "do_sample": False,
            "max_new_tokens": int(generation["force_answer_max_new_tokens"]),
            "suffix": generation["force_answer_suffix"],
        },
        "trajectory": trajectory,
        "schedule_checkpoints": checkpoints,
        "model_audit": model_audit,
        "collection": {
            "worker": worker_id,
            "host": socket.gethostname(),
            "gpu": gpu,
            "device": torch.cuda.get_device_name(device),
            "branch_wall_ms": branch_wall_ms,
            "new_branch_wall_ms": new_branch_wall_ms,
            "reused_checkpoints": reused_checkpoints,
            "new_checkpoints": len(checkpoints) - reused_checkpoints,
            "execution_mode": (
                "incremental_exact_resume_from_capped_13k_source"
                if source is not None
                else "full_generation"
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    atomic_torch_save(artifact, destination)
    return {
        "problem_id": problem_id,
        "dataset": dataset,
        "split": split,
        "dense_tokens": len(dense["tokens"]),
        "dense_success": bool(dense_success),
        "reached_max": bool(dense["reached_max_tokens"]),
        "checkpoints": len(checkpoints),
        "forced_answer_truncated": int(
            sum(len(row["branch_token_ids"]) >= int(generation["force_answer_max_new_tokens"]) for row in rows)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--problem-id",
        action="append",
        default=[],
        help="Operational exact-ID filter; may be repeated and does not alter labels.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-unlocked-legacy",
        action="store_true",
        help=(
            "Explicitly run a historical configuration without the committed runtime "
            "lock. Such outputs are non-formal and receive the legacy fingerprint."
        ),
    )
    parser.add_argument(
        "--task-scope",
        choices=("all", "heldout_extension_targets"),
        default="all",
        help=(
            "Operational collection scope. heldout_extension_targets retains the "
            "frozen protocol and full extension manifest but schedules only held-out "
            "samples that require exact 13K-to-32K continuation."
        ),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    runtime_lock_value = config.get("reproducibility", {}).get("runtime_lock")
    if runtime_lock_value is None and not args.allow_unlocked_legacy:
        raise RuntimeError(
            "formal collection requires reproducibility.runtime_lock; use a committed "
            "locked config, or pass --allow-unlocked-legacy only for historical work"
        )
    reproducibility_audit = None
    if runtime_lock_value is not None:
        deterministic_settings = strict_reproducibility(seed=0, num_threads=1)
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        runtime_identity = environment_provenance(device)
        runtime_lock_path = Path(runtime_lock_value)
        if not runtime_lock_path.is_absolute():
            runtime_lock_path = ROOT / runtime_lock_path
        runtime_lock_audit = enforce_runtime_lock(runtime_lock_path, runtime_identity)
        code_identity = code_provenance(
            ROOT,
            (
                "scripts/collect_deepseek7b_paragraph_v1.py",
                "scripts/deepseek7b_protocol_v1.py",
                "src/reproducibility.py",
            ),
        )
        reproducibility_audit = {
            "settings": deterministic_settings,
            "runtime_lock": runtime_lock_audit,
            "environment": runtime_identity,
            "code": code_identity,
        }
        # Keep the scientific fingerprint independent of logical GPU index and
        # output timestamps while binding it to the exact code and runtime lock.
        fingerprint = canonical_fingerprint(
            {
                "config": config,
                "formal_reproducibility": {
                    "protocol_id": deterministic_settings["protocol_id"],
                    "runtime_lock_id": runtime_lock_audit["lock_id"],
                    "runtime_lock_sha256": sha256_file(runtime_lock_path),
                    "git_commit": code_identity["git"]["commit"],
                    "source_sha256": code_identity["source_sha256"],
                },
            }
        )
    else:
        fingerprint = canonical_fingerprint(config)
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    incremental_targets = incremental_extension_targets(config)
    prepared_root = Path(config["data"]["prepared_root"])
    output_root = Path(config["output_root"]) / "cache"
    task_pool = all_tasks(prepared_root)
    if args.problem_id:
        selected_problem_ids = set(args.problem_id)
        task_pool = [
            task for task in task_pool if str(task[2]["problem_id"]) in selected_problem_ids
        ]
        found_problem_ids = {str(task[2]["problem_id"]) for task in task_pool}
        missing_problem_ids = sorted(selected_problem_ids - found_problem_ids)
        if missing_problem_ids:
            raise ValueError(f"unknown --problem-id values: {missing_problem_ids}")
    if args.task_scope == "heldout_extension_targets":
        task_pool = [
            task
            for task in task_pool
            if task[1] == "heldout"
            and (task[0], task[1], str(task[2]["problem_id"])) in incremental_targets
        ]
    tasks = [
        task
        for index, task in enumerate(task_pool)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    free, total = torch.cuda.mem_get_info(device)
    print(
        json.dumps(
            {
                "status": "loading",
                "worker": args.worker_id,
                "gpu": args.gpu,
                "free_gib": free / 2**30,
                "total_gib": total / 2**30,
                "assigned": len(tasks),
            }
        ),
        flush=True,
    )
    model, tokenizer, model_audit = load_model(Path(config["model"]["local_path"]), device)
    completed = skipped = failures = 0
    started = time.time()
    for dataset, split, record in tasks:
        problem_id = str(record["problem_id"])
        destination = output_root / dataset / split / f"sample_{problem_id}.pt"
        require_incremental = (dataset, split, problem_id) in incremental_targets
        if args.resume and valid(
            destination,
            fingerprint,
            problem_id,
            require_incremental_exact_resume=require_incremental,
        ):
            skipped += 1
            continue
        if destination.exists():
            raise RuntimeError(f"refusing incompatible artifact: {destination}")
        try:
            result = collect_one(
                dataset,
                split,
                record,
                destination,
                config,
                fingerprint,
                model,
                tokenizer,
                model_audit,
                device,
                args.worker_id,
                args.gpu,
                reproducibility_audit,
            )
            completed += 1
            print(json.dumps({"status": "completed", "completed": completed, **result}), flush=True)
        except Exception as error:
            failures += 1
            print(
                json.dumps(
                    {
                        "status": "error",
                        "dataset": dataset,
                        "split": split,
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
        "worker": args.worker_id,
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "completed": completed,
        "skipped": skipped,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
        "protocol_fingerprint": fingerprint,
        "task_scope": args.task_scope,
        "problem_ids": list(args.problem_id),
        "reproducibility": reproducibility_audit,
    }
    summary_path = Path(config["output_root"]) / "workers" / f"{args.worker_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

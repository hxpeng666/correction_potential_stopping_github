#!/usr/bin/env python3
"""Deterministic paragraph collection for Qwen3-14B on GSM8K/MATH OOD data."""
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
sys.path.insert(0, str(ROOT / "scripts"))

from deepseek7b_protocol_v1 import (
    atomic_torch_save,
    canonical_fingerprint,
    generate_dense,
    greedy_branch,
    paragraph_checkpoints,
    prediction,
    prefill_token_cache,
    render_prompt,
    stable_seed,
    success,
    tail_mean,
)
from src.qwen3_reasoning import CheckpointHiddenCapture, load_qwen3
from src.reproducibility import (
    code_provenance,
    enforce_runtime_lock,
    environment_provenance,
    sha256_file,
    sha256_json,
    strict_reproducibility,
)

DATA_LAYOUT = (
    ("gsm8k", "probe_train"),
    ("gsm8k", "calibration"),
    ("gsm8k", "heldout"),
    ("math", "probe_train"),
    ("math", "calibration"),
    ("math500", "heldout"),
    ("aime", "heldout"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def all_tasks(prepared_root: Path) -> list[tuple[str, str, dict[str, Any]]]:
    tasks: list[tuple[str, str, dict[str, Any]]] = []
    for dataset, split in DATA_LAYOUT:
        path = prepared_root / dataset / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        tasks.extend((dataset, split, row) for row in read_jsonl(path))
    return sorted(tasks, key=lambda item: (item[0], item[1], str(item[2]["problem_id"])))


def data_identity(prepared_root: Path) -> dict[str, Any]:
    files = {}
    for dataset, split in DATA_LAYOUT:
        path = prepared_root / dataset / f"{split}.jsonl"
        files[f"{dataset}/{split}.jsonl"] = {
            "rows": len(read_jsonl(path)),
            "sha256": sha256_file(path),
        }
    return {"files": files, "sha256": sha256_json(files)}


def gold_for(dataset: str, record: dict[str, Any]) -> str | None:
    if "gold_answer" in record:
        return str(record["gold_answer"])
    if dataset == "gsm8k":
        marker = str(record["answer"]).rsplit("####", 1)
        return marker[-1].strip().replace(",", "") if marker else None
    raise KeyError(f"no gold answer for {dataset}:{record.get('problem_id')}")


def tokenize_prompt_like_deepseek(tokenizer, prompt_text: str) -> torch.Tensor:
    """Use the frozen DeepSeek two-step prompt-tokenization path exactly."""
    return tokenizer(prompt_text, return_tensors="pt").input_ids


def artifact_path(root: Path, dataset: str, split: str, problem_id: str) -> Path:
    return root / "cache" / dataset / split / f"sample_{problem_id}.pt"


def valid(path: Path, fingerprint: str, problem_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        return (
            value.get("status") == "complete"
            and value.get("protocol_fingerprint") == fingerprint
            and str(value.get("problem_id")) == problem_id
            and value.get("actual_checkpoint_schedule") == "paragraph"
        )
    except Exception:
        return False


def cap_forced_answer(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    dense_tokens: list[int],
    suffix_ids: torch.Tensor,
    maximum: int,
) -> dict[str, Any]:
    """Greedily extract the final answer from the exact capped Dense prefix."""
    dense_ids = torch.tensor([dense_tokens], dtype=torch.long, device=input_ids.device)
    combined = torch.cat((input_ids, dense_ids), dim=1)
    with torch.inference_mode():
        cache, _last_hidden = prefill_token_cache(model, combined, chunk_size=512)
        result = greedy_branch(
            model,
            tokenizer,
            cache,
            prefix_context=int(combined.shape[1]),
            suffix_ids=suffix_ids,
            maximum=maximum,
        )
    del cache, combined, dense_ids, _last_hidden
    return result


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
    data_audit: dict[str, Any],
    device: torch.device,
    worker_id: str,
    gpu: int,
    reproducibility_audit: dict[str, Any],
) -> dict[str, Any]:
    problem_id = str(record["problem_id"])
    problem_seed = stable_seed(int(config["seed"]), problem_id)
    prompt_text = render_prompt(tokenizer, str(record["question"]))
    expected_prompt_protocol = {
        "render": "apply_chat_template_tokenize_false_add_generation_prompt_true",
        "encode": "tokenizer_default_add_special_tokens",
        "reference": "scripts/collect_deepseek7b_paragraph_v1.py",
    }
    if config.get("prompt_tokenization") != expected_prompt_protocol:
        raise ValueError("prompt tokenization is not the frozen DeepSeek path")
    input_ids = tokenize_prompt_like_deepseek(tokenizer, prompt_text).to(device)
    generation = config["generation"]
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
    gold = gold_for(dataset, record)
    raw_dense_prediction = prediction(dataset, dense["text"])
    suffix_ids = tokenizer(
        generation["force_answer_suffix"],
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    cap_branch: dict[str, Any] | None = None
    if bool(dense["reached_max_tokens"]):
        cap_branch = cap_forced_answer(
            model,
            tokenizer,
            input_ids,
            [int(token) for token in dense["tokens"]],
            suffix_ids,
            int(generation["force_answer_max_new_tokens"]),
        )
        dense_prediction = prediction(dataset, cap_branch["text"])
        dense_grader = "forced_answer_at_exact_13k_prefix"
    else:
        dense_prediction = raw_dense_prediction
        dense_grader = "natural_dense_completion"
    dense_success = success(dataset, gold, dense_prediction)

    checkpoints, trajectory = paragraph_checkpoints(tokenizer, dense["tokens"])
    capture_layer = int(config["features"]["layer_zero_based"])
    capture = CheckpointHiddenCapture(model, [capture_layer])
    rows: list[dict[str, Any]] = []
    vectors: list[torch.Tensor] = []
    branch_wall_ms = 0.0
    try:
        with torch.inference_mode():
            prefill = model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=True,
                return_dict=True,
            )
            base_cache = prefill.past_key_values
            del prefill
            previous = 0
            for checkpoint in checkpoints:
                delta = torch.tensor(
                    [dense["tokens"][previous:checkpoint]],
                    dtype=torch.long,
                    device=device,
                )
                mask = torch.ones(
                    (1, int(input_ids.shape[1]) + checkpoint),
                    dtype=torch.long,
                    device=device,
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
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "problem_id": problem_id,
                        "checkpoint": checkpoint,
                        "checkpoint_schedules": ["paragraph"],
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
                    }
                )
                previous = checkpoint
    finally:
        capture.close()

    hidden_size = int(model_audit["hidden_size"])
    hidden = (
        torch.stack(vectors)
        if vectors
        else torch.empty((0, 1, hidden_size), dtype=torch.float16)
    )
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
        "data_identity": data_audit,
        "actual_checkpoint_schedule": "paragraph",
        "checkpoint_protocol": config["checkpoint"],
        "capture_layers": [capture_layer],
        "rows": rows,
        "hidden": hidden,
        "record": record,
        "gold_answer": gold,
        "prompt_text": prompt_text,
        "prompt_tokens": int(input_ids.shape[1]),
        "prompt_token_ids": [int(token) for token in input_ids[0].tolist()],
        "dense": {
            **dense,
            "content_tokens": dense["tokens"],
            "raw_completion_prediction": raw_dense_prediction,
            "prediction": dense_prediction,
            "success": bool(dense_success),
            "grader": dense_grader,
            "cap_forced_answer": cap_branch,
            "reasoning_tokens": len(dense["tokens"]),
        },
        "dense_generation": {
            "requested_max_new_tokens": int(generation["dense_max_new_tokens"]),
            "temperature": float(generation["temperature"]),
            "top_p": float(generation["top_p"]),
            "top_k": int(generation["top_k"]),
            "do_sample": bool(generation["do_sample"]),
            "per_problem_seed": True,
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
        "dense_grader": dense_grader,
        "reached_max": bool(dense["reached_max_tokens"]),
        "checkpoints": len(checkpoints),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--problem-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    strict_settings = strict_reproducibility(seed=0, num_threads=1)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    runtime_identity = environment_provenance(device)
    runtime_lock_path = Path(config["reproducibility"]["runtime_lock"])
    if not runtime_lock_path.is_absolute():
        runtime_lock_path = ROOT / runtime_lock_path
    runtime_lock_audit = enforce_runtime_lock(runtime_lock_path, runtime_identity)
    code_identity = code_provenance(
        ROOT,
        (
            "scripts/collect_qwen3_14b_deterministic_ood_v1.py",
            "scripts/deepseek7b_protocol_v1.py",
            "src/qwen3_reasoning.py",
            "src/reproducibility.py",
        ),
    )
    prepared_root = (args.prepared_root or Path(config["data"]["prepared_root"])).resolve()
    model_path = (args.model_path or Path(config["model"]["local_path"])).resolve()
    output_root = (args.output_root or Path(config["output_root"])).resolve()
    data_audit = data_identity(prepared_root)

    print(
        json.dumps(
            {
                "status": "loading",
                "worker": args.worker_id,
                "gpu": args.gpu,
                "free_gib": torch.cuda.mem_get_info(device)[0] / 2**30,
            }
        ),
        flush=True,
    )
    model, tokenizer, model_audit = load_qwen3(
        model_path,
        device,
        str(config["model"]["dtype"]),
        str(config["model"]["attention_backend"]),
    )
    expected_model = config["model"]
    if (
        model_audit["hidden_size"] != int(expected_model["hidden_size"])
        or model_audit["layers"] != int(expected_model["num_hidden_layers"])
    ):
        raise RuntimeError(f"model/config mismatch: {model_audit}")
    scientific_model_audit = {
        key: value for key, value in model_audit.items() if key != "path"
    }
    fingerprint = canonical_fingerprint(
        {
            "config": config,
            "model": scientific_model_audit,
            "data": data_audit,
            "formal_reproducibility": {
                "protocol_id": strict_settings["protocol_id"],
                "runtime_lock_id": runtime_lock_audit["lock_id"],
                "runtime_lock_sha256": sha256_file(runtime_lock_path),
                "git_commit": code_identity["git"]["commit"],
                "source_sha256": code_identity["source_sha256"],
            },
        }
    )
    reproducibility_audit = {
        "settings": strict_settings,
        "runtime_lock": runtime_lock_audit,
        "environment": runtime_identity,
        "code": code_identity,
    }

    task_pool = all_tasks(prepared_root)
    if args.problem_id:
        selected = set(args.problem_id)
        task_pool = [task for task in task_pool if str(task[2]["problem_id"]) in selected]
        found = {str(task[2]["problem_id"]) for task in task_pool}
        if found != selected:
            raise ValueError(f"unknown problem ids: {sorted(selected - found)}")
    tasks = [
        task for index, task in enumerate(task_pool)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    completed = skipped = failures = 0
    started = time.time()
    for dataset, split, record in tasks:
        problem_id = str(record["problem_id"])
        destination = artifact_path(output_root, dataset, split, problem_id)
        if args.resume and valid(destination, fingerprint, problem_id):
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
                data_audit,
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
        "assigned": len(tasks),
        "completed": completed,
        "skipped": skipped,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
        "protocol_fingerprint": fingerprint,
        "data_identity": data_audit,
        "model_audit": model_audit,
        "reproducibility": reproducibility_audit,
    }
    summary_path = output_root / "workers" / f"{args.worker_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

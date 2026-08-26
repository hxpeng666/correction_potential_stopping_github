#!/usr/bin/env python3
"""Collect paired full-trajectory checkpoint schedules with greedy forced answers."""
from __future__ import annotations

import argparse
import bisect
import gc
import hashlib
import json
import os
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers.cache_utils import DynamicCache

from src.final_paper_inference import atomic_torch_save, prediction_for, success_for
from src.final_paper_replay_cache import raw_semantic_boundaries
from src.mmlu_pro_protocol import parse_answer as parse_mmlu_pro_answer
from src.qwen3_reasoning import CheckpointHiddenCapture, load_qwen3
from src.utils import load_yaml, seed_everything


SCHEDULES = ("sentence", "fixed_budget", "prefix_stride", "lynx_cue", "paragraph", "hybrid")
PARAGRAPH = re.compile(r"\n\s*\n+")


def canonical_fingerprint(config: dict[str, Any]) -> str:
    payload = {
        "protocol_id": config["protocol_id"],
        "model": config["model"],
        "generation": config["generation"],
        "checkpoint": config["checkpoint"],
        "datasets": config["datasets"],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def think_end(tokenizer, content: list[int]) -> tuple[int, str]:
    pattern = list(tokenizer("</think>", add_special_tokens=False).input_ids)
    for start in range(0, len(content) - len(pattern) + 1):
        if content[start : start + len(pattern)] == pattern:
            return start, "first_think_close"
    return len(content), "full_dense_content"


def token_ends(tokenizer, ids: list[int]) -> tuple[str, list[int]]:
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded.input_ids) == ids:
        return text, [int(end) for _start, end in encoded.offset_mapping]
    ends = [
        len(tokenizer.decode(ids[:end], skip_special_tokens=False, clean_up_tokenization_spaces=False))
        for end in range(1, len(ids) + 1)
    ]
    return text, ends


def lynx_cue_positions(tokenizer, ids: list[int], patterns: list[str]) -> list[int]:
    # Exact token-to-character convention used by the public LYNX implementation.
    pieces = [
        tokenizer.decode([token], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for token in ids
    ]
    cumulative: list[int] = []
    text = ""
    for piece in pieces:
        text += piece
        cumulative.append(len(text))
    lowered = text.lower()
    result: set[int] = set()
    for pattern in patterns:
        start = 0
        while True:
            found = lowered.find(pattern.lower(), start)
            if found < 0:
                break
            character_end = found + len(pattern)
            result.add(next((i + 1 for i, end in enumerate(cumulative) if end >= character_end), len(ids)))
            start = character_end
    return sorted(value for value in result if value > 0)


def build_schedules(tokenizer, content: list[int], config: dict[str, Any]) -> tuple[dict[str, list[int]], dict[str, Any]]:
    upper, end_reason = think_end(tokenizer, content)
    limited = content[:upper]
    semantic, decoded = raw_semantic_boundaries(tokenizer, content, upper) if upper else ([], "")
    sentence: list[int] = []
    previous = 0
    gap = int(config["sentence_minimum_gap"])
    for checkpoint in semantic:
        if checkpoint <= upper and checkpoint - previous >= gap:
            sentence.append(checkpoint)
            previous = checkpoint

    fixed = [int(value) for value in config["fixed_token_budgets"] if 0 < int(value) <= upper]
    stride = int(config["prefix_stride"])
    prefix = list(range(stride, upper + 1, stride))
    cues = lynx_cue_positions(tokenizer, limited, [str(x) for x in config["cue_patterns"]])

    text, ends = token_ends(tokenizer, limited) if limited else ("", [])
    paragraph: set[int] = set()
    for match in PARAGRAPH.finditer(text):
        index = bisect.bisect_left(ends, match.end())
        if index < len(ends):
            paragraph.add(index + 1)

    hybrid: list[int] = []
    semantic_set = set(semantic)
    previous = 0
    minimum_gap = int(config["hybrid_minimum_gap"])
    maximum_gap = int(config["hybrid_maximum_gap"])
    for checkpoint in range(1, upper + 1):
        if checkpoint - previous < minimum_gap:
            continue
        if checkpoint in semantic_set or checkpoint - previous >= maximum_gap:
            hybrid.append(checkpoint)
            previous = checkpoint

    schedules = {
        "sentence": sentence,
        "fixed_budget": fixed,
        "prefix_stride": prefix,
        "lynx_cue": cues,
        "paragraph": sorted(paragraph),
        "hybrid": hybrid,
    }
    return schedules, {
        "reasoning_end": upper,
        "reasoning_end_source": end_reason,
        "dense_content_tokens": len(content),
        "decoded_reasoning_chars": len(decoded),
    }


def tail_mean(values: list[float], end: int, width: int = 8) -> float:
    local = values[max(0, end - width) : end]
    return float(sum(local) / len(local)) if local else float("nan")


def finite_or(value: Any, fallback: float) -> float:
    try:
        numeric = float(value)
        return numeric if numeric == numeric and abs(numeric) != float("inf") else float(fallback)
    except (TypeError, ValueError):
        return float(fallback)


def greedy_branch(model, tokenizer, base_cache, prefix_context: int, suffix_ids: torch.Tensor, maximum: int) -> dict[str, Any]:
    cache = DynamicCache.from_legacy_cache(base_cache.to_legacy_cache())
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    started = time.perf_counter()
    mask = torch.ones((1, prefix_context + suffix_ids.shape[1]), dtype=torch.long, device=suffix_ids.device)
    output = model(input_ids=suffix_ids, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True)
    past = output.past_key_values
    tokens = [int(torch.argmax(output.logits[0, -1].float()).item())]
    while len(tokens) < maximum and tokens[-1] not in eos:
        current = torch.tensor([[tokens[-1]]], dtype=torch.long, device=suffix_ids.device)
        mask = torch.ones((1, prefix_context + suffix_ids.shape[1] + len(tokens)), dtype=torch.long, device=suffix_ids.device)
        output = model(input_ids=current, attention_mask=mask, past_key_values=past, use_cache=True, return_dict=True)
        past = output.past_key_values
        tokens.append(int(torch.argmax(output.logits[0, -1].float()).item()))
    generated = tokenizer.decode(tokens, skip_special_tokens=True)
    suffix = tokenizer.decode(suffix_ids[0], skip_special_tokens=True)
    return {"tokens": tokens, "generated_text": generated, "text": suffix + generated, "wall_ms": 1000.0 * (time.perf_counter() - started)}


def destinations(output_root: Path, split: str, problem_id: str, schedules: list[str]) -> dict[str, Path]:
    return {schedule: output_root / "cache" / schedule / split / f"sample_{problem_id}.pt" for schedule in schedules}


def valid(path: Path, fingerprint: str, problem_id: str, schedule: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
        return value.get("status") == "complete" and value.get("protocol_fingerprint") == fingerprint and str(value.get("problem_id")) == problem_id and value.get("actual_checkpoint_schedule") == schedule
    except Exception:
        return False


def collect_one(source_path: Path, output_root: Path, config: dict[str, Any], fingerprint: str, dataset: str, model, tokenizer, audit: dict[str, Any], device: torch.device, gpu: int, worker: str, resume: bool) -> dict[str, Any]:
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    problem_id = str(source["problem_id"])
    split = str(source["split"])
    active_schedules = [str(value) for value in config["checkpoint"]["schedules"]]
    if not active_schedules or any(value not in SCHEDULES for value in active_schedules):
        raise ValueError(f"invalid active schedules: {active_schedules}")
    targets = destinations(output_root, split, problem_id, active_schedules)
    if resume and all(valid(path, fingerprint, problem_id, schedule) for schedule, path in targets.items()):
        return {"status": "skipped", "problem_id": problem_id, "branches": 0}
    existing_bad = [str(path) for schedule, path in targets.items() if path.exists() and not valid(path, fingerprint, problem_id, schedule)]
    if existing_bad:
        raise RuntimeError(f"refusing to overwrite incompatible artifacts: {existing_bad}")

    content = list(source["dense"]["content_tokens"])
    all_schedules, trajectory = build_schedules(tokenizer, content, config["checkpoint"])
    schedules = {name: all_schedules[name] for name in active_schedules}
    union = sorted(set().union(*(set(values) for values in schedules.values())))
    prompt_ids = tokenizer(source["prompt_text"], return_tensors="pt").input_ids.to(device)
    prompt_tokens = int(prompt_ids.shape[1])
    if prompt_tokens != int(source["prompt_tokens"]):
        raise ValueError(f"prompt retokenization mismatch: {source_path}")
    suffix_ids = tokenizer(config["generation"]["force_answer_suffix"], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    capture = CheckpointHiddenCapture(model, [int(config["features"]["layer_zero_based"])])
    reuse = (
        {int(row["checkpoint"]): dict(row) for row in source.get("rows", [])}
        if bool(config["generation"].get("reuse_forced_answer_branches", True))
        else {}
    )
    dense_prediction = source["dense"]["prediction"]
    dense_success = bool(source["dense"]["success"])
    entropies = [float(x) for x in source["dense"].get("entropies_top20", [])]
    rows: dict[int, dict[str, Any]] = {}
    vectors: dict[int, torch.Tensor] = {}
    reused = 0
    new_branches = 0
    branch_wall_ms = 0.0
    try:
        with torch.inference_mode():
            prefill = model.model(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), use_cache=True, return_dict=True)
            base_cache = prefill.past_key_values
            del prefill
            previous = 0
            for checkpoint in union:
                delta = torch.tensor([content[previous:checkpoint]], dtype=torch.long, device=device)
                mask = torch.ones((1, prompt_tokens + checkpoint), dtype=torch.long, device=device)
                capture.begin()
                teacher = model.model(input_ids=delta, attention_mask=mask, past_key_values=base_cache, use_cache=True, return_dict=True)
                vector = capture.finish_cpu().to(torch.float16)
                base_cache = teacher.past_key_values
                del teacher
                vectors[checkpoint] = vector
                if checkpoint in reuse:
                    branch_row = dict(reuse[checkpoint])
                    reused += 1
                else:
                    branch = greedy_branch(model, tokenizer, base_cache, prompt_tokens + checkpoint, suffix_ids, int(config["generation"]["force_answer_max_new_tokens"]))
                    if dataset == "mmlu_pro":
                        prediction = parse_mmlu_pro_answer(branch["text"], len(source["record"]["choices"]))
                        current_success = prediction is not None and prediction == source["gold_answer"]
                    else:
                        prediction = prediction_for("gsm8k", branch["text"])
                        current_success = success_for("gsm8k", source["gold_answer"], prediction)
                    branch_row = {
                        "current_prediction": prediction,
                        "current_success": bool(current_success),
                        "branch_tokens": len(branch["tokens"]),
                        "branch_token_ids": list(branch["tokens"]),
                        "branch_text": branch["text"],
                        "branch_generated_text": branch["generated_text"],
                        "branch_collection_wall_ms": float(branch["wall_ms"]),
                    }
                    new_branches += 1
                    branch_wall_ms += float(branch["wall_ms"])
                current_prediction = branch_row.get("current_prediction")
                current_success = bool(branch_row.get("current_success", False))
                rows[checkpoint] = {
                    **branch_row,
                    "dataset": dataset,
                    "split": split,
                    "problem_id": problem_id,
                    "checkpoint": checkpoint,
                    "checkpoint_schedules": [],
                    "gold_answer": source["gold_answer"],
                    "dense_prediction": dense_prediction,
                    "dense_success": dense_success,
                    "dense_tokens": int(source["dense"]["reasoning_tokens"]),
                    "dense_wall_ms": finite_or(source["dense"].get("wall_ms"), source["dense"]["reasoning_tokens"]),
                    "dense_prefill_cuda_ms": finite_or(source["dense"].get("prefill_cuda_ms"), 0.0),
                    "current_prediction": current_prediction,
                    "current_success": current_success,
                    "consistency": bool(current_prediction is not None and dense_prediction is not None and current_prediction == dense_prediction),
                    "correction": bool((not current_success) and dense_success),
                    "damage": bool(current_success and (not dense_success)),
                    "branch_timing_source": "excluded_greedy_collection",
                    "forced_answer_decoding": "greedy_argmax",
                    "forced_answer_do_sample": False,
                    "prompt_tokens": prompt_tokens,
                    "prefix_context_tokens": prompt_tokens + checkpoint,
                    "prefix_mean_entropy_tail8": tail_mean(entropies, checkpoint),
                    "producer_gpu": gpu,
                }
                previous = checkpoint
    finally:
        capture.close()

    for schedule, checkpoints in schedules.items():
        selected_rows = []
        selected_vectors = []
        for checkpoint in checkpoints:
            row = dict(rows[checkpoint])
            row["checkpoint_schedules"] = ["sentence"]  # compatibility view for the frozen trainer
            row["actual_checkpoint_schedule"] = schedule
            selected_rows.append(row)
            selected_vectors.append(vectors[checkpoint])
        hidden = torch.stack(selected_vectors) if selected_vectors else torch.empty((0, 1, int(audit["hidden_size"])), dtype=torch.float16)
        artifact = {
            "schema_version": 7,
            "status": "complete",
            "protocol_id": config["protocol_id"],
            "protocol_fingerprint": fingerprint,
            "primary_replay_view_fingerprint": fingerprint + ":" + schedule,
            "dataset": dataset,
            "split": split,
            "problem_id": problem_id,
            "dtype": config["model"]["dtype"],
            "seed": int(config["seed"]["global"]),
            "actual_checkpoint_schedule": schedule,
            "checkpoint_protocol": config["checkpoint"],
            "capture_layers": [int(config["features"]["layer_zero_based"])],
            "rows": selected_rows,
            "hidden": hidden,
            "source_dense_artifact": str(source_path.resolve()),
            "source_common_cache_artifact": str(source_path.resolve()),
            "record": source["record"],
            "gold_answer": source["gold_answer"],
            "prompt_text": source["prompt_text"],
            "prompt_tokens": prompt_tokens,
            "dense": source["dense"],
            "forced_answer_decoding": {
                "strategy": "greedy_argmax",
                "do_sample": False,
                "max_new_tokens": int(config["generation"]["force_answer_max_new_tokens"]),
                "suffix": config["generation"]["force_answer_suffix"],
            },
            "trajectory": trajectory,
            "schedule_checkpoints": checkpoints,
            "model_audit": audit,
            "collection": {"worker": worker, "host": socket.gethostname(), "gpu": gpu, "device": torch.cuda.get_device_name(gpu), "new_branches": new_branches, "reused_branches": reused, "branch_wall_ms": branch_wall_ms, "created_at": datetime.now(timezone.utc).isoformat()},
        }
        targets[schedule].parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(artifact, targets[schedule])
    return {"status": "completed", "problem_id": problem_id, "branches": new_branches, "reused": reused, "reasoning_end": trajectory["reasoning_end"], "checkpoints": {key: len(value) for key, value in schedules.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), default="gsm8k")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--assigned-subshard-index", type=int)
    parser.add_argument("--assigned-num-subshards", type=int)
    parser.add_argument("--split", choices=("all", "probe_train", "calibration", "heldout"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    fingerprint = canonical_fingerprint(config)
    output_root = args.output_root or Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    source_root = ROOT / config["datasets"][args.dataset]["source_root"]
    splits = ("probe_train", "calibration", "heldout") if args.split == "all" else (args.split,)
    paths = sorted(path for split in splits for path in (source_root / split).glob("sample_*.pt"))
    assigned = [path for index, path in enumerate(paths) if index % args.num_shards == args.shard_index]
    if (args.assigned_subshard_index is None) != (args.assigned_num_subshards is None):
        raise ValueError("assigned subshard index/count must be provided together")
    if args.assigned_num_subshards is not None:
        if args.assigned_num_subshards <= 0 or not 0 <= args.assigned_subshard_index < args.assigned_num_subshards:
            raise ValueError("invalid assigned subshard index/count")
        assigned = [
            path for index, path in enumerate(assigned)
            if index % args.assigned_num_subshards == args.assigned_subshard_index
        ]
    if args.limit is not None:
        assigned = assigned[: args.limit]
    seed_everything(int(config["seed"]["global"]))
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    free, total = torch.cuda.mem_get_info(device)
    partition = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned_subshard_index": args.assigned_subshard_index,
        "assigned_num_subshards": args.assigned_num_subshards,
    }
    print(json.dumps({"status": "loading", "worker": args.worker_id, "gpu": args.gpu, "free_GiB": free / 2**30, "total_GiB": total / 2**30, "assigned": len(assigned), "partition": partition}), flush=True)
    model, tokenizer, audit = load_qwen3(ROOT / config["model"]["local_path"], device, config["model"]["dtype"], config["model"]["attention_backend"])
    completed = skipped = branches = reused = failures = 0
    started = time.time()
    for source_path in assigned:
        try:
            result = collect_one(source_path, output_root, config, fingerprint, args.dataset, model, tokenizer, audit, device, args.gpu, args.worker_id, args.resume)
            completed += int(result["status"] == "completed")
            skipped += int(result["status"] == "skipped")
            branches += int(result.get("branches", 0))
            reused += int(result.get("reused", 0))
            print(json.dumps({"worker": args.worker_id, "completed": completed, "skipped": skipped, "failures": failures, **result}), flush=True)
        except Exception as error:
            failures += 1
            print(json.dumps({"status": "error", "worker": args.worker_id, "source": str(source_path), "error_type": type(error).__name__, "error": str(error)}), flush=True)
            if isinstance(error, torch.cuda.OutOfMemoryError):
                raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    summary = {"status": "complete" if failures == 0 else "failed", "worker": args.worker_id, "gpu": args.gpu, "partition": partition, "completed": completed, "skipped": skipped, "failures": failures, "new_branches": branches, "reused_branches": reused, "elapsed_seconds": time.time() - started, "protocol_fingerprint": fingerprint}
    summary_path = output_root / "workers" / f"{args.worker_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name("." + summary_path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

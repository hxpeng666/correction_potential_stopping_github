#!/usr/bin/env python3
"""Collect a problem-uniform relative-budget frontier from frozen Dense traces."""
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "src").is_dir():
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers.cache_utils import DynamicCache

from src.final_paper_inference import atomic_torch_save, prediction_for, success_for
from src.mmlu_pro_protocol import parse_answer as parse_mmlu_pro_answer
from src.qwen3_reasoning import load_qwen3
from src.utils import load_yaml, seed_everything


def canonical_fingerprint(config: dict[str, Any], dataset: str) -> str:
    payload = {
        "protocol_id": config["protocol_id"],
        "model": config["model"],
        "generation": config["generation"],
        "budget": config["budget"],
        "dataset": dataset,
        "source_root": config["datasets"][dataset]["source_root"],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def find_subsequence(values: list[int], pattern: list[int], start: int = 0) -> int | None:
    for index in range(start, len(values) - len(pattern) + 1):
        if values[index : index + len(pattern)] == pattern:
            return index
    return None


def thinking_range(tokenizer, content: list[int]) -> tuple[int, int, dict[str, Any]]:
    open_ids = list(tokenizer("<think>", add_special_tokens=False).input_ids)
    close_ids = list(tokenizer("</think>", add_special_tokens=False).input_ids)
    opening = find_subsequence(content, open_ids)
    start = opening + len(open_ids) if opening is not None else 0
    closing = find_subsequence(content, close_ids, start)
    end = closing if closing is not None else len(content)
    return start, end, {
        "opening_found": opening is not None,
        "closing_found": closing is not None,
        "thinking_start": start,
        "thinking_end": end,
        "thinking_tokens": max(0, end - start),
    }


def clone_cache(cache):
    return DynamicCache.from_legacy_cache(cache.to_legacy_cache())


def greedy_branch(model, tokenizer, base_cache, prefix_context: int, suffix_ids, maximum: int):
    cache = clone_cache(base_cache)
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    started = time.perf_counter()
    mask = torch.ones(
        (1, prefix_context + suffix_ids.shape[1]), dtype=torch.long, device=suffix_ids.device
    )
    output = model(
        input_ids=suffix_ids,
        attention_mask=mask,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    past = output.past_key_values
    tokens = [int(torch.argmax(output.logits[0, -1].float()).item())]
    while len(tokens) < maximum and tokens[-1] not in eos:
        current = torch.tensor([[tokens[-1]]], dtype=torch.long, device=suffix_ids.device)
        mask = torch.ones(
            (1, prefix_context + suffix_ids.shape[1] + len(tokens)),
            dtype=torch.long,
            device=suffix_ids.device,
        )
        output = model(
            input_ids=current,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        tokens.append(int(torch.argmax(output.logits[0, -1].float()).item()))
    generated = tokenizer.decode(tokens, skip_special_tokens=True)
    suffix = tokenizer.decode(suffix_ids[0], skip_special_tokens=True)
    return {
        "tokens": tokens,
        "generated_text": generated,
        "text": suffix + generated,
        "terminated_by_eos": bool(tokens and tokens[-1] in eos),
        "wall_ms": 1000.0 * (time.perf_counter() - started),
    }


def parse_branch(dataset: str, branch: dict[str, Any], source: dict[str, Any]):
    if dataset == "mmlu_pro":
        prediction = parse_mmlu_pro_answer(branch["text"], len(source["record"]["choices"]))
        success = prediction is not None and prediction == source["gold_answer"]
    else:
        prediction = prediction_for("gsm8k", branch["text"])
        success = success_for("gsm8k", source["gold_answer"], prediction)
    return prediction, bool(success)


def valid(path: Path, fingerprint: str, problem_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
        return (
            value.get("status") == "complete"
            and value.get("protocol_fingerprint") == fingerprint
            and str(value.get("problem_id")) == problem_id
        )
    except Exception:
        return False


def collect_one(source_path, destination, config, dataset, fingerprint, model, tokenizer, device, gpu, worker):
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    problem_id = str(source["problem_id"])
    if destination.exists() and not valid(destination, fingerprint, problem_id):
        raise RuntimeError(f"refusing to overwrite incompatible artifact: {destination}")
    content = list(source["dense"]["content_tokens"])
    start, end, trace = thinking_range(tokenizer, content)
    span = end - start
    dense_tokens = int(source["dense"]["reasoning_tokens"])
    prompt_ids = tokenizer(source["prompt_text"], return_tensors="pt").input_ids.to(device)
    prompt_tokens = int(prompt_ids.shape[1])
    if prompt_tokens != int(source["prompt_tokens"]):
        raise ValueError(f"prompt retokenization mismatch: {source_path}")

    fractions = [float(value) for value in config["budget"]["retained_fractions"]]
    minimum = int(config["budget"]["minimum_retained_dense_tokens"])
    fraction_to_point: dict[float, int | None] = {}
    for fraction in fractions:
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"retained fraction must lie in (0,1): {fraction}")
        if dense_tokens <= 0:
            fraction_to_point[fraction] = None
        else:
            retained = min(len(content), max(minimum, int(math.floor(fraction * dense_tokens))))
            fraction_to_point[fraction] = retained
    points = sorted({point for point in fraction_to_point.values() if point is not None})

    generation = config["generation"]
    suffix = str(generation["force_answer_suffix"])
    maximum = int(generation["force_answer_max_new_tokens"])
    suffix_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
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
                delta = torch.tensor([content[previous:point]], dtype=torch.long, device=device)
                mask = torch.ones((1, prompt_tokens + point), dtype=torch.long, device=device)
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
                model, tokenizer, base_cache, prompt_tokens + point, suffix_ids, maximum
            )
            previous = point

    dense_success = bool(source["dense"]["success"])
    rows = []
    for fraction in fractions:
        point = fraction_to_point[fraction]
        if point is None:
            rows.append({
                "retained_fraction": fraction,
                "target_reasoning_saving_fraction": 1.0 - fraction,
                "dense_fallback": True,
                "current_prediction": source["dense"]["prediction"],
                "current_success": dense_success,
                "dense_prediction": source["dense"]["prediction"],
                "dense_success": dense_success,
                "dense_tokens": dense_tokens,
                "stop_reasoning_tokens": dense_tokens,
                "stop_total_tokens": dense_tokens,
                "lost_correct": False,
                "helped": False,
            })
            continue
        branch = branches[point]
        prediction, success = parse_branch(dataset, branch, source)
        rows.append({
            "retained_fraction": fraction,
            "target_reasoning_saving_fraction": 1.0 - fraction,
            "dense_fallback": False,
            "thinking_start": start,
            "thinking_end": end,
            "thinking_tokens": span,
            "retained_dense_tokens": point,
            "actual_retained_dense_fraction": point / dense_tokens,
            "checkpoint_after_think_close": bool(point > end),
            "checkpoint": point,
            "current_prediction": prediction,
            "current_success": success,
            "dense_prediction": source["dense"]["prediction"],
            "dense_success": dense_success,
            "dense_tokens": dense_tokens,
            "branch_tokens": len(branch["tokens"]),
            "branch_token_ids": branch["tokens"],
            "branch_text": branch["text"],
            "branch_generated_text": branch["generated_text"],
            "forced_answer_truncated": not branch["terminated_by_eos"],
            "stop_reasoning_tokens": point,
            "stop_total_tokens": point + int(suffix_ids.shape[1]) + len(branch["tokens"]),
            "lost_correct": bool(dense_success and not success),
            "helped": bool((not dense_success) and success),
        })
    artifact = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "dataset": dataset,
        "split": str(source["split"]),
        "problem_id": problem_id,
        "record": source["record"],
        "gold_answer": source["gold_answer"],
        "prompt_text": source["prompt_text"],
        "prompt_tokens": prompt_tokens,
        "dense": source["dense"],
        "source_artifact": str(source_path.resolve()),
        "thinking_range": trace,
        "rows": rows,
        "forced_answer_decoding": {
            "strategy": "greedy_argmax",
            "do_sample": False,
            "max_new_tokens": maximum,
            "suffix": suffix,
        },
        "collection": {
            "worker": worker,
            "host": socket.gethostname(),
            "gpu": gpu,
            "device": torch.cuda.get_device_name(gpu),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(artifact, destination)
    return {"problem_id": problem_id, "rows": len(rows), "branches": len(points)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    output_root = ROOT / config["output_root"]
    source_root = ROOT / config["datasets"][args.dataset]["source_root"] / "heldout"
    paths = sorted(source_root.glob("sample_*.pt"))
    assigned = [path for index, path in enumerate(paths) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        assigned = assigned[: args.limit]
    fingerprint = canonical_fingerprint(config, args.dataset)

    seed_everything(int(config["seed"]))
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    free, total = torch.cuda.mem_get_info(device)
    print(json.dumps({
        "status": "loading",
        "worker": args.worker_id,
        "dataset": args.dataset,
        "gpu": args.gpu,
        "free_GiB": free / 2**30,
        "total_GiB": total / 2**30,
        "assigned": len(assigned),
    }), flush=True)
    model, tokenizer, audit = load_qwen3(
        ROOT / config["model"]["local_path"],
        device,
        config["model"]["dtype"],
        config["model"]["attention_backend"],
    )
    completed = skipped = failures = 0
    started = time.time()
    for source_path in assigned:
        problem_id = source_path.stem.removeprefix("sample_")
        destination = output_root / args.dataset / "heldout" / source_path.name
        try:
            if args.resume and valid(destination, fingerprint, problem_id):
                skipped += 1
                continue
            result = collect_one(
                source_path, destination, config, args.dataset, fingerprint,
                model, tokenizer, device, args.gpu, args.worker_id,
            )
            completed += 1
            print(json.dumps({
                "status": "completed", "worker": args.worker_id,
                "completed": completed, "skipped": skipped, "failures": failures, **result,
            }), flush=True)
        except Exception as error:
            failures += 1
            print(json.dumps({
                "status": "error", "worker": args.worker_id, "source": str(source_path),
                "error_type": type(error).__name__, "error": str(error),
            }), flush=True)
            if isinstance(error, torch.cuda.OutOfMemoryError):
                raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    summary = {
        "status": "complete" if failures == 0 else "failed",
        "worker": args.worker_id,
        "dataset": args.dataset,
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "completed": completed,
        "skipped": skipped,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
        "protocol_fingerprint": fingerprint,
        "model_audit": audit,
    }
    path = output_root / "workers" / f"{args.worker_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Selectively extend capped Qwen3-4B GSM8K held-out rollouts from 4096 to 13000.

The source prefix and its dedicated sampling RNG stream are preserved exactly.
Uncapped held-out artifacts are never regenerated; summarize mode merges the 78
extensions with the 1241 immutable source results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from src.final_paper_inference import atomic_torch_save, prediction_for, success_for
from src.final_paper_replay_cache import task_seed
from run_final_paper_concurrency_worker_v1 import generate_untimed, warmup
from src.qwen3_reasoning import confidence_stats, load_qwen3, sample_token
from src.utils import atomic_json, load_yaml, seed_everything


PROTOCOL_ID = "qwen3_4b_gsm8k_selective13k_v1"
GLOBAL_SEED = 20260803
SOURCE_MAX = 4096
TARGET_MAX = 13000
EXPECTED_HELDOUT = 1319
EXPECTED_SELECTED = 78
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MODEL_PATH = ROOT / "models/Qwen3-4B"
SOURCE_ROOT = (
    ROOT
    / "results/final_paper_greedy_forced_v1/selected_common_cache/gsm8k/merged/heldout"
)
OUTPUT_ROOT = ROOT / "results/qwen3_4b_gsm8k_selective13k_v1"


def canonical_fingerprint(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


PROTOCOL = {
    "protocol_id": PROTOCOL_ID,
    "dataset": "gsm8k",
    "split": "heldout",
    "model": "Qwen/Qwen3-4B",
    "model_dtype": "float16",
    "attention_backend": "sdpa",
    "required_gpu_architecture": "NVIDIA A100 80GB PCIe (matching source rollout hardware)",
    "warmup": "run_final_paper_concurrency_worker_v1.warmup before collection",
    "global_seed": GLOBAL_SEED,
    "selection": "source dense.reached_max_tokens=true and len(tokens)=4096",
    "source_max_new_tokens": SOURCE_MAX,
    "target_max_new_tokens": TARGET_MAX,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "do_sample": True,
    "continuation": "tokenwise KV replay plus exact multinomial RNG fast-forward",
    "uncapped_policy": "retain immutable source artifact",
    "expected_heldout": EXPECTED_HELDOUT,
    "expected_selected": EXPECTED_SELECTED,
}
FINGERPRINT = canonical_fingerprint(PROTOCOL)


def sha256_tokens(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def source_artifacts() -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(SOURCE_ROOT.glob("sample_*.pt"))
    if len(paths) != EXPECTED_HELDOUT:
        raise RuntimeError(f"heldout source count {len(paths)} != {EXPECTED_HELDOUT}")
    values = [(path, torch.load(path, map_location="cpu", weights_only=False)) for path in paths]
    invalid = [str(path) for path, value in values if value.get("status") != "complete"]
    if invalid:
        raise RuntimeError(f"incomplete source artifacts: {invalid[:5]}")
    return values


def selected_artifacts() -> list[tuple[Path, dict[str, Any]]]:
    selected = []
    for path, value in source_artifacts():
        dense = value["dense"]
        if dense.get("reached_max_tokens"):
            if len(dense["tokens"]) != SOURCE_MAX:
                raise RuntimeError(f"capped source has unexpected length: {path}")
            selected.append((path, value))
    if len(selected) != EXPECTED_SELECTED:
        raise RuntimeError(f"selected count {len(selected)} != {EXPECTED_SELECTED}")
    return selected


def output_path(problem_id: str) -> Path:
    return OUTPUT_ROOT / "artifacts" / f"sample_{problem_id}.pt"


def complete_output(path: Path, problem_id: str, source_tokens: list[int]) -> bool:
    if not path.is_file():
        return False
    value = torch.load(path, map_location="cpu", weights_only=False)
    dense = value.get("dense", {})
    return (
        value.get("status") == "complete"
        and value.get("protocol_fingerprint") == FINGERPRINT
        and str(value.get("problem_id")) == problem_id
        and int(dense.get("target_max_new_tokens", -1)) == TARGET_MAX
        and dense.get("tokens", [])[:SOURCE_MAX] == source_tokens
        and dense.get("source_prefix_exact") is True
    )


def advance_sampling_generator(generator: torch.Generator, steps: int, device: torch.device) -> None:
    probabilities = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)
    for _ in range(steps):
        torch.multinomial(probabilities, 1, generator=generator)


def replay_prefix(model, input_ids: torch.Tensor, source_tokens: list[int]):
    output = model.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
        return_dict=True,
    )
    past = output.past_key_values
    del output
    prompt_length = int(input_ids.shape[1])
    last_hidden = None
    for index, token in enumerate(source_tokens, start=1):
        current = torch.tensor([[token]], dtype=torch.long, device=input_ids.device)
        mask = torch.ones((1, prompt_length + index), dtype=torch.long, device=input_ids.device)
        output = model.model(
            input_ids=current,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1:, :]
        del output
    if last_hidden is None:
        raise RuntimeError("empty source prefix")
    return past, last_hidden


def extend_dense(model, tokenizer, input_ids: torch.Tensor, source: dict[str, Any], seed: int):
    source_tokens = [int(value) for value in source["tokens"]]
    logps = [float(value) for value in source["logps"]]
    margins = [float(value) for value in source["margins"]]
    entropies = [float(value) for value in source["entropies_top20"]]
    if not (len(source_tokens) == len(logps) == len(margins) == len(entropies) == SOURCE_MAX):
        raise RuntimeError("source token/statistic lengths are inconsistent")
    device = input_ids.device
    generator = torch.Generator(device=device).manual_seed(seed)
    advance_sampling_generator(generator, SOURCE_MAX, device)
    torch.cuda.synchronize(device)
    replay_started = time.perf_counter()
    past, last_hidden = replay_prefix(model, input_ids, source_tokens)
    logits = model.lm_head(last_hidden)
    del last_hidden
    torch.cuda.synchronize(device)
    replay_wall_ms = 1000.0 * (time.perf_counter() - replay_started)

    tokens = list(source_tokens)
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    continuation_started = time.perf_counter()
    while len(tokens) < TARGET_MAX:
        logp, margin, entropy = confidence_stats(logits)
        token = sample_token(logits, generator, TEMPERATURE, TOP_K, TOP_P)
        tokens.append(token)
        logps.append(logp)
        margins.append(margin)
        entropies.append(entropy)
        if token in eos or len(tokens) >= TARGET_MAX:
            break
        current = torch.tensor([[token]], dtype=torch.long, device=device)
        mask = torch.ones(
            (1, int(input_ids.shape[1]) + len(tokens)), dtype=torch.long, device=device
        )
        output = model.model(
            input_ids=current,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        logits = model.lm_head(output.last_hidden_state[:, -1:, :])
        del output
    torch.cuda.synchronize(device)
    continuation_wall_ms = 1000.0 * (time.perf_counter() - continuation_started)
    if not (len(tokens) == len(logps) == len(margins) == len(entropies)):
        raise RuntimeError("extended token/statistic lengths are inconsistent")
    return {
        "tokens": tokens,
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "logps": logps,
        "margins": margins,
        "entropies_top20": entropies,
        "reasoning_tokens": len(tokens),
        "reached_max_tokens": len(tokens) >= TARGET_MAX,
        "target_max_new_tokens": TARGET_MAX,
        "source_prefix_tokens": SOURCE_MAX,
        "source_prefix_exact": tokens[:SOURCE_MAX] == source_tokens,
        "source_prefix_sha256": sha256_tokens(source_tokens),
        "extended_sha256": sha256_tokens(tokens),
        "incremental_resume": {
            "mode": "tokenwise_kv_replay_without_source_lm_head_or_sampling",
            "rng_fast_forward_multinomial_calls": SOURCE_MAX,
            "replay_wall_ms": replay_wall_ms,
            "continuation_wall_ms": continuation_wall_ms,
        },
    }


def collect(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    selected = selected_artifacts()
    visible = [item for index, item in enumerate(selected) if index % args.num_shards == args.shard_index]
    if args.max_items is not None:
        visible = visible[: args.max_items]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json({"status": "frozen", "protocol": PROTOCOL, "fingerprint": FINGERPRINT}, OUTPUT_ROOT / "PROTOCOL.json")
    seed_everything(GLOBAL_SEED)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model, tokenizer, model_audit = load_qwen3(MODEL_PATH, device, "float16", "sdpa")
    warmup(
        model,
        tokenizer,
        device,
        load_yaml(ROOT / "configs/final_paper_replay_v2_gsm8k_fp16.yaml"),
        "selective13k_validation",
    )
    completed = skipped = 0
    for source_path, source_artifact in visible:
        problem_id = str(source_artifact["problem_id"])
        source_dense = source_artifact["dense"]
        destination = output_path(problem_id)
        if args.resume and complete_output(destination, problem_id, source_dense["tokens"]):
            skipped += 1
            continue
        if destination.exists():
            raise RuntimeError(f"refusing to overwrite incompatible artifact: {destination}")
        encoded = tokenizer(source_artifact["prompt_text"], return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        if int(input_ids.shape[1]) != int(source_artifact["prompt_tokens"]):
            raise RuntimeError(f"prompt token mismatch for {problem_id}")
        dense_seed = task_seed(GLOBAL_SEED, "gsm8k", "heldout", problem_id, "dense")
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            dense = extend_dense(model, tokenizer, input_ids, source_dense, dense_seed)
        dense["prediction"] = prediction_for("gsm8k", dense["text"])
        dense["success"] = success_for("gsm8k", source_artifact["gold_answer"], dense["prediction"])
        dense["peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated(device) / (1024**2)
        artifact = {
            "schema_version": 1,
            "status": "complete",
            "protocol_id": PROTOCOL_ID,
            "protocol_fingerprint": FINGERPRINT,
            "problem_id": problem_id,
            "dataset": "gsm8k",
            "split": "heldout",
            "seed": GLOBAL_SEED,
            "dense_task_seed": dense_seed,
            "gold_answer": source_artifact["gold_answer"],
            "record": source_artifact["record"],
            "prompt_text": source_artifact["prompt_text"],
            "prompt_tokens": int(input_ids.shape[1]),
            "model_audit": model_audit,
            "source_artifact": str(source_path.resolve()),
            "source_protocol_id": source_artifact.get("protocol_id"),
            "source_protocol_fingerprint": source_artifact.get("protocol_fingerprint"),
            "source_dense": {
                "reasoning_tokens": len(source_dense["tokens"]),
                "prediction": source_dense["prediction"],
                "success": bool(source_dense["success"]),
                "reached_max_tokens": bool(source_dense["reached_max_tokens"]),
                "token_sha256": sha256_tokens(source_dense["tokens"]),
            },
            "dense": dense,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_torch_save(artifact, destination)
        completed += 1
        print(json.dumps({
            "problem_id": problem_id,
            "completed": completed,
            "tokens": dense["reasoning_tokens"],
            "success": dense["success"],
            "reached_max": dense["reached_max_tokens"],
            "peak_mib": round(dense["peak_cuda_memory_mib"], 1),
            "replay_s": round(dense["incremental_resume"]["replay_wall_ms"] / 1000, 1),
            "continuation_s": round(dense["incremental_resume"]["continuation_wall_ms"] / 1000, 1),
        }), flush=True)
    summary = {
        "status": "complete",
        "phase": "collect",
        "protocol_fingerprint": FINGERPRINT,
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "visible": len(visible),
        "completed_now": completed,
        "skipped": skipped,
    }
    atomic_json(summary, OUTPUT_ROOT / f"worker_shard{args.shard_index}.json")
    print(json.dumps(summary), flush=True)


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize() -> None:
    sources = source_artifacts()
    selected_ids = {
        str(value["problem_id"])
        for _path, value in sources
        if value["dense"].get("reached_max_tokens")
    }
    rows = []
    missing = []
    prefix_failures = []
    transitions = {"C->C": 0, "C->W": 0, "W->C": 0, "W->W": 0}
    for source_path, source in sources:
        problem_id = str(source["problem_id"])
        old_dense = source["dense"]
        if problem_id in selected_ids:
            path = output_path(problem_id)
            if not complete_output(path, problem_id, old_dense["tokens"]):
                missing.append(problem_id)
                continue
            extension = torch.load(path, map_location="cpu", weights_only=False)
            dense = extension["dense"]
            transition = ("C" if old_dense["success"] else "W") + "->" + ("C" if dense["success"] else "W")
            transitions[transition] += 1
            if dense["tokens"][:SOURCE_MAX] != old_dense["tokens"]:
                prefix_failures.append(problem_id)
            source_kind = "selective_13k_extension"
        else:
            dense = old_dense
            source_kind = "immutable_uncapped_4096_source"
        rows.append({
            "problem_id": problem_id,
            "success": bool(dense["success"]),
            "prediction": dense["prediction"],
            "gold": source["gold_answer"],
            "reasoning_tokens": len(dense["tokens"]),
            "reached_active_cap": bool(dense.get("reached_max_tokens")),
            "source_kind": source_kind,
            "source_artifact": str(source_path.resolve()),
        })
    if missing or prefix_failures or len(rows) != EXPECTED_HELDOUT:
        raise RuntimeError(
            f"audit failed missing={missing[:5]} prefix={prefix_failures[:5]} rows={len(rows)}"
        )
    lengths = [row["reasoning_tokens"] for row in rows]
    correct = sum(int(row["success"]) for row in rows)
    source_correct = sum(int(value["dense"]["success"]) for _path, value in sources)
    source_lengths = [len(value["dense"]["tokens"]) for _path, value in sources]
    new_cap = sum(
        int(row["source_kind"] == "selective_13k_extension" and row["reasoning_tokens"] >= TARGET_MAX)
        for row in rows
    )
    result = {
        "status": "complete",
        "protocol": PROTOCOL,
        "protocol_fingerprint": FINGERPRINT,
        "audit": {
            "heldout_count": len(rows),
            "selected_count": len(selected_ids),
            "complete_extensions": len(selected_ids) - len(missing),
            "exact_source_prefixes": len(selected_ids) - len(prefix_failures),
            "uncapped_source_artifacts_retained": EXPECTED_HELDOUT - len(selected_ids),
        },
        "source_4096": {
            "correct": source_correct,
            "accuracy": source_correct / EXPECTED_HELDOUT,
            "mean_tokens": statistics.fmean(source_lengths),
            "median_tokens": statistics.median(source_lengths),
            "reached_cap": len(selected_ids),
        },
        "selective_13000": {
            "correct": correct,
            "accuracy": correct / EXPECTED_HELDOUT,
            "accuracy_delta": (correct - source_correct) / EXPECTED_HELDOUT,
            "mean_tokens": statistics.fmean(lengths),
            "median_tokens": statistics.median(lengths),
            "p95_tokens": percentile(lengths, 0.95),
            "max_tokens": max(lengths),
            "reached_13000_cap": new_cap,
            "transitions_among_78": transitions,
        },
        "rows": rows,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(result, OUTPUT_ROOT / "DENSE_TEST_RESULTS.json")
    audit = {
        "status": "complete",
        "protocol_fingerprint": FINGERPRINT,
        **result["audit"],
        "result_rows": len(rows),
    }
    atomic_json(audit, OUTPUT_ROOT / "COMPLETION_AUDIT.json")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


def validate(args: argparse.Namespace) -> None:
    source_path, source = selected_artifacts()[0]
    problem_id = str(source["problem_id"])
    extension_path = output_path(problem_id)
    if not complete_output(extension_path, problem_id, source["dense"]["tokens"]):
        raise RuntimeError(f"missing valid smoke extension: {extension_path}")
    extension = torch.load(extension_path, map_location="cpu", weights_only=False)
    expected_tokens = [int(value) for value in extension["dense"]["tokens"]]
    seed_everything(GLOBAL_SEED)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model, tokenizer, model_audit = load_qwen3(MODEL_PATH, device, "float16", "sdpa")
    warmup(
        model,
        tokenizer,
        device,
        load_yaml(ROOT / "configs/final_paper_replay_v2_gsm8k_fp16.yaml"),
        "selective13k_validation",
    )
    encoded = tokenizer(source["prompt_text"], return_tensors="pt")
    dense_seed = task_seed(GLOBAL_SEED, "gsm8k", "heldout", problem_id, "dense")
    with torch.inference_mode():
        trace = generate_untimed(
            model,
            tokenizer,
            encoded.input_ids.to(device),
            encoded.attention_mask.to(device),
            {
                "max_new_tokens": len(expected_tokens),
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
            },
            dense_seed,
        )
    first_mismatch = next(
        (index for index, pair in enumerate(zip(trace.tokens, expected_tokens)) if pair[0] != pair[1]),
        None,
    )
    exact = trace.tokens == expected_tokens
    result = {
        "status": "complete" if exact else "failed",
        "protocol_fingerprint": FINGERPRINT,
        "problem_id": problem_id,
        "source_artifact": str(source_path.resolve()),
        "extension_artifact": str(extension_path.resolve()),
        "tokens_compared": len(expected_tokens),
        "exact_full_generation_match": exact,
        "first_mismatch_zero_based": first_mismatch,
        "reference_sha256": sha256_tokens(trace.tokens),
        "extension_sha256": sha256_tokens(expected_tokens),
        "reference_wall_ms": trace.collection_wall_ms,
        "model_audit": model_audit,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(result, OUTPUT_ROOT / "INCREMENTAL_EQUIVALENCE_VALIDATION.json")
    print(json.dumps(result, indent=2), flush=True)
    if not exact:
        raise RuntimeError(f"incremental continuation mismatch at token {first_mismatch}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("collect", "validate", "summarize"), default="collect")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "collect":
        collect(args)
    elif args.mode == "validate":
        validate(args)
    else:
        summarize()


if __name__ == "__main__":
    main()

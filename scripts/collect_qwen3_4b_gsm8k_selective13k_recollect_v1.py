#!/usr/bin/env python3
"""Recollect only the 78 capped Qwen3-4B GSM8K held-out samples at 13K."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from run_final_paper_concurrency_worker_v1 import generate_untimed, warmup
from src.final_paper_inference import atomic_torch_save, prediction_for, success_for
from src.final_paper_replay_cache import task_seed
from src.qwen3_reasoning import load_qwen3
from src.utils import atomic_json, load_yaml, seed_everything


PROTOCOL_ID = "qwen3_4b_gsm8k_selective13k_recollect_v1"
GLOBAL_SEED = 20260803
SOURCE_MAX = 4096
TARGET_MAX = 13000
EXPECTED_HELDOUT = 1319
EXPECTED_SELECTED = 78
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MODEL_PATH = ROOT / "models/Qwen3-4B"
CONFIG_PATH = ROOT / "configs/final_paper_replay_v2_gsm8k_fp16.yaml"
SOURCE_ROOT = ROOT / "results/final_paper_greedy_forced_v1/selected_common_cache/gsm8k/merged/heldout"
OUTPUT_ROOT = ROOT / "results/qwen3_4b_gsm8k_selective13k_recollect_v1"


PROTOCOL = {
    "protocol_id": PROTOCOL_ID,
    "dataset": "gsm8k",
    "split": "heldout",
    "selection": "only old dense.reached_max_tokens=true and len(tokens)=4096",
    "selected_count": EXPECTED_SELECTED,
    "uncapped_policy": "retain 1241 immutable old results",
    "selected_policy": "recollect from prompt; do not resume old KV or RNG state",
    "comparability_note": "selected trajectories may differ before token 4096, especially across GPU architectures",
    "model": "Qwen/Qwen3-4B",
    "dtype": "float16",
    "attention_backend": "sdpa",
    "global_seed": GLOBAL_SEED,
    "max_new_tokens": TARGET_MAX,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "do_sample": True,
    "generator": "original run_final_paper_concurrency_worker_v1.generate_untimed",
    "warmup": "original worker warmup before collection",
}


def fingerprint(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


PROTOCOL_FINGERPRINT = fingerprint(PROTOCOL)


def token_sha(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def load_sources() -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(SOURCE_ROOT.glob("sample_*.pt"))
    if len(paths) != EXPECTED_HELDOUT:
        raise RuntimeError(f"source count {len(paths)} != {EXPECTED_HELDOUT}")
    result = []
    selected = 0
    for path in paths:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("status") != "complete":
            raise RuntimeError(f"incomplete source: {path}")
        result.append((path, value))
        if value["dense"].get("reached_max_tokens"):
            if len(value["dense"]["tokens"]) != SOURCE_MAX:
                raise RuntimeError(f"unexpected capped length: {path}")
            selected += 1
    if selected != EXPECTED_SELECTED:
        raise RuntimeError(f"selected count {selected} != {EXPECTED_SELECTED}")
    return result


def selected_sources() -> list[tuple[Path, dict[str, Any]]]:
    return [item for item in load_sources() if item[1]["dense"].get("reached_max_tokens")]


def artifact_path(problem_id: str) -> Path:
    return OUTPUT_ROOT / "artifacts" / f"sample_{problem_id}.pt"


def valid_artifact(path: Path, problem_id: str) -> bool:
    if not path.is_file():
        return False
    value = torch.load(path, map_location="cpu", weights_only=False)
    return (
        value.get("status") == "complete"
        and value.get("protocol_fingerprint") == PROTOCOL_FINGERPRINT
        and str(value.get("problem_id")) == problem_id
        and int(value.get("dense", {}).get("max_new_tokens", -1)) == TARGET_MAX
    )


def collect(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    selected = selected_sources()
    if args.problem_ids_file is not None:
        selection_path = args.problem_ids_file.resolve()
        requested = [str(value) for value in json.loads(selection_path.read_text(encoding="utf-8"))]
        if len(requested) != len(set(requested)):
            raise ValueError(f"duplicate IDs in {selection_path}")
        by_id = {str(item[1]["problem_id"]): item for item in selected}
        missing = [problem_id for problem_id in requested if problem_id not in by_id]
        if missing:
            raise KeyError(f"requested IDs are not selected capped samples: {missing}")
        visible = [by_id[problem_id] for problem_id in requested]
    else:
        visible = [item for index, item in enumerate(selected) if index % args.num_shards == args.shard_index]
    if args.max_items is not None:
        visible = visible[: args.max_items]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(
        {"status": "frozen", "protocol": PROTOCOL, "fingerprint": PROTOCOL_FINGERPRINT},
        OUTPUT_ROOT / "PROTOCOL.json",
    )
    seed_everything(GLOBAL_SEED)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    gpu_name = torch.cuda.get_device_name(device)
    model, tokenizer, model_audit = load_qwen3(MODEL_PATH, device, "float16", "sdpa")
    warmup(model, tokenizer, device, load_yaml(CONFIG_PATH), f"recollect_{args.worker_id}")
    generation = {
        "max_new_tokens": TARGET_MAX,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
    }
    completed = skipped = 0
    for source_path, source in visible:
        problem_id = str(source["problem_id"])
        destination = artifact_path(problem_id)
        if args.resume and valid_artifact(destination, problem_id):
            skipped += 1
            continue
        if destination.exists():
            raise RuntimeError(f"refusing to overwrite incompatible artifact: {destination}")
        encoded = tokenizer(source["prompt_text"], return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        if int(input_ids.shape[1]) != int(source["prompt_tokens"]):
            raise RuntimeError(f"prompt token mismatch: {problem_id}")
        dense_seed = task_seed(GLOBAL_SEED, "gsm8k", "heldout", problem_id, "dense")
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            trace = generate_untimed(
                model, tokenizer, input_ids, attention_mask, generation, dense_seed
            )
        tokens = [int(value) for value in trace.tokens]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        prediction = prediction_for("gsm8k", text)
        success = success_for("gsm8k", source["gold_answer"], prediction)
        prefix_match = common_prefix_length(tokens, source["dense"]["tokens"])
        dense = {
            "tokens": tokens,
            "text": text,
            "prediction": prediction,
            "success": bool(success),
            "reasoning_tokens": len(tokens),
            "reached_max_tokens": len(tokens) >= TARGET_MAX,
            "max_new_tokens": TARGET_MAX,
            "logps": trace.logps,
            "margins": trace.margins,
            "entropies_top20": trace.entropies,
            "collection_wall_ms": trace.collection_wall_ms,
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
            "token_sha256": token_sha(tokens),
            "old_4096_common_prefix_tokens": prefix_match,
        }
        artifact = {
            "schema_version": 1,
            "status": "complete",
            "protocol_id": PROTOCOL_ID,
            "protocol_fingerprint": PROTOCOL_FINGERPRINT,
            "dataset": "gsm8k",
            "split": "heldout",
            "problem_id": problem_id,
            "seed": GLOBAL_SEED,
            "dense_task_seed": dense_seed,
            "gpu_index": args.gpu,
            "gpu_name": gpu_name,
            "worker_id": args.worker_id,
            "model_audit": model_audit,
            "source_artifact": str(source_path.resolve()),
            "source_protocol_id": source.get("protocol_id"),
            "source_protocol_fingerprint": source.get("protocol_fingerprint"),
            "record": source["record"],
            "gold_answer": source["gold_answer"],
            "prompt_text": source["prompt_text"],
            "prompt_tokens": int(input_ids.shape[1]),
            "old_dense": {
                "tokens": len(source["dense"]["tokens"]),
                "prediction": source["dense"]["prediction"],
                "success": bool(source["dense"]["success"]),
                "token_sha256": token_sha(source["dense"]["tokens"]),
            },
            "dense": dense,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_torch_save(artifact, destination)
        completed += 1
        print(json.dumps({
            "problem_id": problem_id,
            "completed": completed,
            "tokens": len(tokens),
            "success": bool(success),
            "reached_max": len(tokens) >= TARGET_MAX,
            "old_prefix_match": prefix_match,
            "wall_s": round(trace.collection_wall_ms / 1000, 1),
            "peak_mib": round(dense["peak_cuda_memory_mib"], 1),
        }), flush=True)
    worker = {
        "status": "complete",
        "phase": "collect",
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "worker_id": args.worker_id,
        "gpu": args.gpu,
        "gpu_name": gpu_name,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "visible": len(visible),
        "completed_now": completed,
        "skipped": skipped,
    }
    atomic_json(worker, OUTPUT_ROOT / "workers" / f"{args.worker_id}.json")
    print(json.dumps(worker), flush=True)


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize() -> None:
    sources = load_sources()
    selected_ids = {
        str(value["problem_id"])
        for _path, value in sources
        if value["dense"].get("reached_max_tokens")
    }
    rows = []
    transitions = {"C->C": 0, "C->W": 0, "W->C": 0, "W->W": 0}
    prefix_matches = []
    gpu_counts: dict[str, int] = {}
    missing = []
    for source_path, source in sources:
        problem_id = str(source["problem_id"])
        old = source["dense"]
        if problem_id in selected_ids:
            path = artifact_path(problem_id)
            if not valid_artifact(path, problem_id):
                missing.append(problem_id)
                continue
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            dense = artifact["dense"]
            transition = ("C" if old["success"] else "W") + "->" + ("C" if dense["success"] else "W")
            transitions[transition] += 1
            prefix_matches.append(int(dense["old_4096_common_prefix_tokens"]))
            gpu_counts[artifact["gpu_name"]] = gpu_counts.get(artifact["gpu_name"], 0) + 1
            kind = "selective_13k_recollection"
        else:
            dense = old
            kind = "immutable_uncapped_4096_source"
        rows.append({
            "problem_id": problem_id,
            "success": bool(dense["success"]),
            "prediction": dense["prediction"],
            "gold": source["gold_answer"],
            "reasoning_tokens": len(dense["tokens"]),
            "source_kind": kind,
            "source_artifact": str(source_path.resolve()),
        })
    if missing or len(rows) != EXPECTED_HELDOUT:
        raise RuntimeError(f"incomplete: missing={missing[:10]} rows={len(rows)}")
    old_correct = sum(int(value["dense"]["success"]) for _path, value in sources)
    old_lengths = [len(value["dense"]["tokens"]) for _path, value in sources]
    correct = sum(int(row["success"]) for row in rows)
    lengths = [row["reasoning_tokens"] for row in rows]
    reached_13k = sum(
        int(row["source_kind"] == "selective_13k_recollection" and row["reasoning_tokens"] >= TARGET_MAX)
        for row in rows
    )
    result = {
        "status": "complete",
        "protocol": PROTOCOL,
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "audit": {
            "heldout_count": len(rows),
            "selected_recollected": len(selected_ids),
            "uncapped_retained": EXPECTED_HELDOUT - len(selected_ids),
            "gpu_counts": gpu_counts,
            "warning": PROTOCOL["comparability_note"],
        },
        "old_4096": {
            "correct": old_correct,
            "accuracy": old_correct / EXPECTED_HELDOUT,
            "mean_tokens": statistics.fmean(old_lengths),
            "median_tokens": statistics.median(old_lengths),
            "reached_cap": len(selected_ids),
        },
        "selective_recollect_13000": {
            "correct": correct,
            "accuracy": correct / EXPECTED_HELDOUT,
            "accuracy_delta": (correct - old_correct) / EXPECTED_HELDOUT,
            "mean_tokens": statistics.fmean(lengths),
            "median_tokens": statistics.median(lengths),
            "p95_tokens": percentile(lengths, 0.95),
            "max_tokens": max(lengths),
            "reached_13000_cap": reached_13k,
            "transitions_among_78": transitions,
            "old_prefix_match_median": statistics.median(prefix_matches),
            "old_prefix_match_min": min(prefix_matches),
            "old_prefix_match_max": max(prefix_matches),
        },
        "rows": rows,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(result, OUTPUT_ROOT / "DENSE_TEST_RESULTS.json")
    atomic_json(
        {
            "status": "complete",
            "protocol_fingerprint": PROTOCOL_FINGERPRINT,
            "heldout_count": len(rows),
            "selected_recollected": len(selected_ids),
            "result_rows": len(rows),
        },
        OUTPUT_ROOT / "COMPLETION_AUDIT.json",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("collect", "summarize"), default="collect")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--worker-id", default="worker")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--problem-ids-file", type=Path)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    collect(args) if args.mode == "collect" else summarize()


if __name__ == "__main__":
    main()

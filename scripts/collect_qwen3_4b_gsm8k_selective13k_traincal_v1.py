#!/usr/bin/env python3
"""Selectively recollect old 4096-capped GSM8K train/cal Dense traces at 13K."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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


PROTOCOL_ID = "qwen3_4b_gsm8k_selective13k_method_v1"
GLOBAL_SEED = 20260803
SOURCE_MAX = 4096
TARGET_MAX = 13000
EXPECTED = {"probe_train": 53, "calibration": 26, "heldout": 78}
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MODEL_PATH = ROOT / "models/Qwen3-4B"
WARMUP_CONFIG = ROOT / "configs/final_paper_replay_v2_gsm8k_fp16.yaml"
SOURCE_ROOT = ROOT / "results/final_paper_greedy_forced_v1/selected_common_cache/gsm8k/merged"
PRIOR_HELDOUT_ROOT = ROOT / "results/qwen3_4b_gsm8k_selective13k_recollect_v1/artifacts"
OUTPUT_ROOT = ROOT / "results/qwen3_4b_gsm8k_selective13k_method_v1"
SELECTED_ROOT = OUTPUT_ROOT / "dense_selected"


PROTOCOL = {
    "protocol_id": PROTOCOL_ID,
    "dataset": "gsm8k",
    "splits": ["probe_train", "calibration", "heldout"],
    "selection": "only old dense.reached_max_tokens=true and len(tokens)=4096",
    "selected_counts": EXPECTED,
    "uncapped_policy": "retain old 4096-run artifacts unchanged",
    "selected_policy": "recollect from prompt at 13K; do not resume old KV or RNG state",
    "heldout_policy": "reuse the already audited selective-13K Dense artifacts exactly",
    "model": "Qwen/Qwen3-4B",
    "dtype": "float16",
    "attention_backend": "sdpa",
    "global_seed": GLOBAL_SEED,
    "max_new_tokens": TARGET_MAX,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "do_sample": True,
    "forced_answer": {"strategy": "greedy_argmax", "max_new_tokens": 16},
    "checkpoint_schedule": "paragraph_full_range",
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


def selected_sources(split: str) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted((SOURCE_ROOT / split).glob("sample_*.pt"))
    result = []
    for path in paths:
        value = torch.load(path, map_location="cpu", weights_only=False)
        dense = value["dense"]
        if dense.get("reached_max_tokens"):
            if len(dense["tokens"]) != SOURCE_MAX:
                raise RuntimeError(f"unexpected capped length: {path}")
            result.append((path, value))
    if len(result) != EXPECTED[split]:
        raise RuntimeError(f"{split}: selected {len(result)} != {EXPECTED[split]}")
    return result


def destination(split: str, problem_id: str) -> Path:
    return SELECTED_ROOT / split / f"sample_{problem_id}.pt"


def valid(path: Path, problem_id: str, split: str) -> bool:
    if not path.is_file():
        return False
    value = torch.load(path, map_location="cpu", weights_only=False)
    return (
        value.get("status") == "complete"
        and value.get("protocol_fingerprint") == PROTOCOL_FINGERPRINT
        and str(value.get("problem_id")) == problem_id
        and value.get("split") == split
        and int(value.get("dense", {}).get("max_new_tokens", -1)) == TARGET_MAX
        and "content_tokens" in value.get("dense", {})
    )


def normalized_dense(dense: dict[str, Any], tokenizer) -> dict[str, Any]:
    tokens = [int(value) for value in dense["tokens"]]
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    content = tokens[:-1] if tokens and tokens[-1] in eos else tokens
    result = dict(dense)
    result.update({
        "tokens": tokens,
        "content_tokens": content,
        "reasoning_tokens": len(tokens),
        "max_new_tokens": TARGET_MAX,
        "reached_max_tokens": len(tokens) >= TARGET_MAX,
        "prefill_cuda_ms": dense.get("prefill_cuda_ms"),
        "decode_cuda_ms": dense.get("decode_cuda_ms", []),
        "wall_ms": dense.get("wall_ms"),
        "throughput_collection_wall_ms": dense.get(
            "throughput_collection_wall_ms", dense.get("collection_wall_ms")
        ),
        "token_sha256": dense.get("token_sha256", token_sha(tokens)),
    })
    return result


def prepare_heldout_tokenizer_only() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    selected = selected_sources("heldout")
    prior = {path.stem.removeprefix("sample_"): path for path in PRIOR_HELDOUT_ROOT.glob("sample_*.pt")}
    expected_ids = {str(value["problem_id"]) for _path, value in selected}
    if set(prior) != expected_ids:
        raise RuntimeError("prior heldout IDs do not exactly match the 78 old-capped IDs")
    written = skipped = 0
    for old_path, old in selected:
        problem_id = str(old["problem_id"])
        target = destination("heldout", problem_id)
        if valid(target, problem_id, "heldout"):
            skipped += 1
            continue
        if target.exists():
            raise RuntimeError(f"refusing to overwrite incompatible artifact: {target}")
        previous = torch.load(prior[problem_id], map_location="cpu", weights_only=False)
        dense = normalized_dense(previous["dense"], tokenizer)
        artifact = {
            **previous,
            "schema_version": 2,
            "status": "complete",
            "protocol_id": PROTOCOL_ID,
            "protocol_fingerprint": PROTOCOL_FINGERPRINT,
            "split": "heldout",
            "source_artifact": str(old_path.resolve()),
            "prior_13k_artifact": str(prior[problem_id].resolve()),
            "source_kind": "reused_audited_selective13k_dense",
            "dense": dense,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_torch_save(artifact, target)
        written += 1
    print(json.dumps({"status": "complete", "split": "heldout", "written": written, "skipped": skipped}))


def collect(args: argparse.Namespace) -> None:
    if args.split not in ("probe_train", "calibration"):
        raise ValueError("Dense collection is only needed for probe_train/calibration")
    selected = selected_sources(args.split)
    visible = [item for index, item in enumerate(selected) if index % args.num_shards == args.shard_index]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json({"status": "frozen", "protocol": PROTOCOL, "fingerprint": PROTOCOL_FINGERPRINT}, OUTPUT_ROOT / "PROTOCOL.json")
    seed_everything(GLOBAL_SEED)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    gpu_name = torch.cuda.get_device_name(device)
    model, tokenizer, model_audit = load_qwen3(MODEL_PATH, device, "float16", "sdpa")
    warmup(model, tokenizer, device, load_yaml(WARMUP_CONFIG), f"selective13k_{args.worker_id}")
    generation = {"max_new_tokens": TARGET_MAX, "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K}
    completed = skipped = 0
    for source_path, source in visible:
        problem_id = str(source["problem_id"])
        target = destination(args.split, problem_id)
        if args.resume and valid(target, problem_id, args.split):
            skipped += 1
            continue
        if target.exists():
            raise RuntimeError(f"refusing to overwrite incompatible artifact: {target}")
        encoded = tokenizer(source["prompt_text"], return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        if int(input_ids.shape[1]) != int(source["prompt_tokens"]):
            raise RuntimeError(f"prompt token mismatch: {problem_id}")
        dense_seed = task_seed(GLOBAL_SEED, "gsm8k", args.split, problem_id, "dense")
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            trace = generate_untimed(model, tokenizer, input_ids, attention_mask, generation, dense_seed)
        tokens = [int(value) for value in trace.tokens]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        prediction = prediction_for("gsm8k", text)
        success = success_for("gsm8k", source["gold_answer"], prediction)
        dense = normalized_dense({
            "tokens": tokens,
            "text": text,
            "prediction": prediction,
            "success": bool(success),
            "logps": trace.logps,
            "margins": trace.margins,
            "entropies_top20": trace.entropies,
            "collection_wall_ms": trace.collection_wall_ms,
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
            "old_4096_common_prefix_tokens": common_prefix_length(tokens, source["dense"]["tokens"]),
        }, tokenizer)
        artifact = {
            "schema_version": 2,
            "status": "complete",
            "protocol_id": PROTOCOL_ID,
            "protocol_fingerprint": PROTOCOL_FINGERPRINT,
            "dataset": "gsm8k",
            "split": args.split,
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
            "source_kind": "selective13k_recollection_from_prompt",
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
        atomic_torch_save(artifact, target)
        completed += 1
        print(json.dumps({
            "problem_id": problem_id,
            "split": args.split,
            "completed": completed,
            "tokens": len(tokens),
            "success": bool(success),
            "reached_max": len(tokens) >= TARGET_MAX,
            "wall_s": round(trace.collection_wall_ms / 1000, 1),
            "peak_mib": round(dense["peak_cuda_memory_mib"], 1),
        }), flush=True)
    summary = {
        "status": "complete",
        "worker_id": args.worker_id,
        "split": args.split,
        "gpu": args.gpu,
        "visible": len(visible),
        "completed_now": completed,
        "skipped": skipped,
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
    }
    atomic_json(summary, OUTPUT_ROOT / "workers" / f"{args.worker_id}.json")
    print(json.dumps(summary), flush=True)


def audit_dense() -> None:
    report: dict[str, Any] = {"status": "complete", "splits": {}, "errors": []}
    for split, expected in EXPECTED.items():
        paths = sorted((SELECTED_ROOT / split).glob("sample_*.pt"))
        old_ids = {str(value["problem_id"]) for _path, value in selected_sources(split)}
        actual_ids = set()
        lengths = []
        for path in paths:
            value = torch.load(path, map_location="cpu", weights_only=False)
            problem_id = str(value.get("problem_id"))
            actual_ids.add(problem_id)
            if not valid(path, problem_id, split):
                report["errors"].append(f"invalid artifact: {path}")
            dense = value["dense"]
            lengths.append(len(dense["tokens"]))
            if len(dense["content_tokens"]) not in (len(dense["tokens"]), len(dense["tokens"]) - 1):
                report["errors"].append(f"content-token mismatch: {path}")
        if actual_ids != old_ids or len(paths) != expected:
            report["errors"].append(f"{split}: expected exact {expected} selected IDs, got {len(paths)}")
        report["splits"][split] = {
            "files": len(paths),
            "min_tokens": min(lengths) if lengths else None,
            "max_tokens": max(lengths) if lengths else None,
            "mean_tokens": sum(lengths) / len(lengths) if lengths else None,
            "reached_13k": sum(value >= TARGET_MAX for value in lengths),
        }
    if report["errors"]:
        report["status"] = "failed"
    atomic_json(report, OUTPUT_ROOT / "DENSE_SELECTED_AUDIT.json")
    print(json.dumps(report, indent=2))
    if report["status"] != "complete":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("collect", "prepare-heldout", "audit"), default="collect")
    parser.add_argument("--split", choices=("probe_train", "calibration"), default="probe_train")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--worker-id", default="worker")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "collect":
        collect(args)
    elif args.mode == "prepare-heldout":
        prepare_heldout_tokenizer_only()
    else:
        audit_dense()


if __name__ == "__main__":
    main()

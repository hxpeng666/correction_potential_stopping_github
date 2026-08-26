#!/usr/bin/env python3
"""Collect all checkpoint forced answers for one sample with greedy argmax decoding."""
from __future__ import annotations

import argparse
import gc
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from transformers.cache_utils import DynamicCache

from greedy_forced_common_v1 import (
    artifact_valid,
    atomic_json,
    claim_task,
    finish_task,
    load_config,
    output_path,
    protocol_fingerprint,
    queue_counts,
    resolve,
)
from src.final_paper_inference import atomic_torch_save, prediction_for, success_for
from src.mmlu_pro_protocol import parse_answer as parse_mmlu_pro_answer
from src.qwen3_reasoning import load_qwen3
from src.utils import seed_everything


def greedy_branch_from_legacy_cache(
    model,
    tokenizer,
    legacy_cache,
    *,
    prefix_context: int,
    suffix_ids: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, Any]:
    cache = DynamicCache.from_legacy_cache(
        tuple(
            (key[:, :, :prefix_context, :], value[:, :, :prefix_context, :])
            for key, value in legacy_cache
        )
    )
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    started = time.perf_counter()
    mask = torch.ones(
        (1, prefix_context + suffix_ids.shape[1]),
        dtype=torch.long,
        device=suffix_ids.device,
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
    while len(tokens) < max_new_tokens and tokens[-1] not in eos:
        token_tensor = torch.tensor([[tokens[-1]]], dtype=torch.long, device=suffix_ids.device)
        mask = torch.ones(
            (1, prefix_context + suffix_ids.shape[1] + len(tokens)),
            dtype=torch.long,
            device=suffix_ids.device,
        )
        output = model(
            input_ids=token_tensor,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        tokens.append(int(torch.argmax(output.logits[0, -1].float()).item()))
    torch.cuda.synchronize()
    generated = tokenizer.decode(tokens, skip_special_tokens=True)
    suffix = tokenizer.decode(suffix_ids[0], skip_special_tokens=True)
    return {
        "tokens": tokens,
        "generated_text": generated,
        "text": suffix + generated,
        "wall_ms": 1000.0 * (time.perf_counter() - started),
    }


def warmup(model, tokenizer, device: torch.device) -> None:
    encoded = tokenizer("Warm-up. Answer: ", return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    mask = encoded.attention_mask.to(device)
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=mask, use_cache=True, return_dict=True)
        _ = int(torch.argmax(output.logits[0, -1].float()).item())
    torch.cuda.synchronize()


def collect_one(
    *,
    config: dict[str, Any],
    fingerprint: str,
    source_path: Path,
    destination: Path,
    dataset: str,
    split: str,
    problem_id: str,
    expected_source_fingerprint: str,
    model,
    tokenizer,
    model_audit: dict[str, Any],
    device: torch.device,
    worker_id: str,
    gpu: int,
) -> dict[str, Any]:
    if artifact_valid(destination, fingerprint, problem_id):
        return {"status": "skipped", "branches": 0}
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite incompatible artifact: {destination}")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source.get("status") != "complete" or str(source.get("problem_id")) != problem_id:
        raise ValueError(f"invalid source artifact: {source_path}")
    if source.get("protocol_fingerprint") != expected_source_fingerprint:
        raise ValueError(f"source protocol fingerprint mismatch: {source_path}")
    if source.get("dtype") != "float16" or int(source.get("seed", -1)) != 20260803:
        raise ValueError(f"source dtype/seed mismatch: {source_path}")
    rows = source.get("rows", [])
    checkpoints = [int(row["checkpoint"]) for row in rows]
    if not checkpoints or len(checkpoints) != len(set(checkpoints)):
        raise ValueError(f"missing or duplicate checkpoints: {source_path}")
    prompt_ids = tokenizer(source["prompt_text"], return_tensors="pt").input_ids.to(device)
    prompt_tokens = int(prompt_ids.shape[1])
    if prompt_tokens != int(source["prompt_tokens"]):
        raise ValueError(f"prompt retokenization mismatch: {source_path}")
    maximum = max(checkpoints)
    content = source["dense"]["content_tokens"]
    if maximum > len(content):
        raise ValueError(f"checkpoint beyond dense content: {source_path}")
    dense_tensor = torch.tensor([content[:maximum]], dtype=torch.long, device=device)
    teacher_input = torch.cat([prompt_ids, dense_tensor], dim=1)
    with torch.inference_mode():
        teacher = model.model(
            input_ids=teacher_input,
            attention_mask=torch.ones_like(teacher_input),
            use_cache=True,
            return_dict=True,
        )
    legacy_cache = teacher.past_key_values.to_legacy_cache()
    del teacher
    decoding = config["forced_answer_decoding"]
    suffix_text = str(decoding["suffix"])
    suffix_ids = tokenizer(
        suffix_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    collected_rows = []
    branch_wall_ms = 0.0
    with torch.inference_mode():
        for original in rows:
            checkpoint = int(original["checkpoint"])
            branch = greedy_branch_from_legacy_cache(
                model,
                tokenizer,
                legacy_cache,
                prefix_context=prompt_tokens + checkpoint,
                suffix_ids=suffix_ids,
                max_new_tokens=int(decoding["max_new_tokens"]),
            )
            if dataset == "mmlu_pro":
                option_count = int(source["record"]["option_count"])
                prediction = parse_mmlu_pro_answer(branch["text"], option_count)
            else:
                prediction = prediction_for(dataset, branch["text"])
            current_success = success_for(dataset, source["gold_answer"], prediction)
            dense_prediction = source["dense"]["prediction"]
            row = dict(original)
            row.update(
                {
                    "current_prediction": prediction,
                    "current_success": bool(current_success),
                    "consistency": bool(
                        prediction is not None
                        and dense_prediction is not None
                        and prediction == dense_prediction
                    ),
                    "correction": bool(
                        (not current_success) and source["dense"]["success"]
                    ),
                    "damage": bool(
                        current_success and (not source["dense"]["success"])
                    ),
                    "branch_tokens": len(branch["tokens"]),
                    "branch_token_ids": list(branch["tokens"]),
                    "forced_context_tokens": prompt_tokens
                    + checkpoint
                    + int(suffix_ids.shape[1]),
                    "branch_text": branch["text"],
                    "branch_generated_text": branch["generated_text"],
                    "branch_collection_wall_ms": float(branch["wall_ms"]),
                    "branch_timing_source": "excluded_greedy_collection",
                    "forced_answer_decoding": "greedy_argmax",
                    "forced_answer_do_sample": False,
                    "greedy_worker_gpu": gpu,
                    "greedy_worker_device": torch.cuda.get_device_name(gpu),
                }
            )
            collected_rows.append(row)
            branch_wall_ms += float(branch["wall_ms"])
    merged = dict(source)
    merged.update(
        {
            "schema_version": 6,
            "status": "complete",
            "protocol_id": config["protocol_id"],
            "protocol_fingerprint": fingerprint,
            "parent_protocol_id": config["parent_protocol_id"],
            "source_protocol_id": source.get("protocol_id"),
            "source_protocol_fingerprint": source.get("protocol_fingerprint"),
            "source_common_cache_artifact": str(source_path.resolve()),
            "rows": collected_rows,
            "direct": source["direct"],
            "direct_decoding_unchanged_from_source": True,
            "forced_answer_decoding": dict(decoding),
            "forced_answer_collection": {
                "worker_id": worker_id,
                "host": socket.gethostname(),
                "gpu": gpu,
                "device": torch.cuda.get_device_name(gpu),
                "dtype": "float16",
                "attention_backend": config["model"]["attention_backend"],
                "branches": len(collected_rows),
                "branch_wall_ms": branch_wall_ms,
                "timing_eligible_for_paper": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "model_audit": model_audit,
        }
    )
    if len(collected_rows) != int(merged["hidden"].shape[0]):
        raise RuntimeError(f"row/vector mismatch: {source_path}")
    atomic_torch_save(merged, destination)
    return {"status": "completed", "branches": len(collected_rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_paper_greedy_forced_v1.yaml")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--source-state", choices=("pending", "requires_a100"), default="pending")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--single-source", action="append", type=Path)
    parser.add_argument("--single-output-root", type=Path)
    args = parser.parse_args()
    if bool(args.single_source) != bool(args.single_output_root):
        raise ValueError("--single-source and --single-output-root must be used together")
    config = load_config(args.config)
    fingerprint = protocol_fingerprint(config)
    seed_everything(int(config["seed"]["global"]))
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model, tokenizer, model_audit = load_qwen3(
        resolve(config["model"]["local_path"]),
        device,
        "float16",
        config["model"]["attention_backend"],
    )
    warmup(model, tokenizer, device)
    processed = skipped = failures = oom = branches = 0
    started = time.time()

    if args.single_source:
        for source_path_arg in args.single_source:
            source_path = source_path_arg if source_path_arg.is_absolute() else ROOT / source_path_arg
            source = torch.load(source_path, map_location="cpu", weights_only=False)
            dataset, split, problem_id = (
                str(source["dataset"]),
                str(source["split"]),
                str(source["problem_id"]),
            )
            destination = (
                args.single_output_root
                / dataset
                / split
                / f"sample_{problem_id}.pt"
            )
            expected = config["datasets"][dataset]["source_protocol_fingerprint"]
            result = collect_one(
                config=config,
                fingerprint=fingerprint,
                source_path=source_path,
                destination=destination,
                dataset=dataset,
                split=split,
                problem_id=problem_id,
                expected_source_fingerprint=expected,
                model=model,
                tokenizer=tokenizer,
                model_audit=model_audit,
                device=device,
                worker_id=args.worker_id,
                gpu=args.gpu,
            )
            processed += int(result["status"] == "completed")
            skipped += int(result["status"] == "skipped")
            branches += int(result["branches"])
        print(json.dumps({"status": "complete", "processed": processed, "skipped": skipped, "branches": branches}, indent=2))
        return

    while args.max_samples is None or processed + skipped + failures + oom < args.max_samples:
        claimed_result = claim_task(config, args.worker_id, args.source_state)
        if claimed_result is None:
            break
        payload, claimed_path = claimed_result
        try:
            result = collect_one(
                config=config,
                fingerprint=fingerprint,
                source_path=Path(payload["source_path"]),
                destination=Path(payload["destination"]),
                dataset=str(payload["dataset"]),
                split=str(payload["split"]),
                problem_id=str(payload["problem_id"]),
                expected_source_fingerprint=str(payload["source_protocol_fingerprint"]),
                model=model,
                tokenizer=tokenizer,
                model_audit=model_audit,
                device=device,
                worker_id=args.worker_id,
                gpu=args.gpu,
            )
            processed += int(result["status"] == "completed")
            skipped += int(result["status"] == "skipped")
            branches += int(result["branches"])
            payload.update({"result": result, "worker_id": args.worker_id, "gpu": args.gpu})
            finish_task(config, payload, claimed_path, "done")
        except torch.cuda.OutOfMemoryError as error:
            oom += 1
            payload.update(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "requires_a100_from_gpu": args.gpu,
                }
            )
            finish_task(config, payload, claimed_path, "requires_a100")
            torch.cuda.empty_cache()
        except Exception as error:
            failures += 1
            payload.update({"error_type": type(error).__name__, "error": str(error)})
            finish_task(config, payload, claimed_path, "failed")
            print(json.dumps({"worker": args.worker_id, "problem_id": payload.get("problem_id"), "error": repr(error)}), flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        elapsed = max(time.time() - started, 1e-9)
        live = {
            "status": "running",
            "worker_id": args.worker_id,
            "gpu": args.gpu,
            "device": torch.cuda.get_device_name(args.gpu),
            "source_state": args.source_state,
            "processed": processed,
            "skipped": skipped,
            "failures": failures,
            "oom_requires_a100": oom,
            "branches": branches,
            "elapsed_seconds": elapsed,
            "samples_per_hour": 3600.0 * processed / elapsed,
            "queue": queue_counts(config),
        }
        atomic_json(live, resolve(config["queue_root"]) / "workers" / f"{args.worker_id}.live.json")
        print(json.dumps(live), flush=True)
    summary = {
        "status": "complete" if failures == 0 else "failed",
        "worker_id": args.worker_id,
        "gpu": args.gpu,
        "device": torch.cuda.get_device_name(args.gpu),
        "source_state": args.source_state,
        "processed": processed,
        "skipped": skipped,
        "failures": failures,
        "oom_requires_a100": oom,
        "branches": branches,
        "elapsed_seconds": time.time() - started,
        "queue": queue_counts(config),
    }
    atomic_json(summary, resolve(config["queue_root"]) / "workers" / f"{args.worker_id}.json")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

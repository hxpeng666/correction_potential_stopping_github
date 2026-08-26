#!/usr/bin/env python3
"""Collect BF16 Dense/Direct trajectories under the frozen legacy-v4 seed protocol."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.final_paper_inference import (
    artifact_complete,
    atomic_torch_save,
    demonstrations_by_subject,
    gold_for,
    prediction_for,
    prompt_messages,
    read_jsonl,
    render_prompt,
    resolved_generation,
    success_for,
)
from src.final_paper_protocol import canonical_fingerprint
from src.qwen3_reasoning import generate_trace, load_qwen3
from src.utils import atomic_json, load_yaml, seed_everything


DIRECT_SEED_OFFSET = 104729


def legacy_example_seed(global_seed: int, dataset: str, record: dict) -> int:
    """Legacy GSM8K formula; deterministic declared extension for the new MMLU dataset."""
    if dataset == "gsm8k":
        index = int(record["source_index"])
    else:
        index = int(record["legacy_seed_index"])
    value = int(global_seed) + index * 1009
    if not 0 <= value < 2**63 - 1:
        raise ValueError(f"legacy example seed out of range: {value}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("probe_train", "calibration", "heldout"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--problem-ids-file", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    config = load_yaml(args.config)
    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model, tokenizer, model_audit = load_qwen3(
        ROOT / config["model"]["local_path"],
        device,
        config["model"]["dtype"],
        config["model"]["attention_backend"],
    )
    prepared = ROOT / config["dataset"]["prepared_root"]
    records = read_jsonl(prepared / f"{args.split}.jsonl")
    if args.problem_ids_file is not None:
        selection_path = (
            args.problem_ids_file
            if args.problem_ids_file.is_absolute()
            else ROOT / args.problem_ids_file
        )
        selected_ids = [str(value) for value in json.loads(selection_path.read_text())]
        by_id = {str(record["problem_id"]): record for record in records}
        missing = [value for value in selected_ids if value not in by_id]
        if missing:
            raise KeyError(f"selected problem IDs absent from {args.split}: {missing}")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError(f"duplicate problem IDs in {selection_path}")
        records = [by_id[value] for value in selected_ids]
    if args.num_samples is not None:
        records = records[: args.num_samples]
    demonstrations = (
        demonstrations_by_subject(prepared / "demonstrations.jsonl")
        if args.dataset == "mmlu"
        else None
    )
    output_root = (
        args.output_root
        if args.output_root is not None
        else ROOT / config["output_dir"] / "raw" / f"dense_seed_{args.seed}"
    )
    output = output_root / args.split
    output.mkdir(parents=True, exist_ok=True)
    dense_generation = resolved_generation(
        config, int(config["generation"]["dense_max_new_tokens"])
    )
    direct_generation = resolved_generation(
        config, int(config["generation"]["force_answer_max_new_tokens"])
    )
    completed = 0
    skipped = 0
    maxed = 0
    dense_correct = 0
    direct_correct = 0
    for local_index, record in enumerate(records):
        if local_index % args.num_shards != args.shard_index:
            continue
        problem_id = str(record["problem_id"])
        destination = output / f"sample_{problem_id}.pt"
        if args.resume and artifact_complete(destination, problem_id):
            skipped += 1
            continue
        if destination.exists():
            raise RuntimeError(
                f"refusing to overwrite non-resumable artifact; preserve and diagnose: {destination}"
            )
        messages = prompt_messages(args.dataset, record, config, demonstrations)
        dense_prompt = render_prompt(
            tokenizer, messages, enable_thinking=bool(config["model"]["enable_thinking"])
        )
        dense_encoded = tokenizer(dense_prompt, return_tensors="pt")
        dense_input_ids = dense_encoded.input_ids.to(device)
        dense_mask = dense_encoded.attention_mask.to(device)
        example_seed = legacy_example_seed(args.seed, args.dataset, record)
        with torch.inference_mode():
            trace = generate_trace(
                model,
                tokenizer,
                dense_input_ids,
                dense_mask,
                dense_generation,
                example_seed,
            )
        dense_text = tokenizer.decode(trace.tokens, skip_special_tokens=True)
        gold = gold_for(args.dataset, record)
        dense_prediction = prediction_for(args.dataset, dense_text)
        dense_success = success_for(args.dataset, gold, dense_prediction)

        direct_messages = [dict(message) for message in messages]
        direct_instruction = (
            " Do not show any reasoning. Output only the final result as \\boxed{number}."
            if args.dataset == "gsm8k"
            else (
                " Do not show any reasoning. Output only one option letter as "
                "\\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."
            )
        )
        direct_messages[0]["content"] += direct_instruction
        direct_prefix = str(config["generation"]["direct_answer_prefix"])
        direct_prompt = render_prompt(tokenizer, direct_messages, enable_thinking=False) + direct_prefix
        direct_encoded = tokenizer(direct_prompt, return_tensors="pt")
        direct_input_ids = direct_encoded.input_ids.to(device)
        direct_mask = direct_encoded.attention_mask.to(device)
        with torch.inference_mode():
            direct_trace = generate_trace(
                model,
                tokenizer,
                direct_input_ids,
                direct_mask,
                direct_generation,
                example_seed + DIRECT_SEED_OFFSET,
            )
        direct_generated_text = tokenizer.decode(direct_trace.tokens, skip_special_tokens=True)
        direct_text = direct_prefix + direct_generated_text
        direct_prediction = prediction_for(args.dataset, direct_text)
        direct_success = success_for(args.dataset, gold, direct_prediction)
        reached_max = len(trace.tokens) >= dense_generation["max_new_tokens"]
        artifact = {
            "schema_version": 1,
            "status": "complete",
            "dataset": args.dataset,
            "split": args.split,
            "seed": args.seed,
            "example_seed": example_seed,
            "direct_seed": example_seed + DIRECT_SEED_OFFSET,
            "protocol_id": "legacy_empirical_v4",
            "problem_id": problem_id,
            "record": record,
            "gold_answer": gold,
            "config_fingerprint": canonical_fingerprint(config),
            "model_audit": model_audit,
            "dtype": str(config["model"]["dtype"]),
            "attention_backend": str(config["model"]["attention_backend"]),
            "prompt_text": dense_prompt,
            "prompt_tokens": int(dense_input_ids.shape[1]),
            "dense": {
                "tokens": trace.tokens,
                "text": dense_text,
                "prediction": dense_prediction,
                "success": dense_success,
                "reasoning_tokens": len(trace.tokens),
                "reached_max_tokens": reached_max,
                "prefill_cuda_ms": trace.prefill_cuda_ms,
                "decode_cuda_ms": trace.decode_cuda_ms,
                "wall_ms": trace.wall_ms,
                "logps": trace.logps,
                "margins": trace.margins,
                "entropies_top20": trace.entropies,
            },
            "direct": {
                "prompt_text": direct_prompt,
                "prompt_tokens": int(direct_input_ids.shape[1]),
                "tokens": direct_trace.tokens,
                "text": direct_text,
                "answer_prefix": direct_prefix,
                "generated_text": direct_generated_text,
                "prediction": direct_prediction,
                "success": direct_success,
                "generated_tokens": len(direct_trace.tokens),
                "prefill_cuda_ms": direct_trace.prefill_cuda_ms,
                "decode_cuda_ms": direct_trace.decode_cuda_ms,
                "wall_ms": direct_trace.wall_ms,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_torch_save(artifact, destination)
        completed += 1
        maxed += int(reached_max)
        dense_correct += int(dense_success)
        direct_correct += int(direct_success)
        print(
            json.dumps(
                {
                    "problem_id": problem_id,
                    "completed": completed,
                    "dense_tokens": len(trace.tokens),
                    "dense_success": dense_success,
                    "direct_success": direct_success,
                    "reached_max": reached_max,
                }
            ),
            flush=True,
        )
    summary = {
        "status": "complete",
        "phase": "dense_and_direct",
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "records_visible": len(records),
        "completed_now": completed,
        "skipped_complete": skipped,
        "dense_correct_now": dense_correct,
        "direct_correct_now": direct_correct,
        "reached_max_now": maxed,
        "model": model_audit,
    }
    atomic_json(summary, output / f"summary_shard{args.shard_index}.json")
    atomic_json(
        {"status": "complete", "summary": summary},
        output / f"phase_shard{args.shard_index}.complete",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

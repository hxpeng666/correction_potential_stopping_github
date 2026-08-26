#!/usr/bin/env python3
"""Construct offline forced-answer labels and hidden states on cached Dense traces."""
from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.final_paper_inference import (
    PositionHiddenCapture,
    artifact_complete,
    atomic_torch_save,
    branch_from_legacy_cache,
    prediction_for,
    resolved_generation,
    stable_example_seed,
    success_for,
)
from src.final_paper_protocol import BOUNDARY, checkpoint_schedules
from src.qwen3_reasoning import load_qwen3
from src.utils import atomic_json, load_yaml, seed_everything


def raw_semantic_boundaries(tokenizer, token_ids: list[int], upper: int) -> tuple[list[int], str]:
    """Return boundaries aligned to exact generated token IDs, with a rare safe fallback."""
    limited = token_ids[:upper]
    text = tokenizer.decode(
        limited, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    reencoded = list(encoded.input_ids)
    if reencoded == limited:
        token_ends = [int(end) for _start, end in encoded.offset_mapping]
    else:
        token_ends = []
        for end in range(1, len(limited) + 1):
            prefix = tokenizer.decode(
                limited[:end],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            token_ends.append(len(prefix))
    result = set()
    for match in BOUNDARY.finditer(text):
        position = bisect.bisect_left(token_ends, match.end())
        if position < len(token_ends):
            result.add(position + 1)
    return sorted(result), text


def tail_mean(values: list[float], checkpoint: int, width: int = 8) -> float:
    selected = values[max(0, checkpoint - width):checkpoint]
    return float(np.mean(selected)) if selected else math.nan


def valid_dense_source(path: Path, problem_id: str) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("status") != "complete" or value.get("problem_id") != problem_id:
        raise ValueError(f"invalid Dense source: {path}")
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
    parser.add_argument("--dense-root", type=Path)
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
    dense_root = (
        args.dense_root
        if args.dense_root is not None
        else ROOT / config["output_dir"] / "raw" / f"dense_seed_{args.seed}"
    )
    output_root = (
        args.output_root
        if args.output_root is not None
        else ROOT / config["output_dir"] / "raw" / f"checkpoints_seed_{args.seed}"
    )
    dense_files = sorted((dense_root / args.split).glob("sample_*.pt"))
    if args.num_samples is not None:
        dense_files = dense_files[: args.num_samples]
    output = output_root / args.split
    output.mkdir(parents=True, exist_ok=True)
    capture_layers = [int(x) for x in config["generation"]["capture_layers"]]
    capture = PositionHiddenCapture(model, capture_layers)
    suffix_ids = tokenizer(
        config["generation"]["force_answer_suffix"],
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    branch_generation = resolved_generation(
        config, int(config["generation"]["force_answer_max_new_tokens"])
    )
    protocol = config["checkpoint_protocol"]
    fixed = [int(x) for x in config["generation"]["fixed_budgets"]]
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    completed = 0
    skipped = 0
    total_checkpoints = 0
    no_legal = 0
    shorter_than_minimum = 0

    for local_index, source_path in enumerate(dense_files):
        if local_index % args.num_shards != args.shard_index:
            continue
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        problem_id = str(source["problem_id"])
        destination = output / f"sample_{problem_id}.pt"
        if args.resume and artifact_complete(destination, problem_id):
            skipped += 1
            continue
        if destination.exists():
            raise RuntimeError(
                f"refusing to overwrite non-resumable artifact; preserve and diagnose: {destination}"
            )
        source = valid_dense_source(source_path, problem_id)
        dense_token_ids = [int(x) for x in source["dense"]["tokens"]]
        content_ids = (
            dense_token_ids[:-1]
            if dense_token_ids and dense_token_ids[-1] in eos
            else dense_token_ids
        )
        maximum = min(int(protocol["maximum"]), len(content_ids))
        semantic, decoded_prefix = raw_semantic_boundaries(
            tokenizer, content_ids, maximum
        ) if maximum else ([], "")
        schedules = checkpoint_schedules(
            semantic,
            len(content_ids),
            minimum=int(protocol["minimum"]),
            maximum=int(protocol["maximum"]),
            sentence_gap=int(protocol["sentence_minimum_gap"]),
            hybrid_minimum_gap=int(protocol["hybrid_minimum_gap"]),
            hybrid_maximum_gap=int(protocol["hybrid_maximum_gap"]),
            fixed=fixed,
        )
        union = sorted(set().union(*(set(values) for values in schedules.values())))
        shorter_than_minimum += int(len(content_ids) < int(protocol["minimum"]))
        no_legal += int(not schedules["sentence"])
        rows: list[dict[str, Any]] = []
        vectors: list[torch.Tensor] = []
        if union:
            prompt_ids = tokenizer(
                source["prompt_text"], return_tensors="pt"
            ).input_ids.to(device)
            prompt_tokens = int(prompt_ids.shape[1])
            maximum_checkpoint = max(union)
            dense_tensor = torch.tensor(
                [content_ids[:maximum_checkpoint]], dtype=torch.long, device=device
            )
            teacher_input = torch.cat([prompt_ids, dense_tensor], dim=1)
            absolute_positions = [
                prompt_tokens + checkpoint - 1 for checkpoint in union
            ]
            capture.begin(absolute_positions, device)
            with torch.inference_mode():
                teacher = model.model(
                    input_ids=teacher_input,
                    attention_mask=torch.ones_like(teacher_input),
                    use_cache=True,
                    return_dict=True,
                )
            hidden = capture.finish_cpu()
            legacy_cache = teacher.past_key_values.to_legacy_cache()
            for union_index, checkpoint in enumerate(union):
                branch = branch_from_legacy_cache(
                    model,
                    tokenizer,
                    legacy_cache,
                    prefix_context=prompt_tokens + checkpoint,
                    suffix_ids=suffix_ids,
                    generation=branch_generation,
                    seed=stable_example_seed(
                        args.seed, problem_id, f"forced_{checkpoint}"
                    ),
                )
                prediction = prediction_for(args.dataset, branch["text"])
                current_success = success_for(
                    args.dataset, source["gold_answer"], prediction
                )
                dense_prediction = source["dense"]["prediction"]
                prefix_text = tokenizer.decode(
                    content_ids[:checkpoint],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                schedule_membership = [
                    name for name, values in schedules.items() if checkpoint in values
                ]
                decode_times = source["dense"]["decode_cuda_ms"]
                row = {
                    "problem_id": problem_id,
                    "dataset": args.dataset,
                    "split": args.split,
                    "seed": args.seed,
                    "subject": source["record"].get("subject"),
                    "category": source["record"].get("category"),
                    "checkpoint": checkpoint,
                    "checkpoint_schedules": schedule_membership,
                    "is_sentence_checkpoint": checkpoint in schedules["sentence"],
                    "is_fixed_checkpoint": checkpoint in schedules["fixed"],
                    "is_hybrid_checkpoint": checkpoint in schedules["hybrid"],
                    "gold_answer": source["gold_answer"],
                    "current_prediction": prediction,
                    "current_success": bool(current_success),
                    "dense_prediction": dense_prediction,
                    "dense_success": bool(source["dense"]["success"]),
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
                    "dense_tokens": int(source["dense"]["reasoning_tokens"]),
                    "dense_content_tokens": len(content_ids),
                    "dense_wall_ms": float(source["dense"]["wall_ms"]),
                    "dense_prefill_cuda_ms": float(
                        source["dense"]["prefill_cuda_ms"]
                    ),
                    "prefix_decode_cuda_ms": float(
                        sum(decode_times[: max(0, checkpoint - 1)])
                    ),
                    "prefix_mean_entropy_tail8": tail_mean(
                        source["dense"]["entropies_top20"], checkpoint
                    ),
                    "branch_tokens": len(branch["tokens"]),
                    "branch_wall_ms": float(branch["wall_ms"]),
                    "branch_text": branch["text"],
                    "branch_generated_text": branch["generated_text"],
                    "prefix_text": prefix_text,
                }
                rows.append(row)
                vectors.append(hidden[union_index].to(torch.float16))
        artifact = {
            "schema_version": 1,
            "status": "complete",
            "dataset": args.dataset,
            "split": args.split,
            "seed": args.seed,
            "problem_id": problem_id,
            "source_dense_artifact": str(source_path),
            "model_audit": model_audit,
            "capture_layers": capture_layers,
            "checkpoint_protocol": protocol,
            "schedules": schedules,
            "semantic_boundaries": semantic,
            "dense_content_tokens": len(content_ids),
            "decoded_prefix_for_boundaries": decoded_prefix,
            "rows": rows,
            "hidden": (
                torch.stack(vectors)
                if vectors
                else torch.empty(
                    (0, len(capture_layers), model_audit["hidden_size"]),
                    dtype=torch.float16,
                )
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if len(artifact["rows"]) != len(artifact["hidden"]):
            raise RuntimeError(f"row/vector mismatch for {problem_id}")
        atomic_torch_save(artifact, destination)
        completed += 1
        total_checkpoints += len(rows)
        print(
            json.dumps(
                {
                    "problem_id": problem_id,
                    "completed": completed,
                    "dense_content_tokens": len(content_ids),
                    "checkpoints": {key: len(value) for key, value in schedules.items()},
                    "union": len(union),
                }
            ),
            flush=True,
        )
    capture.close()
    summary = {
        "status": "complete",
        "phase": "offline_checkpoints",
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "dense_files_visible": len(dense_files),
        "completed_now": completed,
        "skipped_complete": skipped,
        "total_union_checkpoints_now": total_checkpoints,
        "no_sentence_checkpoint_now": no_legal,
        "shorter_than_minimum_now": shorter_than_minimum,
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

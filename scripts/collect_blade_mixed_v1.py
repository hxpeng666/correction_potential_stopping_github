#!/usr/bin/env python3
"""Collect BLADE MGRC checkpoints, K16 strict-clean labels, and all-layer states."""
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "src").is_dir():
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from nltk import sent_tokenize
from transformers.cache_utils import DynamicCache

from src.final_paper_inference import atomic_torch_save, prediction_for, success_for
from src.mmlu_pro_protocol import parse_answer as parse_mmlu_pro_answer
from src.qwen3_reasoning import CheckpointHiddenCapture, load_qwen3, sample_token
from src.utils import load_yaml, seed_everything


PARAGRAPH_RE = re.compile(r"\n\s*\n+")


def canonical_fingerprint(config: dict[str, Any], dataset: str) -> str:
    payload = {
        "protocol_id": config["protocol_id"],
        "source": config["source"],
        "model": config["model"],
        "common_scope": config["common_scope"],
        "dataset": dataset,
        "dataset_config": config["datasets"][dataset],
        "checkpoints": config["checkpoints"],
        "strict_clean_supervision": config["strict_clean_supervision"],
        "apls_capture": {
            "decoder_layers": config["model"]["decoder_layers"],
            "hidden_size": config["model"]["hidden_size"],
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def completion_seed(global_seed: int, dataset: str, split: str, problem_id: str, checkpoint: int, index: int) -> int:
    payload = f"{global_seed}:{dataset}:{split}:{problem_id}:{checkpoint}:{index}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def find_subsequence(values: list[int], pattern: list[int], start: int = 0) -> int | None:
    for index in range(start, len(values) - len(pattern) + 1):
        if values[index : index + len(pattern)] == pattern:
            return index
    return None


def reasoning_range(tokenizer, content: list[int]) -> tuple[int, int, dict[str, Any]]:
    open_ids = list(tokenizer("<think>", add_special_tokens=False).input_ids)
    close_ids = list(tokenizer("</think>", add_special_tokens=False).input_ids)
    opening = find_subsequence(content, open_ids)
    start = opening + len(open_ids) if opening is not None else 0
    closing = find_subsequence(content, close_ids, start)
    end = closing if closing is not None else len(content)
    return start, end, {
        "opening_found": opening is not None,
        "closing_found": closing is not None,
        "reasoning_start": start,
        "reasoning_end": end,
        "dense_content_tokens": len(content),
    }


def token_text_and_offsets(tokenizer, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded.input_ids) == ids:
        return text, [(int(a), int(b)) for a, b in encoded.offset_mapping]
    ends = [
        len(tokenizer.decode(ids[:i], skip_special_tokens=False, clean_up_tokenization_spaces=False))
        for i in range(1, len(ids) + 1)
    ]
    return text, list(zip([0] + ends[:-1], ends))


def char_end_to_token_count(offsets: list[tuple[int, int]], char_end: int) -> int:
    index = bisect.bisect_left([end for _start, end in offsets], char_end)
    return min(len(offsets), index + 1)


def sentence_positions(tokenizer, content: list[int], start: int, end: int) -> list[int]:
    ids = content[start:end]
    if not ids:
        return []
    raw, offsets = token_text_and_offsets(tokenizer, ids)
    left = len(raw) - len(raw.lstrip())
    reasoning = raw.strip()
    cursor = 0
    result = []
    for sentence in sent_tokenize(reasoning):
        found = reasoning.find(sentence, cursor)
        if found < 0:
            raise ValueError(f"sentence cannot be mapped back to reasoning: {sentence!r}")
        cursor = found + len(sentence)
        result.append(start + char_end_to_token_count(offsets, left + cursor))
    return sorted(set(position for position in result if start < position <= end))


def doubt_positions(tokenizer, content: list[int], start: int, end: int, cues: list[str]) -> list[dict[str, Any]]:
    ids = content[start:end]
    pieces = [
        tokenizer.decode([token], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for token in ids
    ]
    cumulative = []
    text = ""
    for piece in pieces:
        text += piece
        cumulative.append(len(text))
    lowered = text.lower()
    events: dict[int, set[str]] = defaultdict(set)
    for cue in cues:
        cursor = 0
        while True:
            found = lowered.find(cue.lower(), cursor)
            if found < 0:
                break
            char_end = found + len(cue)
            local_position = next((i + 1 for i, value in enumerate(cumulative) if value >= char_end), len(ids))
            if local_position > 0:
                events[start + local_position].add(cue)
            cursor = char_end
    return [
        {"checkpoint": position, "cues": sorted(values)}
        for position, values in sorted(events.items())
        if start < position <= end
    ]


def source_name(problem_id: str) -> str:
    return f"sample_{problem_id}.pt"


def frozen_paragraph_positions(root: Path, split: str, problem_id: str) -> list[int]:
    path = root / split / source_name(problem_id)
    if not path.is_file():
        return []
    value = torch.load(path, map_location="cpu", weights_only=False)
    return sorted(set(int(x) for x in value.get("schedule_checkpoints", [])))


def candidate_rows(tokenizer, content: list[int], paragraph: list[int], config: dict[str, Any], split: str):
    start, end, trace = reasoning_range(tokenizer, content)
    by_position: dict[int, dict[str, Any]] = {}
    for position in sentence_positions(tokenizer, content, start, end):
        by_position.setdefault(position, {"checkpoint": position, "checkpoint_types": set(), "cues": set()})
        by_position[position]["checkpoint_types"].add("sentence")
    for value in doubt_positions(tokenizer, content, start, end, [str(x) for x in config["checkpoints"]["self_doubt_cues"]]):
        position = int(value["checkpoint"])
        by_position.setdefault(position, {"checkpoint": position, "checkpoint_types": set(), "cues": set()})
        by_position[position]["checkpoint_types"].add("self_doubt")
        by_position[position]["cues"].update(value["cues"])
    dropped_paragraph = 0
    if split == "probe_train":
        for position in paragraph:
            if not start < position <= end:
                dropped_paragraph += 1
                continue
            by_position.setdefault(position, {"checkpoint": position, "checkpoint_types": set(), "cues": set()})
            by_position[position]["checkpoint_types"].add("paragraph")
    rows = []
    for position, value in sorted(by_position.items()):
        types = sorted(value["checkpoint_types"])
        rows.append({
            "checkpoint": position,
            "checkpoint_types": types,
            "is_sentence": "sentence" in types,
            "is_self_doubt": "self_doubt" in types,
            "is_paragraph": "paragraph" in types,
            "cues": sorted(value["cues"]),
        })
    trace.update({
        "sentence_checkpoints": sum(row["is_sentence"] for row in rows),
        "self_doubt_checkpoints": sum(row["is_self_doubt"] for row in rows),
        "paragraph_checkpoints": sum(row["is_paragraph"] for row in rows),
        "union_checkpoints": len(rows),
        "paragraph_outside_reasoning_dropped": dropped_paragraph,
        "paragraph_training_only": True,
    })
    return rows, trace


def clone_cache(cache):
    return DynamicCache.from_legacy_cache(cache.to_legacy_cache())


def branch(
    model,
    tokenizer,
    base_cache,
    prefix_context: int,
    suffix_ids,
    maximum: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
):
    cache = clone_cache(base_cache)
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    generator = torch.Generator(device=suffix_ids.device).manual_seed(seed)
    mask = torch.ones((1, prefix_context + suffix_ids.shape[1]), dtype=torch.long, device=suffix_ids.device)
    output = model(input_ids=suffix_ids, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True)
    past = output.past_key_values
    if do_sample:
        first = sample_token(output.logits, generator, temperature, top_k, top_p)
    else:
        first = int(torch.argmax(output.logits[0, -1].float()).item())
    tokens = [first]
    while len(tokens) < maximum and tokens[-1] not in eos:
        current = torch.tensor([[tokens[-1]]], dtype=torch.long, device=suffix_ids.device)
        mask = torch.ones(
            (1, prefix_context + suffix_ids.shape[1] + len(tokens)), dtype=torch.long, device=suffix_ids.device
        )
        output = model(input_ids=current, attention_mask=mask, past_key_values=past, use_cache=True, return_dict=True)
        past = output.past_key_values
        if do_sample:
            token = sample_token(output.logits, generator, temperature, top_k, top_p)
        else:
            token = int(torch.argmax(output.logits[0, -1].float()).item())
        tokens.append(token)
    generated = tokenizer.decode(tokens, skip_special_tokens=True)
    suffix = tokenizer.decode(suffix_ids[0], skip_special_tokens=True)
    return {
        "seed": seed,
        "tokens": tokens,
        "generated_text": generated,
        "text": suffix + generated,
        "terminated_by_eos": bool(tokens and tokens[-1] in eos),
    }


def parsed(dataset: str, value: dict[str, Any], source: dict[str, Any]):
    if dataset == "mmlu_pro":
        prediction = parse_mmlu_pro_answer(value["text"], len(source["record"]["choices"]))
        success = prediction is not None and prediction == source["gold_answer"]
    else:
        prediction = prediction_for("gsm8k", value["text"])
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
            and tuple(value.get("hidden", torch.empty(0)).shape[1:]) == (36, 2560)
        )
    except Exception:
        return False


def collect_one(source_path, destination, config, dataset, fingerprint, model, tokenizer, audit, device, gpu, worker):
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    problem_id = str(source["problem_id"])
    split = str(source["split"])
    if destination.exists() and not valid(destination, fingerprint, problem_id):
        raise RuntimeError(f"refusing to overwrite incompatible artifact: {destination}")
    content = list(source["dense"]["content_tokens"])
    paragraph_root = ROOT / config["datasets"][dataset]["paragraph_cache_root"]
    paragraph = frozen_paragraph_positions(paragraph_root, split, problem_id)
    specs, trajectory = candidate_rows(tokenizer, content, paragraph, config, split)
    points = [int(row["checkpoint"]) for row in specs]
    prompt_ids = tokenizer(source["prompt_text"], return_tensors="pt").input_ids.to(device)
    prompt_tokens = int(prompt_ids.shape[1])
    if prompt_tokens != int(source["prompt_tokens"]):
        raise ValueError(f"prompt retokenization mismatch: {source_path}")
    suffix = str(config["common_scope"]["force_answer_suffix"])
    suffix_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    strict = config["strict_clean_supervision"]
    maximum = int(strict["max_new_tokens"])
    capture = CheckpointHiddenCapture(model, list(range(int(config["model"]["decoder_layers"]))))
    hidden = []
    rows = []
    try:
        with torch.inference_mode():
            prefill = model.model(
                input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), use_cache=True, return_dict=True
            )
            base_cache = prefill.past_key_values
            del prefill
            previous = 0
            for spec in specs:
                point = int(spec["checkpoint"])
                delta = torch.tensor([content[previous:point]], dtype=torch.long, device=device)
                mask = torch.ones((1, prompt_tokens + point), dtype=torch.long, device=device)
                capture.begin()
                teacher = model.model(
                    input_ids=delta, attention_mask=mask, past_key_values=base_cache,
                    use_cache=True, return_dict=True,
                )
                base_cache = teacher.past_key_values
                vector = capture.finish_cpu().to(torch.float16)
                del teacher
                hidden.append(vector)

                greedy_seed = completion_seed(int(config["seed"]), dataset, split, problem_id, point, -1)
                greedy = branch(
                    model, tokenizer, base_cache, prompt_tokens + point, suffix_ids,
                    int(config["common_scope"]["force_answer_max_new_tokens"]), False,
                    float(strict["temperature"]), int(strict["top_k"]), float(strict["top_p"]), greedy_seed,
                )
                current_prediction, current_success = parsed(dataset, greedy, source)
                sampled = []
                if split in ("probe_train", "calibration"):
                    for index in range(int(strict["completions_per_checkpoint"])):
                        seed = completion_seed(int(config["seed"]), dataset, split, problem_id, point, index)
                        value = branch(
                            model, tokenizer, base_cache, prompt_tokens + point, suffix_ids, maximum, True,
                            float(strict["temperature"]), int(strict["top_k"]), float(strict["top_p"]), seed,
                        )
                        prediction, success = parsed(dataset, value, source)
                        sampled.append({
                            "index": index,
                            "seed": seed,
                            "prediction": prediction,
                            "success": success,
                            "branch_tokens": len(value["tokens"]),
                            "branch_token_ids": value["tokens"],
                            "generated_text": value["generated_text"],
                            "truncated": not value["terminated_by_eos"],
                        })
                correct_count = sum(value["success"] for value in sampled)
                if sampled and correct_count == len(sampled):
                    label = 1
                elif sampled and correct_count == 0:
                    label = 0
                else:
                    label = None
                rows.append({
                    **spec,
                    "rank": len(rows),
                    "dataset": dataset,
                    "split": split,
                    "problem_id": problem_id,
                    "gold_answer": source["gold_answer"],
                    "dense_prediction": source["dense"]["prediction"],
                    "dense_success": bool(source["dense"]["success"]),
                    "dense_tokens": int(source["dense"]["reasoning_tokens"]),
                    "current_prediction": current_prediction,
                    "current_success": current_success,
                    "correction": bool((not current_success) and source["dense"]["success"]),
                    "damage": bool(current_success and not source["dense"]["success"]),
                    "greedy_branch_tokens": len(greedy["tokens"]),
                    "greedy_branch_token_ids": greedy["tokens"],
                    "greedy_branch_text": greedy["text"],
                    "greedy_forced_answer_truncated": not greedy["terminated_by_eos"],
                    "strict_clean_label": label,
                    "strict_clean_correct_count": correct_count if sampled else None,
                    "strict_clean_ambiguous": bool(sampled and label is None),
                    "strict_clean_completions": sampled,
                    "stop_reasoning_tokens": point,
                    "stop_total_tokens": point + int(suffix_ids.shape[1]) + len(greedy["tokens"]),
                })
                previous = point
    finally:
        capture.close()
    layer_count = int(config["model"]["decoder_layers"])
    hidden_tensor = (
        torch.stack(hidden)
        if hidden
        else torch.empty((0, layer_count, int(config["model"]["hidden_size"])), dtype=torch.float16)
    )
    artifact = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "dataset": dataset,
        "method": "blade",
        "schedule_variant": "native_mixed",
        "split": split,
        "problem_id": problem_id,
        "rows": rows,
        "hidden": hidden_tensor,
        "capture_layers_zero_based": list(range(layer_count)),
        "feature_definition": "raw_decoder_block_output_at_mixed_checkpoint",
        "record": source["record"],
        "gold_answer": source["gold_answer"],
        "prompt_text": source["prompt_text"],
        "prompt_tokens": prompt_tokens,
        "dense": source["dense"],
        "source_artifact": str(source_path.resolve()),
        "trajectory": trajectory,
        "strict_clean_supervision": config["strict_clean_supervision"],
        "model_audit": audit,
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
    return {
        "problem_id": problem_id,
        "checkpoints": len(rows),
        "strict_clean": sum(row["strict_clean_label"] is not None for row in rows),
        "ambiguous": sum(row["strict_clean_ambiguous"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--split", choices=("all", "probe_train", "calibration", "heldout"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    output_root = ROOT / config["output_root"]
    source_root = ROOT / config["datasets"][args.dataset]["source_root"]
    splits = ("probe_train", "calibration", "heldout") if args.split == "all" else (args.split,)
    paths = sorted(path for split in splits for path in (source_root / split).glob("sample_*.pt"))
    assigned = [path for index, path in enumerate(paths) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        assigned = assigned[: args.limit]
    fingerprint = canonical_fingerprint(config, args.dataset)
    seed_everything(int(config["seed"]))
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    free, total = torch.cuda.mem_get_info(device)
    print(json.dumps({
        "status": "loading", "worker": args.worker_id, "dataset": args.dataset,
        "gpu": args.gpu, "free_GiB": free / 2**30, "total_GiB": total / 2**30,
        "assigned": len(assigned),
    }), flush=True)
    model, tokenizer, audit = load_qwen3(
        ROOT / config["model"]["local_path"], device,
        config["model"]["dtype"], config["model"]["attention_backend"],
    )
    completed = skipped = failures = checkpoints = 0
    started = time.time()
    for source_path in assigned:
        problem_id = source_path.stem.removeprefix("sample_")
        split = source_path.parent.name
        destination = output_root / args.dataset / "cache" / split / source_path.name
        try:
            if args.resume and valid(destination, fingerprint, problem_id):
                skipped += 1
                continue
            result = collect_one(
                source_path, destination, config, args.dataset, fingerprint,
                model, tokenizer, audit, device, args.gpu, args.worker_id,
            )
            completed += 1
            checkpoints += int(result["checkpoints"])
            print(json.dumps({
                "status": "completed", "worker": args.worker_id, "completed": completed,
                "skipped": skipped, "failures": failures, **result,
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
        "checkpoints": checkpoints,
        "elapsed_seconds": time.time() - started,
        "protocol_fingerprint": fingerprint,
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

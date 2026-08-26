#!/usr/bin/env python3
"""Collect method-faithful checkpoint data on frozen Qwen3-4B Dense traces.

This collector deliberately keeps the language-model/data protocol fixed while changing
the checkpoint representation and forced-exit mechanism required by each paper.
"""
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
from src.qwen3_reasoning import CheckpointHiddenCapture, load_qwen3
from src.utils import load_yaml, seed_everything


METHODS = ("learn_to_stop", "lynx", "thought_calibration")
SCHEDULES = ("native", "paragraph")
PARAGRAPH_RE = re.compile(r"\n\s*\n+")


def canonical_fingerprint(config: dict[str, Any], dataset: str, method: str) -> str:
    payload = {
        "protocol_id": config["protocol_id"],
        "model": config["model"],
        "common_scope": config["common_scope"],
        "dataset": dataset,
        "method": method,
        "method_config": config["methods"][method],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def find_subsequence(values: list[int], pattern: list[int], start: int = 0) -> int | None:
    if not pattern:
        return None
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
    starts = [0] + ends[:-1]
    return text, list(zip(starts, ends))


def char_end_to_token_count(offsets: list[tuple[int, int]], char_end: int) -> int:
    ends = [end for _start, end in offsets]
    index = bisect.bisect_left(ends, char_end)
    return min(len(offsets), index + 1)


def nltk_sentence_positions(tokenizer, content: list[int], start: int, end: int) -> list[int]:
    ids = content[start:end]
    if not ids:
        return []
    raw, offsets = token_text_and_offsets(tokenizer, ids)
    left = len(raw) - len(raw.lstrip())
    reasoning = raw.strip()
    cursor = 0
    positions: list[int] = []
    for sentence in sent_tokenize(reasoning):
        found = reasoning.find(sentence, cursor)
        if found < 0:
            raise ValueError(f"NLTK sentence cannot be mapped back to reasoning: {sentence!r}")
        cursor = found + len(sentence)
        absolute_char_end = left + cursor
        positions.append(start + char_end_to_token_count(offsets, absolute_char_end))
    return sorted(set(value for value in positions if start < value <= end))


def paragraph_positions(tokenizer, content: list[int], end: int) -> list[int]:
    ids = content[:end]
    if not ids:
        return []
    text, offsets = token_text_and_offsets(tokenizer, ids)
    positions = {
        char_end_to_token_count(offsets, match.end())
        for match in PARAGRAPH_RE.finditer(text)
    }
    return sorted(value for value in positions if 0 < value <= end)


def thought_sections(
    tokenizer, content: list[int], start: int, end: int
) -> list[dict[str, Any]]:
    """Return paragraph sections inside the think block, including the terminal section."""
    ids = content[start:end]
    if not ids:
        return []
    text, offsets = token_text_and_offsets(tokenizer, ids)
    delimiters = list(PARAGRAPH_RE.finditer(text))
    result: list[dict[str, Any]] = []
    previous_char = 0
    for match in delimiters:
        section_start = previous_char
        section_end = match.start()
        checkpoint_char = match.end()
        previous_char = match.end()
        if text[section_start:section_end].strip():
            token_start = bisect.bisect_right([b for _a, b in offsets], section_start)
            feature_end = char_end_to_token_count(offsets, section_end)
            checkpoint = char_end_to_token_count(offsets, checkpoint_char)
            result.append({
                "text": text[section_start:section_end],
                "feature_start": start + token_start,
                "feature_end": start + feature_end,
                "checkpoint": start + checkpoint,
                "terminal": False,
            })
    if text[previous_char:].strip():
        token_start = bisect.bisect_right([b for _a, b in offsets], previous_char)
        result.append({
            "text": text[previous_char:],
            "feature_start": start + token_start,
            "feature_end": end,
            "checkpoint": end,
            "terminal": True,
        })
    return result


def lynx_cues(tokenizer, ids: list[int], patterns: list[str]) -> list[dict[str, Any]]:
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
    by_position: dict[int, str] = {}
    for pattern in patterns:
        cursor = 0
        while True:
            found = lowered.find(pattern.lower(), cursor)
            if found < 0:
                break
            char_end = found + len(pattern)
            position = next(
                (i + 1 for i, value in enumerate(cumulative) if value >= char_end),
                len(ids),
            )
            by_position.setdefault(position, pattern)
            cursor = char_end
    return [
        {
            "feature_position": position,
            "prefix_position": max(0, position - 1),
            "event": by_position[position],
        }
        for position in sorted(by_position)
        if position > 0
    ]


def source_name(problem_id: str) -> str:
    return f"sample_{problem_id}.pt"


def frozen_paragraph_positions(root: Path, split: str, problem_id: str) -> list[int] | None:
    path = root / split / source_name(problem_id)
    if not path.is_file():
        return None
    value = torch.load(path, map_location="cpu", weights_only=False)
    return [int(x) for x in value.get("schedule_checkpoints", [])]


def load_reuse_rows(
    roots: list[Path], split: str, problem_id: str, prompt_text: str, content: list[int]
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for root in roots:
        path = root / split / source_name(problem_id)
        if not path.is_file():
            continue
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("prompt_text") != prompt_text:
            continue
        artifact_content = list(artifact.get("dense", {}).get("content_tokens", []))
        if artifact_content != content:
            continue
        decoding = artifact.get("forced_answer_decoding", {})
        if decoding and decoding.get("strategy") != "greedy_argmax":
            continue
        for row in artifact.get("rows", []):
            checkpoint = int(row["checkpoint"])
            if row.get("current_prediction") is not None or row.get("branch_text"):
                rows.setdefault(checkpoint, dict(row))
    return rows


def clone_cache(cache):
    return DynamicCache.from_legacy_cache(cache.to_legacy_cache())


def greedy_branch(
    model,
    tokenizer,
    base_cache,
    prefix_context: int,
    suffix_ids: torch.Tensor,
    maximum: int,
) -> dict[str, Any]:
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
        "wall_ms": 1000.0 * (time.perf_counter() - started),
    }


def branch_to_row(
    branch: dict[str, Any], dataset: str, source: dict[str, Any]
) -> dict[str, Any]:
    if dataset == "mmlu_pro":
        prediction = parse_mmlu_pro_answer(branch["text"], len(source["record"]["choices"]))
        success = prediction is not None and prediction == source["gold_answer"]
    else:
        prediction = prediction_for("gsm8k", branch["text"])
        success = success_for("gsm8k", source["gold_answer"], prediction)
    return {
        "current_prediction": prediction,
        "current_success": bool(success),
        "branch_tokens": len(branch["tokens"]),
        "branch_token_ids": list(branch["tokens"]),
        "branch_text": branch["text"],
        "branch_generated_text": branch["generated_text"],
        "branch_collection_wall_ms": float(branch["wall_ms"]),
    }


def valid(path: Path, fingerprint: str, method: str, schedule: str, problem_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
        return (
            value.get("status") == "complete"
            and value.get("protocol_fingerprint") == fingerprint
            and value.get("method") == method
            and value.get("schedule_variant") == schedule
            and str(value.get("problem_id")) == problem_id
        )
    except Exception:
        return False


def destination(output_root: Path, dataset: str, method: str, schedule: str, split: str, problem_id: str) -> Path:
    return output_root / dataset / method / schedule / "cache" / split / source_name(problem_id)


def candidate_specs(
    tokenizer,
    content: list[int],
    method: str,
    config: dict[str, Any],
    frozen_paragraph: list[int] | None,
    split: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[dict[str, Any]]]:
    start, end, trace = reasoning_range(tokenizer, content)
    computed_paragraph = paragraph_positions(tokenizer, content, end)
    if frozen_paragraph is not None and computed_paragraph != frozen_paragraph:
        raise ValueError(
            f"paragraph schedule drift: computed={computed_paragraph[:20]} frozen={frozen_paragraph[:20]}"
        )
    paragraph = frozen_paragraph if frozen_paragraph is not None else computed_paragraph
    control_sections: list[dict[str, Any]] = []

    if method == "learn_to_stop":
        native_positions = nltk_sentence_positions(tokenizer, content, start, end)
        schedules = {
            "native": [
                {"feature_position": p, "prefix_position": p, "checkpoint_kind": "nltk_sentence_end"}
                for p in native_positions
            ],
            "paragraph": [
                {"feature_position": p, "prefix_position": p, "checkpoint_kind": "paragraph_end"}
                for p in paragraph
            ],
        }
    elif method == "lynx":
        native = lynx_cues(
            tokenizer,
            content[:end],
            [str(x) for x in config["methods"][method]["cue_patterns"]],
        )
        # The public LYNX dataset builders add a synthetic checkpoint at 70% of
        # a rollout when no natural cue is available.  Keep that training/calibration
        # fallback, but never introduce it at held-out inference, where the official
        # evaluator falls back to the Dense rollout if there is no cue.  We place the
        # fallback at 70% of the Qwen thinking span rather than after </think>, so it
        # remains a deployable reasoning checkpoint under the frozen Qwen3 protocol.
        if not native and split in ("probe_train", "calibration"):
            span = max(1, end - start)
            position = min(end, max(start + 1, start + int(0.7 * span)))
            native = [{
                "feature_position": position,
                "prefix_position": max(0, position - 1),
                "event": "synthetic",
            }]
            trace["lynx_synthetic_no_cue_fallback"] = True
        else:
            trace["lynx_synthetic_no_cue_fallback"] = False
        for row in native:
            row["checkpoint_kind"] = (
                "lynx_synthetic_70pct"
                if row.get("event") == "synthetic"
                else "lynx_cue_before_token"
            )
        schedules = {
            "native": native,
            "paragraph": [
                {"feature_position": p, "prefix_position": p, "event": "paragraph", "checkpoint_kind": "paragraph_end"}
                for p in paragraph
            ],
        }
    elif method == "thought_calibration":
        sections = thought_sections(tokenizer, content, start, end)
        control_sections = sections
        paragraph_set = set(paragraph)
        native = []
        para = []
        for index, section in enumerate(sections):
            base = {
                "feature_position": int(section["checkpoint"]),
                "prefix_position": int(section["checkpoint"]),
                "feature_start": int(section["feature_start"]),
                "feature_end": int(section["feature_end"]),
                "section_index": index,
            }
            lowered = str(section["text"]).lower()
            if "wait" in lowered or "but" in lowered:
                native.append({**base, "checkpoint_kind": "thought_wait_or_but_section"})
            if int(section["checkpoint"]) in paragraph_set:
                para.append({**base, "checkpoint_kind": "paragraph_end"})
        schedules = {"native": native, "paragraph": para}
    else:
        raise ValueError(method)

    for schedule in schedules:
        dedup: dict[tuple[int, int], dict[str, Any]] = {}
        for row in schedules[schedule]:
            key = (int(row["feature_position"]), int(row["prefix_position"]))
            dedup.setdefault(key, row)
        schedules[schedule] = list(dedup.values())
    trace.update({
        "native_checkpoints": len(schedules["native"]),
        "paragraph_checkpoints": len(schedules["paragraph"]),
    })
    return schedules, trace, control_sections


def collect_one(
    source_path: Path,
    output_root: Path,
    config: dict[str, Any],
    dataset: str,
    method: str,
    fingerprint: str,
    model,
    tokenizer,
    model_audit: dict[str, Any],
    device: torch.device,
    gpu: int,
    worker: str,
    resume: bool,
) -> dict[str, Any]:
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    problem_id = str(source["problem_id"])
    split = str(source["split"])
    targets = {
        schedule: destination(output_root, dataset, method, schedule, split, problem_id)
        for schedule in SCHEDULES
    }
    if resume and all(valid(path, fingerprint, method, schedule, problem_id) for schedule, path in targets.items()):
        return {"status": "skipped", "problem_id": problem_id, "new_branches": 0}
    for schedule, path in targets.items():
        if path.exists() and not valid(path, fingerprint, method, schedule, problem_id):
            raise RuntimeError(f"refusing to overwrite incompatible artifact: {path}")
    pending_schedules = {
        schedule
        for schedule, path in targets.items()
        if not (resume and valid(path, fingerprint, method, schedule, problem_id))
    }

    content = list(source["dense"]["content_tokens"])
    dataset_config = config["datasets"][dataset]
    paragraph_root = ROOT / dataset_config["paragraph_cache_root"]
    frozen_paragraph = frozen_paragraph_positions(paragraph_root, split, problem_id)
    schedules, trajectory, control_sections = candidate_specs(
        tokenizer, content, method, config, frozen_paragraph, split
    )
    prompt_ids = tokenizer(source["prompt_text"], return_tensors="pt").input_ids.to(device)
    prompt_tokens = int(prompt_ids.shape[1])
    if prompt_tokens != int(source["prompt_tokens"]):
        raise ValueError(f"prompt retokenization mismatch: {source_path}")

    if method == "lynx":
        suffix = str(config["methods"][method]["force_answer_suffix"])
        maximum = int(config["methods"][method]["force_answer_max_new_tokens"])
        reuse: dict[int, dict[str, Any]] = {}
        hook_layers = [11, 17, 23]
    else:
        suffix = str(config["common_scope"]["force_answer_suffix"])
        maximum = int(config["common_scope"]["force_answer_max_new_tokens"])
        roots = [ROOT / value for value in dataset_config.get("reuse_cache_roots", [])]
        reuse = load_reuse_rows(roots, split, problem_id, source["prompt_text"], content)
        hook_layers = []
    suffix_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    all_specs = [
        {**row, "schedule": schedule}
        for schedule, rows in schedules.items()
        if schedule in pending_schedules
        for row in rows
    ]
    feature_at: dict[int, list[dict[str, Any]]] = defaultdict(list)
    branch_at: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in all_specs:
        feature_at[int(row["feature_position"])].append(row)
        branch_at[int(row["prefix_position"])].append(row)
    # Mean-step features need every paragraph endpoint as a forward control point so
    # the exact step token span remains available in teacher.last_hidden_state.
    section_control = {int(row["checkpoint"]) for row in control_sections}
    controls = sorted(set(feature_at) | set(branch_at) | section_control)
    if any(point < 0 or point > trajectory["reasoning_end"] for point in controls):
        raise ValueError(f"checkpoint outside reasoning range for {problem_id}")

    capture = CheckpointHiddenCapture(model, hook_layers) if hook_layers else None
    hidden_by_key: dict[tuple[str, int, int], torch.Tensor] = {}
    branch_by_prefix: dict[int, dict[str, Any]] = {}
    new_branches = reused_branches = 0
    branch_wall_ms = 0.0
    try:
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
            for point in controls:
                teacher = None
                if point > previous:
                    delta = torch.tensor([content[previous:point]], dtype=torch.long, device=device)
                    mask = torch.ones((1, prompt_tokens + point), dtype=torch.long, device=device)
                    if capture is not None and point in feature_at:
                        capture.begin()
                    teacher = model.model(
                        input_ids=delta,
                        attention_mask=mask,
                        past_key_values=base_cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    base_cache = teacher.past_key_values

                if point in feature_at:
                    if teacher is None:
                        raise RuntimeError("feature checkpoint cannot be at position zero")
                    if method == "lynx":
                        intermediate = capture.finish_cpu()
                        final = teacher.last_hidden_state[0, -1].detach().float().cpu().unsqueeze(0)
                        vector = torch.cat([intermediate, final], dim=0).to(torch.float16)
                        for spec in feature_at[point]:
                            key = (spec["schedule"], int(spec["feature_position"]), int(spec["prefix_position"]))
                            hidden_by_key[key] = vector
                    elif method == "learn_to_stop":
                        vector = teacher.last_hidden_state[0, -1].detach().float().cpu().unsqueeze(0).to(torch.float16)
                        for spec in feature_at[point]:
                            key = (spec["schedule"], int(spec["feature_position"]), int(spec["prefix_position"]))
                            hidden_by_key[key] = vector
                    else:
                        for spec in feature_at[point]:
                            absolute_start = int(spec["feature_start"])
                            absolute_end = int(spec["feature_end"])
                            local_start = absolute_start - previous
                            local_end = absolute_end - previous
                            if not (0 <= local_start < local_end <= teacher.last_hidden_state.shape[1]):
                                raise ValueError(
                                    f"thought span unavailable at {point}: abs=({absolute_start},{absolute_end}) previous={previous}"
                                )
                            vector = (
                                teacher.last_hidden_state[0, local_start:local_end]
                                .detach().float().mean(dim=0).cpu().unsqueeze(0).to(torch.float16)
                            )
                            key = (spec["schedule"], int(spec["feature_position"]), int(spec["prefix_position"]))
                            hidden_by_key[key] = vector

                if point in branch_at:
                    if point in reuse:
                        branch_by_prefix[point] = dict(reuse[point])
                        reused_branches += 1
                    else:
                        branch = greedy_branch(
                            model,
                            tokenizer,
                            base_cache,
                            prompt_tokens + point,
                            suffix_ids,
                            maximum,
                        )
                        branch_by_prefix[point] = branch_to_row(branch, dataset, source)
                        new_branches += 1
                        branch_wall_ms += float(branch["wall_ms"])
                if teacher is not None:
                    del teacher
                previous = point
    finally:
        if capture is not None:
            capture.close()

    dense_prediction = source["dense"]["prediction"]
    dense_success = bool(source["dense"]["success"])
    dense_tokens = int(source["dense"]["reasoning_tokens"])
    for schedule, specs in schedules.items():
        path = targets[schedule]
        if resume and valid(path, fingerprint, method, schedule, problem_id):
            continue
        rows: list[dict[str, Any]] = []
        vectors: list[torch.Tensor] = []
        for rank, spec in enumerate(specs):
            feature_position = int(spec["feature_position"])
            prefix_position = int(spec["prefix_position"])
            key = (schedule, feature_position, prefix_position)
            branch = dict(branch_by_prefix[prefix_position])
            current_prediction = branch.get("current_prediction")
            current_success = bool(branch.get("current_success", False))
            row = {
                **branch,
                **spec,
                "dataset": dataset,
                "split": split,
                "problem_id": problem_id,
                "rank": rank,
                "checkpoint": prefix_position,
                "feature_position": feature_position,
                "prefix_position": prefix_position,
                "gold_answer": source["gold_answer"],
                "dense_prediction": dense_prediction,
                "dense_success": dense_success,
                "dense_tokens": dense_tokens,
                "current_prediction": current_prediction,
                "current_success": current_success,
                "consistency": bool(
                    current_prediction is not None
                    and dense_prediction is not None
                    and current_prediction == dense_prediction
                ),
                "correction": bool((not current_success) and dense_success),
                "damage": bool(current_success and (not dense_success)),
                "forced_answer_decoding": "greedy_argmax",
                "forced_answer_do_sample": False,
                "stop_reasoning_tokens": prefix_position,
                "stop_total_tokens": prefix_position + int(suffix_ids.shape[1]) + int(branch.get("branch_tokens", 0)),
            }
            rows.append(row)
            vectors.append(hidden_by_key[key])
        layer_count = 4 if method == "lynx" else 1
        hidden = (
            torch.stack(vectors)
            if vectors
            else torch.empty((0, layer_count, int(model_audit["hidden_size"])), dtype=torch.float16)
        )
        artifact = {
            "schema_version": 1,
            "status": "complete",
            "protocol_id": config["protocol_id"],
            "protocol_fingerprint": fingerprint,
            "primary_replay_view_fingerprint": f"{fingerprint}:{schedule}",
            "dataset": dataset,
            "method": method,
            "schedule_variant": schedule,
            "split": split,
            "problem_id": problem_id,
            "rows": rows,
            "hidden": hidden,
            "capture_layers_one_based": (
                config["methods"][method]["feature_layers_one_based"]
                if method == "lynx" else [int(model_audit["layers"])]
            ),
            "feature_definition": config["methods"][method]["representation"],
            "record": source["record"],
            "gold_answer": source["gold_answer"],
            "prompt_text": source["prompt_text"],
            "prompt_tokens": prompt_tokens,
            "dense": source["dense"],
            "source_artifact": str(source_path.resolve()),
            "trajectory": trajectory,
            "method_config": config["methods"][method],
            "forced_answer_decoding": {
                "strategy": "greedy_argmax",
                "do_sample": False,
                "max_new_tokens": maximum,
                "suffix": suffix,
            },
            "model_audit": model_audit,
            "collection": {
                "worker": worker,
                "host": socket.gethostname(),
                "gpu": gpu,
                "device": torch.cuda.get_device_name(gpu),
                "new_branches": new_branches,
                "reused_branches": reused_branches,
                "branch_wall_ms": branch_wall_ms,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(artifact, path)
    return {
        "status": "completed",
        "problem_id": problem_id,
        "new_branches": new_branches,
        "reused_branches": reused_branches,
        "checkpoints": {name: len(rows) for name, rows in schedules.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
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
    fingerprint = canonical_fingerprint(config, args.dataset, args.method)
    output_root = ROOT / config["output_root"]
    source_root = ROOT / config["datasets"][args.dataset]["source_root"]
    splits = ("probe_train", "calibration", "heldout") if args.split == "all" else (args.split,)
    paths = sorted(path for split in splits for path in (source_root / split).glob("sample_*.pt"))
    assigned = [path for index, path in enumerate(paths) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        assigned = assigned[: args.limit]

    seed_everything(int(config["seed"]["global"]))
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    free, total = torch.cuda.mem_get_info(device)
    print(json.dumps({
        "status": "loading",
        "worker": args.worker_id,
        "dataset": args.dataset,
        "method": args.method,
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

    completed = skipped = failures = new_branches = reused_branches = 0
    started = time.time()
    for path in assigned:
        try:
            result = collect_one(
                path, output_root, config, args.dataset, args.method, fingerprint,
                model, tokenizer, audit, device, args.gpu, args.worker_id, args.resume,
            )
            completed += int(result["status"] == "completed")
            skipped += int(result["status"] == "skipped")
            new_branches += int(result.get("new_branches", 0))
            reused_branches += int(result.get("reused_branches", 0))
            print(json.dumps({
                "worker": args.worker_id,
                "completed": completed,
                "skipped": skipped,
                "failures": failures,
                **result,
            }), flush=True)
        except Exception as error:
            failures += 1
            print(json.dumps({
                "status": "error",
                "worker": args.worker_id,
                "source": str(path),
                "error_type": type(error).__name__,
                "error": str(error),
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
        "method": args.method,
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "completed": completed,
        "skipped": skipped,
        "failures": failures,
        "new_branches": new_branches,
        "reused_branches": reused_branches,
        "elapsed_seconds": time.time() - started,
        "protocol_fingerprint": fingerprint,
    }
    summary_path = output_root / "workers" / f"{args.worker_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(f".{summary_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

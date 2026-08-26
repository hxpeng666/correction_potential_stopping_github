#!/usr/bin/env python3
"""Reproduce the self-verification probe data construction with Qwen3-4B as judge.

The paper's chunking, answer/no-answer extraction, nearest-chunk merging, labels, and
representations are preserved.  Per the common experimental scope, Gemini is replaced
only as the extractor/judge by the same frozen Qwen3-4B.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import collect_literature_method_data_v1 as common

ROOT = common.ROOT

import torch

from src.final_paper_inference import atomic_torch_save, prediction_for, success_for
from src.mmlu_pro_protocol import parse_answer as parse_mmlu_pro_answer
from src.qwen3_reasoning import load_qwen3
from src.utils import load_yaml, seed_everything


EVALUATION_PROMPT = """Given several chunks of a reasoning trace, along with a ground-truth answer,
independently evaluate each chunk. If a chunk reaches a result at the end, return
the intermediate result; otherwise, return None if the chunk does not contain an
intermediate result (e.g., pure reflections).
Then, if an intermediate answer exists, compare it to the ground-truth answer. If
the intermediate result in the chunk equals the ground-truth answer, return True;
if the intermediate result in the chunk does not equal the ground-truth answer,
return False; if no intermediate answer exists, return None.
Output in JSON format:
[
{{"id": "1", "result": "6 + 9i" / null, "correctness": true / false / null}},
...
]
Input chunks:
{chunks}
Ground-truth answer: {answer}
"""


def path_chunks(
    tokenizer, content: list[int], start: int, end: int, keywords: list[str]
) -> list[dict[str, Any]]:
    sections = common.thought_sections(tokenizer, content, start, end)
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for section in sections:
        lowered = str(section["text"]).lower()
        begins_path = any(keyword.lower() in lowered for keyword in keywords)
        if begins_path and current:
            chunks.append(current)
            current = []
        current.append(section)
    if current:
        chunks.append(current)
    result = []
    for index, values in enumerate(chunks):
        result.append({
            "id": str(index + 1),
            "start": int(values[0]["feature_start"]),
            "end": int(values[-1]["feature_end"]),
            "boundary_end": int(values[-1]["checkpoint"]),
            "text": "\n\n".join(str(value["text"]).strip() for value in values).strip(),
            "section_indices": list(range(
                int(sections.index(values[0])), int(sections.index(values[-1])) + 1
            )),
        })
    return result


def labeler_prompt(chunks: list[dict[str, Any]], answer: Any) -> str:
    rendered = "\n\n".join(
        f"## Chunk {value['id']}\n{value['text']}" for value in chunks
    )
    return EVALUATION_PROMPT.format(chunks=rendered, answer=answer)


@torch.inference_mode()
def greedy_fresh(model, tokenizer, prompt: str, maximum: int = 768) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(model.device)
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        max_new_tokens=maximum,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0, inputs.shape[1] :], skip_special_tokens=True)


def parse_json_rows(text: str, expected: int) -> list[dict[str, Any]] | None:
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        # Repair only the paper's Python-like None/True/False spellings.
        repaired = re.sub(r"\bNone\b", "null", text[start : end + 1])
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        try:
            value = json.loads(repaired)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list):
        return None
    by_id = {str(row.get("id")): row for row in value if isinstance(row, dict)}
    result = []
    for index in range(1, expected + 1):
        row = by_id.get(str(index))
        if row is None:
            return None
        result.append({
            "id": str(index),
            "result": row.get("result"),
            "correctness": row.get("correctness"),
        })
    return result


def judge_chunks(model, tokenizer, chunks: list[dict[str, Any]], answer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not chunks:
        return [], {"mode": "empty", "raw": ""}
    prompt = labeler_prompt(chunks, answer)
    raw = greedy_fresh(model, tokenizer, prompt)
    parsed = parse_json_rows(raw, len(chunks))
    mode = "batched"
    fallback_raw = []
    if parsed is None:
        mode = "per_chunk_fallback"
        parsed = []
        for chunk in chunks:
            local = [{**chunk, "id": "1"}]
            local_raw = greedy_fresh(model, tokenizer, labeler_prompt(local, answer), maximum=384)
            local_parsed = parse_json_rows(local_raw, 1)
            fallback_raw.append(local_raw)
            if local_parsed is None:
                parsed.append({"id": chunk["id"], "result": None, "correctness": None})
            else:
                parsed.append({"id": chunk["id"], **{k: local_parsed[0][k] for k in ("result", "correctness")}})
    return parsed, {"mode": mode, "raw": raw, "fallback_raw": fallback_raw}


def merge_answerless(
    chunks: list[dict[str, Any]], judgments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    answer_indices = [
        index for index, row in enumerate(judgments)
        if row.get("result") is not None and row.get("correctness") is not None
    ]
    if not answer_indices:
        return []
    assignments: dict[int, list[int]] = {index: [] for index in answer_indices}
    for index in range(len(chunks)):
        closest = min(answer_indices, key=lambda candidate: (abs(candidate - index), candidate))
        assignments[closest].append(index)
    result = []
    for answer_index in answer_indices:
        members = assignments[answer_index]
        first, last = min(members), max(members)
        judgment = judgments[answer_index]
        result.append({
            "start": int(chunks[first]["start"]),
            "end": int(chunks[last]["end"]),
            "boundary_end": int(chunks[last]["boundary_end"]),
            "answer_chunk_index": answer_index,
            "member_chunk_indices": members,
            "intermediate_answer": judgment["result"],
            "probe_label": bool(judgment["correctness"]),
            "evaluation_success": bool(judgment["evaluation_success"]),
        })
    return sorted(result, key=lambda row: row["end"])


def attach_frozen_evaluation_correctness(
    judgments: list[dict[str, Any]], dataset: str, source: dict[str, Any]
) -> list[dict[str, Any]]:
    """Keep 4B judge labels intact; attach parser-only correctness for final metrics."""
    result = []
    for row in judgments:
        updated = dict(row)
        answer = row.get("result")
        if answer is None:
            updated["evaluation_success"] = False
        elif dataset == "mmlu_pro":
            prediction = parse_mmlu_pro_answer(
                f"\\boxed{{{answer}}}", len(source["record"]["choices"])
            )
            if prediction is None:
                # The semantic extractor may return the option text instead of its letter.
                normalized = re.sub(r"\s+", " ", str(answer)).strip().strip(".()[]{}\"").lower()
                matches = []
                for index, choice in enumerate(source["record"]["choices"]):
                    choice_normalized = re.sub(r"\s+", " ", str(choice)).strip().strip(".()[]{}\"").lower()
                    if normalized == choice_normalized or (
                        len(choice_normalized) >= 4 and choice_normalized in normalized
                    ):
                        matches.append(chr(ord("A") + index))
                prediction = matches[0] if len(set(matches)) == 1 else None
            if prediction is None:
                updated["evaluation_success"] = False
                updated["evaluation_correctness_source"] = "frozen_parser_unresolved"
            else:
                updated["evaluation_success"] = bool(prediction == source["gold_answer"])
                updated["evaluation_correctness_source"] = "frozen_choice_match"
        else:
            prediction = prediction_for("gsm8k", f"\\boxed{{{answer}}}")
            updated["evaluation_success"] = bool(
                success_for("gsm8k", source["gold_answer"], prediction)
            )
            updated["evaluation_correctness_source"] = "frozen_numeric_parser"
        result.append(updated)
    return result


def capture_final_hidden(
    model,
    prompt_ids: torch.Tensor,
    content: list[int],
    positions: list[int],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    result = {}
    with torch.inference_mode():
        prefill = model.model(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            use_cache=True,
            return_dict=True,
        )
        cache = prefill.past_key_values
        del prefill
        previous = 0
        for position in sorted(set(positions)):
            delta = torch.tensor([content[previous:position]], dtype=torch.long, device=device)
            output = model.model(
                input_ids=delta,
                attention_mask=torch.ones((1, prompt_ids.shape[1] + position), dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = output.past_key_values
            result[position] = output.last_hidden_state[0, -1].detach().float().cpu().unsqueeze(0).to(torch.float16)
            del output
            previous = position
    return result


def valid(path: Path, fingerprint: str, schedule: str, problem_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
        return (
            value.get("status") == "complete"
            and value.get("protocol_fingerprint") == fingerprint
            and value.get("method") == "self_verification"
            and value.get("schedule_variant") == schedule
            and str(value.get("problem_id")) == problem_id
        )
    except Exception:
        return False


def collect_one(
    source_path: Path,
    output_root: Path,
    config: dict[str, Any],
    dataset: str,
    fingerprint: str,
    model,
    tokenizer,
    audit: dict[str, Any],
    device: torch.device,
    gpu: int,
    worker: str,
    resume: bool,
) -> dict[str, Any]:
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    problem_id, split = str(source["problem_id"]), str(source["split"])
    targets = {
        schedule: output_root / dataset / "self_verification" / schedule / "cache" / split / f"sample_{problem_id}.pt"
        for schedule in common.SCHEDULES
    }
    if resume and all(valid(path, fingerprint, schedule, problem_id) for schedule, path in targets.items()):
        return {"status": "skipped", "problem_id": problem_id}
    for schedule, path in targets.items():
        if path.exists() and not valid(path, fingerprint, schedule, problem_id):
            raise RuntimeError(f"refusing to overwrite incompatible artifact: {path}")

    content = list(source["dense"]["content_tokens"])
    start, end, trajectory = common.reasoning_range(tokenizer, content)
    method_config = config["methods"]["self_verification"]
    chunks = path_chunks(tokenizer, content, start, end, [str(x) for x in method_config["path_keywords"]])
    judgments, judge_audit = judge_chunks(model, tokenizer, chunks, source["gold_answer"])
    judgments = attach_frozen_evaluation_correctness(judgments, dataset, source)
    merged = merge_answerless(chunks, judgments)
    dataset_config = config["datasets"][dataset]
    frozen_paragraph = common.frozen_paragraph_positions(
        ROOT / dataset_config["paragraph_cache_root"], split, problem_id
    ) or []
    reuse = common.load_reuse_rows(
        [ROOT / value for value in dataset_config.get("reuse_cache_roots", [])],
        split,
        problem_id,
        source["prompt_text"],
        content,
    )
    native_specs = []
    paragraph_specs = []
    for chunk_index, chunk in enumerate(merged):
        native_specs.append({
            "checkpoint": int(chunk["end"]),
            "probe_label": bool(chunk["probe_label"]),
            "evaluation_success": bool(chunk["evaluation_success"]),
            "intermediate_answer": chunk["intermediate_answer"],
            "merged_chunk_index": chunk_index,
            "checkpoint_kind": "reasoning_path_chunk_end",
        })
        for checkpoint in frozen_paragraph:
            if int(chunk["start"]) <= checkpoint <= int(chunk["boundary_end"]):
                paragraph_specs.append({
                    "checkpoint": int(checkpoint),
                    "probe_label": bool(chunk["probe_label"]),
                    "upcoming_evaluation_success": bool(chunk["evaluation_success"]),
                    "upcoming_intermediate_answer": chunk["intermediate_answer"],
                    "merged_chunk_index": chunk_index,
                    "checkpoint_kind": "paragraph_end_upcoming_answer_label",
                })
    schedules = {"native": native_specs, "paragraph": paragraph_specs}
    positions = [int(row["checkpoint"]) for rows in schedules.values() for row in rows]
    prompt_ids = tokenizer(source["prompt_text"], return_tensors="pt").input_ids.to(device)
    hidden = capture_final_hidden(model, prompt_ids, content, positions, device) if positions else {}

    # Paragraph deployment uses the common greedy forced answer at the current prefix.
    suffix = str(config["common_scope"]["force_answer_suffix"])
    suffix_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    missing = sorted(set(row["checkpoint"] for row in paragraph_specs if row["checkpoint"] not in reuse))
    generated: dict[int, dict[str, Any]] = {}
    if missing:
        with torch.inference_mode():
            prefill = model.model(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), use_cache=True, return_dict=True)
            cache = prefill.past_key_values
            del prefill
            previous = 0
            for point in missing:
                delta = torch.tensor([content[previous:point]], dtype=torch.long, device=device)
                teacher = model.model(
                    input_ids=delta,
                    attention_mask=torch.ones((1, prompt_ids.shape[1] + point), dtype=torch.long, device=device),
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = teacher.past_key_values
                branch = common.greedy_branch(
                    model, tokenizer, cache, int(prompt_ids.shape[1]) + point, suffix_ids,
                    int(config["common_scope"]["force_answer_max_new_tokens"]),
                )
                generated[point] = common.branch_to_row(branch, dataset, source)
                del teacher
                previous = point

    for schedule, specs in schedules.items():
        rows, vectors = [], []
        for rank, spec in enumerate(sorted(specs, key=lambda row: row["checkpoint"])):
            checkpoint = int(spec["checkpoint"])
            if schedule == "native":
                current_prediction = spec["intermediate_answer"]
                current_success = bool(spec["evaluation_success"])
                branch = {
                    "current_prediction": current_prediction,
                    "current_success": current_success,
                    "branch_tokens": 0,
                    "branch_text": str(current_prediction),
                    "branch_generated_text": str(current_prediction),
                    "branch_collection_wall_ms": 0.0,
                }
                total_tokens = checkpoint
            else:
                branch = dict(reuse.get(checkpoint) or generated[checkpoint])
                current_prediction = branch.get("current_prediction")
                current_success = bool(branch.get("current_success", False))
                total_tokens = checkpoint + int(suffix_ids.shape[1]) + int(branch.get("branch_tokens", 0))
            rows.append({
                **branch,
                **spec,
                "dataset": dataset,
                "split": split,
                "problem_id": problem_id,
                "rank": rank,
                "dense_prediction": source["dense"]["prediction"],
                "dense_success": bool(source["dense"]["success"]),
                "dense_tokens": int(source["dense"]["reasoning_tokens"]),
                "current_prediction": current_prediction,
                "current_success": current_success,
                "stop_reasoning_tokens": checkpoint,
                "stop_total_tokens": total_tokens,
            })
            vectors.append(hidden[checkpoint])
        hidden_tensor = torch.stack(vectors) if vectors else torch.empty((0, 1, int(audit["hidden_size"])), dtype=torch.float16)
        artifact = {
            "schema_version": 1,
            "status": "complete",
            "protocol_id": config["protocol_id"],
            "protocol_fingerprint": fingerprint,
            "dataset": dataset,
            "method": "self_verification",
            "schedule_variant": schedule,
            "split": split,
            "problem_id": problem_id,
            "rows": rows,
            "hidden": hidden_tensor,
            "capture_layers_one_based": [int(audit["layers"])],
            "feature_definition": method_config["representation"],
            "record": source["record"],
            "gold_answer": source["gold_answer"],
            "prompt_text": source["prompt_text"],
            "prompt_tokens": int(prompt_ids.shape[1]),
            "dense": source["dense"],
            "source_artifact": str(source_path.resolve()),
            "trajectory": trajectory,
            "preliminary_chunks": chunks,
            "chunk_judgments": judgments,
            "merged_chunks": merged,
            "labeler_audit": {"model": "same_frozen_qwen3_4b", **judge_audit},
            "method_config": method_config,
            "model_audit": audit,
            "collection": {
                "worker": worker,
                "host": socket.gethostname(),
                "gpu": gpu,
                "device": torch.cuda.get_device_name(gpu),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        targets[schedule].parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(artifact, targets[schedule])
    return {
        "status": "completed",
        "problem_id": problem_id,
        "preliminary_chunks": len(chunks),
        "answer_chunks": len(merged),
        "native_checkpoints": len(native_specs),
        "paragraph_checkpoints": len(paragraph_specs),
        "judge_mode": judge_audit["mode"],
    }


def main() -> None:
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
    fingerprint = common.canonical_fingerprint(config, args.dataset, "self_verification")
    output_root = ROOT / config["output_root"]
    source_root = ROOT / config["datasets"][args.dataset]["source_root"]
    splits = common.SPLITS if hasattr(common, "SPLITS") else (("probe_train", "calibration", "heldout") if args.split == "all" else (args.split,))
    if args.split != "all":
        splits = (args.split,)
    paths = sorted(path for split in splits for path in (source_root / split).glob("sample_*.pt"))
    assigned = [path for index, path in enumerate(paths) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        assigned = assigned[: args.limit]
    seed_everything(int(config["seed"]["global"]))
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(json.dumps({"status": "loading", "worker": args.worker_id, "dataset": args.dataset, "gpu": args.gpu, "assigned": len(assigned)}), flush=True)
    model, tokenizer, audit = load_qwen3(
        ROOT / config["model"]["local_path"], device, config["model"]["dtype"], config["model"]["attention_backend"]
    )
    completed = skipped = failures = 0
    started = time.time()
    for path in assigned:
        try:
            result = collect_one(path, output_root, config, args.dataset, fingerprint, model, tokenizer, audit, device, args.gpu, args.worker_id, args.resume)
            completed += int(result["status"] == "completed")
            skipped += int(result["status"] == "skipped")
            print(json.dumps({"worker": args.worker_id, "completed": completed, "skipped": skipped, "failures": failures, **result}), flush=True)
        except Exception as error:
            failures += 1
            print(json.dumps({"status": "error", "worker": args.worker_id, "source": str(path), "error_type": type(error).__name__, "error": str(error)}), flush=True)
            if isinstance(error, torch.cuda.OutOfMemoryError):
                raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    summary = {
        "status": "complete" if failures == 0 else "failed",
        "worker": args.worker_id,
        "dataset": args.dataset,
        "method": "self_verification",
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "completed": completed,
        "skipped": skipped,
        "failures": failures,
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

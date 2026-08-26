#!/usr/bin/env python3
"""Run paired/interleaved actual online timing for Dense and frozen workpoints."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

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
    stable_example_seed,
    success_for,
)
from src.final_paper_online import (
    generate_online_dense,
    generate_online_stopped,
    load_probe_bundle,
)
from src.final_paper_probe import transition_name
from src.qwen3_reasoning import load_qwen3
from src.utils import atomic_json, gpu_telemetry, load_yaml, seed_everything


def select_records(
    records: list[dict[str, Any]],
    problem_ids_file: Path | None,
    maximum: int | None,
) -> list[dict[str, Any]]:
    if problem_ids_file is not None:
        path = problem_ids_file if problem_ids_file.is_absolute() else ROOT / problem_ids_file
        selected = [str(value) for value in json.loads(path.read_text(encoding="utf-8"))]
        by_id = {str(row["problem_id"]): row for row in records}
        missing = [value for value in selected if value not in by_id]
        if missing:
            raise KeyError(f"selected IDs absent from split: {missing}")
        records = [by_id[value] for value in selected]
    if maximum is not None:
        records = records[:maximum]
    return records


def normalize_dense_result(
    dataset: str,
    gold: str | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    prediction = prediction_for(dataset, result["text"])
    return {
        **result,
        "answer_text": result["text"],
        "reasoning_text": result["text"],
        "prediction": prediction,
        "success": success_for(dataset, gold, prediction),
        "coverage": False,
    }


def normalize_strategy_result(
    dataset: str,
    gold: str | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    prediction = prediction_for(dataset, result["answer_text"])
    return {
        **result,
        "prediction": prediction,
        "success": success_for(dataset, gold, prediction),
        "coverage": bool(result["stopped"]),
    }


def summarize(paths: list[Path], workpoints: list[str]) -> dict[str, Any]:
    observations: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ["dense", *workpoints]
    }
    for path in paths:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        for name, repetitions in artifact["runs"].items():
            observations[name].extend(repetitions)
    dense_by_key = {
        (row["problem_id"], int(row["repeat"])): row
        for row in observations["dense"]
    }
    output = {}
    for name, rows in observations.items():
        wall = np.asarray([row["wall_ms"] for row in rows], dtype=float)
        tokens = np.asarray([row["total_generated_tokens"] for row in rows], dtype=float)
        success = np.asarray([row["success"] for row in rows], dtype=float)
        if name == "dense":
            paired_dense_wall = wall
            paired_dense_tokens = tokens
            transitions = {key: 0 for key in ("W_to_C", "C_to_W", "W_to_W", "C_to_C")}
            coverage = 0.0
            fallback = 0.0
            overhead = 0.0
        else:
            paired = [
                dense_by_key[(row["problem_id"], int(row["repeat"]))]
                for row in rows
            ]
            paired_dense_wall = np.asarray([row["wall_ms"] for row in paired], dtype=float)
            paired_dense_tokens = np.asarray(
                [row["total_generated_tokens"] for row in paired], dtype=float
            )
            transitions = {key: 0 for key in ("W_to_C", "C_to_W", "W_to_W", "C_to_C")}
            for row, dense in zip(rows, paired):
                if row["stopped"]:
                    transitions[transition_name(
                        bool(row["success"]), bool(dense["success"])
                    )] += 1
            coverage = float(np.mean([row["stopped"] for row in rows]))
            fallback = float(np.mean([row["fallback"] for row in rows]))
            overhead = float(np.mean([row["stopper_overhead_ms"] for row in rows]))
        output[name] = {
            "observations": len(rows),
            "problems": len({row["problem_id"] for row in rows}),
            "accuracy": float(success.mean()),
            "mean_latency_ms": float(wall.mean()),
            "median_latency_ms": float(np.median(wall)),
            "p95_latency_ms": float(np.percentile(wall, 95)),
            "mean_latency_reduction": float(1.0 - wall.mean() / paired_dense_wall.mean()),
            "p95_latency_reduction": float(
                1.0 - np.percentile(wall, 95) / np.percentile(paired_dense_wall, 95)
            ),
            "mean_generated_tokens": float(tokens.mean()),
            "token_reduction": float(1.0 - tokens.mean() / paired_dense_tokens.mean()),
            "coverage": coverage,
            "fallback_rate": fallback,
            "mean_stopper_overhead_ms": overhead,
            "transitions": transitions,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--split", choices=("probe_train", "calibration", "heldout"), default="heldout")
    parser.add_argument("--problem-ids-file", type=Path)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--workpoints", nargs="+", default=["strict", "balanced", "aggressive"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

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
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("base model is not frozen")
    probe_path = args.probe if args.probe.is_absolute() else ROOT / args.probe
    probe_bundle = load_probe_bundle(probe_path, device)
    if probe_bundle["run_spec"]["method"] != "correction":
        raise ValueError("final online workpoints require correction-potential probe")
    if probe_bundle["run_spec"]["schedule"] != "sentence":
        raise ValueError("final online deployment requires sentence schedule")
    workpoint_specs = {}
    for name in args.workpoints:
        if name not in probe_bundle["online_workpoints"]:
            raise KeyError(f"probe has no frozen workpoint {name}")
        value = probe_bundle["online_workpoints"][name]
        workpoint_specs[name] = {
            "threshold": float(value["calibration"]["threshold"]),
            "direction": str(probe_bundle["run_spec"]["stop_direction"]),
            "family": value["family"],
            "key": value["key"],
        }

    prepared = ROOT / config["dataset"]["prepared_root"]
    records = select_records(
        read_jsonl(prepared / f"{args.split}.jsonl"),
        args.problem_ids_file,
        args.num_samples,
    )
    demonstrations = (
        demonstrations_by_subject(prepared / "demonstrations.jsonl")
        if args.dataset == "mmlu"
        else None
    )
    dense_generation = resolved_generation(
        config, int(config["generation"]["dense_max_new_tokens"])
    )
    branch_generation = resolved_generation(
        config, int(config["generation"]["force_answer_max_new_tokens"])
    )
    suffix_ids = tokenizer(
        config["generation"]["force_answer_suffix"],
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    raw_output = output / "raw"
    raw_output.mkdir(parents=True, exist_ok=True)

    warmup_ids = []
    for record in records[: min(args.warmup, len(records))]:
        messages = prompt_messages(args.dataset, record, config, demonstrations)
        prompt = render_prompt(
            tokenizer, messages, enable_thinking=bool(config["model"]["enable_thinking"])
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            generate_online_dense(
                model,
                tokenizer,
                encoded.input_ids.to(device),
                encoded.attention_mask.to(device),
                generation=dense_generation,
                seed=stable_example_seed(
                    args.seed, str(record["problem_id"]), "online_warmup"
                ),
            )
        warmup_ids.append(str(record["problem_id"]))
        print(json.dumps({
            "phase": "warmup",
            "completed": len(warmup_ids),
            "required": min(args.warmup, len(records)),
            "problem_id": record["problem_id"],
        }), flush=True)

    telemetry_start = gpu_telemetry(args.gpu)
    method_names = ["dense", *args.workpoints]
    completed = 0
    skipped = 0
    for example_index, record in enumerate(records):
        problem_id = str(record["problem_id"])
        destination = raw_output / f"sample_{problem_id}.pt"
        if args.resume and artifact_complete(destination, problem_id):
            skipped += 1
            continue
        if destination.exists():
            raise RuntimeError(f"refusing to overwrite incomplete online artifact: {destination}")
        messages = prompt_messages(args.dataset, record, config, demonstrations)
        prompt = render_prompt(
            tokenizer, messages, enable_thinking=bool(config["model"]["enable_thinking"])
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        gold = gold_for(args.dataset, record)
        runs: dict[str, list[dict[str, Any]]] = {name: [] for name in method_names}
        execution_order = []
        for repeat in range(args.repeats):
            rotation = (example_index + repeat) % len(method_names)
            order = method_names[rotation:] + method_names[:rotation]
            execution_order.append(order)
            for order_index, name in enumerate(order):
                dense_seed = stable_example_seed(args.seed, problem_id, "dense")
                with torch.inference_mode():
                    if name == "dense":
                        result = normalize_dense_result(
                            args.dataset,
                            gold,
                            generate_online_dense(
                                model,
                                tokenizer,
                                input_ids,
                                attention_mask,
                                generation=dense_generation,
                                seed=dense_seed,
                            ),
                        )
                    else:
                        spec = workpoint_specs[name]
                        result = normalize_strategy_result(
                            args.dataset,
                            gold,
                            generate_online_stopped(
                                model,
                                tokenizer,
                                input_ids,
                                attention_mask,
                                dense_generation=dense_generation,
                                branch_generation=branch_generation,
                                suffix_ids=suffix_ids,
                                probe_bundle=probe_bundle,
                                threshold=spec["threshold"],
                                direction=spec["direction"],
                                checkpoint_protocol=config["checkpoint_protocol"],
                                dense_seed=dense_seed,
                                branch_seed_for_checkpoint=lambda checkpoint, pid=problem_id: stable_example_seed(
                                    args.seed, pid, f"forced_{checkpoint}"
                                ),
                            ),
                        )
                result.update({
                    "problem_id": problem_id,
                    "repeat": repeat,
                    "order_index": order_index,
                    "method": name,
                })
                runs[name].append(result)
        artifact = {
            "schema_version": 1,
            "status": "complete",
            "dataset": args.dataset,
            "split": args.split,
            "problem_id": problem_id,
            "record": record,
            "gold_answer": gold,
            "prompt_tokens": int(input_ids.shape[1]),
            "workpoint_specs": workpoint_specs,
            "execution_order": execution_order,
            "runs": runs,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_torch_save(artifact, destination)
        completed += 1
        print(json.dumps({
            "phase": "timed",
            "completed_now": completed,
            "skipped": skipped,
            "problem_id": problem_id,
            "latency_ms": {
                name: [round(row["wall_ms"], 3) for row in values]
                for name, values in runs.items()
            },
        }), flush=True)

    paths = sorted(raw_output.glob("sample_*.pt"))
    expected_ids = {str(record["problem_id"]) for record in records}
    found_ids = {
        str(torch.load(path, map_location="cpu", weights_only=False)["problem_id"])
        for path in paths
    }
    missing = sorted(expected_ids - found_ids)
    if missing:
        raise RuntimeError(f"online timing missing samples: {missing[:10]}")
    payload = {
        "status": "complete",
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "repeats": args.repeats,
        "warmup_examples": len(warmup_ids),
        "warmup_problem_ids": warmup_ids,
        "paired_interleaved": True,
        "batch_size": 1,
        "concurrent_requests": 0,
        "attention_backend": config["model"]["attention_backend"],
        "dtype": config["model"]["dtype"],
        "cuda_graph": False,
        "model": model_audit,
        "probe": str(probe_path),
        "workpoint_specs": workpoint_specs,
        "gpu_telemetry_start": telemetry_start,
        "gpu_telemetry_end": gpu_telemetry(args.gpu),
        "summary": summarize(
            [path for path in paths if torch.load(
                path, map_location="cpu", weights_only=False
            )["problem_id"] in expected_ids],
            args.workpoints,
        ),
    }
    atomic_json(payload, output / "online_summary.json")
    atomic_json(
        {
            "status": "complete",
            "samples": len(records),
            "repeats": args.repeats,
            "artifacts": ["online_summary.json", "raw/"],
        },
        output / "phase.complete",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

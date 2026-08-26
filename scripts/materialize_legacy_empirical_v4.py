#!/usr/bin/env python3
"""Apply the frozen A100 cost model to immutable legacy BF16 semantic artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_paper_inference import atomic_torch_save
from src.final_paper_protocol import canonical_fingerprint
from src.utils import atomic_json, load_yaml


def design(prompt: float, reasoning: float) -> np.ndarray:
    p = float(prompt) / 4096.0
    r = float(reasoning) / 4096.0
    return np.asarray([1.0, p, p*p, p*p*p, r, p*r, r*r, p*p*r, p*r*r, r*r*r], dtype=np.float64)


def predict(coefficients: np.ndarray, prompt: float, reasoning: float) -> float:
    return max(float(design(prompt, reasoning) @ coefficients), 0.001)


def complete(path: Path, problem_id: str, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    value = torch.load(path, map_location="cpu", weights_only=False)
    return value.get("status") == "complete" and value.get("problem_id") == problem_id and value.get("legacy_view_fingerprint") == fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("probe_train", "calibration", "heldout"), required=True)
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cost-model", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    config_fingerprint = canonical_fingerprint(config)
    cost = json.loads(args.cost_model.read_text(encoding="utf-8"))
    if cost.get("status") != "frozen" or not cost["validation_gate"]["passed"]:
        raise ValueError("A100 cost model is not frozen/passed")
    coefficients = np.asarray(cost["unified_cumulative_prefix_model"]["coefficients"], dtype=np.float64)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["model"]["local_path"], local_files_only=True, trust_remote_code=False)
    suffix_tokens = len(tokenizer(config["generation"]["force_answer_suffix"], add_special_tokens=False).input_ids)
    view_fingerprint = canonical_fingerprint({
        "protocol": "legacy_empirical_v4",
        "config_fingerprint": config_fingerprint,
        "cost_timing_fingerprint": cost["timing_selection_fingerprint"],
        "formula": "prefill + interpolated-prefix + branch; no probe-check overhead",
    })
    dense_output = args.output_root / "dense" / args.split
    checkpoint_output = args.output_root / "checkpoints" / args.split
    dense_output.mkdir(parents=True, exist_ok=True)
    checkpoint_output.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = sorted((args.checkpoint_root / args.split).glob("sample_*.pt"))
    completed_now = skipped = 0
    for checkpoint_path in checkpoint_paths:
        checkpoint_artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        problem_id = str(checkpoint_artifact["problem_id"])
        dense_path = args.dense_root / args.split / f"sample_{problem_id}.pt"
        dense_artifact = torch.load(dense_path, map_location="cpu", weights_only=False)
        if dense_artifact.get("config_fingerprint") != config_fingerprint or checkpoint_artifact.get("config_fingerprint") != config_fingerprint:
            raise ValueError(f"config fingerprint mismatch for {problem_id}")
        if dense_artifact.get("dtype") != "bfloat16" or checkpoint_artifact.get("dtype") != "bfloat16":
            raise ValueError(f"non-BF16 artifact in legacy-v4: {problem_id}")
        dense_destination = dense_output / dense_path.name
        checkpoint_destination = checkpoint_output / checkpoint_path.name
        if args.resume and complete(dense_destination, problem_id, view_fingerprint) and complete(checkpoint_destination, problem_id, view_fingerprint):
            skipped += 1
            continue
        prompt_tokens = int(dense_artifact["prompt_tokens"])
        dense_steps = int(dense_artifact["dense"]["reasoning_tokens"])
        dense_cost = predict(coefficients, prompt_tokens, dense_steps)
        dense_view = dict(dense_artifact)
        dense_view["legacy_view_fingerprint"] = view_fingerprint
        dense_view["latency_label"] = "A100 single-request replay-estimated latency"
        dense_view["dense"] = dict(dense_artifact["dense"])
        dense_view["dense"]["wall_ms"] = dense_cost
        dense_view["dense"]["measured_collection_wall_ms"] = dense_artifact["dense"].get("wall_ms")
        direct = dict(dense_artifact["direct"])
        direct["measured_collection_wall_ms"] = direct.get("wall_ms")
        direct["wall_ms"] = predict(coefficients, int(direct["prompt_tokens"]), int(direct["generated_tokens"]))
        dense_view["direct"] = direct
        atomic_torch_save(dense_view, dense_destination)
        prefill_cost = predict(coefficients, prompt_tokens, 0)
        rows = []
        for source_row in checkpoint_artifact["rows"]:
            row = dict(source_row)
            checkpoint = int(row["checkpoint"])
            prefix_total = predict(coefficients, prompt_tokens, checkpoint)
            full_stop = predict(coefficients, prompt_tokens, checkpoint + suffix_tokens + int(row["branch_tokens"]))
            row["dense_wall_ms"] = dense_cost
            row["dense_prefill_cuda_ms"] = prefill_cost
            row["prefix_decode_cuda_ms"] = max(prefix_total - prefill_cost, 0.0)
            row["branch_wall_ms"] = max(full_stop - prefix_total, 0.0)
            row["latency_label"] = "A100 single-request replay-estimated latency"
            rows.append(row)
        checkpoint_view = dict(checkpoint_artifact)
        checkpoint_view["source_dense_artifact"] = str(dense_destination.resolve())
        checkpoint_view["rows"] = rows
        checkpoint_view["legacy_view_fingerprint"] = view_fingerprint
        checkpoint_view["latency_label"] = "A100 single-request replay-estimated latency"
        checkpoint_view["checkpoint_probe_overhead_ms"] = 0.0
        atomic_torch_save(checkpoint_view, checkpoint_destination)
        completed_now += 1
    summary = {
        "status": "complete",
        "dataset": args.dataset,
        "split": args.split,
        "visible": len(checkpoint_paths),
        "completed_now": completed_now,
        "skipped": skipped,
        "view_fingerprint": view_fingerprint,
        "latency_label": "A100 single-request replay-estimated latency",
        "checkpoint_probe_overhead_ms": 0.0,
    }
    atomic_json(summary, args.output_root / f"materialize_{args.dataset}_{args.split}.complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

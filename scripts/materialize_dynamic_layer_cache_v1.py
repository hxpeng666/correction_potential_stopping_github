#!/usr/bin/env python3
"""Materialize layer 8/20/35 hidden states on immutable cached Dense prefixes.

This utility never resamples Dense reasoning or forced-answer branches.  It
teacher-forces the exact cached token prefix through the same frozen model and
writes a separate replay view containing three decoder-layer readouts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.final_paper_inference import PositionHiddenCapture, atomic_torch_save
from src.final_paper_protocol import canonical_fingerprint
from src.qwen3_reasoning import load_qwen3
from src.utils import atomic_json


CAPTURE_LAYERS = [8, 20, 35]


def source_manifest(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "problem_id": str(artifact.get("problem_id")),
        "protocol_fingerprint": artifact.get("protocol_fingerprint"),
        "primary_replay_view_fingerprint": artifact.get(
            "primary_replay_view_fingerprint"
        ),
    }


def complete_destination(path: Path, problem_id: str, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    value = torch.load(path, map_location="cpu", weights_only=False)
    return (
        value.get("status") == "complete"
        and str(value.get("problem_id")) == problem_id
        and value.get("capture_layers") == CAPTURE_LAYERS
        and value.get("layer_ablation_capture_fingerprint") == fingerprint
    )


def parity_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    candidate = candidate.float()
    if reference.numel() == 0:
        return {
            "rows": 0,
            "min_cosine": 1.0,
            "mean_cosine": 1.0,
            "max_relative_l2": 0.0,
            "mean_relative_l2": 0.0,
            "max_absolute_difference": 0.0,
        }
    cosine = torch.nn.functional.cosine_similarity(reference, candidate, dim=-1)
    relative = torch.linalg.vector_norm(candidate - reference, dim=-1) / torch.linalg.vector_norm(
        reference, dim=-1
    ).clamp_min(1e-12)
    return {
        "rows": int(reference.shape[0]),
        "min_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
        "max_relative_l2": float(relative.max()),
        "mean_relative_l2": float(relative.mean()),
        "max_absolute_difference": float((candidate - reference).abs().max()),
    }


def validate_source(path: Path, artifact: dict[str, Any]) -> tuple[list[dict[str, Any]], torch.Tensor]:
    if artifact.get("status") != "complete":
        raise ValueError(f"incomplete source: {path}")
    if artifact.get("dtype") != "float16":
        raise ValueError(f"dtype mismatch in {path}: {artifact.get('dtype')}")
    if artifact.get("capture_layers") != [20]:
        raise ValueError(f"expected layer-20 source in {path}")
    rows = artifact.get("rows", [])
    hidden = artifact.get("hidden")
    if not torch.is_tensor(hidden) or hidden.ndim != 3:
        raise ValueError(f"invalid hidden tensor in {path}")
    if hidden.shape[0] != len(rows) or hidden.shape[1] != 1 or hidden.shape[2] != 2560:
        raise ValueError(f"row/hidden mismatch in {path}: rows={len(rows)} hidden={tuple(hidden.shape)}")
    checkpoints = [int(row["checkpoint"]) for row in rows]
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError(f"duplicate checkpoints in {path}")
    if checkpoints != sorted(checkpoints):
        raise ValueError(f"non-monotone checkpoints in {path}")
    return rows, hidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--split", choices=("probe_train", "calibration", "heldout"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/Qwen3-4B")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model, tokenizer, model_audit = load_qwen3(
        args.model_path, device, "float16", "sdpa"
    )
    capture = PositionHiddenCapture(model, CAPTURE_LAYERS)
    source_dir = args.source_root / args.split
    output_dir = args.output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(source_dir.glob("sample_*.pt"))
    if args.num_samples is not None:
        paths = paths[: args.num_samples]
    if not paths:
        raise FileNotFoundError(source_dir)

    completed = 0
    skipped = 0
    parity_rows = 0
    min_cosine = 1.0
    max_relative_l2 = 0.0
    max_absolute_difference = 0.0
    started = time.monotonic()
    try:
        for local_index, source_path in enumerate(paths):
            if local_index % args.num_shards != args.shard_index:
                continue
            source = torch.load(source_path, map_location="cpu", weights_only=False)
            problem_id = str(source.get("problem_id"))
            rows, reference_hidden = validate_source(source_path, source)
            invocation = {
                "schema": "dynamic_layer_cache_v1",
                "dataset": args.dataset,
                "split": args.split,
                "source": source_manifest(source_path, source),
                "model_metadata_fingerprint": model_audit["metadata_fingerprint"],
                "dtype": "float16",
                "attention_backend": "sdpa",
                "capture_layers": CAPTURE_LAYERS,
                "teacher_forced_dense_tokens": True,
            }
            fingerprint = canonical_fingerprint(invocation)
            destination = output_dir / source_path.name
            if args.resume and complete_destination(destination, problem_id, fingerprint):
                skipped += 1
                continue
            if destination.exists():
                raise RuntimeError(f"refusing to overwrite incompatible output: {destination}")

            if rows:
                checkpoints = [int(row["checkpoint"]) for row in rows]
                prompt_ids = tokenizer(
                    source["prompt_text"], return_tensors="pt"
                ).input_ids.to(device)
                prompt_tokens = int(prompt_ids.shape[1])
                if prompt_tokens != int(source["prompt_tokens"]):
                    raise ValueError(
                        f"prompt token mismatch for {problem_id}: {prompt_tokens} != {source['prompt_tokens']}"
                    )
                content_count = int(source["dense_content_tokens"])
                dense_tokens = [int(value) for value in source["dense"]["tokens"][:content_count]]
                maximum_checkpoint = max(checkpoints)
                if maximum_checkpoint > len(dense_tokens):
                    raise ValueError(f"checkpoint exceeds Dense prefix for {problem_id}")
                dense_tensor = torch.tensor(
                    [dense_tokens[:maximum_checkpoint]], dtype=torch.long, device=device
                )
                teacher_input = torch.cat([prompt_ids, dense_tensor], dim=1)
                positions = [prompt_tokens + checkpoint - 1 for checkpoint in checkpoints]
                capture.begin(positions, device)
                with torch.inference_mode():
                    model.model(
                        input_ids=teacher_input,
                        attention_mask=torch.ones_like(teacher_input),
                        use_cache=False,
                        return_dict=True,
                    )
                hidden = capture.finish_cpu().to(torch.float16)
            else:
                hidden = torch.empty((0, len(CAPTURE_LAYERS), 2560), dtype=torch.float16)

            parity = parity_metrics(reference_hidden[:, 0], hidden[:, 1])
            if parity["min_cosine"] < 0.9999 or parity["max_relative_l2"] > 0.01:
                raise RuntimeError(f"layer-20 parity gate failed for {problem_id}: {parity}")
            parity_rows += int(parity["rows"])
            min_cosine = min(min_cosine, parity["min_cosine"])
            max_relative_l2 = max(max_relative_l2, parity["max_relative_l2"])
            max_absolute_difference = max(
                max_absolute_difference, parity["max_absolute_difference"]
            )

            artifact = dict(source)
            artifact.update(
                {
                    "capture_layers": CAPTURE_LAYERS,
                    "hidden": hidden,
                    "source_layer20_replay_artifact": str(source_path.resolve()),
                    "layer_ablation_capture": invocation,
                    "layer_ablation_capture_fingerprint": fingerprint,
                    "layer20_teacher_force_parity": parity,
                    "layer_ablation_created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            if len(artifact["rows"]) != int(artifact["hidden"].shape[0]):
                raise RuntimeError(f"row/vector mismatch after capture: {problem_id}")
            if not torch.isfinite(artifact["hidden"]).all():
                raise RuntimeError(f"NaN/Inf hidden after capture: {problem_id}")
            atomic_torch_save(artifact, destination)
            completed += 1
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                json.dumps(
                    {
                        "problem_id": problem_id,
                        "completed": completed,
                        "skipped": skipped,
                        "samples_per_hour": 3600.0 * completed / elapsed,
                        "rows": len(rows),
                        "parity": parity,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        capture.close()

    elapsed = time.monotonic() - started
    summary = {
        "status": "complete",
        "dataset": args.dataset,
        "split": args.split,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "visible_source_files": len(paths),
        "completed_now": completed,
        "skipped_complete": skipped,
        "elapsed_seconds": elapsed,
        "samples_per_hour": 3600.0 * completed / elapsed if completed and elapsed else None,
        "capture_layers": CAPTURE_LAYERS,
        "parity_rows": parity_rows,
        "min_layer20_cosine": min_cosine,
        "max_layer20_relative_l2": max_relative_l2,
        "max_layer20_absolute_difference": max_absolute_difference,
        "model": model_audit,
    }
    atomic_json(summary, output_dir / f"summary_shard{args.shard_index}.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

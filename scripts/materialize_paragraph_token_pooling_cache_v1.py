#!/usr/bin/env python3
"""Materialize paragraph-checkpoint token-pooling readouts from frozen Dense traces.

The source cache already contains the immutable Dense token sequence, forced-answer
labels, split assignment, and paragraph checkpoints.  This script never samples or
regenerates answers.  It teacher-forces the cached prefix once, captures layer 20,
and stores several causal readouts for each existing checkpoint.
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

import torch
import torch.nn.functional as F

from src.final_paper_inference import PositionHiddenCapture, atomic_torch_save
from src.final_paper_replay_cache import raw_semantic_boundaries
from src.qwen3_reasoning import load_qwen3
from src.utils import atomic_json


LAYER = 20
HIDDEN_SIZE = 2560
REPRESENTATIONS = (
    "boundary",
    "preboundary_nonblank",
    "last4_noncontrol_mean",
    "last8_noncontrol_mean",
    "sentence_mean",
    "paragraph_mean",
    "last8_noncontrol_ln_mean",
    "paragraph_ln_mean",
)


def canonical_fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def source_manifest(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "problem_id": str(artifact.get("problem_id")),
        "protocol_fingerprint": artifact.get("protocol_fingerprint"),
        "primary_replay_view_fingerprint": artifact.get(
            "primary_replay_view_fingerprint"
        ),
    }


def complete_destination(path: Path, problem_id: str, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return (
        value.get("status") == "complete"
        and str(value.get("problem_id")) == problem_id
        and value.get("representation_names") == list(REPRESENTATIONS)
        and value.get("token_pooling_fingerprint") == fingerprint
    )


def validate_source(path: Path, artifact: dict[str, Any]) -> tuple[list[dict[str, Any]], torch.Tensor]:
    if artifact.get("status") != "complete":
        raise ValueError(f"incomplete source: {path}")
    if artifact.get("actual_checkpoint_schedule") != "paragraph":
        raise ValueError(f"expected paragraph source: {path}")
    if artifact.get("capture_layers") != [LAYER]:
        raise ValueError(f"expected layer-{LAYER} source: {path}")
    rows = artifact.get("rows", [])
    hidden = artifact.get("hidden")
    if not torch.is_tensor(hidden) or tuple(hidden.shape) != (len(rows), 1, HIDDEN_SIZE):
        raise ValueError(f"row/hidden mismatch: {path} {getattr(hidden, 'shape', None)}")
    checkpoints = [int(row["checkpoint"]) for row in rows]
    if checkpoints != sorted(set(checkpoints)):
        raise ValueError(f"invalid checkpoint ordering: {path}")
    return rows, hidden


def token_is_noncontrol(tokenizer, token_id: int) -> bool:
    if token_id in set(tokenizer.all_special_ids):
        return False
    piece = tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return bool(piece.strip())


def last_indices(mask: list[bool], stop: int, count: int, *, start: int = 0) -> list[int]:
    values = [index for index in range(max(0, start), min(stop, len(mask))) if mask[index]]
    return values[-count:]


def safe_indices(values: list[int], fallback: int) -> list[int]:
    return values if values else [fallback]


def mean_vectors(values: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return values[torch.tensor(indices, dtype=torch.long)].float().mean(dim=0)


def ln_mean_vectors(values: torch.Tensor, indices: list[int]) -> torch.Tensor:
    selected = values[torch.tensor(indices, dtype=torch.long)].float()
    selected = F.layer_norm(selected, (selected.shape[-1],))
    return selected.mean(dim=0)


def exact_sampling_stats(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[list[float], list[float]]:
    """Return p_max and top1-top2 probability gap after actual sampling filters."""
    values = logits.float() / float(temperature)
    k = min(int(top_k), values.shape[-1]) if top_k > 0 else values.shape[-1]
    top_values, _ = torch.topk(values, k=k, dim=-1)
    top_values, _ = torch.sort(top_values, descending=True, dim=-1)
    if top_p < 1.0:
        initial = torch.softmax(top_values, dim=-1)
        remove = initial.cumsum(dim=-1) - initial > float(top_p)
        top_values = top_values.masked_fill(remove, -torch.inf)
    probabilities = torch.softmax(top_values, dim=-1)
    pmax = probabilities[:, 0]
    gap = pmax - probabilities[:, 1] if probabilities.shape[1] > 1 else pmax
    return pmax.cpu().tolist(), gap.cpu().tolist()


def representation_rows(
    tokenizer,
    content: list[int],
    checkpoints: list[int],
    content_hidden: torch.Tensor,
) -> torch.Tensor:
    noncontrol = [token_is_noncontrol(tokenizer, token) for token in content]
    semantic, _ = raw_semantic_boundaries(tokenizer, content, len(content))
    semantic = sorted(set(int(value) for value in semantic))
    output: list[torch.Tensor] = []
    previous_paragraph = 0
    for checkpoint in checkpoints:
        boundary = checkpoint - 1
        if boundary < 0 or boundary >= len(content_hidden):
            raise ValueError(f"checkpoint {checkpoint} outside captured prefix")

        before = last_indices(noncontrol, boundary, 1)
        recent4 = last_indices(noncontrol, checkpoint, 4)
        recent8 = last_indices(noncontrol, checkpoint, 8)
        paragraph = last_indices(
            noncontrol, checkpoint, max(checkpoint - previous_paragraph, 1),
            start=previous_paragraph,
        )

        ends = [value for value in semantic if value <= checkpoint]
        if ends:
            sentence_end = ends[-1]
            earlier = [value for value in semantic if value < sentence_end]
            sentence_start = earlier[-1] if earlier else 0
            sentence = last_indices(
                noncontrol,
                sentence_end,
                max(sentence_end - sentence_start, 1),
                start=sentence_start,
            )
        else:
            sentence = recent8

        before = safe_indices(before, boundary)
        recent4 = safe_indices(recent4, before[-1])
        recent8 = safe_indices(recent8, before[-1])
        sentence = safe_indices(sentence, recent8[-1])
        paragraph = safe_indices(paragraph, recent8[-1])
        vectors = torch.stack(
            [
                content_hidden[boundary].float(),
                content_hidden[before[-1]].float(),
                mean_vectors(content_hidden, recent4),
                mean_vectors(content_hidden, recent8),
                mean_vectors(content_hidden, sentence),
                mean_vectors(content_hidden, paragraph),
                ln_mean_vectors(content_hidden, recent8),
                ln_mean_vectors(content_hidden, paragraph),
            ]
        )
        output.append(vectors)
        previous_paragraph = checkpoint
    if not output:
        return torch.empty((0, len(REPRESENTATIONS), HIDDEN_SIZE), dtype=torch.float16)
    return torch.stack(output).to(torch.float16)


def parity_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.numel() == 0:
        return {"rows": 0, "min_cosine": 1.0, "max_relative_l2": 0.0}
    reference = reference.float()
    candidate = candidate.float()
    cosine = F.cosine_similarity(reference, candidate, dim=-1)
    relative = torch.linalg.vector_norm(candidate - reference, dim=-1) / torch.linalg.vector_norm(
        reference, dim=-1
    ).clamp_min(1e-12)
    return {
        "rows": int(reference.shape[0]),
        "min_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
        "max_relative_l2": float(relative.max()),
        "mean_relative_l2": float(relative.mean()),
    }


def chunked_teacher_force(
    model,
    capture: PositionHiddenCapture,
    prompt_ids: torch.Tensor,
    content: list[int],
    checkpoints: list[int],
    device: torch.device,
    *,
    chunk_size: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[float]]:
    """Causally recapture cached tokens with bounded activation memory."""
    prompt_tokens = int(prompt_ids.shape[1])
    capture.begin([prompt_tokens - 1], device)
    with torch.inference_mode():
        prompt_output = model.model(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            use_cache=True,
            return_dict=True,
        )
    prompt_hidden = capture.finish_cpu()[0, 0]
    past = prompt_output.past_key_values
    del prompt_output

    content_hidden_parts: list[torch.Tensor] = []
    stats: dict[int, tuple[float, float]] = {}
    maximum = len(content)
    checkpoint_set = set(checkpoints)
    for start in range(0, maximum, chunk_size):
        end = min(start + chunk_size, maximum)
        chunk = torch.tensor([content[start:end]], dtype=torch.long, device=device)
        attention_mask = torch.ones(
            (1, prompt_tokens + end), dtype=torch.long, device=device
        )
        capture.begin(range(end - start), device)
        with torch.inference_mode():
            output = model.model(
                input_ids=chunk,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            local_checkpoints = [
                checkpoint
                for checkpoint in range(start + 1, end + 1)
                if checkpoint in checkpoint_set
            ]
            if local_checkpoints:
                local_positions = torch.tensor(
                    [checkpoint - 1 - start for checkpoint in local_checkpoints],
                    dtype=torch.long,
                    device=device,
                )
                selected = output.last_hidden_state[0].index_select(0, local_positions)
                logits = model.lm_head(selected).float()
                pmax, gap = exact_sampling_stats(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                stats.update(
                    {
                        checkpoint: (float(pmax[index]), float(gap[index]))
                        for index, checkpoint in enumerate(local_checkpoints)
                    }
                )
                del selected, logits
        content_hidden_parts.append(capture.finish_cpu()[:, 0])
        past = output.past_key_values
        del output, chunk, attention_mask
    content_hidden = torch.cat(content_hidden_parts, dim=0)
    if tuple(content_hidden.shape) != (maximum, HIDDEN_SIZE):
        raise ValueError(f"chunked hidden shape mismatch: {content_hidden.shape}")
    if set(stats) != checkpoint_set:
        raise ValueError(f"missing checkpoint sampling stats: {checkpoint_set - set(stats)}")
    return (
        prompt_hidden,
        content_hidden,
        [stats[checkpoint][0] for checkpoint in checkpoints],
        [stats[checkpoint][1] for checkpoint in checkpoints],
    )


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
    parser.add_argument("--problem-id")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    if args.chunk_size <= 0:
        raise ValueError("chunk size must be positive")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model, tokenizer, model_audit = load_qwen3(
        args.model_path, device, "float16", "sdpa"
    )
    capture = PositionHiddenCapture(model, [LAYER])
    paths = sorted((args.source_root / args.split).glob("sample_*.pt"))
    if args.problem_id is not None:
        paths = [path for path in paths if path.stem == f"sample_{args.problem_id}"]
    if args.num_samples is not None:
        paths = paths[: args.num_samples]
    assigned = [path for index, path in enumerate(paths) if index % args.num_shards == args.shard_index]
    if not assigned:
        raise FileNotFoundError(args.source_root / args.split)
    output_dir = args.output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = skipped = checkpoint_rows = 0
    minimum_cosine = 1.0
    maximum_relative_l2 = 0.0
    started = time.monotonic()
    try:
        for source_path in assigned:
            source = torch.load(source_path, map_location="cpu", weights_only=False)
            rows, reference_hidden = validate_source(source_path, source)
            problem_id = str(source["problem_id"])
            invocation = {
                "schema": "paragraph_token_pooling_cache_chunked_v2",
                "source": source_manifest(source_path, source),
                "model_metadata_fingerprint": model_audit["metadata_fingerprint"],
                "layer_zero_based": LAYER,
                "dtype": "float16",
                "attention_backend": "sdpa",
                "teacher_force_execution": {
                    "mode": "causal_kv_chunked",
                    "chunk_size": args.chunk_size,
                },
                "representations": list(REPRESENTATIONS),
                "sampling_distribution": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                },
            }
            fingerprint = canonical_fingerprint(invocation)
            destination = output_dir / source_path.name
            if args.resume and complete_destination(destination, problem_id, fingerprint):
                skipped += 1
                continue
            if destination.exists():
                raise RuntimeError(f"refusing to overwrite incompatible output: {destination}")

            prompt_ids = tokenizer(source["prompt_text"], return_tensors="pt").input_ids.to(device)
            prompt_tokens = int(prompt_ids.shape[1])
            if prompt_tokens != int(source["prompt_tokens"]):
                raise ValueError(f"prompt token mismatch: {problem_id}")
            content = [int(value) for value in source["dense"]["content_tokens"]]
            checkpoints = [int(row["checkpoint"]) for row in rows]
            maximum = max(checkpoints, default=0)
            if maximum > len(content):
                raise ValueError(f"checkpoint exceeds Dense content: {problem_id}")

            enriched_rows = [dict(row) for row in rows]
            if checkpoints:
                prompt_hidden, content_hidden, pmax, probability_gap = chunked_teacher_force(
                    model,
                    capture,
                    prompt_ids,
                    content[:maximum],
                    checkpoints,
                    device,
                    chunk_size=args.chunk_size,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                hidden = representation_rows(
                    tokenizer, content[:maximum], checkpoints, content_hidden
                )
                for index, row in enumerate(enriched_rows):
                    row["sampling_pmax"] = float(pmax[index])
                    row["sampling_probability_gap"] = float(probability_gap[index])
                    row["checkpoint_type"] = "paragraph"
            else:
                capture.begin([prompt_tokens - 1], device)
                with torch.inference_mode():
                    prompt_output = model.model(
                        input_ids=prompt_ids,
                        attention_mask=torch.ones_like(prompt_ids),
                        use_cache=False,
                        return_dict=True,
                    )
                prompt_hidden = capture.finish_cpu()[0, 0]
                del prompt_output
                hidden = torch.empty(
                    (0, len(REPRESENTATIONS), HIDDEN_SIZE), dtype=torch.float16
                )

            prompt_hidden_ln = F.layer_norm(
                prompt_hidden.float(), (HIDDEN_SIZE,)
            ).to(torch.float16)
            parity = parity_metrics(reference_hidden[:, 0], hidden[:, 0])
            # 2080-Ti and A100 SDPA kernels differ slightly; this gate matches the
            # previously audited paragraph layer recapture tolerance.
            if parity["min_cosine"] < 0.9998 or parity["max_relative_l2"] > 0.02:
                raise RuntimeError(f"boundary parity failed for {problem_id}: {parity}")
            checkpoint_rows += len(rows)
            minimum_cosine = min(minimum_cosine, parity["min_cosine"])
            maximum_relative_l2 = max(maximum_relative_l2, parity["max_relative_l2"])

            artifact = dict(source)
            artifact.update(
                {
                    "rows": enriched_rows,
                    "capture_layers": [LAYER],
                    "representation_names": list(REPRESENTATIONS),
                    "hidden": hidden,
                    "prompt_hidden_ln": prompt_hidden_ln,
                    "token_pooling_invocation": invocation,
                    "token_pooling_fingerprint": fingerprint,
                    "source_paragraph_artifact": str(source_path.resolve()),
                    "boundary_teacher_force_parity": parity,
                    "token_pooling_created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            if not torch.isfinite(hidden).all() or not torch.isfinite(prompt_hidden_ln).all():
                raise RuntimeError(f"NaN/Inf representation: {problem_id}")
            atomic_torch_save(artifact, destination)
            completed += 1
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                json.dumps(
                    {
                        "problem_id": problem_id,
                        "completed": completed,
                        "skipped": skipped,
                        "rows": len(rows),
                        "samples_per_hour": 3600.0 * completed / elapsed,
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
        "assigned": len(assigned),
        "completed_now": completed,
        "skipped_complete": skipped,
        "checkpoint_rows": checkpoint_rows,
        "representations": list(REPRESENTATIONS),
        "minimum_boundary_cosine": minimum_cosine,
        "maximum_boundary_relative_l2": maximum_relative_l2,
        "elapsed_seconds": elapsed,
        "samples_per_hour": 3600.0 * completed / elapsed if completed and elapsed else None,
        "model": model_audit,
    }
    atomic_json(summary, output_dir / f"summary_shard{args.shard_index}.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

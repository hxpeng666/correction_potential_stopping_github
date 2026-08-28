#!/usr/bin/env python3
"""Generate paired greedy answer branches for several frozen suffixes at once."""
from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from deepseek7b_protocol_v1 import greedy_branch, last_boxed, load_model, prediction, success
from src.reproducibility import (
    code_provenance,
    enforce_runtime_lock,
    environment_provenance,
    sha256_json,
    strict_reproducibility,
)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def valid(path: Path, fingerprint: str, problem_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        return (
            value.get("status") == "complete"
            and value.get("collection_fingerprint") == fingerprint
            and value.get("problem_id") == problem_id
        )
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    determinism = strict_reproducibility(seed=0, num_threads=1)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("selection_uses_labels") is not False:
        raise ValueError("invalid sample manifest")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    runtime = environment_provenance(device)
    runtime_lock = enforce_runtime_lock(ROOT / config["reproducibility"]["runtime_lock"], runtime)
    provenance = code_provenance(
        ROOT,
        (
            "configs/deepseek7b_deterministic_exit_suffix_v1.yaml",
            "scripts/collect_deepseek7b_deterministic_suffix_v1.py",
            "scripts/deepseek7b_protocol_v1.py",
            "src/reproducibility.py",
        ),
    )
    variants = {
        key: value["suffix"]
        for key, value in config["forced_answer"]["variants"].items()
        if value.get("role") != "frozen_reference"
    }
    maximum = int(config["forced_answer"]["max_new_tokens"])
    collection_fingerprint = sha256_json(
        {
            "protocol_id": config["protocol_id"],
            "git_commit": provenance["git"]["commit"],
            "manifest_fingerprint": manifest["manifest_fingerprint"],
            "variants": variants,
            "maximum": maximum,
            "decoding": "greedy_argmax",
        }
    )
    entries = [
        row for index, row in enumerate(manifest["entries"])
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        entries = entries[: args.limit]
    model, tokenizer, model_audit = load_model(Path(config["model"]["local_path"]), device)
    suffix_ids = {
        label: tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        for label, text in variants.items()
    }
    completed = skipped = branches = failures = 0
    for entry in entries:
        problem_id = str(entry["problem_id"])
        target = args.output_root / entry["dataset"] / entry["split"] / f"{problem_id}.pt"
        if args.resume and valid(target, collection_fingerprint, problem_id):
            skipped += 1
            print(json.dumps({"status": "skipped", "problem_id": problem_id}), flush=True)
            continue
        if target.exists():
            raise RuntimeError(f"refusing to overwrite incompatible output: {target}")
        try:
            source_path = Path(entry["source_path"])
            source = torch.load(source_path, map_location="cpu", weights_only=False)
            if str(source["problem_id"]) != problem_id:
                raise AssertionError("problem identity mismatch")
            selected = set(map(int, entry["checkpoints"]))
            source_rows = {
                int(row["checkpoint"]): row for row in source["rows"]
                if int(row["checkpoint"]) in selected
            }
            if set(source_rows) != selected:
                raise AssertionError(f"checkpoint mismatch: {problem_id}")
            prompt = tokenizer(source["prompt_text"], return_tensors="pt").input_ids.to(device)
            if int(prompt.shape[1]) != int(source["prompt_tokens"]):
                raise AssertionError(f"prompt token mismatch: {problem_id}")
            content = list(source["dense"]["content_tokens"])
            records: list[dict[str, Any]] = []
            with torch.inference_mode():
                prefill = model.model(
                    input_ids=prompt,
                    attention_mask=torch.ones_like(prompt),
                    use_cache=True,
                    return_dict=True,
                )
                cache = prefill.past_key_values
                del prefill
                previous = 0
                for checkpoint in sorted(selected):
                    delta = torch.tensor([content[previous:checkpoint]], dtype=torch.long, device=device)
                    mask = torch.ones((1, int(prompt.shape[1]) + checkpoint), dtype=torch.long, device=device)
                    teacher = model.model(
                        input_ids=delta,
                        attention_mask=mask,
                        past_key_values=cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    cache = teacher.past_key_values
                    del teacher
                    original = source_rows[checkpoint]
                    variant_records: dict[str, Any] = {}
                    for label, ids in suffix_ids.items():
                        branch = greedy_branch(
                            model,
                            tokenizer,
                            cache,
                            prefix_context=int(prompt.shape[1]) + checkpoint,
                            suffix_ids=ids,
                            maximum=maximum,
                        )
                        parsed = prediction(str(source["dataset"]), branch["text"])
                        correct = success(str(source["dataset"]), source["gold_answer"], parsed)
                        variant_records[label] = {
                            "prediction": parsed,
                            "success": bool(correct),
                            "complete_boxed": last_boxed(branch["text"]) is not None,
                            "grader_parseable": parsed is not None,
                            "branch_tokens": len(branch["tokens"]),
                            "branch_token_ids": list(branch["tokens"]),
                            "branch_text": branch["text"],
                            "max_hit": len(branch["tokens"]) >= maximum,
                        }
                        branches += 1
                    records.append(
                        {
                            "checkpoint": checkpoint,
                            "dense_success": bool(original["dense_success"]),
                            "reference": {
                                "prediction": original.get("current_prediction"),
                                "success": bool(original["current_success"]),
                                "complete_boxed": last_boxed(str(original.get("branch_text", ""))) is not None,
                                "grader_parseable": original.get("current_prediction") is not None,
                                "branch_tokens": int(original.get("branch_tokens", 0)),
                                "branch_text": str(original.get("branch_text", "")),
                                "max_hit": bool(original.get("forced_answer_max_hit", False)),
                            },
                            "variants": variant_records,
                        }
                    )
                    previous = checkpoint
            output = {
                "status": "complete",
                "protocol_id": config["protocol_id"],
                "collection_fingerprint": collection_fingerprint,
                "manifest_fingerprint": manifest["manifest_fingerprint"],
                "problem_id": problem_id,
                "dataset": entry["dataset"],
                "split": entry["split"],
                "source_path": str(source_path.resolve()),
                "source_protocol_fingerprint": entry["source_protocol_fingerprint"],
                "dense_reasoning_tokens": int(entry["dense_reasoning_tokens"]),
                "dense_success": bool(entry["dense_success"]),
                "records": records,
                "variants": variants,
                "maximum": maximum,
                "code_identity": provenance,
                "determinism": determinism,
                "runtime_lock": runtime_lock,
                "runtime": runtime,
                "model_audit": model_audit,
                "collection": {
                    "worker": args.worker_id,
                    "host": socket.gethostname(),
                    "gpu": args.gpu,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            }
            atomic_torch_save(output, target)
            completed += 1
            print(json.dumps({"status": "completed", "problem_id": problem_id, "rows": len(records)}), flush=True)
        except Exception as error:
            failures += 1
            print(json.dumps({"status": "error", "problem_id": problem_id, "type": type(error).__name__, "error": str(error)}), flush=True)
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    marker = args.output_root / "workers" / f"{args.worker_id}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "status": "complete" if failures == 0 else "failed",
                "assigned": len(entries),
                "completed": completed,
                "skipped": skipped,
                "branches": branches,
                "failures": failures,
                "collection_fingerprint": collection_fingerprint,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()


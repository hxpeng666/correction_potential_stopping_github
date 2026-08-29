#!/usr/bin/env python3
"""Audit every non-engine experimental choice against a Transformers artifact."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def model_scientific(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "path"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformers", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = torch.load(args.transformers, map_location="cpu", weights_only=False)
    candidate = torch.load(args.vllm, map_location="cpu", weights_only=False)
    pairs = {
        "dataset": (reference.get("dataset"), candidate.get("dataset")),
        "split": (reference.get("split"), candidate.get("split")),
        "problem_id": (reference.get("problem_id"), candidate.get("problem_id")),
        "record": (reference.get("record"), candidate.get("record")),
        "gold_answer": (reference.get("gold_answer"), candidate.get("gold_answer")),
        "dtype": (reference.get("dtype"), candidate.get("dtype")),
        "base_seed": (reference.get("seed"), candidate.get("seed")),
        "problem_seed": (reference.get("problem_seed"), candidate.get("problem_seed")),
        "prompt_text": (reference.get("prompt_text"), candidate.get("prompt_text")),
        "prompt_tokens": (reference.get("prompt_tokens"), candidate.get("prompt_tokens")),
        "prompt_token_ids": (
            reference.get("prompt_token_ids"),
            candidate.get("prompt_token_ids"),
        ),
        "checkpoint_protocol": (
            reference.get("checkpoint_protocol"),
            candidate.get("checkpoint_protocol"),
        ),
        "checkpoint_schedule": (
            reference.get("actual_checkpoint_schedule"),
            candidate.get("actual_checkpoint_schedule"),
        ),
        "capture_layers": (
            reference.get("capture_layers"),
            candidate.get("capture_layers"),
        ),
        "dense_generation": (
            reference.get("dense_generation"),
            candidate.get("dense_generation"),
        ),
        "forced_answer_decoding": (
            reference.get("forced_answer_decoding"),
            candidate.get("forced_answer_decoding"),
        ),
        "data_identity": (
            reference.get("data_identity"),
            candidate.get("data_identity"),
        ),
        "model_identity": (
            model_scientific(reference.get("model_audit", {})),
            model_scientific(candidate.get("model_audit", {})),
        ),
    }
    checks = {name: left == right for name, (left, right) in pairs.items()}
    engine = candidate.get("vllm_engine", {})
    checks.update(
        {
            "vllm_single_process": engine.get("multiprocessing") is False,
            "vllm_sync_scheduler": engine.get("async_scheduling") is False,
            "vllm_eager_no_cudagraph": engine.get("enforce_eager") is True,
            "vllm_profile_declared": isinstance(engine.get("profile"), str),
            "vllm_phase_settings_declared": set(engine.get("phases", {}))
            == {"dense", "hidden", "branches"},
            "vllm_small_batch_for_40gb": all(
                int(settings.get("max_num_seqs", 999)) <= 2
                for settings in engine.get("phases", {}).values()
            ),
            "vllm_layer20_aux_mapping": (
                engine.get("requested_zero_based_decoder_layer") == 20
                and engine.get("vllm_aux_hidden_state_layer_ids") == [21]
            ),
            "vllm_forbidden_binary_extensions_absent": engine.get(
                "forbidden_optional_packages_absent"
            )
            == ["flash-attn", "xformers"],
            "hidden_replay_token_ids_exact": candidate.get(
                "hidden_replay_audit", {}
            ).get("token_ids_exact")
            is True,
        }
    )
    deliberately_replaced = {
        "dense_token_ids": "vLLM sampler/kernel replaces manual Transformers decode",
        "dense_text_and_entropy": "derived from the vLLM rollout",
        "hidden_tensor_values": "vLLM native model runner and hidden extractor",
        "paragraph_positions": "same algorithm applied to the replaced Dense token sequence",
        "greedy_branch_token_ids": "vLLM greedy decode replaces Transformers greedy_branch",
        "runtime": "vLLM 0.18.1 requires Torch 2.10/CUDA 12.8 and vLLM attention kernels",
    }
    payload = {
        "status": "complete" if all(checks.values()) else "failed",
        "all_non_engine_protocol_exact": bool(all(checks.values())),
        "checks": checks,
        "deliberately_replaced": deliberately_replaced,
        "artifacts": {
            "transformers": str(args.transformers.resolve()),
            "vllm": str(args.vllm.resolve()),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

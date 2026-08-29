#!/usr/bin/env python3
"""Three-pass, fail-closed full-vLLM Qwen3 paragraph collector.

Pass 1 samples the frozen Dense rollout.  Pass 2 replays the exact prompt plus
sampled token IDs through vLLM's native hidden-state extractor and selects only
the frozen paragraph checkpoint positions.  Pass 3 runs every greedy suffix
branch through vLLM and materializes the legacy-compatible scientific artifact.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from packaging.utils import canonicalize_name
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from deepseek7b_protocol_v1 import (
    atomic_torch_save,
    canonical_fingerprint,
    paragraph_checkpoints,
    prediction,
    render_prompt,
    stable_seed,
    success,
    tail_mean,
)
from src.qwen3_reasoning import inspect_qwen3
from src.reproducibility import (
    code_provenance,
    enforce_runtime_lock,
    environment_provenance,
    sha256_file,
    sha256_json,
    strict_reproducibility,
)

DATA_LAYOUT = (
    ("gsm8k", "probe_train"),
    ("gsm8k", "calibration"),
    ("gsm8k", "heldout"),
    ("math", "probe_train"),
    ("math", "calibration"),
    ("math500", "heldout"),
    ("aime", "heldout"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def all_tasks(prepared_root: Path) -> list[tuple[str, str, dict[str, Any]]]:
    tasks: list[tuple[str, str, dict[str, Any]]] = []
    for dataset, split in DATA_LAYOUT:
        path = prepared_root / dataset / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        tasks.extend((dataset, split, row) for row in read_jsonl(path))
    return sorted(tasks, key=lambda item: (item[0], item[1], str(item[2]["problem_id"])))


def data_identity(prepared_root: Path) -> dict[str, Any]:
    files = {}
    for dataset, split in DATA_LAYOUT:
        path = prepared_root / dataset / f"{split}.jsonl"
        files[f"{dataset}/{split}.jsonl"] = {
            "rows": len(read_jsonl(path)),
            "sha256": sha256_file(path),
        }
    return {"files": files, "sha256": sha256_json(files)}


def gold_for(dataset: str, record: dict[str, Any]) -> str | None:
    if "gold_answer" in record:
        return str(record["gold_answer"])
    if dataset == "gsm8k":
        marker = str(record["answer"]).rsplit("####", 1)
        return marker[-1].strip().replace(",", "") if marker else None
    raise KeyError(f"no gold answer for {dataset}:{record.get('problem_id')}")


def artifact_path(root: Path, dataset: str, split: str, problem_id: str) -> Path:
    return root / "cache" / dataset / split / f"sample_{problem_id}.pt"


def stage_path(
    root: Path, worker: str, stage: str, dataset: str, split: str, problem_id: str
) -> Path:
    return root / "staging" / worker / stage / dataset / split / f"sample_{problem_id}.pt"


def valid_stage(path: Path, fingerprint: str, problem_id: str, stage: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        return (
            value.get("status") == "complete"
            and value.get("stage") == stage
            and value.get("protocol_fingerprint") == fingerprint
            and str(value.get("problem_id")) == problem_id
        )
    except Exception:
        return False


def valid_artifact(path: Path, fingerprint: str, problem_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        return (
            value.get("status") == "complete"
            and value.get("protocol_fingerprint") == fingerprint
            and str(value.get("problem_id")) == problem_id
            and value.get("actual_checkpoint_schedule") == "paragraph"
        )
    except Exception:
        return False


def enforce_vllm_environment(config: dict[str, Any]) -> dict[str, str]:
    expected = {
        str(key): str(value)
        for key, value in config["reproducibility"]["required_environment"].items()
    }
    mismatches = {
        key: {"expected": value, "actual": os.environ.get(key)}
        for key, value in expected.items()
        if os.environ.get(key) != value
    }
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        mismatches["CUDA_VISIBLE_DEVICES"] = {
            "expected": "one physical GPU index",
            "actual": visible,
        }
    if mismatches:
        raise RuntimeError(
            "full-vLLM deterministic environment is not frozen: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return expected


def enforce_full_environment_lock(path: Path) -> dict[str, Any]:
    expected: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"invalid full environment lock line: {line}")
        name, version = line.split("==", 1)
        canonical = canonicalize_name(name)
        if canonical in expected:
            raise ValueError(f"duplicate package in full environment lock: {canonical}")
        expected[canonical] = version
    observed: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for distribution in importlib.metadata.distributions():
        canonical = canonicalize_name(str(distribution.metadata["Name"]))
        version = str(distribution.version)
        if canonical in observed:
            duplicates.setdefault(canonical, [observed[canonical]]).append(version)
        observed[canonical] = version
    missing = {name: version for name, version in expected.items() if name not in observed}
    extra = {name: version for name, version in observed.items() if name not in expected}
    changed = {
        name: {"expected": expected[name], "actual": observed[name]}
        for name in expected.keys() & observed.keys()
        if expected[name] != observed[name]
    }
    if missing or extra or changed or duplicates:
        raise RuntimeError(
            "full Python environment lock mismatch: "
            + json.dumps(
                {
                    "missing": missing,
                    "extra": extra,
                    "changed": changed,
                    "duplicates": duplicates,
                },
                sort_keys=True,
            )
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "package_count": len(expected),
        "exact": True,
    }


def vllm_engine_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = config["vllm"]
    observed = importlib.metadata.version("vllm")
    if observed != str(frozen["version"]):
        raise RuntimeError(f"vLLM version mismatch: expected {frozen['version']} got {observed}")
    required = {
        "engine": "v1",
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "attention_backend": "FLASH_ATTN",
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "max_num_seqs": 1,
        "request_batch_size": 1,
        "engine_seed": 0,
        "gpu_memory_utilization": 0.47,
    }
    mismatches = {
        key: {"expected": value, "actual": frozen.get(key)}
        for key, value in required.items()
        if frozen.get(key) != value
    }
    if mismatches:
        raise RuntimeError("vLLM engine config drift: " + json.dumps(mismatches, sort_keys=True))
    forbidden = [
        str(value) for value in config["reproducibility"]["forbidden_optional_packages"]
    ]
    present = {}
    for package in forbidden:
        try:
            present[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if present:
        raise RuntimeError(
            "forbidden binary extension packages are installed: "
            + json.dumps(present, sort_keys=True)
        )
    hidden = frozen["hidden_extractor"]
    requested_layer = int(config["features"]["layer_zero_based"])
    if (
        int(hidden["requested_zero_based_decoder_layer"]) != requested_layer
        or [int(value) for value in hidden["vllm_aux_hidden_state_layer_ids"]]
        != [requested_layer + 1]
        or hidden["vllm_layer_mapping"]
        != "aux_id_equals_zero_based_decoder_layer_plus_one"
    ):
        raise RuntimeError("vLLM auxiliary hidden-state layer mapping drift")
    return {
        "version": observed,
        "engine": frozen["engine"],
        "tensor_parallel_size": int(frozen["tensor_parallel_size"]),
        "dtype": frozen["dtype"],
        "attention_backend": frozen["attention_backend"],
        "enforce_eager": bool(frozen["enforce_eager"]),
        "compilation": frozen["compilation"],
        "enable_prefix_caching": bool(frozen["enable_prefix_caching"]),
        "enable_chunked_prefill": bool(frozen["enable_chunked_prefill"]),
        "async_scheduling": bool(frozen["async_scheduling"]),
        "max_model_len": int(frozen["max_model_len"]),
        "max_num_batched_tokens": int(frozen["max_num_batched_tokens"]),
        "max_num_seqs": int(frozen["max_num_seqs"]),
        "request_batch_size": int(frozen["request_batch_size"]),
        "engine_seed": int(frozen["engine_seed"]),
        "gpu_memory_utilization": float(frozen["gpu_memory_utilization"]),
        "multiprocessing": False,
        "forbidden_optional_packages_absent": forbidden,
        "requested_zero_based_decoder_layer": requested_layer,
        "vllm_aux_hidden_state_layer_ids": [requested_layer + 1],
    }


def make_llm(
    model_path: Path,
    config: dict[str, Any],
    gpu_memory_utilization: float,
    *,
    hidden_directory: Path | None = None,
):
    from vllm import LLM

    frozen = config["vllm"]
    common: dict[str, Any] = {
        "model": str(model_path),
        "tokenizer": str(model_path),
        "tokenizer_mode": "auto",
        "trust_remote_code": False,
        "tensor_parallel_size": int(frozen["tensor_parallel_size"]),
        "dtype": str(frozen["dtype"]),
        "seed": int(frozen["engine_seed"]),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "enforce_eager": bool(frozen["enforce_eager"]),
        "disable_custom_all_reduce": True,
        "max_model_len": int(frozen["max_model_len"]),
        "max_num_batched_tokens": int(frozen["max_num_batched_tokens"]),
        "max_num_seqs": int(frozen["max_num_seqs"]),
        "enable_prefix_caching": False,
        "enable_chunked_prefill": bool(frozen["enable_chunked_prefill"]),
        "async_scheduling": bool(frozen["async_scheduling"]),
        "disable_log_stats": True,
        "attention_backend": str(frozen["attention_backend"]),
        "compilation_config": {"mode": 0},
    }
    if hidden_directory is not None:
        hidden_directory.mkdir(parents=True, exist_ok=True)
        layer_ids = [
            int(value)
            for value in frozen["hidden_extractor"][
                "vllm_aux_hidden_state_layer_ids"
            ]
        ]
        common["speculative_config"] = {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": layer_ids}
            },
        }
        common["kv_transfer_config"] = {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": str(hidden_directory.resolve())
            },
        }
    return LLM(**common)


def release_llm(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    for owner in (llm, engine, core):
        shutdown = getattr(owner, "shutdown", None)
        if callable(shutdown):
            shutdown()
            break
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def assert_torch_determinism() -> None:
    checks = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark_disabled": not torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32_disabled": not torch.backends.cudnn.allow_tf32,
        "matmul_tf32_disabled": not torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_highest": torch.get_float32_matmul_precision() == "highest",
    }
    if not all(checks.values()):
        raise RuntimeError("vLLM changed deterministic torch flags: " + json.dumps(checks))


def top20_entropies(logprobs: Any, expected: int) -> list[float]:
    if logprobs is None or len(logprobs) != expected:
        raise RuntimeError(
            f"vLLM logprob/token mismatch: {None if logprobs is None else len(logprobs)} != {expected}"
        )
    result: list[float] = []
    for index, step in enumerate(logprobs):
        values = sorted(
            (float(value.logprob) for value in step.values()), reverse=True
        )[:20]
        if len(values) != 20 or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"invalid top20 logprobs at token {index}: count={len(values)}")
        tensor = torch.tensor(values, dtype=torch.float32)
        probabilities = torch.softmax(tensor, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        result.append(float(entropy))
    return result


def state_update(path: Path, **values: Any) -> None:
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    current.update(values)
    current["updated_at"] = utc_now()
    atomic_json(current, path)


def dense_pass(
    tasks: list[tuple[str, str, dict[str, Any]]],
    output_root: Path,
    worker: str,
    fingerprint: str,
    config: dict[str, Any],
    model_path: Path,
    tokenizer: Any,
    gpu_memory_utilization: float,
    state_path_value: Path,
) -> None:
    from vllm import SamplingParams

    pending = []
    for dataset, split, record in tasks:
        problem_id = str(record["problem_id"])
        final = artifact_path(output_root, dataset, split, problem_id)
        stage = stage_path(output_root, worker, "dense", dataset, split, problem_id)
        if valid_artifact(final, fingerprint, problem_id) or valid_stage(
            stage, fingerprint, problem_id, "dense"
        ):
            continue
        if final.exists() or stage.exists():
            raise RuntimeError(f"refusing incompatible Dense destination for {problem_id}")
        pending.append((dataset, split, record, stage))
    state_update(state_path_value, phase="dense", phase_assigned=len(pending), phase_completed=0)
    if not pending:
        return
    llm = make_llm(model_path, config, gpu_memory_utilization)
    assert_torch_determinism()
    generation = config["generation"]
    try:
        batch_size = int(config["vllm"]["request_batch_size"])
        for start in range(0, len(pending), batch_size):
            local = pending[start : start + batch_size]
            prepared = []
            for dataset, split, record, stage in local:
                problem_id = str(record["problem_id"])
                prompt_text = render_prompt(tokenizer, str(record["question"]))
                prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()
                if len(prompt_ids) > int(config["vllm"]["audited_max_prompt_tokens"]):
                    raise RuntimeError(
                        f"prompt exceeds audited maximum for {problem_id}: {len(prompt_ids)}"
                    )
                if (
                    len(prompt_ids) + int(generation["dense_max_new_tokens"])
                    > int(config["vllm"]["max_model_len"])
                ):
                    raise RuntimeError(f"Dense request exceeds max_model_len for {problem_id}")
                problem_seed = stable_seed(int(config["seed"]), problem_id)
                params = SamplingParams(
                    n=1,
                    temperature=float(generation["temperature"]),
                    top_p=float(generation["top_p"]),
                    top_k=int(generation["top_k"]),
                    seed=problem_seed,
                    max_tokens=int(generation["dense_max_new_tokens"]),
                    logprobs=20,
                    detokenize=False,
                )
                prepared.append(
                    (dataset, split, record, stage, problem_id, prompt_text, prompt_ids, problem_seed, params)
                )
            started = time.perf_counter()
            requests = llm.generate(
                [
                    {"prompt_token_ids": [int(token) for token in value[6]]}
                    for value in prepared
                ],
                [value[8] for value in prepared],
                use_tqdm=False,
            )
            batch_wall_ms = 1000.0 * (time.perf_counter() - started)
            if len(requests) != len(prepared):
                raise RuntimeError(f"vLLM Dense batch output mismatch: {len(requests)} != {len(prepared)}")
            for offset, (request, values) in enumerate(zip(requests, prepared), start=1):
                dataset, split, record, stage, problem_id, prompt_text, prompt_ids, problem_seed, _ = values
                count = start + offset
                if len(request.outputs) != 1:
                    raise RuntimeError(f"unexpected vLLM output count for {problem_id}")
                output = request.outputs[0]
                tokens = [int(token) for token in output.token_ids]
                maximum = int(generation["dense_max_new_tokens"])
                if not tokens or len(tokens) > maximum:
                    raise RuntimeError(f"invalid Dense length for {problem_id}: {len(tokens)}")
                if [int(token) for token in request.prompt_token_ids] != prompt_ids:
                    raise RuntimeError(f"vLLM changed prompt token IDs for {problem_id}")
                dense = {
                    "tokens": tokens,
                    "text": tokenizer.decode(tokens, skip_special_tokens=True),
                    "entropies_top20": top20_entropies(output.logprobs, len(tokens)),
                    "wall_ms": batch_wall_ms / len(prepared),
                    "reached_max_tokens": len(tokens) == maximum,
                    "vllm_finish_reason": output.finish_reason,
                }
                checkpoints, trajectory = paragraph_checkpoints(tokenizer, tokens)
                value = {
                    "status": "complete",
                    "stage": "dense",
                    "protocol_fingerprint": fingerprint,
                    "dataset": dataset,
                    "split": split,
                    "problem_id": problem_id,
                    "problem_seed": problem_seed,
                    "record": record,
                    "prompt_text": prompt_text,
                    "prompt_token_ids": prompt_ids,
                    "dense": dense,
                    "schedule_checkpoints": checkpoints,
                    "trajectory": trajectory,
                }
                atomic_torch_save(value, stage)
                state_update(
                    state_path_value,
                    phase="dense",
                    phase_assigned=len(pending),
                    phase_completed=count,
                    latest_problem_id=problem_id,
                )
                print(
                    json.dumps(
                        {
                            "status": "dense_complete",
                            "completed": count,
                            "assigned": len(pending),
                            "problem_id": problem_id,
                            "dense_tokens": len(tokens),
                            "reached_max": len(tokens) == maximum,
                        }
                    ),
                    flush=True,
                )
    finally:
        release_llm(llm)


def hidden_pass(
    tasks: list[tuple[str, str, dict[str, Any]]],
    output_root: Path,
    worker: str,
    fingerprint: str,
    config: dict[str, Any],
    model_path: Path,
    gpu_memory_utilization: float,
    state_path_value: Path,
) -> None:
    from vllm import SamplingParams

    pending = []
    for dataset, split, record in tasks:
        problem_id = str(record["problem_id"])
        final = artifact_path(output_root, dataset, split, problem_id)
        dense = stage_path(output_root, worker, "dense", dataset, split, problem_id)
        hidden = stage_path(output_root, worker, "hidden", dataset, split, problem_id)
        if valid_artifact(final, fingerprint, problem_id) or valid_stage(
            hidden, fingerprint, problem_id, "hidden"
        ):
            continue
        if not valid_stage(dense, fingerprint, problem_id, "dense"):
            raise RuntimeError(f"missing compatible Dense stage for {problem_id}")
        if hidden.exists():
            raise RuntimeError(f"refusing incompatible hidden stage: {hidden}")
        pending.append((dataset, split, record, dense, hidden))
    state_update(state_path_value, phase="hidden", phase_assigned=len(pending), phase_completed=0)
    if not pending:
        return
    scratch = output_root / "hidden_connector_scratch" / worker
    llm = make_llm(
        model_path,
        config,
        gpu_memory_utilization,
        hidden_directory=scratch,
    )
    assert_torch_determinism()
    params = SamplingParams(temperature=0.0, seed=0, max_tokens=1, detokenize=False)
    hidden_size = int(config["model"]["hidden_size"])
    try:
        batch_size = int(config["vllm"]["request_batch_size"])
        for start in range(0, len(pending), batch_size):
            local = pending[start : start + batch_size]
            prepared = []
            for dataset, split, record, dense_path, hidden_path in local:
                stage = torch.load(dense_path, map_location="cpu", weights_only=False)
                prompt_ids = [int(token) for token in stage["prompt_token_ids"]]
                dense_tokens = [int(token) for token in stage["dense"]["tokens"]]
                replay_ids = prompt_ids + dense_tokens
                if len(replay_ids) + 1 > int(config["vllm"]["max_model_len"]):
                    raise RuntimeError(
                        f"replay exceeds max_model_len for {stage['problem_id']}: {len(replay_ids)}"
                    )
                prepared.append((dataset, split, record, hidden_path, stage, prompt_ids, replay_ids))
            requests = llm.generate(
                [{"prompt_token_ids": value[6]} for value in prepared],
                [params for _ in prepared],
                use_tqdm=False,
            )
            if len(requests) != len(prepared):
                raise RuntimeError(f"vLLM hidden batch output mismatch: {len(requests)} != {len(prepared)}")
            for offset, (request, values) in enumerate(zip(requests, prepared), start=1):
                dataset, split, record, hidden_path, stage, prompt_ids, replay_ids = values
                count = start + offset
                if [int(token) for token in request.prompt_token_ids] != replay_ids:
                    raise RuntimeError(f"hidden replay prompt changed for {stage['problem_id']}")
                kv_params = request.kv_transfer_params or {}
                filename = kv_params.get("hidden_states_path")
                if not filename:
                    raise RuntimeError(f"hidden connector returned no path for {stage['problem_id']}")
                connector_path = Path(filename)
                with safe_open(connector_path, framework="pt", device="cpu") as handle:
                    token_ids = handle.get_tensor("token_ids")
                    all_hidden = handle.get_tensor("hidden_states")
                connector_path.unlink()
                if token_ids.tolist() != replay_ids:
                    raise RuntimeError(f"hidden connector token IDs diverged for {stage['problem_id']}")
                expected_shape = (len(replay_ids), 1, hidden_size)
                if tuple(all_hidden.shape) != expected_shape:
                    raise RuntimeError(
                        f"hidden shape mismatch for {stage['problem_id']}: "
                        f"{tuple(all_hidden.shape)} != {expected_shape}"
                    )
                checkpoints = [int(value) for value in stage["schedule_checkpoints"]]
                indices = [len(prompt_ids) + checkpoint - 1 for checkpoint in checkpoints]
                selected = (
                    all_hidden[indices].float().to(torch.float16).contiguous()
                    if indices
                    else torch.empty((0, 1, hidden_size), dtype=torch.float16)
                )
                value = {
                    "status": "complete",
                    "stage": "hidden",
                    "protocol_fingerprint": fingerprint,
                    "dataset": dataset,
                    "split": split,
                    "problem_id": str(stage["problem_id"]),
                    "hidden": selected,
                    "hidden_shape": list(selected.shape),
                    "capture_layers": [int(config["features"]["layer_zero_based"])],
                    "replay_token_count": len(replay_ids),
                    "replay_token_ids_sha256": sha256_json(replay_ids),
                    "selection_indices": indices,
                    "token_ids_exact": True,
                }
                atomic_torch_save(value, hidden_path)
                state_update(
                    state_path_value,
                    phase="hidden",
                    phase_assigned=len(pending),
                    phase_completed=count,
                    latest_problem_id=str(stage["problem_id"]),
                )
                print(
                    json.dumps(
                        {
                            "status": "hidden_complete",
                            "completed": count,
                            "assigned": len(pending),
                            "problem_id": stage["problem_id"],
                            "hidden_shape": list(selected.shape),
                        }
                    ),
                    flush=True,
                )
                del all_hidden, selected
    finally:
        release_llm(llm)


def branch_result(tokenizer: Any, suffix_ids: list[int], output: Any, wall_ms: float) -> dict[str, Any]:
    tokens = [int(token) for token in output.token_ids]
    generated = tokenizer.decode(tokens, skip_special_tokens=True)
    suffix = tokenizer.decode(suffix_ids, skip_special_tokens=True)
    return {
        "tokens": tokens,
        "generated_text": generated,
        "text": suffix + generated,
        "wall_ms": wall_ms,
        "vllm_finish_reason": output.finish_reason,
    }


def finalize_pass(
    tasks: list[tuple[str, str, dict[str, Any]]],
    output_root: Path,
    worker: str,
    physical_gpu: int,
    fingerprint: str,
    config: dict[str, Any],
    model_path: Path,
    tokenizer: Any,
    model_audit: dict[str, Any],
    data_audit: dict[str, Any],
    reproducibility_audit: dict[str, Any],
    engine_audit: dict[str, Any],
    gpu_memory_utilization: float,
    state_path_value: Path,
) -> tuple[int, int]:
    from vllm import SamplingParams

    pending = []
    skipped = 0
    for dataset, split, record in tasks:
        problem_id = str(record["problem_id"])
        final = artifact_path(output_root, dataset, split, problem_id)
        if valid_artifact(final, fingerprint, problem_id):
            skipped += 1
            continue
        if final.exists():
            raise RuntimeError(f"refusing incompatible final artifact: {final}")
        dense = stage_path(output_root, worker, "dense", dataset, split, problem_id)
        hidden = stage_path(output_root, worker, "hidden", dataset, split, problem_id)
        if not valid_stage(dense, fingerprint, problem_id, "dense"):
            raise RuntimeError(f"missing Dense stage for finalization: {problem_id}")
        if not valid_stage(hidden, fingerprint, problem_id, "hidden"):
            raise RuntimeError(f"missing hidden stage for finalization: {problem_id}")
        pending.append((dataset, split, record, dense, hidden, final))
    state_update(state_path_value, phase="branches", phase_assigned=len(pending), phase_completed=0)
    if not pending:
        return 0, skipped
    llm = make_llm(model_path, config, gpu_memory_utilization)
    assert_torch_determinism()
    generation = config["generation"]
    suffix_ids = tokenizer(
        generation["force_answer_suffix"], add_special_tokens=False
    ).input_ids
    params = SamplingParams(
        temperature=0.0,
        seed=0,
        max_tokens=int(generation["force_answer_max_new_tokens"]),
        detokenize=False,
    )
    completed = 0
    try:
        for dataset, split, record, dense_path, hidden_path, final in pending:
            dense_stage = torch.load(dense_path, map_location="cpu", weights_only=False)
            hidden_stage = torch.load(hidden_path, map_location="cpu", weights_only=False)
            prompt_ids = [int(token) for token in dense_stage["prompt_token_ids"]]
            dense = dict(dense_stage["dense"])
            dense_tokens = [int(token) for token in dense["tokens"]]
            checkpoints = [int(value) for value in dense_stage["schedule_checkpoints"]]
            prompts = [
                {"prompt_token_ids": prompt_ids + dense_tokens[:checkpoint] + suffix_ids}
                for checkpoint in checkpoints
            ]
            cap_index: int | None = None
            if bool(dense["reached_max_tokens"]):
                cap_index = len(prompts)
                prompts.append({"prompt_token_ids": prompt_ids + dense_tokens + suffix_ids})
            for prompt in prompts:
                if (
                    len(prompt["prompt_token_ids"])
                    + int(generation["force_answer_max_new_tokens"])
                    > int(config["vllm"]["max_model_len"])
                ):
                    raise RuntimeError(
                        f"branch exceeds max_model_len for {record['problem_id']}"
                    )
            started = time.perf_counter()
            requests = llm.generate(prompts, params, use_tqdm=False) if prompts else []
            total_wall_ms = 1000.0 * (time.perf_counter() - started)
            if len(requests) != len(prompts):
                raise RuntimeError(
                    f"vLLM branch output mismatch for {record['problem_id']}: "
                    f"{len(requests)} != {len(prompts)}"
                )
            for request, prompt in zip(requests, prompts):
                if [int(token) for token in request.prompt_token_ids] != prompt[
                    "prompt_token_ids"
                ]:
                    raise RuntimeError(
                        f"vLLM changed branch prompt token IDs for {record['problem_id']}"
                    )
                if len(request.outputs) != 1:
                    raise RuntimeError(
                        f"unexpected branch output count for {record['problem_id']}"
                    )
            each_wall_ms = total_wall_ms / len(requests) if requests else 0.0
            branches = [
                branch_result(tokenizer, suffix_ids, request.outputs[0], each_wall_ms)
                for request in requests
            ]
            cap_branch = branches[cap_index] if cap_index is not None else None
            gold = gold_for(dataset, record)
            raw_dense_prediction = prediction(dataset, dense["text"])
            if cap_branch is not None:
                dense_prediction = prediction(dataset, cap_branch["text"])
                dense_grader = "forced_answer_at_exact_13k_prefix"
            else:
                dense_prediction = raw_dense_prediction
                dense_grader = "natural_dense_completion"
            dense_success = success(dataset, gold, dense_prediction)
            rows: list[dict[str, Any]] = []
            for index, checkpoint in enumerate(checkpoints):
                branch = branches[index]
                current_prediction = prediction(dataset, branch["text"])
                current_success = success(dataset, gold, current_prediction)
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "problem_id": str(record["problem_id"]),
                        "checkpoint": checkpoint,
                        "checkpoint_schedules": ["paragraph"],
                        "actual_checkpoint_schedule": "paragraph",
                        "gold_answer": gold,
                        "dense_prediction": dense_prediction,
                        "dense_success": bool(dense_success),
                        "dense_tokens": len(dense_tokens),
                        "dense_wall_ms": float(dense["wall_ms"]),
                        "current_prediction": current_prediction,
                        "current_success": bool(current_success),
                        "consistency": bool(
                            current_prediction is not None
                            and dense_prediction is not None
                            and current_prediction == dense_prediction
                        ),
                        "correction": bool((not current_success) and dense_success),
                        "damage": bool(current_success and (not dense_success)),
                        "branch_tokens": len(branch["tokens"]),
                        "branch_token_ids": branch["tokens"],
                        "branch_text": branch["text"],
                        "branch_generated_text": branch["generated_text"],
                        "branch_collection_wall_ms": float(branch["wall_ms"]),
                        "forced_answer_decoding": "greedy_argmax",
                        "forced_answer_do_sample": False,
                        "prompt_tokens": len(prompt_ids),
                        "prefix_context_tokens": len(prompt_ids) + checkpoint,
                        "prefix_mean_entropy_tail8": tail_mean(
                            dense["entropies_top20"], checkpoint
                        ),
                        "producer_gpu": physical_gpu,
                    }
                )
            dense.update(
                {
                    "content_tokens": dense_tokens,
                    "raw_completion_prediction": raw_dense_prediction,
                    "prediction": dense_prediction,
                    "success": bool(dense_success),
                    "grader": dense_grader,
                    "cap_forced_answer": cap_branch,
                    "reasoning_tokens": len(dense_tokens),
                }
            )
            artifact = {
                "schema_version": 1,
                "status": "complete",
                "protocol_id": config["protocol_id"],
                "protocol_fingerprint": fingerprint,
                "primary_replay_view_fingerprint": fingerprint + ":paragraph",
                "dataset": dataset,
                "split": split,
                "problem_id": str(record["problem_id"]),
                "dtype": "bfloat16",
                "seed": int(config["seed"]),
                "problem_seed": int(dense_stage["problem_seed"]),
                "reproducibility": reproducibility_audit,
                "vllm_engine": engine_audit,
                "data_identity": data_audit,
                "actual_checkpoint_schedule": "paragraph",
                "checkpoint_protocol": config["checkpoint"],
                "capture_layers": [int(config["features"]["layer_zero_based"])],
                "rows": rows,
                "hidden": hidden_stage["hidden"],
                "record": record,
                "gold_answer": gold,
                "prompt_text": dense_stage["prompt_text"],
                "prompt_tokens": len(prompt_ids),
                "prompt_token_ids": prompt_ids,
                "dense": dense,
                "dense_generation": {
                    "requested_max_new_tokens": int(generation["dense_max_new_tokens"]),
                    "temperature": float(generation["temperature"]),
                    "top_p": float(generation["top_p"]),
                    "top_k": int(generation["top_k"]),
                    "do_sample": bool(generation["do_sample"]),
                    "per_problem_seed": True,
                },
                "forced_answer_decoding": {
                    "strategy": "greedy_argmax",
                    "do_sample": False,
                    "max_new_tokens": int(generation["force_answer_max_new_tokens"]),
                    "suffix": generation["force_answer_suffix"],
                },
                "trajectory": dense_stage["trajectory"],
                "schedule_checkpoints": checkpoints,
                "model_audit": model_audit,
                "hidden_replay_audit": {
                    key: hidden_stage[key]
                    for key in (
                        "replay_token_count",
                        "replay_token_ids_sha256",
                        "selection_indices",
                        "token_ids_exact",
                    )
                },
                "collection": {
                    "worker": worker,
                    "host": socket.gethostname(),
                    "gpu": physical_gpu,
                    "device": reproducibility_audit["environment"]["gpu"]["name"],
                    "branch_wall_ms": total_wall_ms,
                    "created_at": utc_now(),
                },
            }
            atomic_torch_save(artifact, final)
            completed += 1
            state_update(
                state_path_value,
                phase="branches",
                phase_assigned=len(pending),
                phase_completed=completed,
                latest_problem_id=str(record["problem_id"]),
            )
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "completed": completed,
                        "assigned": len(pending),
                        "problem_id": record["problem_id"],
                        "dense_tokens": len(dense_tokens),
                        "dense_success": bool(dense_success),
                        "dense_grader": dense_grader,
                        "checkpoints": len(checkpoints),
                    }
                ),
                flush=True,
            )
    finally:
        release_llm(llm)
    return completed, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--problem-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phase", choices=("dense", "hidden", "branches"), required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not math.isclose(
        args.gpu_memory_utilization,
        float(config["vllm"]["gpu_memory_utilization"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "gpu_memory_utilization is not the frozen value: "
            f"{args.gpu_memory_utilization} != {config['vllm']['gpu_memory_utilization']}"
        )
    frozen_environment = enforce_vllm_environment(config)
    strict_settings = strict_reproducibility(seed=0, num_threads=1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible GPU, got {torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    runtime_identity = environment_provenance(device)
    runtime_lock_path = Path(config["reproducibility"]["runtime_lock"])
    if not runtime_lock_path.is_absolute():
        runtime_lock_path = ROOT / runtime_lock_path
    runtime_lock_audit = enforce_runtime_lock(runtime_lock_path, runtime_identity)
    full_environment_lock_path = Path(
        config["reproducibility"]["full_environment_lock"]
    )
    if not full_environment_lock_path.is_absolute():
        full_environment_lock_path = ROOT / full_environment_lock_path
    full_environment_lock_audit = enforce_full_environment_lock(
        full_environment_lock_path
    )
    engine_audit = vllm_engine_audit(config)
    code_identity = code_provenance(
        ROOT,
        (
            "configs/qwen3_14b_deterministic_ood13k_vllm_full_v1.yaml",
            "configs/qwen3_14b_vllm_full_v1_requirements.txt",
            "scripts/collect_qwen3_14b_vllm_full_v1.py",
            "scripts/run_qwen3_14b_vllm_worker_v1.py",
            "scripts/deepseek7b_protocol_v1.py",
            "src/qwen3_reasoning.py",
            "src/reproducibility.py",
        ),
    )
    prepared_root = (args.prepared_root or Path(config["data"]["prepared_root"])).resolve()
    model_path = (args.model_path or Path(config["model"]["local_path"])).resolve()
    output_root = (args.output_root or Path(config["output_root"])).resolve()
    data_audit = data_identity(prepared_root)
    model_audit = inspect_qwen3(model_path)
    if (
        model_audit["hidden_size"] != int(config["model"]["hidden_size"])
        or model_audit["layers"] != int(config["model"]["num_hidden_layers"])
    ):
        raise RuntimeError(f"model/config mismatch: {model_audit}")
    scientific_model_audit = {key: value for key, value in model_audit.items() if key != "path"}
    fingerprint = canonical_fingerprint(
        {
            "config": config,
            "model": scientific_model_audit,
            "data": data_audit,
            "formal_reproducibility": {
                "protocol_id": strict_settings["protocol_id"],
                "runtime_lock_id": runtime_lock_audit["lock_id"],
                "runtime_lock_sha256": sha256_file(runtime_lock_path),
                "full_environment_lock_sha256": full_environment_lock_audit["sha256"],
                "git_commit": code_identity["git"]["commit"],
                "source_sha256": code_identity["source_sha256"],
                "vllm_engine": engine_audit,
            },
        }
    )
    reproducibility_audit = {
        "settings": strict_settings,
        "vllm_environment": frozen_environment,
        "runtime_lock": runtime_lock_audit,
        "full_environment_lock": full_environment_lock_audit,
        "environment": runtime_identity,
        "code": code_identity,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    expected_prompt_protocol = {
        "render": "apply_chat_template_tokenize_false_add_generation_prompt_true",
        "encode": "tokenizer_default_add_special_tokens",
        "reference": "scripts/collect_deepseek7b_paragraph_v1.py",
        "thinking_mode": "qwen3_chat_template_default_enabled",
    }
    if config.get("prompt_tokenization") != expected_prompt_protocol:
        raise ValueError("prompt tokenization is not the frozen DeepSeek path")

    task_pool = all_tasks(prepared_root)
    if args.problem_id:
        selected = set(args.problem_id)
        task_pool = [task for task in task_pool if str(task[2]["problem_id"]) in selected]
        found = {str(task[2]["problem_id"]) for task in task_pool}
        if found != selected:
            raise ValueError(f"unknown problem ids: {sorted(selected - found)}")
    tasks = [
        task
        for index, task in enumerate(task_pool)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    state_path_value = output_root / "workers" / f"{args.worker_id}.state.json"
    state_update(
        state_path_value,
        status="running",
        worker=args.worker_id,
        physical_gpu=args.physical_gpu,
        visible_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        assigned=len(tasks),
        protocol_fingerprint=fingerprint,
        active_phase=args.phase,
        phase_started_at=utc_now(),
    )
    started = time.time()
    failures = 0
    try:
        if args.phase == "dense":
            dense_pass(
                tasks,
                output_root,
                args.worker_id,
                fingerprint,
                config,
                model_path,
                tokenizer,
                args.gpu_memory_utilization,
                state_path_value,
            )
        elif args.phase == "hidden":
            hidden_pass(
                tasks,
                output_root,
                args.worker_id,
                fingerprint,
                config,
                model_path,
                args.gpu_memory_utilization,
                state_path_value,
            )
        else:
            completed, skipped = finalize_pass(
                tasks,
                output_root,
                args.worker_id,
                args.physical_gpu,
                fingerprint,
                config,
                model_path,
                tokenizer,
                model_audit,
                data_audit,
                reproducibility_audit,
                engine_audit,
                args.gpu_memory_utilization,
                state_path_value,
            )
    except Exception as error:
        failures = 1
        state_update(
            state_path_value,
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    if args.phase != "branches":
        state_update(
            state_path_value,
            status="phase_complete",
            phase=f"{args.phase}_complete",
            phase_completed_at=utc_now(),
        )
        print(
            json.dumps(
                {
                    "status": "phase_complete",
                    "phase": args.phase,
                    "worker": args.worker_id,
                    "assigned": len(tasks),
                    "protocol_fingerprint": fingerprint,
                    "elapsed_seconds": time.time() - started,
                }
            ),
            flush=True,
        )
        return
    summary = {
        "status": "complete" if failures == 0 else "failed",
        "worker": args.worker_id,
        "physical_gpu": args.physical_gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned": len(tasks),
        "completed": completed,
        "skipped": skipped,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
        "protocol_fingerprint": fingerprint,
        "data_identity": data_audit,
        "model_audit": model_audit,
        "vllm_engine": engine_audit,
        "reproducibility": reproducibility_audit,
    }
    summary_path = output_root / "workers" / f"{args.worker_id}.json"
    atomic_json(summary, summary_path)
    state_update(state_path_value, status="complete", phase="complete")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

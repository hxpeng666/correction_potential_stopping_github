"""Strict reproducibility helpers for probe-training experiments.

The experiment contract is intentionally narrower than generic PyTorch
reproducibility: a formal run must come from a clean Git commit and must use the
same deterministic CUDA/CPU settings.  Result files can therefore be traced to
one code revision and compared by content hash rather than by rounded metrics.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


DETERMINISM_PROTOCOL_ID = "cps_strict_determinism_v1"
REQUIRED_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def sha256_state_dict(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
        digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True
    ).strip()


def git_provenance(root: str | Path, *, require_clean: bool = True) -> dict[str, Any]:
    project = Path(root).resolve()
    commit = _git(project, "rev-parse", "HEAD")
    branch = _git(project, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(project, "status", "--porcelain", "--untracked-files=all")
    dirty_entries = [line for line in status.splitlines() if line.strip()]
    if require_clean and dirty_entries:
        preview = "\n".join(dirty_entries[:20])
        raise RuntimeError(
            "formal experiment requires a clean Git worktree; commit or remove "
            f"these changes first:\n{preview}"
        )
    try:
        remote = _git(project, "remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        remote = None
    return {
        "commit": commit,
        "branch": branch,
        "remote": remote,
        "dirty": bool(dirty_entries),
        "dirty_entries": dirty_entries,
    }


def code_provenance(root: str | Path, relative_paths: Iterable[str]) -> dict[str, Any]:
    project = Path(root).resolve()
    files = {}
    for relative in sorted(set(relative_paths)):
        path = project / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing provenance source: {path}")
        files[relative] = sha256_file(path)
    return {
        "git": git_provenance(project, require_clean=True),
        "source_sha256": files,
    }


def strict_reproducibility(seed: int = 0, *, num_threads: int = 1) -> dict[str, Any]:
    mismatches = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in REQUIRED_ENVIRONMENT.items()
        if os.environ.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "strict reproducibility environment is not frozen: "
            + json.dumps(mismatches, sort_keys=True)
        )
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(num_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch allows this setting only before inter-op work begins.
            pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "protocol_id": DETERMINISM_PROTOCOL_ID,
        "seed": seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "num_threads": torch.get_num_threads(),
        "environment": dict(REQUIRED_ENVIRONMENT),
    }


def environment_provenance(device: torch.device | None = None) -> dict[str, Any]:
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "PyYAML",
        "transformers",
        "huggingface-hub",
        "safetensors",
        "tokenizers",
        "vllm",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if device is not None and device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        result["gpu"] = {
            "logical_index": index,
            "name": properties.name,
            "total_memory": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
        try:
            property_uuid = getattr(properties, "uuid", None)
            if property_uuid is None:
                uuid = None
            else:
                raw_uuid = str(property_uuid)
                uuid = raw_uuid if raw_uuid.startswith("GPU-") else f"GPU-{raw_uuid}"
            driver = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip().splitlines()[0].strip()
            if uuid is None:
                line = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={index}",
                        "--query-gpu=uuid",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip().splitlines()[0]
                uuid = line.strip()
            result["gpu"].update({"uuid": uuid, "driver": driver})
        except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
            result["gpu"].update({"uuid": None, "driver": None})
    return result


def enforce_runtime_lock(
    lock_path: str | Path,
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the scientific runtime exactly matches a lock file.

    GPU UUIDs are represented as an allow-list so two separately certified,
    otherwise identical A100s can be used.  Every other locked field must match
    exactly.  A changed driver or package version therefore requires a new lock,
    Git commit, and bitwise reproducibility gate rather than a silent rerun.
    """
    path = Path(lock_path).resolve()
    expected = json.loads(path.read_text(encoding="utf-8"))
    mismatches: dict[str, Any] = {}

    for key in ("python", "platform", "cuda_runtime", "cudnn"):
        if expected.get(key) != observed.get(key):
            mismatches[key] = {
                "expected": expected.get(key),
                "observed": observed.get(key),
            }

    expected_packages = expected.get("packages", {})
    observed_packages = observed.get("packages", {})
    for name, version in expected_packages.items():
        if observed_packages.get(name) != version:
            mismatches[f"packages.{name}"] = {
                "expected": version,
                "observed": observed_packages.get(name),
            }

    expected_gpu = expected.get("gpu")
    observed_gpu = observed.get("gpu")
    if expected_gpu is not None:
        if observed_gpu is None:
            mismatches["gpu"] = {"expected": expected_gpu, "observed": None}
        else:
            exact_gpu_fields = ("name", "total_memory", "compute_capability", "driver")
            for key in exact_gpu_fields:
                if expected_gpu.get(key) != observed_gpu.get(key):
                    mismatches[f"gpu.{key}"] = {
                        "expected": expected_gpu.get(key),
                        "observed": observed_gpu.get(key),
                    }
            allowed_uuids = expected_gpu.get("allowed_uuids", [])
            if observed_gpu.get("uuid") not in allowed_uuids:
                mismatches["gpu.uuid"] = {
                    "expected_one_of": allowed_uuids,
                    "observed": observed_gpu.get("uuid"),
                }

    if mismatches:
        raise RuntimeError(
            "runtime does not match the committed reproducibility lock: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "status": "matched",
        "lock_id": expected.get("lock_id"),
        "path": str(path),
        "sha256": sha256_file(path),
    }


def deterministic_subprocess_environment(seed: int = 0) -> dict[str, str]:
    if seed != 0:
        raise ValueError("strict v1 currently freezes seed=0")
    environment = os.environ.copy()
    environment.update(REQUIRED_ENVIRONMENT)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    return environment

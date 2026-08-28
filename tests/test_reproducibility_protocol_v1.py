from __future__ import annotations

import subprocess

import numpy as np
import pytest
import torch

from src.reproducibility import (
    REQUIRED_ENVIRONMENT,
    deterministic_subprocess_environment,
    enforce_runtime_lock,
    git_provenance,
    sha256_array,
    sha256_state_dict,
)


def test_array_hash_includes_shape_and_dtype() -> None:
    values = np.arange(6, dtype=np.float32)
    assert sha256_array(values) != sha256_array(values.reshape(2, 3))
    assert sha256_array(values) != sha256_array(values.astype(np.float64))
    assert sha256_array(values.copy()) == sha256_array(values)


def test_state_hash_is_key_order_independent() -> None:
    left = {
        "b": torch.tensor([2.0], dtype=torch.float32),
        "a": torch.tensor([1.0], dtype=torch.float32),
    }
    right = {"a": left["a"].clone(), "b": left["b"].clone()}
    assert sha256_state_dict(left) == sha256_state_dict(right)
    right["b"][0] = 3.0
    assert sha256_state_dict(left) != sha256_state_dict(right)


def test_deterministic_subprocess_environment_is_frozen() -> None:
    environment = deterministic_subprocess_environment(seed=0)
    for key, value in REQUIRED_ENVIRONMENT.items():
        assert environment[key] == value
    with pytest.raises(ValueError):
        deterministic_subprocess_environment(seed=1)


def test_dirty_git_worktree_is_rejected(tmp_path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "test"], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "freeze"], check=True)
    assert git_provenance(tmp_path, require_clean=True)["dirty"] is False
    tracked.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        git_provenance(tmp_path, require_clean=True)


def test_runtime_lock_matches_only_certified_environment(tmp_path) -> None:
    observed = {
        "python": "3.12.12",
        "executable": "/env/bin/python",
        "platform": "linux",
        "packages": {"torch": "2.7.1", "numpy": "2.2.6"},
        "cuda_runtime": "12.6",
        "cudnn": 90501,
        "gpu": {
            "name": "NVIDIA A100 80GB PCIe",
            "total_memory": 84974239744,
            "compute_capability": [8, 0],
            "uuid": "GPU-certified-1",
            "driver": "550.120",
        },
    }
    lock = {
        "lock_id": "test-lock",
        **{key: observed[key] for key in (
            "python", "platform", "packages",
            "cuda_runtime", "cudnn",
        )},
        "gpu": {
            **{key: observed["gpu"][key] for key in (
                "name", "total_memory", "compute_capability", "driver",
            )},
            "allowed_uuids": ["GPU-certified-0", "GPU-certified-1"],
        },
    }
    path = tmp_path / "runtime.json"
    path.write_text(__import__("json").dumps(lock), encoding="utf-8")
    assert enforce_runtime_lock(path, observed)["status"] == "matched"

    changed = {**observed, "packages": {**observed["packages"], "torch": "2.8.0"}}
    with pytest.raises(RuntimeError, match="runtime does not match"):
        enforce_runtime_lock(path, changed)

from __future__ import annotations

import subprocess

import numpy as np
import pytest
import torch

from src.reproducibility import (
    REQUIRED_ENVIRONMENT,
    deterministic_subprocess_environment,
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

"""Shared protocol, queue, and artifact helpers for greedy forced-answer v1."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]

from src.final_paper_protocol import canonical_fingerprint
from src.utils import load_yaml


QUEUE_STATES = ("pending", "claimed", "done", "failed", "requires_a100")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(resolve(path))


def protocol_contract(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": config["protocol_id"],
        "parent_protocol_id": config["parent_protocol_id"],
        "seed": config["seed"],
        "model": config["model"],
        "forced_answer_decoding": config["forced_answer_decoding"],
        "preservation": config["preservation"],
        "datasets": config["datasets"],
    }


def protocol_fingerprint(config: dict[str, Any]) -> str:
    return canonical_fingerprint(protocol_contract(config))


def task_name(dataset: str, split: str, problem_id: str) -> str:
    raw = f"{dataset}:{split}:{problem_id}:greedy_forced_v1"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest() + ".json"


def queue_dirs(config: dict[str, Any]) -> dict[str, Path]:
    root = resolve(config["queue_root"])
    result = {state: root / state for state in QUEUE_STATES}
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    (root / "workers").mkdir(parents=True, exist_ok=True)
    return result


def output_path(
    config: dict[str, Any], dataset: str, split: str, problem_id: str
) -> Path:
    return (
        resolve(config["cache_root"])
        / dataset
        / "merged"
        / split
        / f"sample_{problem_id}.pt"
    )


def artifact_valid(path: Path, fingerprint: str, problem_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    decoding = value.get("forced_answer_decoding", {})
    return (
        value.get("status") == "complete"
        and str(value.get("problem_id")) == str(problem_id)
        and value.get("protocol_fingerprint") == fingerprint
        and decoding.get("strategy") == "greedy_argmax"
        and decoding.get("do_sample") is False
    )


def source_split_path(dataset_config: dict[str, Any], split: str) -> Path:
    """Resolve either legacy ``merged/<split>`` or direct ``<split>`` views."""
    source_root = resolve(dataset_config["source_selected_cache_root"])
    subdirectory = str(dataset_config.get("source_samples_subdir", "merged"))
    return source_root / subdirectory / split


def atomic_json(payload: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, prefix=target.name, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def ensure_task(config: dict[str, Any], payload: dict[str, Any]) -> bool:
    directories = queue_dirs(config)
    name = task_name(payload["dataset"], payload["split"], payload["problem_id"])
    if any((directories[state] / name).exists() for state in QUEUE_STATES):
        return False
    target = directories["pending"] / name
    payload = {**payload, "task_name": name, "created_at": now()}
    try:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        return True
    except FileExistsError:
        return False


def claim_task(config: dict[str, Any], worker_id: str, source_state: str = "pending"):
    directories = queue_dirs(config)
    for source in directories[source_state].glob("*.json"):
        claimed = directories["claimed"] / f"{source.stem}.{worker_id}.{os.getpid()}.json"
        try:
            os.replace(source, claimed)
        except FileNotFoundError:
            continue
        payload = json.loads(claimed.read_text(encoding="utf-8"))
        payload["claim_pid"] = os.getpid()
        payload["claim_host"] = socket.gethostname()
        payload["claimed_at"] = now()
        atomic_json(payload, claimed)
        return payload, claimed
    return None


def finish_task(
    config: dict[str, Any], payload: dict[str, Any], claimed: Path, state: str
) -> None:
    if state not in {"done", "failed", "requires_a100"}:
        raise ValueError(state)
    directories = queue_dirs(config)
    clean = {key: value for key, value in payload.items() if not key.startswith("claim_")}
    clean["finished_at"] = now()
    destination = directories[state] / clean["task_name"]
    atomic_json(clean, destination)
    claimed.unlink(missing_ok=True)


def queue_counts(config: dict[str, Any]) -> dict[str, int]:
    directories = queue_dirs(config)
    return {
        state: sum(1 for _ in directories[state].glob("*.json"))
        for state in QUEUE_STATES
    }


def recover_stale_claims(config: dict[str, Any]) -> int:
    directories = queue_dirs(config)
    recovered = 0
    current_host = socket.gethostname()
    for claimed in directories["claimed"].glob("*.json"):
        payload = json.loads(claimed.read_text(encoding="utf-8"))
        pid = int(payload.get("claim_pid", -1))
        host = payload.get("claim_host")
        if host == current_host and pid > 0 and Path(f"/proc/{pid}").exists():
            continue
        destination = directories["pending"] / payload["task_name"]
        if destination.exists():
            raise RuntimeError(f"duplicate pending/claimed task: {destination}")
        os.replace(claimed, destination)
        recovered += 1
    return recovered

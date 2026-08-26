"""Shared immutable-cache, deterministic-seed, checkpoint, and filesystem-queue utilities."""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from src.final_paper_protocol import BOUNDARY, canonical_fingerprint
from src.qwen3_reasoning import inspect_qwen3
from src.utils import atomic_json, load_yaml


TASK_KINDS = ("dense", "branch")
BRANCH_DIRECT = -1


def task_seed(
    global_seed: int,
    dataset: str,
    split: str,
    sample_id: str,
    checkpoint: int | str,
) -> int:
    payload = f"{global_seed}:{dataset}:{split}:{sample_id}:{checkpoint}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def protocol_fingerprint(config_path: Path, split_manifest: Path, model_root: Path) -> str:
    config = load_yaml(config_path)
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    model = inspect_qwen3(model_root)
    protected = {
        "protocol_id": config["protocol_id"],
        "seed": config["seed"],
        "dataset": config["dataset"],
        "model": config["model"],
        "generation": config["generation"],
        "checkpoint_protocol": config["checkpoint_protocol"],
        "prompt": config["prompt"],
        "split_fingerprint": manifest["fingerprint"],
        "model_metadata_fingerprint": model["metadata_fingerprint"],
    }
    return canonical_fingerprint(protected)


def artifact_matches(path: Path, *, problem_id: str, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return (
        artifact.get("status") == "complete"
        and str(artifact.get("problem_id")) == str(problem_id)
        and artifact.get("protocol_fingerprint") == fingerprint
    )


def task_name(payload: dict[str, Any]) -> str:
    checkpoint = payload.get("checkpoint", "dense")
    raw = (
        f"{payload['dataset']}:{payload['split']}:{payload['problem_id']}:"
        f"{checkpoint}:{payload['kind']}"
    )
    return hashlib.sha256(raw.encode()).hexdigest() + ".json"


def task_directories(queue_root: Path, kind: str) -> dict[str, Path]:
    if kind not in TASK_KINDS:
        raise ValueError(kind)
    root = queue_root / kind
    result = {name: root / name for name in ("pending", "claimed", "done", "failed", "requires_a100")}
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def ensure_task(queue_root: Path, payload: dict[str, Any]) -> bool:
    directories = task_directories(queue_root, str(payload["kind"]))
    name = task_name(payload)
    if any((directories[state] / name).exists() for state in directories):
        return False
    temporary = directories["pending"] / f".{name}.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.link(temporary, directories["pending"] / name)
        created = True
    except FileExistsError:
        created = False
    finally:
        temporary.unlink(missing_ok=True)
    return created


def claim_task(
    queue_root: Path,
    kind: str,
    worker_id: str,
    *,
    source_state: str = "pending",
) -> tuple[dict[str, Any], Path] | None:
    directories = task_directories(queue_root, kind)
    source = directories[source_state]
    for path in sorted(source.glob("*.json")):
        claimed = directories["claimed"] / f"{path.stem}.{worker_id}.{os.getpid()}.json"
        try:
            os.replace(path, claimed)
        except FileNotFoundError:
            continue
        payload = json.loads(claimed.read_text(encoding="utf-8"))
        payload["_original_task_name"] = path.name
        return payload, claimed
    return None


def finish_task(claimed: Path, payload: dict[str, Any], queue_root: Path, state: str = "done") -> None:
    if state not in {"done", "failed", "requires_a100"}:
        raise ValueError(state)
    directories = task_directories(queue_root, str(payload["kind"]))
    # A successful branch artifact is the immutable completion record. Avoid
    # creating a second tiny JSON file for every checkpoint at full scale.
    if state == "done" and str(payload["kind"]) == "branch":
        claimed.unlink(missing_ok=True)
        return
    original = str(payload["_original_task_name"])
    payload = {key: value for key, value in payload.items() if not key.startswith("_")}
    payload["finished_at_unix"] = time.time()
    temporary = directories[state] / f".{original}.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, directories[state] / original)
    claimed.unlink(missing_ok=True)


def requeue_claim(claimed: Path, payload: dict[str, Any], queue_root: Path) -> None:
    directories = task_directories(queue_root, str(payload["kind"]))
    original = str(payload["_original_task_name"])
    os.replace(claimed, directories["pending"] / original)


def recover_claims(queue_root: Path, kind: str) -> int:
    directories = task_directories(queue_root, kind)
    recovered = 0
    for path in directories["claimed"].glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        original = payload.get("_original_task_name")
        if not original:
            parts = path.name.split(".")
            original = parts[0] + ".json"
        destination = directories["pending"] / str(original)
        if destination.exists():
            raise RuntimeError(f"duplicate pending/claimed task: {destination}")
        os.replace(path, destination)
        recovered += 1
    return recovered


def queue_counts(queue_root: Path, kind: str) -> dict[str, int]:
    directories = task_directories(queue_root, kind)
    return {state: len(list(path.glob("*.json"))) for state, path in directories.items()}


def raw_semantic_boundaries(tokenizer, token_ids: list[int], upper: int) -> tuple[list[int], str]:
    limited = token_ids[:upper]
    text = tokenizer.decode(limited, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded.input_ids) == limited:
        token_ends = [int(end) for _start, end in encoded.offset_mapping]
    else:
        token_ends = []
        for end in range(1, len(limited) + 1):
            prefix = tokenizer.decode(
                limited[:end],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            token_ends.append(len(prefix))
        text = tokenizer.decode(
            limited, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    checkpoints = set()
    for match in BOUNDARY.finditer(text):
        position = bisect.bisect_left(token_ends, match.end())
        if position < len(token_ends):
            checkpoints.add(position + 1)
    return sorted(checkpoints), text


def schedules_for_trace(
    tokenizer,
    content_ids: list[int],
    *,
    minimum: int,
    maximum: int,
    sentence_gap: int,
    fixed: Iterable[int],
) -> tuple[dict[str, list[int]], str]:
    upper = min(maximum, len(content_ids))
    semantic, decoded = raw_semantic_boundaries(tokenizer, content_ids, upper) if upper else ([], "")
    sentence = []
    previous = 0
    for checkpoint in semantic:
        if minimum <= checkpoint <= upper and checkpoint - previous >= sentence_gap:
            sentence.append(checkpoint)
            previous = checkpoint
    fixed_values = [int(value) for value in fixed if minimum <= int(value) <= upper]
    return {"sentence": sentence, "fixed": fixed_values}, decoded


def tail_mean(values: list[float], end: int, width: int = 8) -> float:
    local = values[max(0, end - width):end]
    return float(sum(local) / len(local)) if local else float("nan")


def cache_paths(cache_root: Path, split: str, problem_id: str) -> dict[str, Path]:
    return {
        "dense": cache_root / "dense" / split / f"sample_{problem_id}.pt",
        "branches": cache_root / "branches" / split / problem_id,
        "merged": cache_root / "merged" / split / f"sample_{problem_id}.pt",
    }


def branch_path(cache_root: Path, split: str, problem_id: str, checkpoint: int) -> Path:
    name = "direct.pt" if checkpoint == BRANCH_DIRECT else f"checkpoint_{checkpoint:04d}.pt"
    return cache_root / "branches" / split / problem_id / name


def select_records(path: Path, problem_ids: set[str] | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if problem_ids is not None:
        rows = [row for row in rows if str(row["problem_id"]) in problem_ids]
    return rows

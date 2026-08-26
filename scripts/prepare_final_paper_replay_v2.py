#!/usr/bin/env python3
"""Create immutable single-seed replay-v2 splits without touching legacy splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_paper_protocol import MMLU_SUBJECTS, canonical_fingerprint, normalize_question
from src.utils import atomic_json

SEED = 20260803


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            line = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
            handle.write(line)
            digest.update(line)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count, digest.hexdigest()


def order_key(dataset: str, subject: str, row: dict[str, Any]) -> str:
    value = f"{SEED}:{dataset}:{subject}:{row['problem_id']}"
    return hashlib.sha256(value.encode()).hexdigest()


def save_dataset(output: Path, rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    files = {}
    for split, rows in rows_by_split.items():
        count, digest = atomic_jsonl(rows, output / f"{split}.jsonl")
        files[split] = {"count": count, "sha256": digest, "problem_ids": [str(row["problem_id"]) for row in rows]}
    return files


def prepare_gsm(source: Path, output: Path, split_output: Path) -> dict[str, Any]:
    train = read_jsonl(source / "probe_train.jsonl") + read_jsonl(source / "calibration.jsonl")
    by_source = {int(row["source_index"]): row for row in train}
    if len(by_source) != 7473:
        raise ValueError(f"GSM8K official train accounting failed: {len(by_source)}")
    ordered = sorted(by_source.values(), key=lambda row: order_key("gsm8k", "", row))
    probe, calibration = ordered[:5000], ordered[5000:6000]
    heldout = read_jsonl(source / "heldout.jsonl")
    rows = {"probe_train": probe, "calibration": calibration, "heldout": heldout}
    files = save_dataset(output, rows)
    ids = {name: set(value["problem_ids"]) for name, value in files.items()}
    if ids["probe_train"] & ids["calibration"]:
        raise AssertionError("GSM8K probe/calibration overlap")
    manifest = {
        "schema_version": 2,
        "protocol_id": "final_paper_replay_v2",
        "dataset": "openai/gsm8k",
        "seed": SEED,
        "selection": "sha256(seed,dataset,split-independent sample_id) order",
        "files": files,
        "unused_official_train_problem_ids": [str(row["problem_id"]) for row in ordered[6000:]],
        "invariants": {
            "probe_train": 5000,
            "calibration": 1000,
            "official_test": 1319,
            "test_used_for_training_threshold_or_scaler": False,
        },
    }
    manifest["fingerprint"] = canonical_fingerprint(manifest)
    atomic_json(manifest, split_output / "gsm8k_split.json")
    return manifest


def quota(total: int) -> dict[str, int]:
    base, remainder = divmod(total, len(MMLU_SUBJECTS))
    return {subject: base + int(index < remainder) for index, subject in enumerate(MMLU_SUBJECTS)}


def prepare_mmlu(source: Path, output: Path, split_output: Path) -> dict[str, Any]:
    auxiliary = read_jsonl(source / "probe_train.jsonl") + read_jsonl(source / "calibration.jsonl")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in auxiliary:
        groups[str(row["subject"])].append(row)
    if set(groups) != set(MMLU_SUBJECTS):
        raise ValueError("MMLU auxiliary subject coverage is not 57/57")
    probe_quota, calibration_quota = quota(4000), quota(1000)
    probe, calibration = [], []
    subject_manifest = {}
    for subject in MMLU_SUBJECTS:
        ordered = sorted(groups[subject], key=lambda row: order_key("mmlu", subject, row))
        p, c = probe_quota[subject], calibration_quota[subject]
        if len(ordered) < p + c:
            raise ValueError(f"insufficient routed auxiliary data for {subject}: {len(ordered)} < {p+c}")
        probe.extend(ordered[:p])
        calibration.extend(ordered[p:p+c])
        subject_manifest[subject] = {
            "probe_train_count": p,
            "calibration_count": c,
            "probe_train_problem_ids": [str(row["problem_id"]) for row in ordered[:p]],
            "calibration_problem_ids": [str(row["problem_id"]) for row in ordered[p:p+c]],
        }
    demonstrations = read_jsonl(source / "demonstrations.jsonl")
    heldout = read_jsonl(source / "heldout.jsonl")
    rows = {
        "probe_train": probe,
        "calibration": calibration,
        "demonstrations": demonstrations,
        "heldout": heldout,
    }
    files = save_dataset(output, rows)
    protected = {
        normalize_question(row["question"])
        for split in ("demonstrations", "heldout")
        for row in rows[split]
    }
    selected_text = {
        normalize_question(row["question"])
        for split in ("probe_train", "calibration")
        for row in rows[split]
    }
    overlap = sorted(protected & selected_text)
    ids = {name: set(files[name]["problem_ids"]) for name in ("probe_train", "calibration", "heldout")}
    counts = Counter(row["subject"] for row in heldout)
    if overlap or ids["probe_train"] & ids["calibration"] or ids["heldout"] & (ids["probe_train"] | ids["calibration"]):
        raise AssertionError("MMLU leakage/overlap audit failed")
    manifest = {
        "schema_version": 2,
        "protocol_id": "final_paper_replay_v2",
        "dataset": "cais/mmlu",
        "seed": SEED,
        "selection": "within-routed-subject sha256(seed,dataset,subject,sample_id) order with balanced quotas",
        "subjects": list(MMLU_SUBJECTS),
        "subject_manifest": subject_manifest,
        "files": files,
        "heldout_subject_counts": dict(sorted(counts.items())),
        "invariants": {
            "probe_train": 4000,
            "calibration": 1000,
            "demonstrations": 285,
            "official_test": 14042,
            "subjects": 57,
            "normalized_question_overlap_with_dev_or_test": 0,
            "test_used_for_training_threshold_or_scaler": False,
        },
    }
    manifest["fingerprint"] = canonical_fingerprint(manifest)
    atomic_json(manifest, split_output / "mmlu_split.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-data", type=Path, default=Path("data/final_paper_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/final_paper_replay_v2"))
    parser.add_argument("--split-output", type=Path, default=Path("results/final_paper_replay_v2/splits"))
    args = parser.parse_args()
    legacy = args.legacy_data if args.legacy_data.is_absolute() else ROOT / args.legacy_data
    output = args.output if args.output.is_absolute() else ROOT / args.output
    split_output = args.split_output if args.split_output.is_absolute() else ROOT / args.split_output
    gsm = prepare_gsm(legacy / "gsm8k", output / "gsm8k", split_output)
    mmlu = prepare_mmlu(legacy / "mmlu", output / "mmlu", split_output)
    atomic_json({
        "status": "complete",
        "seed": SEED,
        "gsm8k_fingerprint": gsm["fingerprint"],
        "mmlu_fingerprint": mmlu["fingerprint"],
    }, split_output / "prepare.complete")
    print(json.dumps({"gsm8k": gsm["fingerprint"], "mmlu": mmlu["fingerprint"]}, indent=2))


if __name__ == "__main__":
    main()

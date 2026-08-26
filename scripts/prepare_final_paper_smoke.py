#!/usr/bin/env python3
"""Create deterministic, category-balanced ID lists for final-paper smoke tests."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_paper_inference import read_jsonl
from src.final_paper_protocol import MMLU_CATEGORIES
from src.utils import atomic_json


SEED = 20260803
PER_SPLIT = 8


def hash_key(split: str, problem_id: str) -> str:
    return hashlib.sha256(
        f"final_paper_smoke:{SEED}:{split}:{problem_id}".encode("utf-8")
    ).hexdigest()


def gsm_ids(rows: list[dict], split: str) -> list[str]:
    ordered = sorted(
        (str(row["problem_id"]) for row in rows),
        key=lambda value: hash_key(split, value),
    )
    return ordered[:PER_SPLIT]


def mmlu_ids(rows: list[dict], split: str) -> list[str]:
    selected: list[str] = []
    for category in ("STEM", "Humanities", "Social Sciences", "Other"):
        candidates = sorted(
            (row for row in rows if row["category"] == category),
            key=lambda row: hash_key(split, str(row["problem_id"])),
        )
        used_subjects: set[str] = set()
        local = []
        for row in candidates:
            subject = str(row["subject"])
            if subject in used_subjects:
                continue
            used_subjects.add(subject)
            local.append(str(row["problem_id"]))
            if len(local) == 2:
                break
        if len(local) != 2:
            raise ValueError(f"cannot select two subjects for {category}/{split}")
        selected.extend(local)
    if len(selected) != PER_SPLIT or len(set(selected)) != PER_SPLIT:
        raise ValueError(f"invalid MMLU smoke selection for {split}")
    return selected


def main() -> None:
    output = ROOT / "results/final_paper_v1_smoke/splits"
    manifest = {
        "status": "complete",
        "seed": SEED,
        "samples_per_dataset_split": PER_SPLIT,
        "selection": {},
    }
    for dataset in ("gsm8k", "mmlu"):
        prepared = ROOT / f"data/final_paper_v1/{dataset}"
        manifest["selection"][dataset] = {}
        for split in ("probe_train", "calibration", "heldout"):
            rows = read_jsonl(prepared / f"{split}.jsonl")
            ids = gsm_ids(rows, split) if dataset == "gsm8k" else mmlu_ids(rows, split)
            by_id = {str(row["problem_id"]): row for row in rows}
            path = output / f"{dataset}_{split}_ids.json"
            atomic_json(ids, path)
            metadata = {
                "path": str(path.relative_to(ROOT)),
                "problem_ids": ids,
            }
            if dataset == "mmlu":
                metadata["subjects"] = [by_id[value]["subject"] for value in ids]
                metadata["categories"] = [by_id[value]["category"] for value in ids]
                if set(metadata["categories"]) != set(MMLU_CATEGORIES):
                    raise ValueError(f"MMLU smoke misses a category in {split}")
            manifest["selection"][dataset][split] = metadata
    atomic_json(manifest, output / "final_paper_smoke_selection.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

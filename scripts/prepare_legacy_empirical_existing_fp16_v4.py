#!/usr/bin/env python3
"""Create immutable, stratified views over the already-generated FP16 cache."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results/final_paper_replay_v4/legacy_empirical_protocol_existing_fp16_seed20260803_train1000_cal500_mmlu1000"
SOURCE_ROOT = ROOT / "results/final_paper_replay_v2/cache"
OLD_SCOPE = ROOT / "results/final_paper_replay_v3/degraded_train1000_cal500_mmlu1000_final_paper/SCOPE_MANIFEST.json"
MMLU_SPLIT = ROOT / "results/final_paper_replay_v2/splits/mmlu_split.json"
EXPECTED = {
    "gsm8k": {"probe_train": 1000, "calibration": 500, "heldout": 1319},
    "mmlu": {"probe_train": 1000, "calibration": 500, "heldout": 1000},
}


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def available(dataset: str, split: str) -> set[str]:
    return {
        path.stem.removeprefix("sample_")
        for path in (SOURCE_ROOT / dataset / "merged" / split).glob("sample_*.pt")
    }


def stratified_existing(
    manifest: dict[str, Any], split: str, total: int, seed: int,
) -> tuple[list[str], dict[str, list[str]]]:
    key = f"{split}_problem_ids"
    subjects = list(manifest["subjects"])
    present = available("mmlu", split)
    base, extras = divmod(total, len(subjects))
    extra_subjects = set(np.random.default_rng(seed).permutation(subjects)[:extras].tolist())
    by_subject: dict[str, list[str]] = {}
    for subject in subjects:
        candidates = [
            str(value) for value in manifest["subject_manifest"][subject][key]
            if str(value) in present
        ]
        needed = base + int(subject in extra_subjects)
        if len(candidates) < needed:
            raise RuntimeError(f"existing cache insufficient for {split}/{subject}: {len(candidates)}/{needed}")
        by_subject[subject] = candidates[:needed]
    selected = [value for subject in subjects for value in by_subject[subject]]
    if len(selected) != total or len(set(selected)) != total:
        raise RuntimeError(f"invalid stratified selection for {split}: {len(selected)}")
    return selected, by_subject


def link_selection(dataset: str, split: str, ids: list[str]) -> None:
    destination_root = RUN_ROOT / "selected_common_cache" / dataset / "merged" / split
    destination_root.mkdir(parents=True, exist_ok=True)
    for problem_id in ids:
        source = (SOURCE_ROOT / dataset / "merged" / split / f"sample_{problem_id}.pt").resolve()
        destination = destination_root / source.name
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.is_symlink():
            if destination.resolve() != source:
                raise RuntimeError(f"refusing to replace mismatched symlink: {destination}")
            continue
        if destination.exists():
            raise RuntimeError(f"refusing to replace existing selection artifact: {destination}")
        os.symlink(source, destination)


def main() -> None:
    phase_marker = RUN_ROOT / "phases/prepare_existing_cache.complete"
    frozen_scope = RUN_ROOT / "SCOPE_MANIFEST.json"
    if phase_marker.is_file() and frozen_scope.is_file():
        marker = json.loads(phase_marker.read_text(encoding="utf-8"))
        scope = json.loads(frozen_scope.read_text(encoding="utf-8"))
        selected = sum(1 for _ in (RUN_ROOT / "selected_common_cache").glob("*/merged/*/sample_*.pt"))
        if marker.get("status") == "complete" and scope.get("status") == "frozen" and selected == 5319:
            print(json.dumps({
                "status": "skipped_frozen", "scope_fingerprint": scope["scope_fingerprint"],
                "selected_samples": selected,
            }, ensure_ascii=False, indent=2))
            return
    old_scope = json.loads(OLD_SCOPE.read_text(encoding="utf-8"))
    mmlu_manifest = json.loads(MMLU_SPLIT.read_text(encoding="utf-8"))
    ids: dict[str, dict[str, list[str]]] = {
        "gsm8k": {
            split: [str(value) for value in old_scope["datasets"]["gsm8k"][f"{split}_problem_ids"]]
            for split in ("probe_train", "calibration", "heldout")
        },
        "mmlu": {},
    }
    train_ids, train_by_subject = stratified_existing(mmlu_manifest, "probe_train", 1000, 20260803)
    calibration_ids, calibration_by_subject = stratified_existing(mmlu_manifest, "calibration", 500, 20260804)
    ids["mmlu"]["probe_train"] = train_ids
    ids["mmlu"]["calibration"] = calibration_ids
    ids["mmlu"]["heldout"] = [
        str(value) for value in old_scope["datasets"]["mmlu"]["heldout_problem_ids"]
    ]
    for dataset, splits in ids.items():
        for split, values in splits.items():
            if len(values) != EXPECTED[dataset][split] or len(set(values)) != len(values):
                raise RuntimeError(f"scope count/duplicate mismatch {dataset}/{split}")
            if not set(values) <= available(dataset, split):
                raise RuntimeError(f"scope contains unavailable cache ID {dataset}/{split}")
            link_selection(dataset, split, values)
        if set(splits["probe_train"]) & set(splits["calibration"]):
            raise RuntimeError(f"train/calibration overlap for {dataset}")
        if (set(splits["probe_train"]) | set(splits["calibration"])) & set(splits["heldout"]):
            raise RuntimeError(f"train-or-calibration/test overlap for {dataset}")
    scope = {
        "status": "frozen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope_id": "legacy_empirical_existing_fp16_seed20260803_train1000_cal500_mmlu1000",
        "allowed_protocol_overrides": {
            "dtype": "float16 (existing cache)",
            "generation_seed": "existing per-task deterministic seed protocol",
        },
        "source_cache": str(SOURCE_ROOT),
        "datasets": {
            "gsm8k": {
                "selection": "existing previously frozen train1000/cal500/full-test scope",
                **{f"{split}_problem_ids": values for split, values in ids["gsm8k"].items()},
            },
            "mmlu": {
                "selection": "existing-cache-only subject-stratified 1000/500; frozen balanced test1000",
                "probe_train_source": "auxiliary_train",
                "calibration_source": "auxiliary_train (distribution shift relative to official test)",
                "heldout_source": "official test",
                "probe_train_by_subject": train_by_subject,
                "calibration_by_subject": calibration_by_subject,
                **{f"{split}_problem_ids": values for split, values in ids["mmlu"].items()},
            },
        },
    }
    scope["scope_fingerprint"] = fingerprint(scope)
    atomic_json(scope, RUN_ROOT / "SCOPE_MANIFEST.json")
    atomic_json({
        "status": "stopped_by_user_protocol_override",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_bf16_samples_preserved": 200,
        "reason": "User approved reuse of existing FP16/current-seed cache; no further BF16 collection.",
        "replacement_run": str(RUN_ROOT),
    }, ROOT / "results/final_paper_replay_v4/legacy_empirical_protocol_train1000_cal500_mmlu1000/ABORTED_BY_PROTOCOL_OVERRIDE.json")
    atomic_json({
        "status": "complete", "scope_fingerprint": scope["scope_fingerprint"],
        "symlinked_samples": sum(EXPECTED[dataset][split] for dataset in EXPECTED for split in EXPECTED[dataset]),
        "model_generation_performed": False,
    }, RUN_ROOT / "phases/prepare_existing_cache.complete")
    print(json.dumps({
        "status": "complete", "scope_fingerprint": scope["scope_fingerprint"],
        "mmlu_train_subject_counts": dict(sorted(Counter({k: len(v) for k, v in train_by_subject.items()}).items())),
        "mmlu_calibration_subject_counts": dict(sorted(Counter({k: len(v) for k, v in calibration_by_subject.items()}).items())),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

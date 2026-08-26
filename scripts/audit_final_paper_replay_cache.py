#!/usr/bin/env python3
"""Fail-closed integrity, alignment, fingerprint, and leakage audit for replay-v2 cache."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.final_paper_protocol import MMLU_SUBJECTS, normalize_question
from src.final_paper_replay_cache import BRANCH_DIRECT, branch_path, protocol_fingerprint
from src.utils import atomic_json, load_yaml


CONFIGS = {
    "gsm8k": Path("configs/final_paper_replay_v2_gsm8k_fp16.yaml"),
    "mmlu": Path("configs/final_paper_replay_v2_mmlu_fp16.yaml"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-base", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache_base = args.cache_base if args.cache_base.is_absolute() else ROOT / args.cache_base
    output = args.output if args.output.is_absolute() else ROOT / args.output
    selection = None
    if args.selection:
        path = args.selection if args.selection.is_absolute() else ROOT / args.selection
        selection = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    evidence: dict[str, Any] = {"datasets": {}}
    for dataset, config_path in CONFIGS.items():
        config = load_yaml(config_path)
        prepared = ROOT / config["dataset"]["prepared_root"]
        manifest_path = ROOT / f"results/final_paper_replay_v2/splits/{dataset}_split.json"
        expected_fingerprint = protocol_fingerprint(
            ROOT / config_path,
            manifest_path,
            ROOT / config["model"]["local_path"],
        )
        cache_root = cache_base / dataset
        dataset_evidence = {}
        ids_by_split = {}
        texts_by_split = {}
        for split in ("probe_train", "calibration", "heldout"):
            rows = read_jsonl(prepared / f"{split}.jsonl")
            allowed = None if selection is None else set(selection[dataset][split])
            if allowed is not None:
                rows = [row for row in rows if str(row["problem_id"]) in allowed]
            expected_ids = {str(row["problem_id"]) for row in rows}
            ids_by_split[split] = expected_ids
            texts_by_split[split] = {normalize_question(row["question"]) for row in rows}
            merged_paths = sorted((cache_root / "merged" / split).glob("sample_*.pt"))
            found_ids = set()
            checkpoints = 0
            branch_files = 0
            for path in merged_paths:
                artifact = torch.load(path, map_location="cpu", weights_only=False)
                problem_id = str(artifact.get("problem_id"))
                found_ids.add(problem_id)
                if artifact.get("protocol_fingerprint") != expected_fingerprint:
                    failures.append(f"{dataset}/{split}/{problem_id}: fingerprint mismatch")
                if artifact.get("dtype") != "float16":
                    failures.append(f"{dataset}/{split}/{problem_id}: dtype != float16")
                if artifact.get("capture_layers") != [20]:
                    failures.append(f"{dataset}/{split}/{problem_id}: capture_layers != [20]")
                rows_local = artifact.get("rows", [])
                hidden = artifact.get("hidden")
                if not torch.is_tensor(hidden) or len(rows_local) != int(hidden.shape[0]):
                    failures.append(f"{dataset}/{split}/{problem_id}: row/vector mismatch")
                    continue
                if hidden.ndim != 3 or tuple(hidden.shape[1:]) != (1, 2560):
                    failures.append(f"{dataset}/{split}/{problem_id}: hidden shape {tuple(hidden.shape)}")
                if not np.isfinite(hidden.float().numpy()).all():
                    failures.append(f"{dataset}/{split}/{problem_id}: nonfinite hidden")
                keys = [(dataset, split, problem_id, int(row["checkpoint"])) for row in rows_local]
                if len(keys) != len(set(keys)):
                    failures.append(f"{dataset}/{split}/{problem_id}: duplicate cache key")
                direct = branch_path(cache_root, split, problem_id, BRANCH_DIRECT)
                if not direct.is_file():
                    failures.append(f"{dataset}/{split}/{problem_id}: missing Direct")
                branch_files += 1 + len(rows_local)
                checkpoints += len(rows_local)
                if any(row.get("branch_timing_source") != "excluded_2080_ti" for row in rows_local):
                    failures.append(f"{dataset}/{split}/{problem_id}: invalid branch timing source")
            if found_ids != expected_ids:
                missing = sorted(expected_ids - found_ids)
                extra = sorted(found_ids - expected_ids)
                failures.append(
                    f"{dataset}/{split}: sample mismatch missing={len(missing)} extra={len(extra)}"
                )
            dataset_evidence[split] = {
                "expected_samples": len(expected_ids),
                "merged_samples": len(found_ids),
                "checkpoints": checkpoints,
                "branch_files_expected": branch_files,
            }
        if ids_by_split["probe_train"] & ids_by_split["calibration"]:
            failures.append(f"{dataset}: probe/calibration ID overlap")
        if ids_by_split["heldout"] & (ids_by_split["probe_train"] | ids_by_split["calibration"]):
            failures.append(f"{dataset}: heldout ID overlap")
        if dataset == "mmlu":
            protected = texts_by_split["heldout"] | {
                normalize_question(row["question"])
                for row in read_jsonl(prepared / "demonstrations.jsonl")
            }
            if protected & (texts_by_split["probe_train"] | texts_by_split["calibration"]):
                failures.append("mmlu: normalized question leakage")
            heldout = read_jsonl(prepared / "heldout.jsonl")
            if set(Counter(row["subject"] for row in heldout)) != set(MMLU_SUBJECTS):
                failures.append("mmlu: heldout subject coverage !=57")
        evidence["datasets"][dataset] = dataset_evidence
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "mode": args.mode,
        "latency_label": "A100 replay-estimated latency",
        "failures": failures,
        "evidence": evidence,
    }
    atomic_json(payload, output)
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(4)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""创建确定性的链路检查与时间校准 ID 清单。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_paper_inference import read_jsonl
from src.final_paper_protocol import MMLU_SUBJECTS
from src.utils import atomic_json


def ordered(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"20260803:{salt}:{row['problem_id']}".encode()
        ).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/final_paper_replay_v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/final_paper_replay_v2/selections"),
    )
    args = parser.parse_args()
    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    output = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root

    manifest: dict[str, Any] = {
        "seed": 20260803,
        "files": {},
        "selections": {"gsm8k": {}, "mmlu": {}},
    }
    for split in ("probe_train", "calibration", "heldout"):
        gsm = ordered(read_jsonl(data_root / "gsm8k" / f"{split}.jsonl"), f"gsm:{split}")
        path = output / f"gsm8k_{split}_smoke_ids.json"
        values = [str(row["problem_id"]) for row in gsm[:20]]
        atomic_json(values, path)
        manifest["files"][path.name] = len(values)
        manifest["selections"]["gsm8k"][split] = values

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(data_root / "mmlu" / f"{split}.jsonl"):
            groups[str(row["subject"])].append(row)
        values = [
            str(ordered(groups[subject], f"mmlu:{split}:{subject}")[0]["problem_id"])
            for subject in MMLU_SUBJECTS
        ]
        path = output / f"mmlu_{split}_smoke_ids.json"
        atomic_json(values, path)
        manifest["files"][path.name] = len(values)
        manifest["selections"]["mmlu"][split] = values

    gsm_probe = ordered(
        read_jsonl(data_root / "gsm8k" / "probe_train.jsonl"),
        "gsm:timing",
    )
    gsm_timing = [str(row["problem_id"]) for row in gsm_probe[:200]]
    path = output / "gsm8k_probe_train_timing_ids.json"
    atomic_json(gsm_timing, path)
    manifest["files"][path.name] = len(gsm_timing)

    mmlu_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(data_root / "mmlu" / "probe_train.jsonl"):
        mmlu_groups[str(row["subject"])].append(row)
    mmlu_timing = [
        str(row["problem_id"])
        for subject in MMLU_SUBJECTS
        for row in ordered(mmlu_groups[subject], f"mmlu:timing:{subject}")[:8]
    ]
    path = output / "mmlu_probe_train_timing_ids.json"
    atomic_json(mmlu_timing, path)
    manifest["files"][path.name] = len(mmlu_timing)
    manifest["timing_note"] = (
        "Timing IDs are drawn only from probe_train: 200 GSM8K and "
        "8 per MMLU subject (456 total). They are independent of policy calibration."
    )
    atomic_json(manifest, output / "selection_manifest.json")
    atomic_json(manifest["selections"], output / "smoke_selection.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

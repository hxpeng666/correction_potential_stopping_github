#!/usr/bin/env python3
"""冻结 MMLU-Pro validation 五样例与官方 test 分层 1k held-out。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from src.final_paper_protocol import canonical_fingerprint, normalize_question
from src.mmlu_pro_protocol import answer_letter, valid_letters
from src.utils import atomic_json, load_yaml


def stable_order(row: dict[str, Any], seed: int) -> str:
    return hashlib.sha256(
        f"{seed}:mmlu_pro:test:{row['category']}:{row['question_id']}".encode()
    ).hexdigest()


def to_record(row: dict[str, Any], split: str) -> dict[str, Any]:
    choices = [str(value) for value in row["options"] if value is not None]
    valid_letters(len(choices))
    gold = answer_letter(row["answer_index"], len(choices))
    declared = str(row["answer"]).strip().upper()
    if declared != gold:
        raise ValueError(f"answer/answer_index 不一致：{row['question_id']} {declared} != {gold}")
    result = {
        "problem_id": f"mmlu_pro_{split}_{row['question_id']}",
        "question_id": str(row["question_id"]),
        "source_split": split,
        "source": str(row.get("src", "")),
        "category": str(row["category"]),
        "subject": str(row["category"]),
        "question": str(row["question"]),
        "choices": choices,
        "answer": gold,
        "answer_index": int(row["answer_index"]),
        "option_count": len(choices),
    }
    result["record_fingerprint"] = canonical_fingerprint(result)
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_paper_mmlu_pro_transfer_v1.yaml")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)
    output = args.output_root or ROOT / config["dataset"]["prepared_root"]
    results = args.results_root or ROOT / config["output_root"]
    manifest_path = results / "splits" / "mmlu_pro_split.json"
    if args.resume and manifest_path.is_file() and (output / "heldout.jsonl").is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") == "frozen" and previous.get("protocol_id") == config["protocol_id"]:
            print(json.dumps({"status": "skipped_frozen", "manifest": str(manifest_path)}))
            return
        raise RuntimeError("已有 MMLU-Pro 划分与当前协议不兼容")
    for path in (output / "demonstrations.jsonl", output / "heldout.jsonl", manifest_path):
        if path.exists():
            raise RuntimeError(f"拒绝覆盖已有文件：{path}")

    dataset = load_dataset(config["dataset"]["name"])
    validation = dataset["validation"]
    test = dataset["test"]
    demos = [to_record(dict(row), "validation") for row in validation]
    demo_counts = Counter(row["category"] for row in demos)
    if len(demo_counts) != 14 or set(demo_counts.values()) != {5}:
        raise ValueError(f"validation 不是 14 类各 5 条：{dict(demo_counts)}")
    demo_questions = {normalize_question(row["question"]) for row in demos}

    seed_value = config["seed"]
    seed = int(seed_value["global"] if isinstance(seed_value, dict) else seed_value)
    categories = sorted(set(str(value) for value in test["category"]))
    total = int(config["dataset"]["heldout_count"])
    base, remainder = divmod(total, len(categories))
    quotas = {category: base + int(index < remainder) for index, category in enumerate(categories)}
    selected: list[dict[str, Any]] = []
    candidate_counts: dict[str, int] = {}
    duplicate_excluded: dict[str, int] = {}
    for category in categories:
        local = []
        duplicate_excluded[category] = 0
        for raw in test:
            if str(raw["category"]) != category:
                continue
            record = to_record(dict(raw), "test")
            if normalize_question(record["question"]) in demo_questions:
                duplicate_excluded[category] += 1
                continue
            local.append(record)
        candidate_counts[category] = len(local)
        local.sort(key=lambda row: stable_order(row, seed))
        if len(local) < quotas[category]:
            raise ValueError(f"{category} 不足目标配额 {quotas[category]}")
        selected.extend(local[: quotas[category]])
    selected.sort(key=lambda row: (row["category"], stable_order(row, seed)))
    if len(selected) != total or len({row["problem_id"] for row in selected}) != total:
        raise ValueError("MMLU-Pro held-out 数量或 ID 唯一性错误")

    write_jsonl(output / "demonstrations.jsonl", demos)
    write_jsonl(output / "heldout.jsonl", selected)
    smoke = []
    for category in categories:
        smoke.append(next(row for row in selected if row["category"] == category))
    write_jsonl(output / "smoke_heldout.jsonl", smoke)
    fingerprint_payload = {
        "protocol_id": config["protocol_id"],
        "seed": seed,
        "dataset": config["dataset"]["name"],
        "validation_fingerprint": getattr(validation, "_fingerprint", None),
        "test_fingerprint": getattr(test, "_fingerprint", None),
        "demonstration_ids": [row["problem_id"] for row in demos],
        "heldout_ids": [row["problem_id"] for row in selected],
        "quotas": quotas,
    }
    manifest = {
        "status": "frozen",
        "protocol_id": config["protocol_id"],
        "report_label": config["report_label"],
        "global_seed": seed,
        "dataset_name": config["dataset"]["name"],
        "dataset_splits": {
            "validation": {"rows": len(validation), "fingerprint": getattr(validation, "_fingerprint", None)},
            "test": {"rows": len(test), "fingerprint": getattr(test, "_fingerprint", None)},
        },
        "demonstrations": {
            "source": "official_validation",
            "count": len(demos),
            "per_category": dict(sorted(demo_counts.items())),
            "used_for_training_or_calibration": False,
        },
        "probe_train": {"source": "frozen_existing_MMLU_auxiliary_train", "count": 1000},
        "policy_calibration": {"source": "frozen_existing_MMLU_auxiliary_train", "count": 500},
        "heldout": {
            "source": "official_test",
            "count": len(selected),
            "per_category": dict(sorted(Counter(row["category"] for row in selected).items())),
            "quotas": quotas,
            "selection": "within-category SHA256(global_seed,dataset,split,category,question_id)",
            "used_for_training_calibration_or_threshold_selection": False,
        },
        "candidate_counts": candidate_counts,
        "validation_question_duplicates_excluded": duplicate_excluded,
        "fingerprint": canonical_fingerprint(fingerprint_payload),
        "paths": {
            "demonstrations": str((output / "demonstrations.jsonl").resolve()),
            "heldout": str((output / "heldout.jsonl").resolve()),
            "smoke_heldout": str((output / "smoke_heldout.jsonl").resolve()),
        },
        "distribution_note": "MMLU-trained probes and thresholds are transferred unchanged to MMLU-Pro; this is not in-domain calibration.",
    }
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

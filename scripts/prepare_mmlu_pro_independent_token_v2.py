#!/usr/bin/env python3
"""建立 MMLU-Pro 1000/500/1000 独立同源固定划分。

旧 MMLU-Pro-800 只进入 probe-train；calibration 与 final-test 都来自其余官方
test 样本，并按 category x 长度四分位分层，避免此前 OOF/source-shift 语义错误。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_paper_protocol import canonical_fingerprint, normalize_question
from src.mmlu_pro_protocol import answer_letter, valid_letters
from src.utils import atomic_json, load_yaml


def make_record(raw: dict[str, Any], source_index: int, source_split: str) -> dict[str, Any]:
    choices = [str(value) for value in raw["options"] if value is not None]
    valid_letters(len(choices))
    gold = answer_letter(raw["answer_index"], len(choices))
    if str(raw["answer"]).strip().upper() != gold:
        raise ValueError(f"答案索引不一致：{raw['question_id']}")
    value = {
        "problem_id": f"mmlu_pro_test_{raw['question_id']}",
        "question_id": str(raw["question_id"]),
        "source_index": int(source_index),
        "source_split": source_split,
        "source": str(raw.get("src", "")),
        "category": str(raw["category"]),
        "subject": str(raw["category"]),
        "question": str(raw["question"]),
        "choices": choices,
        "answer": gold,
        "answer_index": int(raw["answer_index"]),
        "option_count": len(choices),
    }
    value["length_chars"] = len(value["question"]) + sum(len(choice) for choice in choices)
    value["record_fingerprint"] = canonical_fingerprint(value)
    return value


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def stratified_pick(rows: list[dict[str, Any]], count: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = np.asarray([row["stratum"] for row in rows])
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=count, random_state=seed)
    chosen, remaining = next(splitter.split(np.zeros(len(rows)), labels))
    return [rows[int(i)] for i in chosen], [rows[int(i)] for i in remaining]


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = np.asarray([row["length_chars"] for row in rows], dtype=np.float64)
    return {
        "count": len(rows),
        "category": dict(sorted(Counter(row["category"] for row in rows).items())),
        "length_bin": dict(sorted(Counter(str(row["length_quartile"]) for row in rows).items())),
        "stratum": dict(sorted(Counter(row["stratum"] for row in rows).items())),
        "answer_index": dict(sorted(Counter(str(row["answer_index"]) for row in rows).items())),
        "option_count": dict(sorted(Counter(str(row["option_count"]) for row in rows).items())),
        "length_chars": {key: float(value) for key, value in zip(("mean", "p50", "p95"), (lengths.mean(), np.median(lengths), np.percentile(lengths, 95)))},
    }


def total_variation(a: dict[str, int], b: dict[str, int]) -> float:
    keys = set(a) | set(b)
    na, nb = max(sum(a.values()), 1), max(sum(b.values()), 1)
    return 0.5 * sum(abs(a.get(key, 0) / na - b.get(key, 0) / nb) for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_paper_mmlu_pro_independent_token_v2.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(ROOT / args.config)
    output = ROOT / config["dataset"]["prepared_root"]
    results = ROOT / config["output_root"]
    manifest_path = results / "splits" / "mmlu_pro_independent_split.json"
    required = [output / f"{name}.jsonl" for name in ("probe_train", "calibration", "heldout", "all_selected", "demonstrations")]
    if args.resume and manifest_path.is_file() and all(path.is_file() for path in required):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "frozen" and manifest.get("protocol_id") == config["protocol_id"]:
            print(json.dumps({"status": "skipped_frozen", "manifest": str(manifest_path)}))
            return
        raise RuntimeError("已有划分与当前协议不兼容")
    for path in [*required, output / "smoke_new.jsonl", manifest_path]:
        if path.exists():
            raise RuntimeError(f"拒绝覆盖已有文件：{path}")

    dataset = load_dataset(config["dataset"]["name"])
    validation, test = dataset["validation"], dataset["test"]
    demos = [make_record(dict(raw), index, "validation") for index, raw in enumerate(validation)]
    demo_counts = Counter(row["category"] for row in demos)
    if len(demo_counts) != 14 or set(demo_counts.values()) != {5}:
        raise ValueError("MMLU-Pro validation 必须是14类各5条演示")
    demo_questions = {normalize_question(row["question"]) for row in demos}
    valid = []
    for index, raw in enumerate(test):
        try:
            row = make_record(dict(raw), index, "test")
        except (KeyError, TypeError, ValueError):
            continue
        if normalize_question(row["question"]) not in demo_questions:
            valid.append(row)
    if len({row["problem_id"] for row in valid}) != len(valid):
        raise ValueError("官方 test problem ID 重复")

    boundaries = np.quantile([row["length_chars"] for row in valid], [0.25, 0.50, 0.75])
    for row in valid:
        row["length_quartile"] = int(np.searchsorted(boundaries, row["length_chars"], side="right"))
        row["stratum"] = f"{row['category']}|L{row['length_quartile']}"

    old_manifest_path = ROOT / config["dataset"]["old_split_manifest"]
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_ids = set(map(str, old_manifest["learnstop_sampling"]["selected_source_indices"]))
    # 历史 manifest 同时保存 source index 和 problem ID；以 problem ID 为权威。
    old_problem_ids = set(old_manifest.get("selected_ids", []))
    if not old_problem_ids:
        old_problem_ids = {f"mmlu_pro_test_{row['question_id']}" for row in valid if str(row["source_index"]) in old_ids}
    old_rows = [row for row in valid if row["problem_id"] in old_problem_ids]
    if len(old_rows) != int(config["dataset"]["old_reuse_count"]):
        raise ValueError(f"旧800题识别失败：{len(old_rows)}")
    remaining = [row for row in valid if row["problem_id"] not in old_problem_ids]
    seed = int(config["seed"]["dataset_sampling"])

    # 先固定同源 calibration+test 池，再按完全相同的 strata 切成 1:2。
    evaluation_pool, after_evaluation = stratified_pick(remaining, 1500, seed)
    calibration, heldout = stratified_pick(evaluation_pool, 500, seed + 1)
    train_extra, _ = stratified_pick(after_evaluation, 200, seed + 2)
    probe_train = old_rows + train_extra
    for role, rows in (("probe_train", probe_train), ("calibration", calibration), ("heldout", heldout)):
        for row in rows:
            row["policy_role"] = role
            row["reused_from_mmlu_pro_800"] = row["problem_id"] in old_problem_ids
        rows.sort(key=lambda value: value["problem_id"])

    role_sets = [set(row["problem_id"] for row in rows) for rows in (probe_train, calibration, heldout)]
    if [len(value) for value in role_sets] != [1000, 500, 1000] or any(role_sets[i] & role_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("1000/500/1000 数量或互斥检查失败")
    if any(row["source_split"] != "test" for row in probe_train + calibration + heldout):
        raise ValueError("train/cal/test 必须全部来自同一 official test source")

    all_selected = sorted(probe_train + calibration + heldout, key=lambda value: value["problem_id"])
    new_rows = [row for row in all_selected if not row["reused_from_mmlu_pro_800"]]
    smoke_new = []
    for category in sorted(set(row["category"] for row in new_rows)):
        smoke_new.append(next(row for row in new_rows if row["category"] == category))
    write_jsonl(output / "demonstrations.jsonl", demos)
    write_jsonl(output / "probe_train.jsonl", probe_train)
    write_jsonl(output / "calibration.jsonl", calibration)
    write_jsonl(output / "heldout.jsonl", heldout)
    write_jsonl(output / "all_selected.jsonl", all_selected)
    write_jsonl(output / "new_collection.jsonl", new_rows)
    write_jsonl(output / "smoke_new.jsonl", smoke_new)

    summaries = {name: distribution(rows) for name, rows in (("probe_train", probe_train), ("calibration", calibration), ("heldout", heldout))}
    cal_test_tv = {
        field: total_variation(summaries["calibration"][field], summaries["heldout"][field])
        for field in ("category", "length_bin", "stratum", "answer_index", "option_count")
    }
    payload = {
        "protocol_id": config["protocol_id"],
        "seed": seed,
        "roles": {name: [row["problem_id"] for row in rows] for name, rows in (("probe_train", probe_train), ("calibration", calibration), ("heldout", heldout))},
        "length_boundaries": boundaries.tolist(),
    }
    manifest = {
        "status": "frozen",
        "protocol_id": config["protocol_id"],
        "source": {"dataset": config["dataset"]["name"], "train": "official test", "calibration": "official test", "heldout": "official test", "validation_usage": "5-shot demonstrations only", "test_fingerprint": getattr(test, "_fingerprint", None), "validation_fingerprint": getattr(validation, "_fingerprint", None)},
        "counts": {"probe_train": 1000, "calibration": 500, "heldout": 1000, "old_cache_reused_in_probe_train_only": 800, "new_collection": len(new_rows)},
        "selection": {"seed": seed, "algorithm": "StratifiedShuffleSplit(category x global length quartile)", "length_quartile_boundaries": boundaries.tolist()},
        "distributions": summaries,
        "calibration_vs_heldout_total_variation": cal_test_tv,
        "role_ids": payload["roles"],
        "demonstrations": {"source": "official validation", "count": len(demos), "per_category": dict(sorted(demo_counts.items())), "used_for_probe_or_policy": False},
        "leakage_audit": {"pairwise_overlap": 0, "normalized_demo_overlap": 0, "old_800_roles": ["probe_train"], "heldout_used_for_selection": False},
        "fingerprint": canonical_fingerprint(payload),
    }
    atomic_json(manifest, manifest_path)
    atomic_json({"status": "complete", "calibration_vs_heldout_total_variation": cal_test_tv, "passed": max(cal_test_tv["category"], cal_test_tv["length_bin"]) <= 0.05}, results / "CAL_TEST_DISTRIBUTION_AUDIT.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

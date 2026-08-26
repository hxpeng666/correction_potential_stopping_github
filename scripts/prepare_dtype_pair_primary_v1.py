#!/usr/bin/env python3
"""在不读取 held-out 结果的前提下冻结 BF16–FP16 配对样本清单。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_paper_protocol import MMLU_SUBJECTS, canonical_fingerprint
from src.final_paper_replay_cache import ensure_task, task_seed
from src.utils import atomic_json, load_yaml


EXPECTED_MODEL_FINGERPRINT = (
    "1444257116723c44a60884afd78b095b48727cf4a7e17b69d85e55970070d863"
)


def stable_rank(seed: int, *values: object) -> str:
    payload = ":".join([str(seed), "dtype_pair", *map(str, values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_pool(directory: Path, dataset: str, split: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("sample_*.pt")):
        value = torch.load(path, map_location="cpu", weights_only=False)
        problem_id = str(value.get("problem_id"))
        if value.get("status") != "complete":
            raise ValueError(f"缓存不完整：{path}")
        if value.get("dataset") != dataset or value.get("split") != split:
            raise ValueError(f"dataset/split 错位：{path}")
        if value.get("dtype") != "float16" or int(value.get("seed", -1)) != seed:
            raise ValueError(f"dtype/seed 不匹配：{path}")
        model_fingerprint = value.get("model_audit", {}).get("metadata_fingerprint")
        if model_fingerprint != EXPECTED_MODEL_FINGERPRINT:
            raise ValueError(f"模型指纹不匹配：{path}")
        if 20 not in [int(layer) for layer in value.get("capture_layers", [])]:
            raise ValueError(f"缺少 layer 20：{path}")
        dense = value["dense"]
        tokens = dense.get("tokens")
        if not isinstance(tokens, list) or len(tokens) != int(dense["reasoning_tokens"]):
            raise ValueError(f"Dense token 长度不一致：{path}")
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "sample_id": problem_id,
                "subject": value.get("record", {}).get("subject"),
                "category": value.get("record", {}).get("category"),
                "prompt_tokens": int(value["prompt_tokens"]),
                "dense_tokens_fp16": int(dense["reasoning_tokens"]),
                "reached_4096_fp16": bool(dense.get("reached_max_tokens", False)),
                "source_fp16_artifact": str(path.resolve()),
                "source_protocol_fingerprint": str(value["protocol_fingerprint"]),
                "prompt_sha256": hashlib.sha256(
                    str(value["prompt_text"]).encode("utf-8")
                ).hexdigest(),
                "dense_task_seed": task_seed(seed, dataset, split, problem_id, "dense"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"没有 FP16 候选缓存：{directory}")
    return rows


def bin_number(value: int, cuts: np.ndarray) -> int:
    return int(np.searchsorted(cuts, value, side="right"))


def round_robin_strata(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    candidates = [row for row in rows if not row["reached_4096_fp16"]]
    if count > len(candidates):
        raise ValueError(f"请求 {count} 条非触顶样本，但候选只有 {len(candidates)} 条")
    prompt_cuts = np.quantile([row["prompt_tokens"] for row in candidates], [1 / 3, 2 / 3])
    dense_cuts = np.quantile([row["dense_tokens_fp16"] for row in candidates], [1 / 3, 2 / 3])
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = (
            bin_number(row["prompt_tokens"], prompt_cuts),
            bin_number(row["dense_tokens_fp16"], dense_cuts),
        )
        groups[key].append(row)
    for key in groups:
        groups[key].sort(key=lambda row: stable_rank(seed, row["sample_id"]))
    ordered_keys = sorted(groups)
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for key in ordered_keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
                progressed = True
        if not progressed:
            raise RuntimeError("分层轮转提前耗尽")
    return selected


def diverse_subject_pick(
    rows: list[dict[str, Any]], count: int, seed: int, excluded: set[str]
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["sample_id"] not in excluded and not row["reached_4096_fp16"]
    ]
    candidates.sort(
        key=lambda row: (
            row["dense_tokens_fp16"],
            row["prompt_tokens"],
            stable_rank(seed, row["sample_id"]),
        )
    )
    if len(candidates) < count:
        raise ValueError("学科候选不足")
    if count == 1:
        indices = [len(candidates) // 2]
    else:
        indices = np.linspace(0, len(candidates) - 1, count).round().astype(int).tolist()
    result: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    for index in indices:
        while index in used_indices and index + 1 < len(candidates):
            index += 1
        used_indices.add(index)
        result.append(candidates[index])
    return result


def replace_with_capped(
    selected: list[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    targets: dict[str, int],
    seed: int,
    *,
    preserve_subject: bool,
) -> list[dict[str, Any]]:
    result = list(selected)
    selected_ids = {row["sample_id"] for row in result}
    for split, target in targets.items():
        capped = sorted(
            [
                row
                for row in pools[split]
                if row["reached_4096_fp16"] and row["sample_id"] not in selected_ids
                and (
                    not preserve_subject
                    or any(
                        candidate["split"] == split
                        and str(candidate.get("subject")) == str(row.get("subject"))
                        and not candidate["reached_4096_fp16"]
                        for candidate in result
                    )
                )
            ],
            key=lambda row: stable_rank(seed, "capped", row["sample_id"]),
        )[:target]
        if len(capped) < target:
            raise ValueError(f"{split} 触顶候选不足：{len(capped)} < {target}")
        for replacement in capped:
            removable = [
                row
                for row in result
                if row["split"] == split
                and not row["reached_4096_fp16"]
                and (
                    not preserve_subject
                    or str(row.get("subject")) == str(replacement.get("subject"))
                )
            ]
            if not removable:
                raise ValueError(f"无法为触顶样本保持 split/subject：{replacement['sample_id']}")
            removed = max(
                removable,
                key=lambda row: (
                    row["dense_tokens_fp16"], stable_rank(seed, "remove", row["sample_id"])
                ),
            )
            result.remove(removed)
            selected_ids.remove(removed["sample_id"])
            result.append(replacement)
            selected_ids.add(replacement["sample_id"])
    return result


def select_gsm8k(pools: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    selected = (
        round_robin_strata(pools["probe_train"], 150, seed)
        + round_robin_strata(pools["calibration"], 50, seed)
    )
    return replace_with_capped(
        selected, pools, {"probe_train": 8, "calibration": 2}, seed,
        preserve_subject=False,
    )


def select_mmlu(pools: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    by_split_subject: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for split, rows in pools.items():
        for row in rows:
            by_split_subject[(split, str(row["subject"]))].append(row)
    subjects = list(MMLU_SUBJECTS)
    if any(not by_split_subject[("probe_train", subject)] for subject in subjects):
        raise ValueError("MMLU probe-train 未覆盖全部 57 学科")
    if any(not by_split_subject[("calibration", subject)] for subject in subjects):
        raise ValueError("MMLU calibration 未覆盖全部 57 学科")
    calibration_subjects = set(
        sorted(subjects, key=lambda subject: stable_rank(seed, "cal", subject))[:50]
    )
    extra_subjects = set(
        sorted(subjects, key=lambda subject: stable_rank(seed, "extra", subject))[:29]
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for subject in subjects:
        calibration_count = 1 if subject in calibration_subjects else 0
        train_count = 3 - calibration_count + (1 if subject in extra_subjects else 0)
        chosen_train = diverse_subject_pick(
            by_split_subject[("probe_train", subject)], train_count, seed, selected_ids
        )
        selected.extend(chosen_train)
        selected_ids.update(row["sample_id"] for row in chosen_train)
        if calibration_count:
            chosen_cal = diverse_subject_pick(
                by_split_subject[("calibration", subject)], 1, seed, selected_ids
            )
            selected.extend(chosen_cal)
            selected_ids.update(row["sample_id"] for row in chosen_cal)
    selected = replace_with_capped(
        selected, pools, {"probe_train": 3, "calibration": 1}, seed,
        preserve_subject=True,
    )
    counts = Counter(row["split"] for row in selected)
    if counts != Counter({"probe_train": 150, "calibration": 50}):
        raise AssertionError(f"MMLU split 数量错误：{counts}")
    subject_counts = Counter(str(row["subject"]) for row in selected)
    if set(subject_counts) != set(subjects) or not all(value in (3, 4) for value in subject_counts.values()):
        raise AssertionError("MMLU 配对样本没有按 57 学科分层为每科 3/4 条")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/final_paper_primary_v1.yaml")
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("results/final_paper_primary_v1/dtype_pair_audit"),
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    config = load_yaml(config_path)
    if config.get("primary") is not True or config.get("status") != "pending_dtype_pair_audit":
        raise ValueError("主配置不处于 dtype 配对审计阶段")
    seed = int(config["seed"]["global"])
    all_selected: list[dict[str, Any]] = []
    pool_summaries: dict[str, Any] = {}
    for dataset in ("gsm8k", "mmlu"):
        selected_root = ROOT / config["datasets"][dataset]["selected_cache_root"] / "merged"
        pools = {
            split: load_pool(selected_root / split, dataset, split, seed)
            for split in ("probe_train", "calibration")
        }
        chosen = select_gsm8k(pools, seed) if dataset == "gsm8k" else select_mmlu(pools, seed)
        if len(chosen) != 200 or len({row["sample_id"] for row in chosen}) != 200:
            raise AssertionError(f"{dataset} 配对数量或唯一性错误")
        all_selected.extend(chosen)
        pool_summaries[dataset] = {
            "candidate_counts": {split: len(rows) for split, rows in pools.items()},
            "selected_split_counts": dict(Counter(row["split"] for row in chosen)),
            "selected_subject_counts": dict(sorted(Counter(str(row["subject"]) for row in chosen if row["subject"]).items())),
            "selected_reached_4096": sum(row["reached_4096_fp16"] for row in chosen),
            "prompt_tokens": {
                "min": min(row["prompt_tokens"] for row in chosen),
                "median": float(np.median([row["prompt_tokens"] for row in chosen])),
                "max": max(row["prompt_tokens"] for row in chosen),
            },
            "dense_tokens_fp16": {
                "min": min(row["dense_tokens_fp16"] for row in chosen),
                "median": float(np.median([row["dense_tokens_fp16"] for row in chosen])),
                "max": max(row["dense_tokens_fp16"] for row in chosen),
            },
        }
    protected = {
        "protocol_id": config["protocol_id"],
        "global_seed": seed,
        "selection_rule": "GSM 150/50 prompt×length strata; MMLU 150/50, 57 subjects × 3/4",
        "heldout_used": False,
        "samples": [
            {
                key: row[key]
                for key in (
                    "dataset", "split", "sample_id", "subject", "prompt_tokens",
                    "dense_tokens_fp16", "reached_4096_fp16", "source_protocol_fingerprint",
                    "prompt_sha256", "dense_task_seed",
                )
            }
            for row in sorted(all_selected, key=lambda row: (row["dataset"], row["split"], row["sample_id"]))
        ],
    }
    fingerprint = canonical_fingerprint(protected)
    manifest = {
        "status": "frozen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_fingerprint": fingerprint,
        "heldout_used": False,
        "selection_fields_allowed": [
            "dataset", "split", "sample_id", "subject", "prompt_tokens",
            "dense_tokens_fp16", "reached_4096_fp16",
        ],
        "selection_fields_forbidden": [
            "dense_success", "forced_answer", "W_to_C", "probe_score", "heldout_result",
        ],
        "summary": pool_summaries,
        "samples": sorted(all_selected, key=lambda row: (row["dataset"], row["split"], row["sample_id"])),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "PAIR_SAMPLE_MANIFEST.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("selection_fingerprint") != fingerprint:
            raise RuntimeError(f"拒绝覆盖不同的既有配对清单：{manifest_path}")
        manifest = previous
    else:
        atomic_json(manifest, manifest_path)
    queue_root = output_root / "queue"
    created = 0
    for row in manifest["samples"]:
        payload = {
            "kind": "dense",
            "dataset": row["dataset"],
            "split": row["split"],
            "problem_id": row["sample_id"],
            "source_fp16_artifact": row["source_fp16_artifact"],
            "selection_fingerprint": fingerprint,
            "global_seed": seed,
            "output_root": str((output_root / "bfloat16").resolve()),
        }
        created += int(ensure_task(queue_root, payload))
    summary = {
        "status": "complete",
        "selection_fingerprint": fingerprint,
        "sample_count": len(manifest["samples"]),
        "new_queue_tasks": created,
        "manifest": str(manifest_path),
        "queue_root": str(queue_root),
    }
    atomic_json(summary, output_root / "prepare.complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

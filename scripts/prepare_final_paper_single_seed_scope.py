#!/usr/bin/env python3
"""从 replay-v2 父划分生成当前单种子旧经验协议的不可变选择层。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def canonical_fingerprint(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mmlu_subject(problem_id: str, subjects: list[str]) -> str:
    matches = [subject for subject in subjects if problem_id.startswith(f"mmlu_{subject}_test_")]
    if len(matches) != 1:
        raise ValueError(f"无法从 problem_id 唯一解析 MMLU 学科：{problem_id}")
    return matches[0]


def select_mmlu_stratified(
    parent: dict[str, Any], split: str, total: int, seed: int
) -> dict[str, list[str]]:
    """按学科取固定前缀，并用固定 RNG 决定哪些学科多取一题。"""
    subjects = list(parent["subjects"])
    base, remainder = divmod(total, len(subjects))
    extra_subjects = set(
        np.random.default_rng(seed).permutation(subjects)[:remainder].tolist()
    )
    result = {}
    key = f"{split}_problem_ids"
    for subject in subjects:
        quota = base + int(subject in extra_subjects)
        candidates = parent["subject_manifest"][subject][key]
        if len(candidates) < quota:
            raise ValueError(f"{subject} 的 {split} 候选不足：{len(candidates)} < {quota}")
        result[subject] = list(candidates[:quota])
    return result


def select_mmlu_heldout(
    parent: dict[str, Any], seed: int, total: int = 1000
) -> dict[str, list[str]]:
    subjects = list(parent["subjects"])
    grouped = {subject: [] for subject in subjects}
    for problem_id in parent["files"]["heldout"]["problem_ids"]:
        grouped[mmlu_subject(str(problem_id), subjects)].append(str(problem_id))
    base, remainder = divmod(total, len(subjects))
    selected = {}
    for index, subject in enumerate(subjects):
        quota = base + int(index < remainder)
        ranked = sorted(
            grouped[subject],
            key=lambda problem_id: hashlib.sha256(
                f"{seed}:mmlu:heldout:{problem_id}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < quota:
            raise ValueError(f"{subject} 的 held-out 候选不足：{len(ranked)} < {quota}")
        selected[subject] = ranked[:quota]
    return selected


def group_reference_ids(
    parent: dict[str, Any], split: str, problem_ids: list[str]
) -> dict[str, list[str]]:
    """按父清单学科映射验证并分组仓库内冻结 ID。"""
    if split == "heldout":
        grouped = {subject: [] for subject in parent["subjects"]}
        for problem_id in problem_ids:
            grouped[mmlu_subject(problem_id, parent["subjects"])].append(problem_id)
        return grouped
    wanted = set(problem_ids)
    grouped: dict[str, list[str]] = {}
    seen: set[str] = set()
    key = f"{split}_problem_ids"
    for subject in parent["subjects"]:
        subject_ids = [
            str(value)
            for value in parent["subject_manifest"][subject][key]
            if str(value) in wanted
        ]
        grouped[subject] = subject_ids
        seen.update(subject_ids)
    if seen != wanted:
        missing = sorted(wanted - seen)
        raise ValueError(f"冻结 {split} ID 不在父清单中：{missing[:5]}")
    return grouped


def load_reference_ids(root: Path, dataset: str, split: str) -> list[str]:
    path = root / f"{dataset}_{split}_ids.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError(f"冻结 ID 文件必须是列表：{path}")
    return [str(value) for value in values]


def build_scope(
    gsm: dict[str, Any],
    mmlu: dict[str, Any],
    seed: int,
    reference_root: Path | None = None,
) -> dict[str, Any]:
    if seed != 20260803:
        raise ValueError("论文协议只允许 seed=20260803")
    if reference_root is None:
        reference_root = ROOT / "splits/legacy_empirical_v4_train1000_cal500_mmlu1k"
    gsm_probe = load_reference_ids(reference_root, "gsm8k", "probe_train")
    gsm_calibration = load_reference_ids(reference_root, "gsm8k", "calibration")
    gsm_heldout = load_reference_ids(reference_root, "gsm8k", "heldout")
    mmlu_probe = load_reference_ids(reference_root, "mmlu", "probe_train")
    mmlu_calibration = load_reference_ids(reference_root, "mmlu", "calibration")
    mmlu_heldout = load_reference_ids(reference_root, "mmlu", "heldout")
    expected = {
        "gsm8k": {"probe_train": 1000, "calibration": 500, "heldout": 1319},
        "mmlu": {"probe_train": 1000, "calibration": 500, "heldout": 1000},
    }
    all_ids = {
        "gsm8k": {"probe_train": gsm_probe, "calibration": gsm_calibration, "heldout": gsm_heldout},
        "mmlu": {"probe_train": mmlu_probe, "calibration": mmlu_calibration, "heldout": mmlu_heldout},
    }
    for dataset, splits in all_ids.items():
        for split, values in splits.items():
            if len(values) != expected[dataset][split] or len(set(values)) != len(values):
                raise ValueError(f"{dataset}/{split} 冻结 ID 数量或唯一性错误")
        if set(splits["probe_train"]) & set(splits["calibration"]):
            raise ValueError(f"{dataset} probe/calibration 重叠")
        if (set(splits["probe_train"]) | set(splits["calibration"])) & set(splits["heldout"]):
            raise ValueError(f"{dataset} train/calibration 与 heldout 重叠")
    for split, values in all_ids["gsm8k"].items():
        parent_ids = set(str(value) for value in gsm["files"][split]["problem_ids"])
        if not set(values) <= parent_ids:
            raise ValueError(f"GSM8K {split} 冻结 ID 不在父清单中")
    mmlu_probe_by_subject = group_reference_ids(mmlu, "probe_train", mmlu_probe)
    mmlu_calibration_by_subject = group_reference_ids(mmlu, "calibration", mmlu_calibration)
    mmlu_heldout_by_subject = group_reference_ids(mmlu, "heldout", mmlu_heldout)
    scope = {
        "schema_version": 1,
        "protocol_id": "legacy_empirical_v4_train1000_cal500_mmlu1k",
        "seed": seed,
        "selection_is_output_independent": True,
        "datasets": {
            "gsm8k": {
                "parent_split_fingerprint": gsm["fingerprint"],
                "selection": "父划分固定顺序的前 1000 个 probe、前 500 个 calibration；完整 official test",
                "probe_train_problem_ids": gsm_probe,
                "calibration_problem_ids": gsm_calibration,
                "heldout_problem_ids": gsm_heldout,
                "probe_train_count": len(gsm_probe),
                "calibration_count": len(gsm_calibration),
                "heldout_count": len(gsm_heldout),
            },
            "mmlu": {
                "parent_split_fingerprint": mmlu["fingerprint"],
                "selection": "probe/calibration 按 57 学科分层取 1000/500；heldout 为固定 MMLU-1k",
                "subjects": list(mmlu["subjects"]),
                "probe_train_by_subject": mmlu_probe_by_subject,
                "calibration_by_subject": mmlu_calibration_by_subject,
                "heldout_by_subject": mmlu_heldout_by_subject,
                "probe_train_problem_ids": mmlu_probe,
                "calibration_problem_ids": mmlu_calibration,
                "heldout_problem_ids": mmlu_heldout,
                "probe_train_count": len(mmlu_probe),
                "calibration_count": len(mmlu_calibration),
                "heldout_count": len(mmlu_heldout),
                "probe_train_source": "auxiliary_train",
                "calibration_source": "auxiliary_train（相对 official test 存在分布偏移）",
                "heldout_source": "official test",
                "heldout_subject_counts": {
                    subject: len(ids) for subject, ids in mmlu_heldout_by_subject.items()
                },
            },
        },
        "invariants": {
            "gsm8k_official_test_complete_1319": len(gsm_heldout) == 1319,
            "mmlu_all_57_subjects_covered": len(mmlu_heldout_by_subject) == 57,
            "mmlu_heldout_17_or_18_per_subject": set(
                len(ids) for ids in mmlu_heldout_by_subject.values()
            ) == {17, 18},
            "heldout_not_used_for_probe_scaler_or_thresholds": True,
        },
    }
    scope["scope_fingerprint"] = canonical_fingerprint(scope)
    return scope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--gsm-parent", type=Path, default=Path("splits/gsm8k_split.json"))
    parser.add_argument("--mmlu-parent", type=Path, default=Path("splits/mmlu_split.json"))
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("splits/legacy_empirical_v4_train1000_cal500_mmlu1k"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/legacy_empirical_v4/splits"),
    )
    args = parser.parse_args()
    gsm_path = args.gsm_parent if args.gsm_parent.is_absolute() else ROOT / args.gsm_parent
    mmlu_path = args.mmlu_parent if args.mmlu_parent.is_absolute() else ROOT / args.mmlu_parent
    reference = args.reference_root if args.reference_root.is_absolute() else ROOT / args.reference_root
    output = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    scope = build_scope(load_json(gsm_path), load_json(mmlu_path), args.seed, reference)
    atomic_json(scope, output / "scope_manifest.json")
    audit_selection = {
        dataset: {
            split: details[f"{split}_problem_ids"]
            for split in ("probe_train", "calibration", "heldout")
        }
        for dataset, details in scope["datasets"].items()
    }
    atomic_json(audit_selection, output / "audit_selection.json")
    for dataset, details in scope["datasets"].items():
        for split in ("probe_train", "calibration", "heldout"):
            atomic_json(details[f"{split}_problem_ids"], output / f"{dataset}_{split}_ids.json")
    print(json.dumps({
        "status": "complete",
        "scope_fingerprint": scope["scope_fingerprint"],
        "counts": {
            dataset: {
                split: details[f"{split}_count"]
                for split in ("probe_train", "calibration", "heldout")
            }
            for dataset, details in scope["datasets"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

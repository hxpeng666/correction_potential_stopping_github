#!/usr/bin/env python3
"""从 replay-v2 父划分生成本轮单种子论文实验的不可变选择层。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

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


def select_mmlu_probe(parent: dict[str, Any], total: int = 2000) -> dict[str, list[str]]:
    subjects = list(parent["subjects"])
    base, remainder = divmod(total, len(subjects))
    result = {}
    for index, subject in enumerate(subjects):
        quota = base + int(index < remainder)
        candidates = parent["subject_manifest"][subject]["probe_train_problem_ids"]
        if len(candidates) < quota:
            raise ValueError(f"{subject} 的 probe 候选不足：{len(candidates)} < {quota}")
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


def build_scope(gsm: dict[str, Any], mmlu: dict[str, Any], seed: int) -> dict[str, Any]:
    if seed != 20260803:
        raise ValueError("论文协议只允许 seed=20260803")
    gsm_probe = list(gsm["files"]["probe_train"]["problem_ids"][:2000])
    gsm_calibration = list(gsm["files"]["calibration"]["problem_ids"])
    gsm_heldout = list(gsm["files"]["heldout"]["problem_ids"])
    mmlu_probe_by_subject = select_mmlu_probe(mmlu, 2000)
    mmlu_probe = [item for subject in mmlu["subjects"] for item in mmlu_probe_by_subject[subject]]
    mmlu_heldout_by_subject = select_mmlu_heldout(mmlu, seed, 1000)
    mmlu_heldout = [
        item for subject in mmlu["subjects"] for item in mmlu_heldout_by_subject[subject]
    ]
    scope = {
        "schema_version": 1,
        "protocol_id": "final_paper_single_seed_mmlu1k_v1",
        "seed": seed,
        "selection_is_output_independent": True,
        "datasets": {
            "gsm8k": {
                "parent_split_fingerprint": gsm["fingerprint"],
                "selection": "父划分固定哈希顺序的前 2000 个 probe；完整 calibration 和 official test",
                "probe_train_problem_ids": gsm_probe,
                "calibration_problem_ids": gsm_calibration,
                "heldout_problem_ids": gsm_heldout,
                "probe_train_count": len(gsm_probe),
                "calibration_count": len(gsm_calibration),
                "heldout_count": len(gsm_heldout),
            },
            "mmlu": {
                "parent_split_fingerprint": mmlu["fingerprint"],
                "selection": "probe 按 57 学科平衡取 2000；heldout 学科内固定哈希平衡取 MMLU-1k",
                "subjects": list(mmlu["subjects"]),
                "probe_train_by_subject": mmlu_probe_by_subject,
                "heldout_by_subject": mmlu_heldout_by_subject,
                "probe_train_problem_ids": mmlu_probe,
                "calibration_problem_ids": list(mmlu["files"]["calibration"]["problem_ids"]),
                "heldout_problem_ids": mmlu_heldout,
                "probe_train_count": len(mmlu_probe),
                "calibration_count": int(mmlu["files"]["calibration"]["count"]),
                "heldout_count": len(mmlu_heldout),
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
        "--output-root", type=Path, default=Path("splits/final_paper_single_seed_mmlu1k_v1")
    )
    args = parser.parse_args()
    gsm_path = args.gsm_parent if args.gsm_parent.is_absolute() else ROOT / args.gsm_parent
    mmlu_path = args.mmlu_parent if args.mmlu_parent.is_absolute() else ROOT / args.mmlu_parent
    output = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    scope = build_scope(load_json(gsm_path), load_json(mmlu_path), args.seed)
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

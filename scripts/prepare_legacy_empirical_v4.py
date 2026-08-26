#!/usr/bin/env python3
"""冻结legacy-v4数据清单，并生成协议/缓存兼容性审计。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from src.final_paper_protocol import MMLU_SUBJECTS, answer_letter, canonical_fingerprint, mmlu_category, normalize_question
from src.utils import atomic_json

SEED = 20260803
RUN_ROOT = ROOT / "results/final_paper_replay_v4/legacy_empirical_protocol_train1000_cal500_mmlu1000"
DATA_ROOT = ROOT / "data/final_paper_replay_v4/legacy_empirical_protocol_train1000_cal500_mmlu1000"
PARENT_DATA = ROOT / "data/final_paper_replay_v2"
PARENT_SCOPE = ROOT / "results/final_paper_replay_v3/degraded_train1000_cal500_mmlu1000_final_paper/SCOPE_MANIFEST.json"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stable_order(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def add_seed_index(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["legacy_seed_index"] = int(hashlib.sha256(str(row["problem_id"]).encode()).hexdigest()[:12], 16)
    return result


def select_by_ids(rows: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    by_id = {str(row["problem_id"]): row for row in rows}
    missing = [value for value in ids if value not in by_id]
    if missing:
        raise KeyError(f"missing frozen IDs: {missing[:10]}")
    return [add_seed_index(by_id[value]) for value in ids]


def select_stratified(rows: list[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subject"])].append(row)
    if set(grouped) != set(MMLU_SUBJECTS):
        raise ValueError(f"probe-train pool subject coverage is {len(grouped)}/57")
    base, remainder = divmod(total, len(MMLU_SUBJECTS))
    extra_subjects = set(sorted(MMLU_SUBJECTS, key=lambda value: stable_order(f"train_quota:{value}"))[:remainder])
    selected = []
    for subject in MMLU_SUBJECTS:
        quota = base + int(subject in extra_subjects)
        ordered = sorted(grouped[subject], key=lambda row: stable_order(str(row["problem_id"])))
        if len(ordered) < quota:
            raise ValueError(f"insufficient probe-train rows for {subject}: {len(ordered)}/{quota}")
        selected.extend(add_seed_index(row) for row in ordered[:quota])
    selected.sort(key=lambda row: (row["subject"], stable_order(row["problem_id"])))
    if len(selected) != total:
        raise AssertionError(len(selected))
    return selected


def validation_rows(excluded_questions: set[str]) -> list[dict[str, Any]]:
    dataset = load_dataset(
        "cais/mmlu",
        "all",
        revision=MMLU_REVISION,
        cache_dir=str(ROOT / ".final_paper_cache"),
        download_mode="reuse_dataset_if_exists",
    )["validation"]
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set(excluded_questions)
    for index, row in enumerate(dataset):
        subject = str(row["subject"])
        normalized = normalize_question(row["question"])
        if normalized in seen:
            continue
        seen.add(normalized)
        problem_id = f"mmlu_{subject}_validation_{index:05d}"
        item = {
            "problem_id": problem_id,
            "question": str(row["question"]),
            "choices": [str(value) for value in row["choices"]],
            "answer": answer_letter(row["answer"]),
            "subject": subject,
            "category": mmlu_category(subject),
            "source_split": "validation",
            "source_index": index,
            "selection_role": "calibration",
        }
        item["record_fingerprint"] = canonical_fingerprint(item)
        candidates[subject].append(add_seed_index(item))
    if set(candidates) != set(MMLU_SUBJECTS):
        raise ValueError("official validation does not cover all 57 subjects")
    eligible_for_extra = [subject for subject in MMLU_SUBJECTS if len(candidates[subject]) >= 9]
    if len(eligible_for_extra) < 44:
        raise ValueError(f"only {len(eligible_for_extra)} subjects can supply a ninth validation row")
    extra_subjects = set(sorted(eligible_for_extra, key=lambda value: stable_order(f"quota:{value}"))[:44])
    selected = []
    for subject in MMLU_SUBJECTS:
        quota = 9 if subject in extra_subjects else 8
        ordered = sorted(candidates[subject], key=lambda row: stable_order(row["problem_id"]))
        if len(ordered) < quota:
            raise ValueError(f"insufficient validation rows for {subject}: {len(ordered)}/{quota}")
        selected.extend(ordered[:quota])
    selected.sort(key=lambda row: (row["subject"], stable_order(row["problem_id"])))
    if len(selected) != 500:
        raise AssertionError(len(selected))
    return selected


def write_audit(scope: dict[str, Any]) -> None:
    differences = {
        "status": "AUDITED_REGENERATION_REQUIRED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "legacy_target": {
            "dtype": "bfloat16",
            "attention_backend": "sdpa",
            "gsm8k_dense_seed": "20260803 + source_index * 1009",
            "branch_seed": "example_seed + checkpoint * 7919",
            "probe_seed": 0,
            "internal_split": "numpy.random.default_rng(0), problem-level 80/20",
            "threshold": "101 quantiles + no-stop sentinel; empirical absolute B={0,1,2,4,10}",
            "workpoints": {"strict": 1, "balanced": 2, "aggressive": 4},
        },
        "current_v3": {
            "dtype": "float16",
            "generation_seed": "sha256(global_seed,dataset,split,sample_id,checkpoint)",
            "probe_seed": 20260803,
            "internal_split": "subject/hash split seeded by 20260803",
            "threshold": "Bonferroni simultaneous UCB with disabled policy",
            "mmlu_calibration_source": "auxiliary_train",
        },
        "cache_reuse": {
            "a100_cost_model": "REUSE",
            "frozen_gsm8k_and_mmlu_test_ids": "REUSE",
            "mmlu_dev_demonstrations_and_parser": "REUSE",
            "v2_v3_dense_hidden_forced_answers": "REJECT: FP16 and seed fingerprint mismatch",
            "final_paper_v1_partial_dense": "REJECT: historical hash dense seed is not the mandated source_index formula",
            "probe_and_policy_outputs": "REJECT: input trajectories/training/calibration protocol changed",
        },
        "mmlu_source_fix": {
            "probe_train": "auxiliary_train, frozen 1000",
            "calibration": "official validation, stratified 500, 57/57 subjects",
            "heldout": "official test, frozen balanced 1000, 57/57 subjects",
        },
        "scope_fingerprint": scope["scope_fingerprint"],
    }
    atomic_json(differences, RUN_ROOT / "legacy_protocol_diff.json")
    markdown = """# Legacy协议审计

## 结论

当前 replay-v3 结果不能直接作为 legacy 主结果。它使用 FP16、哈希任务种子、seed=20260803 的内部训练划分、按比例经验预算及 Bonferroni simultaneous-bound 工作点；目标协议要求 BF16、GSM8K 数值样本种子、probe seed=0、`default_rng(0)` 问题级80/20划分和绝对事件预算 `B={0,1,2,4,10}`。

## 可直接复用

- 冻结的 GSM8K official-test 1,319 ID与当前 train/calibration ID清单；
- 冻结的57学科均衡 MMLU-1k test ID与 MMLU dev 5-shot demonstrations；
- 当前冻结的 MCQ prompt、选项顺序与答案解析器；
- 已验证的 A100 single-request replay成本模型。

## 不能复用

- replay-v2/v3 Dense、hidden、entropy与forced-answer缓存：dtype 为FP16且逐样本seed规则不同；
- partial final_paper_v1 Dense：采集入口使用SHA256 seed，而非本轮明确指定的 `20260803 + source_index*1009`；
- 现有probe/scaler/threshold/policy结果：输入轨迹、内部划分、训练seed和阈值协议均不一致。

## 已确认一致或可直接迁移

- 模型权重、Qwen3原生Transformers、SDPA、thinking、temperature/top-p/top-k与4096/16 token上限；
- GSM8K prompt及Decimal答案判定顺序；
- sentence边界正则、64/768范围、8-token gap与first-hit停止；
- layer `[8,20,35]`、主layer 20、5126维特征及MLP结构；
- Correction加权point loss与beta=0.5 soft-min trajectory loss公式；
- 四种target的正类与停止方向。

## 必须修复

- 重新采集BF16且legacy seed一致的Dense/Direct/teacher-forced hidden/forced-answer公共缓存；
- MMLU calibration从auxiliary_train改为official validation的57学科分层500题；
- 恢复probe seed=0及`default_rng(0)`问题级80/20内部划分；
- 恢复101分位点加no-stop sentinel、绝对B预算及30%到90% coverage目标；
- 修复Dense fallback，使coverage、token/wall saving和transition stop count严格为零；
- 分开命名historical empirical与可选formal-certified结果；
- 使用同一A100成本模型重算全部方法，并补充question-level paired 10,000 bootstrap。

## 泄漏边界

GSM8K test保持完整封存；MMLU test保持已冻结的均衡1k。MMLU calibration仅来自official validation，dev只用于5-shot，auxiliary_train只用于probe拟合。任何held-out输出均不参与epoch或阈值选择。
"""
    path = RUN_ROOT / "LEGACY_PROTOCOL_AUDIT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(markdown, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parent_scope = json.loads(PARENT_SCOPE.read_text(encoding="utf-8"))
    datasets = parent_scope["datasets"]
    gsm_source = {
        split: read_jsonl(PARENT_DATA / "gsm8k" / f"{split}.jsonl")
        for split in ("probe_train", "calibration", "heldout")
    }
    gsm = {
        split: select_by_ids(gsm_source[split], datasets["gsm8k"][f"{split}_problem_ids"])
        for split in ("probe_train", "calibration", "heldout")
    }
    mmlu_source = {
        split: read_jsonl(PARENT_DATA / "mmlu" / f"{split}.jsonl")
        for split in ("probe_train", "heldout")
    }
    demos = [add_seed_index(row) for row in read_jsonl(PARENT_DATA / "mmlu/demonstrations.jsonl")]
    mmlu_train = select_stratified(mmlu_source["probe_train"], 1000)
    mmlu_test = select_by_ids(mmlu_source["heldout"], datasets["mmlu"]["heldout_problem_ids"])
    excluded = {normalize_question(row["question"]) for row in mmlu_train + mmlu_test + demos}
    mmlu_calibration = validation_rows(excluded)
    for split, rows in gsm.items():
        atomic_jsonl(rows, DATA_ROOT / "gsm8k" / f"{split}.jsonl")
    atomic_jsonl(mmlu_train, DATA_ROOT / "mmlu/probe_train.jsonl")
    atomic_jsonl(mmlu_calibration, DATA_ROOT / "mmlu/calibration.jsonl")
    atomic_jsonl(mmlu_test, DATA_ROOT / "mmlu/heldout.jsonl")
    atomic_jsonl(demos, DATA_ROOT / "mmlu/demonstrations.jsonl")
    all_sets = {
        "gsm8k": gsm,
        "mmlu": {"probe_train": mmlu_train, "calibration": mmlu_calibration, "heldout": mmlu_test},
    }
    for dataset, splits in all_sets.items():
        id_sets = {name: {row["problem_id"] for row in rows} for name, rows in splits.items()}
        if any(id_sets[a] & id_sets[b] for a in id_sets for b in id_sets if a < b):
            raise ValueError(f"problem ID leakage in {dataset}")
    subject_counts = {
        split: dict(sorted(Counter(row["subject"] for row in rows).items()))
        for split, rows in all_sets["mmlu"].items()
    }
    scope = {
        "schema_version": 1,
        "protocol_id": "legacy_empirical_v4",
        "seed": SEED,
        "datasets": {
            dataset: {
                f"{split}_problem_ids": [row["problem_id"] for row in rows]
                for split, rows in splits.items()
            }
            for dataset, splits in all_sets.items()
        },
        "mmlu_subject_counts": subject_counts,
        "mmlu_sources": {"probe_train": "auxiliary_train", "calibration": "validation", "heldout": "test", "demonstrations": "dev"},
        "invariants": {
            "gsm8k_counts": [1000, 500, 1319],
            "mmlu_counts": [1000, 500, 1000],
            "mmlu_subjects_each_split": [len(subject_counts[name]) for name in ("probe_train", "calibration", "heldout")],
            "test_used_for_training_scaling_or_thresholds": False,
        },
    }
    scope["scope_fingerprint"] = canonical_fingerprint(scope)
    atomic_json(scope, RUN_ROOT / "SCOPE_MANIFEST.json")
    write_audit(scope)
    atomic_json({"status": "complete", "scope_fingerprint": scope["scope_fingerprint"]}, RUN_ROOT / "phases/prepare_and_audit.complete")
    print(json.dumps({
        "status": "complete",
        "scope_fingerprint": scope["scope_fingerprint"],
        "counts": {dataset: {split: len(rows) for split, rows in splits.items()} for dataset, splits in all_sets.items()},
        "mmlu_sources": scope["mmlu_sources"],
        "mmlu_subject_coverage": {split: len(values) for split, values in subject_counts.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

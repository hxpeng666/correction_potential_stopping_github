#!/usr/bin/env python3
"""准备不可变的单随机种子 GSM8K 与 MMLU 论文数据划分。

受保护的测试答案会复制到评估文件中，但绝不进入辅助学科路由器、探针训练、
验证或阈值校准。固定快照中的 MMLU ``auxiliary_train`` 学科列为空；本实现
根据其与开发集及验证集 TF-IDF 中心的相似度确定性分配学科，然后按轮转顺序
为每个学科分配 120 道不重复的辅助题，最后应用预注册的平衡 4,000/1,000
配额。该限制以及每个被选中的来源行都会记录在 ``mmlu_split.json`` 中。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.final_paper_protocol import (
    MMLU_SUBJECTS,
    answer_letter,
    canonical_fingerprint,
    mmlu_category,
    normalize_question,
    stable_hash,
)

GSM_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


def resolve_snapshot(
    explicit: Path | None,
    repo_id: str,
    revision: str,
) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )
    )


def order_key(seed: int, dataset: str, subject: str, problem_id: str) -> str:
    value = f"{seed}:{dataset}:{subject}:{problem_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def balanced_quota(total: int) -> dict[str, int]:
    base, remainder = divmod(total, len(MMLU_SUBJECTS))
    return {
        subject: base + int(index < remainder)
        for index, subject in enumerate(MMLU_SUBJECTS)
    }


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            line = (
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            handle.write(line)
            digest.update(line)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count, digest.hexdigest()


def dataframe_rows(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def serial_choices(value: Any) -> list[str]:
    choices = [str(item) for item in list(value)]
    if len(choices) != 4:
        raise ValueError(f"expected four choices, found {len(choices)}")
    return choices


def record_fingerprint(question: str, choices: list[str], answer: str) -> str:
    return canonical_fingerprint(
        {"question": normalize_question(question), "choices": choices, "answer": answer}
    )


def prepare_gsm8k(source: Path, output: Path, split_path: Path, seed: int) -> dict[str, Any]:
    train = dataframe_rows(source / "train-00000-of-00001.parquet")
    test = dataframe_rows(source / "test-00000-of-00001.parquet")
    if len(train) != 7473 or len(test) != 1319:
        raise ValueError(f"unexpected GSM8K counts: train={len(train)}, test={len(test)}")
    def convert(frame: pd.DataFrame, ids: Iterable[int], source_split: str):
        for index in ids:
            row = frame.iloc[index]
            yield {
                "problem_id": f"gsm8k_{source_split}_{index:05d}",
                "source_index": index,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
            }

    train_rows = list(convert(train, range(len(train)), "train"))
    ordered = sorted(
        train_rows,
        key=lambda row: order_key(seed, "gsm8k", "", str(row["problem_id"])),
    )
    rows_by_split = {
        "probe_train": ordered[:5000],
        "calibration": ordered[5000:6000],
        "heldout": list(convert(test, range(len(test)), "test")),
    }
    files: dict[str, Any] = {}
    for name, rows in rows_by_split.items():
        path = output / f"{name}.jsonl"
        count, fingerprint = atomic_jsonl(rows, path)
        files[name] = {
            "count": count,
            "sha256": fingerprint,
            "problem_ids": [str(row["problem_id"]) for row in rows],
        }
    probe_ids = [int(row["source_index"]) for row in rows_by_split["probe_train"]]
    calibration_ids = [
        int(row["source_index"]) for row in rows_by_split["calibration"]
    ]
    unused_ids = [str(row["problem_id"]) for row in ordered[6000:]]
    if set(probe_ids) & set(calibration_ids):
        raise AssertionError("GSM8K probe/calibration overlap")
    if len(set(probe_ids + calibration_ids)) + len(unused_ids) != 7473:
        raise AssertionError("GSM8K official train accounting failed")
    split = {
        "schema_version": 2,
        "protocol_id": "final_paper_replay_v2",
        "dataset": "openai/gsm8k",
        "seed": seed,
        "selection": "sha256(seed,dataset,split-independent sample_id) order",
        "files": files,
        "unused_official_train_problem_ids": unused_ids,
        "invariants": {
            "probe_train": 5000,
            "calibration": 1000,
            "official_test": 1319,
            "test_used_for_training_threshold_or_scaler": False,
        },
    }
    split["fingerprint"] = canonical_fingerprint(split)
    atomic_json(split, split_path)
    return split


def load_mmlu_subject(source: Path, subject: str, split: str) -> pd.DataFrame:
    return dataframe_rows(source / subject / f"{split}-00000-of-00001.parquet")


def mmlu_rows(
    frame: pd.DataFrame,
    *,
    subject: str,
    source_split: str,
    selected_indices: list[int] | None = None,
):
    indices = list(range(len(frame))) if selected_indices is None else selected_indices
    for index in indices:
        row = frame.iloc[index]
        choices = serial_choices(row["choices"])
        letter = answer_letter(row["answer"])
        yield {
            "problem_id": f"mmlu_{subject}_{source_split}_{index:05d}",
            "subject": subject,
            "category": mmlu_category(subject),
            "source_split": source_split,
            "source_index": int(index),
            "question": str(row["question"]),
            "choices": choices,
            "answer": letter,
            "record_fingerprint": record_fingerprint(str(row["question"]), choices, letter),
        }


def prepare_mmlu(
    source: Path,
    output: Path,
    split_path: Path,
    seed: int,
    max_features: int,
) -> dict[str, Any]:
    dev = {subject: load_mmlu_subject(source, subject, "dev") for subject in MMLU_SUBJECTS}
    validation = {
        subject: load_mmlu_subject(source, subject, "validation")
        for subject in MMLU_SUBJECTS
    }
    test = {subject: load_mmlu_subject(source, subject, "test") for subject in MMLU_SUBJECTS}
    if any(len(dev[subject]) != 5 for subject in MMLU_SUBJECTS):
        raise ValueError("every MMLU subject must have exactly five dev demonstrations")
    if sum(len(test[x]) for x in MMLU_SUBJECTS) != 14042:
        raise ValueError("MMLU test must contain exactly 14042 questions")

    protected_text = {
        normalize_question(str(row["question"]))
        for frames in (dev, validation, test)
        for frame in frames.values()
        for _, row in frame.iterrows()
    }
    auxiliary = dataframe_rows(source / "all" / "auxiliary_train-00000-of-00001.parquet")
    source_subject_values = sorted(set(auxiliary["subject"].fillna("").astype(str)))
    if source_subject_values != [""]:
        raise ValueError(
            "routing protocol was preregistered for the known blank auxiliary subject field; "
            f"found {source_subject_values[:10]}"
        )

    unique_aux_indices: list[int] = []
    unique_aux_text: list[str] = []
    seen = set(protected_text)
    overlap_removed = 0
    duplicate_removed = 0
    for index, question in enumerate(auxiliary["question"].astype(str)):
        normalized = normalize_question(question)
        if normalized in protected_text:
            overlap_removed += 1
            continue
        if normalized in seen:
            duplicate_removed += 1
            continue
        seen.add(normalized)
        unique_aux_indices.append(index)
        unique_aux_text.append(normalized)
    if len(unique_aux_indices) < 57 * 120:
        raise ValueError("not enough deduplicated MMLU auxiliary examples")

    reference_text: list[str] = []
    reference_subject: list[str] = []
    for subject in MMLU_SUBJECTS:
        for frame in (dev[subject], validation[subject]):
            for question in frame["question"].astype(str):
                reference_text.append(normalize_question(question))
                reference_subject.append(subject)

    vectorizer = TfidfVectorizer(
        lowercase=False,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    vectorizer.fit(unique_aux_text + reference_text)
    references = vectorizer.transform(reference_text)
    auxiliary_matrix = vectorizer.transform(unique_aux_text)
    centroids = []
    reference_subject_array = np.asarray(reference_subject)
    for subject in MMLU_SUBJECTS:
        positions = np.flatnonzero(reference_subject_array == subject)
        centroids.append(np.asarray(references[positions].mean(axis=0), dtype=np.float32))
    centroid_matrix = normalize(np.concatenate(centroids, axis=0), norm="l2")
    similarities = np.asarray(auxiliary_matrix @ centroid_matrix.T, dtype=np.float32)

    tie_hashes = np.asarray(
        [int(stable_hash(text)[:16], 16) for text in unique_aux_text], dtype=np.uint64
    )
    rankings = [
        np.lexsort((tie_hashes, -similarities[:, subject_index]))
        for subject_index in range(len(MMLU_SUBJECTS))
    ]
    pointers = np.zeros(len(MMLU_SUBJECTS), dtype=np.int64)
    used: set[int] = set()
    selected: dict[str, list[int]] = {subject: [] for subject in MMLU_SUBJECTS}
    routing_scores: dict[tuple[str, int], float] = {}
    for _round in range(120):
        for subject_index, subject in enumerate(MMLU_SUBJECTS):
            ranking = rankings[subject_index]
            pointer = int(pointers[subject_index])
            while pointer < len(ranking) and int(ranking[pointer]) in used:
                pointer += 1
            if pointer >= len(ranking):
                raise RuntimeError(f"exhausted unique auxiliary examples for {subject}")
            local_index = int(ranking[pointer])
            pointers[subject_index] = pointer + 1
            used.add(local_index)
            source_index = int(unique_aux_indices[local_index])
            selected[subject].append(source_index)
            routing_scores[(subject, source_index)] = float(
                similarities[local_index, subject_index]
            )

    def routed_rows(subject: str, indices: list[int], role: str):
        for source_index in indices:
            row = auxiliary.iloc[source_index]
            choices = serial_choices(row["choices"])
            letter = answer_letter(row["answer"])
            yield {
                "problem_id": f"mmlu_{subject}_auxiliary_{source_index:05d}",
                "subject": subject,
                "category": mmlu_category(subject),
                "source_split": "auxiliary_train",
                "source_index": source_index,
                "routing_role": role,
                "routing_method": "tfidf_dev_validation_centroid_round_robin",
                "routing_score": routing_scores[(subject, source_index)],
                "question": str(row["question"]),
                "choices": choices,
                "answer": letter,
                "record_fingerprint": record_fingerprint(
                    str(row["question"]), choices, letter
                ),
            }

    probe_quota = balanced_quota(4000)
    calibration_quota = balanced_quota(1000)
    probe_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    selected_rows_by_subject: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for subject in MMLU_SUBJECTS:
        initial_routed = list(
            routed_rows(subject, selected[subject][:100], "probe_train")
        ) + list(
            routed_rows(subject, selected[subject][100:120], "calibration")
        )
        candidates = sorted(
            initial_routed,
            key=lambda row: order_key(
                seed,
                "mmlu",
                subject,
                str(row["problem_id"]),
            ),
        )
        p_count = probe_quota[subject]
        c_count = calibration_quota[subject]
        local_probe = candidates[:p_count]
        local_calibration = candidates[p_count:p_count + c_count]
        probe_rows.extend(local_probe)
        calibration_rows.extend(local_calibration)
        selected_rows_by_subject[subject] = {
            "probe_train": local_probe,
            "calibration": local_calibration,
        }
    demo_rows = [
        row
        for subject in MMLU_SUBJECTS
        for row in mmlu_rows(dev[subject], subject=subject, source_split="dev")
    ]
    heldout_rows = [
        row
        for subject in MMLU_SUBJECTS
        for row in mmlu_rows(test[subject], subject=subject, source_split="test")
    ]

    files = {}
    for name, rows in (
        ("probe_train", probe_rows),
        ("calibration", calibration_rows),
        ("demonstrations", demo_rows),
        ("heldout", heldout_rows),
    ):
        path = output / f"{name}.jsonl"
        count, fingerprint = atomic_jsonl(rows, path)
        files[name] = {
            "count": count,
            "sha256": fingerprint,
            "problem_ids": [str(row["problem_id"]) for row in rows],
        }

    subject_manifest = {}
    for subject in MMLU_SUBJECTS:
        subject_manifest[subject] = {
            "probe_train_count": probe_quota[subject],
            "calibration_count": calibration_quota[subject],
            "probe_train_problem_ids": [
                str(row["problem_id"])
                for row in selected_rows_by_subject[subject]["probe_train"]
            ],
            "calibration_problem_ids": [
                str(row["problem_id"])
                for row in selected_rows_by_subject[subject]["calibration"]
            ],
        }
    split = {
        "schema_version": 2,
        "protocol_id": "final_paper_replay_v2",
        "dataset": "cais/mmlu",
        "seed": seed,
        "subjects": list(MMLU_SUBJECTS),
        "selection": "within-routed-subject sha256(seed,dataset,subject,sample_id) order with balanced quotas",
        "subject_manifest": subject_manifest,
        "files": files,
        "heldout_subject_counts": {
            subject: len(test[subject]) for subject in MMLU_SUBJECTS
        },
        "invariants": {
            "probe_train": 4000,
            "calibration": 1000,
            "demonstrations": 285,
            "official_test": 14042,
            "subjects": 57,
            "normalized_question_overlap_with_dev_or_test": 0,
            "test_used_for_training_threshold_or_scaler": False,
        },
    }
    split["fingerprint"] = canonical_fingerprint(split)
    atomic_json(split, split_path)
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gsm-snapshot",
        type=Path,
        help="可选的本地固定快照；未指定时下载或复用 Hugging Face 缓存。",
    )
    parser.add_argument(
        "--mmlu-snapshot",
        type=Path,
        help="可选的本地固定快照；未指定时下载或复用 Hugging Face 缓存。",
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        default=ROOT / "data/final_paper_replay_v2",
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=ROOT / "results/final_paper_replay_v2/splits",
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--tfidf-max-features", type=int, default=100000)
    parser.add_argument(
        "--reference-splits-root",
        type=Path,
        default=ROOT / "splits",
        help="随仓库发布的不可变清单；数据划分漂移时立即中止。",
    )
    args = parser.parse_args()

    if args.seed != 20260803:
        raise ValueError("the published single-seed protocol fixes seed=20260803")
    gsm_source = resolve_snapshot(
        args.gsm_snapshot,
        "openai/gsm8k",
        GSM_REVISION,
    ) / "main"
    mmlu_source = resolve_snapshot(
        args.mmlu_snapshot,
        "cais/mmlu",
        MMLU_REVISION,
    )
    gsm = prepare_gsm8k(
        gsm_source,
        args.data_output / "gsm8k",
        args.split_output / "gsm8k_split.json",
        args.seed,
    )
    print(json.dumps({"phase": "gsm8k", "fingerprint": gsm["fingerprint"]}), flush=True)
    mmlu = prepare_mmlu(
        mmlu_source,
        args.data_output / "mmlu",
        args.split_output / "mmlu_split.json",
        args.seed,
        args.tfidf_max_features,
    )
    print(json.dumps({"phase": "mmlu", "fingerprint": mmlu["fingerprint"]}), flush=True)
    reference_root = (
        args.reference_splits_root
        if args.reference_splits_root.is_absolute()
        else ROOT / args.reference_splits_root
    )
    for dataset, generated in (("gsm8k", gsm), ("mmlu", mmlu)):
        reference_path = reference_root / f"{dataset}_split.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if generated["fingerprint"] != reference["fingerprint"]:
            raise RuntimeError(
                f"{dataset} split drift: generated={generated['fingerprint']} "
                f"reference={reference['fingerprint']}"
            )
    atomic_json(
        {
            "status": "complete",
            "gsm8k_fingerprint": gsm["fingerprint"],
            "mmlu_fingerprint": mmlu["fingerprint"],
        },
        args.split_output / "prepare_final_paper_data.complete",
    )


if __name__ == "__main__":
    main()

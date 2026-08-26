#!/usr/bin/env python3
"""Prepare immutable full GSM8K and MMLU final-paper data/split manifests.

The protected test answers are copied into evaluation files but never enter the
auxiliary subject router, probe training, validation, or threshold calibration.
MMLU's cached auxiliary_train has an empty subject column. We therefore assign a
subject deterministically using TF-IDF similarity to dev+validation centroids,
then allocate 120 unique auxiliary questions per subject in round-robin order.
This limitation and every selected source row are recorded in mmlu_split.json.
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
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
DEFAULT_GSM = HF_HOME / "hub" / f"datasets--openai--gsm8k/snapshots/{GSM_REVISION}/main"
DEFAULT_MMLU = HF_HOME / "hub" / f"datasets--cais--mmlu/snapshots/{MMLU_REVISION}"


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
    permutation = np.random.default_rng(seed).permutation(len(train))
    probe_ids = permutation[:6473].astype(int).tolist()
    calibration_ids = permutation[6473:].astype(int).tolist()
    if len(probe_ids) != 6473 or len(calibration_ids) != 1000:
        raise AssertionError("invalid GSM8K split sizes")
    if set(probe_ids) & set(calibration_ids) or len(set(probe_ids + calibration_ids)) != 7473:
        raise AssertionError("GSM8K split is not a complete disjoint permutation")

    def convert(frame: pd.DataFrame, ids: list[int], source_split: str):
        for index in ids:
            row = frame.iloc[index]
            yield {
                "problem_id": f"gsm8k_{source_split}_{index:05d}",
                "source_index": index,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
            }

    files: dict[str, Any] = {}
    for name, frame, ids, source_split in (
        ("probe_train", train, probe_ids, "train"),
        ("calibration", train, calibration_ids, "train"),
        ("heldout", test, list(range(len(test))), "test"),
    ):
        path = output / f"{name}.jsonl"
        count, fingerprint = atomic_jsonl(convert(frame, ids, source_split), path)
        files[name] = {
            "path": str(path.relative_to(ROOT)),
            "count": count,
            "sha256": fingerprint,
        }
    split = {
        "schema_version": 1,
        "dataset": "openai/gsm8k",
        "config": "main",
        "hub_revision": GSM_REVISION,
        "seed": seed,
        "permutation_algorithm": "numpy.random.default_rng(seed).permutation(7473)",
        "probe_train_source_indices": probe_ids,
        "calibration_source_indices": calibration_ids,
        "heldout_source_indices": list(range(len(test))),
        "files": files,
        "invariants": {
            "official_train_used": 7473,
            "official_train_dropped": 0,
            "official_test_used": 1319,
            "official_test_used_for_fitting_or_thresholds": False,
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

    probe_rows = []
    calibration_rows = []
    for subject in MMLU_SUBJECTS:
        probe_rows.extend(routed_rows(subject, selected[subject][:100], "probe_train"))
        calibration_rows.extend(
            routed_rows(subject, selected[subject][100:120], "calibration")
        )
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
            "path": str(path.relative_to(ROOT)),
            "count": count,
            "sha256": fingerprint,
        }

    subject_manifest = {}
    for subject in MMLU_SUBJECTS:
        subject_manifest[subject] = {
            "category": mmlu_category(subject),
            "probe_train_source_indices": selected[subject][:100],
            "calibration_source_indices": selected[subject][100:120],
            "dev_source_indices": list(range(len(dev[subject]))),
            "validation_count_used_for_unlabeled_routing_centroid": len(validation[subject]),
            "test_source_indices": list(range(len(test[subject]))),
            "test_count": len(test[subject]),
        }
    split = {
        "schema_version": 1,
        "dataset": "cais/mmlu",
        "hub_revision": MMLU_REVISION,
        "seed": seed,
        "subjects": list(MMLU_SUBJECTS),
        "subject_manifest": subject_manifest,
        "files": files,
        "auxiliary_source": {
            "total_rows": len(auxiliary),
            "source_subject_values": source_subject_values,
            "subject_metadata_limitation": (
                "The official cached auxiliary_train subject column is blank for all rows."
            ),
            "routing_method": "TF-IDF cosine similarity to dev+validation subject centroids; deterministic unique round-robin quota allocation.",
            "routing_uses_test_labels": False,
            "routing_uses_test_text_for_subject_scoring": False,
            "test_text_used_only_for_required_exact normalized-question deduplication": True,
            "tfidf_max_features": max_features,
            "eligible_unique_rows": len(unique_aux_indices),
            "protected_overlap_rows_removed": overlap_removed,
            "within_auxiliary_duplicates_removed": duplicate_removed,
        },
        "invariants": {
            "probe_train_per_subject": 100,
            "calibration_per_subject": 20,
            "demonstrations_per_subject": 5,
            "official_test_total": 14042,
            "official_test_used_for_fitting_or_thresholds": False,
            "probe_calibration_overlap": False,
        },
    }
    split["fingerprint"] = canonical_fingerprint(split)
    atomic_json(split, split_path)
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm-snapshot", type=Path, default=DEFAULT_GSM)
    parser.add_argument("--mmlu-snapshot", type=Path, default=DEFAULT_MMLU)
    parser.add_argument("--data-output", type=Path, default=ROOT / "data/final_paper_v1")
    parser.add_argument(
        "--split-output", type=Path, default=ROOT / "results/final_paper_v1/splits"
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--tfidf-max-features", type=int, default=100000)
    args = parser.parse_args()

    gsm = prepare_gsm8k(
        args.gsm_snapshot,
        args.data_output / "gsm8k",
        args.split_output / "gsm8k_split.json",
        args.seed,
    )
    print(json.dumps({"phase": "gsm8k", "fingerprint": gsm["fingerprint"]}), flush=True)
    mmlu = prepare_mmlu(
        args.mmlu_snapshot,
        args.data_output / "mmlu",
        args.split_output / "mmlu_split.json",
        args.seed,
        args.tfidf_max_features,
    )
    print(json.dumps({"phase": "mmlu", "fingerprint": mmlu["fingerprint"]}), flush=True)
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

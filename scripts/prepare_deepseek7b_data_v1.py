#!/usr/bin/env python3
"""Prepare v2 problem-level splits with category-balanced MATH supervision."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd



def last_boxed(text: str) -> str | None:
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(starts):
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            return text[match.end() : index - 1].strip()
    unbraced = re.findall(r"\\boxed\s+([^\s$.,]+)", text)
    if unbraced:
        return unbraced[-1].strip()
    return None


def normalize_math(value: str | None) -> str | None:
    if value is None:
        return None
    return (
        value.strip()
        .replace("$", "")
        .replace("\\!", "")
        .replace("\\dfrac", "\\frac")
        .replace("\\tfrac", "\\frac")
        .replace("\\left", "")
        .replace("\\right", "")
        .replace("\\,", "")
        .replace(" ", "")
    )


MATH_CATEGORIES = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def stratified_sample(rows: list[dict], count: int, seed: int) -> list[dict]:
    if count > len(rows):
        raise ValueError(f"requested {count} from {len(rows)} rows")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(f"{row.get('category')}::{row.get('level')}", []).append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    result: list[dict] = []
    keys = sorted(groups)
    while len(result) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(result) < count:
                result.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    rng.shuffle(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--math-snapshot", type=Path, required=True)
    parser.add_argument("--math500-snapshot", type=Path, required=True)
    parser.add_argument("--legacy-math-split-root", type=Path, required=True)
    parser.add_argument("--aime-parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_root
    source_gsm = (
        args.project_root
        / "data/final_paper_replay_v4/legacy_empirical_protocol_train1000_cal500_mmlu1000/gsm8k"
    )
    counts: dict[str, dict[str, int]] = {"gsm8k": {}, "math": {}, "aime": {}}
    for split in ("probe_train", "calibration", "heldout"):
        rows = read_jsonl(source_gsm / f"{split}.jsonl")
        write_jsonl(output / "gsm8k" / f"{split}.jsonl", rows)
        counts["gsm8k"][split] = len(rows)

    math_by_split: dict[str, list[dict]] = {"train": [], "test": []}
    for category in MATH_CATEGORIES:
        for source_split in ("train", "test"):
            path = args.math_snapshot / category / f"{source_split}-00000-of-00001.parquet"
            frame = pd.read_parquet(path)
            for source_index, row in frame.iterrows():
                gold = normalize_math(last_boxed(str(row.solution)))
                if gold is None:
                    raise ValueError(f"missing boxed gold in {path}:{source_index}")
                math_by_split[source_split].append(
                    {
                        "dataset": "math",
                        "problem_id": f"math_{source_split}_{category}_{int(source_index):05d}",
                        "question": str(row.problem),
                        "gold_answer": gold,
                        "solution": str(row.solution),
                        "category": category,
                        "level": str(row.level),
                        "source_index": int(source_index),
                    }
                )
    # Preserve every member of the already-frozen v1 fit/calibration split,
    # then deterministically fill within each category.  This keeps all valid
    # cached artifacts while making the new protocol exactly 200/100 per class.
    legacy_train = read_jsonl(args.legacy_math_split_root / "probe_train.jsonl")
    legacy_calibration = read_jsonl(args.legacy_math_split_root / "calibration.jsonl")
    raw_by_id = {row["problem_id"]: row for row in math_by_split["train"]}
    if len(raw_by_id) != len(math_by_split["train"]):
        raise ValueError("duplicate Hendrycks MATH train problem IDs")
    legacy_train_ids = {row["problem_id"] for row in legacy_train}
    legacy_calibration_ids = {row["problem_id"] for row in legacy_calibration}
    if legacy_train_ids & legacy_calibration_ids:
        raise ValueError("legacy MATH fit/calibration overlap")
    if not (legacy_train_ids | legacy_calibration_ids) <= set(raw_by_id):
        raise ValueError("legacy MATH split contains IDs absent from source snapshot")

    def stable_local_seed(category: str, role: str) -> int:
        digest = hashlib.sha256(f"20260820:{category}:{role}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    probe_train: list[dict] = []
    calibration: list[dict] = []
    category_counts: dict[str, dict[str, int]] = {}
    for category in MATH_CATEGORIES:
        retained_train = [row for row in legacy_train if row["category"] == category]
        retained_calibration = [
            row for row in legacy_calibration if row["category"] == category
        ]
        if len(retained_train) > 200 or len(retained_calibration) > 100:
            raise ValueError(f"legacy category exceeds v2 quota: {category}")
        used = {
            row["problem_id"] for row in retained_train + retained_calibration
        }
        available = [
            row
            for row in math_by_split["train"]
            if row["category"] == category and row["problem_id"] not in used
        ]
        train_fill = stratified_sample(
            available,
            200 - len(retained_train),
            stable_local_seed(category, "probe_train"),
        )
        train_fill_ids = {row["problem_id"] for row in train_fill}
        available = [row for row in available if row["problem_id"] not in train_fill_ids]
        calibration_fill = stratified_sample(
            available,
            100 - len(retained_calibration),
            stable_local_seed(category, "calibration"),
        )
        local_train = retained_train + train_fill
        local_calibration = retained_calibration + calibration_fill
        if len(local_train) != 200 or len(local_calibration) != 100:
            raise AssertionError(category)
        probe_train.extend(local_train)
        calibration.extend(local_calibration)
        category_counts[category] = {
            "probe_train": len(local_train),
            "calibration": len(local_calibration),
            "retained_v1_probe_train": len(retained_train),
            "retained_v1_calibration": len(retained_calibration),
        }
    random.Random(20260820).shuffle(probe_train)
    random.Random(20260821).shuffle(calibration)
    if {row["problem_id"] for row in probe_train} & {
        row["problem_id"] for row in calibration
    }:
        raise ValueError("v2 MATH fit/calibration overlap")
    for split, rows in (("probe_train", probe_train), ("calibration", calibration)):
        write_jsonl(output / "math" / f"{split}.jsonl", rows)
        counts["math"][split] = len(rows)

    math500_path = args.math500_snapshot / "test.jsonl"
    math500_source = read_jsonl(math500_path)
    subject_map = {
        "Algebra": "algebra",
        "Counting & Probability": "counting_and_probability",
        "Geometry": "geometry",
        "Intermediate Algebra": "intermediate_algebra",
        "Number Theory": "number_theory",
        "Prealgebra": "prealgebra",
        "Precalculus": "precalculus",
    }
    math500_rows = []
    for source_index, row in enumerate(math500_source):
        subject = str(row["subject"])
        if subject not in subject_map:
            raise ValueError(f"unknown MATH-500 subject: {subject}")
        unique_id = str(row["unique_id"])
        safe_id = re.sub(r"[^A-Za-z0-9]+", "_", unique_id).strip("_")
        math500_rows.append(
            {
                "dataset": "math500",
                "problem_id": f"math500_{safe_id}",
                "question": str(row["problem"]),
                "gold_answer": normalize_math(str(row["answer"])),
                "solution": str(row["solution"]),
                "category": subject_map[subject],
                "level": str(row["level"]),
                "source_index": source_index,
                "source_unique_id": unique_id,
            }
        )
    if len(math500_rows) != 500 or len({row["problem_id"] for row in math500_rows}) != 500:
        raise ValueError("MATH-500 must contain exactly 500 unique problems")
    supervised_questions = {row["question"] for row in probe_train + calibration}
    if supervised_questions & {row["question"] for row in math500_rows}:
        raise ValueError("MATH-500 overlaps MATH supervision by exact problem text")
    write_jsonl(output / "math500" / "heldout.jsonl", math500_rows)
    counts["math500"] = {"heldout": len(math500_rows)}

    aime_frame = pd.read_parquet(args.aime_parquet)
    aime_rows = []
    for source_index, row in aime_frame.iterrows():
        aime_rows.append(
            {
                "dataset": "aime",
                "problem_id": f"aime2024_{int(row['id']):03d}",
                "question": str(row["problem"]),
                "gold_answer": normalize_math(str(row["answer"])),
                "solution": str(row.get("solution", "")),
                "category": "aime_2024",
                "level": "competition",
                "source_index": int(source_index),
                "source_url": str(row.get("url", "")),
            }
        )
    write_jsonl(output / "aime" / "heldout.jsonl", aime_rows)
    counts["aime"]["heldout"] = len(aime_rows)

    identities: dict[str, str] = {}
    for path in sorted(output.glob("*/*.jsonl")):
        identities[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "status": "complete",
        "split_unit": "problem",
        "seed": 20260820,
        "counts": counts,
        "math_protocol": {
            "unit": "problem",
            "categories": list(MATH_CATEGORIES),
            "per_category": {"probe_train": 200, "calibration": 100},
            "category_counts": category_counts,
            "fit_calibration_source": "Hendrycks MATH train only",
            "ood_tests": ["MATH-500", "AIME 2024"],
        },
        "aime_protocol": "MATH-trained and MATH-calibrated transfer; AIME 2024 OOD heldout only",
        "source": {
            "gsm8k": str(source_gsm.resolve()),
            "math": str(args.math_snapshot.resolve()),
            "math500": str(args.math500_snapshot.resolve()),
            "legacy_math_split_root": str(args.legacy_math_split_root.resolve()),
            "aime": "HuggingFaceH4/aime_2024@995c88376b099d66f99f79b8f87be99b3eb09864",
        },
        "sha256": identities,
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

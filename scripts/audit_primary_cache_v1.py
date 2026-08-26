#!/usr/bin/env python3
"""审计论文主范围的公共缓存、replay 视图、协议指纹和数据泄漏。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_paper_protocol import MMLU_SUBJECTS, normalize_question
from src.utils import atomic_json, load_yaml


def token_hash(tokens: list[int]) -> str:
    return hashlib.sha256(
        ",".join(map(str, tokens)).encode("utf-8")
    ).hexdigest()


def audit_artifact(
    path: Path,
    dataset: str,
    split: str,
    dtype: str,
    expected_protocol_id: str,
    expected_protocol_fingerprint: str,
) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    errors = []
    problem_id = str(value.get("problem_id"))
    if value.get("status") != "complete":
        errors.append("status")
    if value.get("dataset") != dataset or value.get("split") != split:
        errors.append("dataset_or_split")
    if value.get("dtype") != dtype:
        errors.append("dtype")
    if int(value.get("seed", value.get("global_seed", -1))) != 20260803:
        errors.append("global_seed")
    if value.get("attention_backend") != "sdpa":
        errors.append("attention_backend")
    if value.get("protocol_id") != expected_protocol_id:
        errors.append("cache_protocol_id")
    if value.get("protocol_fingerprint") != expected_protocol_fingerprint:
        errors.append("cache_protocol_fingerprint")
    if value.get("model_audit", {}).get("metadata_fingerprint") != (
        "1444257116723c44a60884afd78b095b48727cf4a7e17b69d85e55970070d863"
    ):
        errors.append("model_fingerprint")
    dense = value.get("dense", {})
    tokens = dense.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != int(dense.get("reasoning_tokens", -1)):
        errors.append("dense_token_length")
        tokens = []
    rows = value.get("rows", [])
    hidden = value.get("hidden")
    if not torch.is_tensor(hidden) or int(hidden.shape[0]) != len(rows):
        errors.append("row_vector_alignment")
    elif hidden.numel() and not torch.isfinite(hidden.float()).all():
        errors.append("hidden_nonfinite")
    checkpoints = [int(row["checkpoint"]) for row in rows]
    if len(checkpoints) != len(set(checkpoints)):
        errors.append("duplicate_checkpoint")
    sentence = sorted(
        int(row["checkpoint"])
        for row in rows
        if bool(row.get("is_sentence_checkpoint", True))
    )
    if any(value < 64 or value > 768 for value in sentence):
        errors.append("sentence_checkpoint_range")
    if any(right - left < 8 for left, right in zip(sentence[:-1], sentence[1:])):
        errors.append("sentence_checkpoint_gap")
    if "schedules" in value and sorted(map(int, value["schedules"].get("sentence", []))) != sentence:
        errors.append("sentence_schedule_mismatch")
    if any("prefix_mean_entropy_tail8" not in row for row in rows):
        errors.append("missing_entropy")
    if any("current_success" not in row or "branch_tokens" not in row for row in rows):
        errors.append("missing_forced_branch")
    return {
        "problem_id": problem_id,
        "question_normalized": normalize_question(value.get("record", {}).get("question", "")),
        "subject": value.get("record", {}).get("subject"),
        "category": value.get("record", {}).get("category"),
        "source_split": value.get("record", {}).get("source_split"),
        "protocol_fingerprint": value.get("protocol_fingerprint"),
        "token_hash": token_hash(tokens),
        "dense_tokens": len(tokens),
        "sentence_checkpoints": len(sentence),
        "no_legal_sentence_checkpoint": not sentence,
        "reached_4096": bool(dense.get("reached_max_tokens", False)),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/final_paper_primary_v1.yaml"))
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config if args.config.is_absolute() else ROOT / args.config)
    dtype = str(config["model"]["dtype"])
    if dtype not in {"float16", "bfloat16"}:
        raise RuntimeError("主 dtype 尚未冻结")
    all_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    checks: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    for dataset in ("gsm8k", "mmlu"):
        source_root = ROOT / config["datasets"][dataset]["selected_cache_root"] / "merged"
        expected_protocol_id = str(config["datasets"][dataset]["selected_cache_protocol_id"])
        expected_protocol_fingerprint = str(
            config["datasets"][dataset]["selected_cache_protocol_fingerprint"]
        )
        all_rows[dataset] = {}
        checks[dataset] = {}
        expected = {
            "probe_train": int(config["datasets"][dataset]["probe_train"]),
            "calibration": int(config["datasets"][dataset]["policy_calibration"]),
            "heldout": int(config["datasets"][dataset]["heldout"]),
        }
        for split, count in expected.items():
            paths = sorted((source_root / split).glob("sample_*.pt"))
            if len(paths) != count:
                errors.append({"dataset": dataset, "split": split, "error": f"count {len(paths)} != {count}"})
            rows = [
                audit_artifact(
                    path,
                    dataset,
                    split,
                    dtype,
                    expected_protocol_id,
                    expected_protocol_fingerprint,
                )
                for path in paths
            ]
            for row, path in zip(rows, paths):
                if row["errors"]:
                    errors.append({"dataset": dataset, "split": split, "path": str(path), "errors": row["errors"]})
            ids = [row["problem_id"] for row in rows]
            if len(ids) != len(set(ids)):
                errors.append({"dataset": dataset, "split": split, "error": "duplicate sample ID"})
            replay_mismatches = 0
            overhead_missing = 0
            if args.replay_root is not None:
                for row in rows:
                    replay_path = args.replay_root / dataset / split / f"sample_{row['problem_id']}.pt"
                    if not replay_path.is_file():
                        replay_mismatches += 1
                        continue
                    replay = torch.load(replay_path, map_location="cpu", weights_only=False)
                    if token_hash(replay["dense"]["tokens"]) != row["token_hash"]:
                        replay_mismatches += 1
                    if "includes_boundary" not in str(replay.get("policy_cost_mode", "")):
                        overhead_missing += 1
                    if any("sentence_replay_stop_wall_ms" not in item for item in replay["rows"]):
                        overhead_missing += 1
            subject_counts = Counter(str(row["subject"]) for row in rows if row["subject"] is not None)
            source_counts = Counter(str(row["source_split"]) for row in rows if row["source_split"] is not None)
            checks[dataset][split] = {
                "expected": count,
                "observed": len(rows),
                "protocol_fingerprints": sorted(set(str(row["protocol_fingerprint"]) for row in rows)),
                "expected_protocol_id": expected_protocol_id,
                "expected_protocol_fingerprint": expected_protocol_fingerprint,
                "subject_counts": dict(sorted(subject_counts.items())),
                "subjects_covered": len(subject_counts),
                "source_counts": dict(source_counts),
                "sentence_checkpoints": sum(row["sentence_checkpoints"] for row in rows),
                "no_legal_sentence_checkpoint": sum(row["no_legal_sentence_checkpoint"] for row in rows),
                "reached_4096": sum(row["reached_4096"] for row in rows),
                "replay_mismatches": replay_mismatches,
                "replay_overhead_missing": overhead_missing,
            }
            if replay_mismatches or overhead_missing:
                errors.append({"dataset": dataset, "split": split, "replay_mismatches": replay_mismatches, "overhead_missing": overhead_missing})
            all_rows[dataset][split] = rows
        split_id_sets = {
            split: {row["problem_id"] for row in rows}
            for split, rows in all_rows[dataset].items()
        }
        split_question_sets = {
            split: {row["question_normalized"] for row in rows if row["question_normalized"]}
            for split, rows in all_rows[dataset].items()
        }
        for left, right in (("probe_train", "calibration"), ("probe_train", "heldout"), ("calibration", "heldout")):
            if split_id_sets[left] & split_id_sets[right]:
                errors.append({"dataset": dataset, "error": f"ID overlap {left}/{right}"})
            if split_question_sets[left] & split_question_sets[right]:
                errors.append({"dataset": dataset, "error": f"normalized question overlap {left}/{right}"})
    mmlu_test_counts = checks["mmlu"]["heldout"]["subject_counts"]
    if set(mmlu_test_counts) != set(MMLU_SUBJECTS) or any(value not in (17, 18) for value in mmlu_test_counts.values()):
        errors.append({"dataset": "mmlu", "error": "MMLU-1k subject counts are not 17/18 over 57 subjects"})
    distribution_shift = (
        checks["mmlu"]["calibration"]["source_counts"] == {"auxiliary_train": 500}
        and checks["mmlu"]["heldout"]["source_counts"] == {"test": 1000}
    )
    if not distribution_shift:
        errors.append({"dataset": "mmlu", "error": "unexpected calibration/test source audit"})
    payload = {
        "status": "passed" if not errors else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config["protocol_id"],
        "dtype_override": dtype,
        "generation_reused": dtype == "float16",
        "model_generation_performed": dtype == "bfloat16",
        "checks": checks,
        "mmlu_distribution_shift": {
            "present": distribution_shift,
            "report_label": "MMLU-1k distribution-shift",
            "calibration_source": "auxiliary_train",
            "heldout_source": "official test",
        },
        "heldout_used_for_training_scaler_epoch_threshold_or_dtype": False,
        "checkpoint_overhead_included": args.replay_root is not None,
        "errors": errors,
    }
    atomic_json(payload, args.output)
    print(json.dumps({"status": payload["status"], "errors": len(errors)}, indent=2))
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()

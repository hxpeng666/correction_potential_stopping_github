#!/usr/bin/env python3
"""Fail-closed audit for DeepSeek paragraph cache artifacts."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from deepseek7b_protocol_v1 import canonical_fingerprint, success


def isolated_success(dataset: str, gold: str | None, predicted: str | None) -> bool:
    """Recheck a disagreement in a clean interpreter to avoid SymPy parser-state drift."""
    payload = json.dumps(
        {"dataset": dataset, "gold": gold, "predicted": predicted},
        ensure_ascii=False,
    )
    code = (
        "import json,sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "from deepseek7b_protocol_v1 import success; "
        "d=json.load(sys.stdin); "
        "print(json.dumps(bool(success(d['dataset'],d['gold'],d['predicted']))))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve().parent)],
        input=payload,
        text=True,
        capture_output=True,
        timeout=300,
        check=True,
    )
    return bool(json.loads(completed.stdout.strip().splitlines()[-1]))


def line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fingerprint = canonical_fingerprint(config)
    prepared = Path(config["data"]["prepared_root"])
    cache = Path(config["output_root"]) / "cache"
    summary = {"status": "complete", "protocol_fingerprint": fingerprint, "splits": {}}
    errors: list[str] = []
    extension = config.get("selective_dense_extension", {})
    extension_enabled = bool(extension.get("enabled", False))
    extension_manifest: dict = {}
    eligible: dict[tuple[str, str, str], dict] = {}
    extension_counts = {
        "eligible": 0,
        "generated_at_32768": 0,
        "reused_noncapped": 0,
        "prefix_identity_verified": 0,
    }
    if extension_enabled:
        manifest_path = Path(extension["manifest"])
        if not manifest_path.is_file():
            errors.append(f"missing selective extension manifest: {manifest_path}")
        else:
            extension_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            eligible = {
                (str(item["dataset"]), str(item["split"]), str(item["problem_id"])): item
                for item in extension_manifest.get("eligible", [])
            }
            extension_counts["eligible"] = len(eligible)
            if (
                extension_manifest.get("source_dense_max_new_tokens")
                != extension.get("source_dense_max_new_tokens")
                or extension_manifest.get("target_dense_max_new_tokens")
                != extension.get("target_dense_max_new_tokens")
                or extension_manifest.get("eligible_count") != len(eligible)
            ):
                errors.append("selective extension manifest/config mismatch")
    for dataset in ("gsm8k", "math", "math500", "aime"):
        for split in ("probe_train", "calibration", "heldout"):
            source = prepared / dataset / f"{split}.jsonl"
            if not source.is_file():
                continue
            source_rows = [
                json.loads(line) for line in source.open(encoding="utf-8") if line.strip()
            ]
            expected = len(source_rows)
            expected_records = {str(row["problem_id"]): row for row in source_rows}
            if len(expected_records) != expected:
                errors.append(f"{dataset}/{split}: duplicate problem IDs in prepared split")
            paths = sorted((cache / dataset / split).glob("sample_*.pt"))
            row = {
                "expected": expected,
                "artifacts": len(paths),
                "problems_with_checkpoints": 0,
                "dense_fallback_no_checkpoint": 0,
                "checkpoints": 0,
                "dense_correct": 0,
                "dense_reached_max": 0,
                "forced_answer_missing": 0,
                "forced_answer_at_48_limit": 0,
            }
            if len(paths) != expected:
                errors.append(f"{dataset}/{split}: expected {expected}, found {len(paths)}")
            for path in paths:
                try:
                    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                    if value.get("status") != "complete":
                        raise ValueError("status is not complete")
                    if value.get("protocol_fingerprint") != fingerprint:
                        raise ValueError("protocol fingerprint mismatch")
                    if value.get("actual_checkpoint_schedule") != "paragraph":
                        raise ValueError("schedule is not paragraph")
                    problem_id = str(value.get("problem_id"))
                    identity = (dataset, split, problem_id)
                    if value.get("dataset") != dataset or value.get("split") != split:
                        raise ValueError("dataset/split mismatch")
                    expected_record = expected_records.get(problem_id)
                    if expected_record is None:
                        raise ValueError("problem absent from prepared split")
                    actual_record = value.get("record", {})
                    for field in ("problem_id", "question"):
                        if str(actual_record.get(field)) != str(expected_record.get(field)):
                            raise ValueError(f"record {field} mismatch")
                    expected_gold = (
                        str(expected_record["gold_answer"])
                        if "gold_answer" in expected_record
                        else str(expected_record["answer"])
                        .rsplit("####", 1)[-1]
                        .strip()
                        .replace(",", "")
                    )
                    if str(value.get("gold_answer")) != expected_gold:
                        raise ValueError("gold answer mismatch")
                    if extension_enabled:
                        dense_generation = value.get("dense_generation", {})
                        target_budget = int(extension["target_dense_max_new_tokens"])
                        source_budget = int(extension["source_dense_max_new_tokens"])
                        if int(dense_generation.get("requested_max_new_tokens", -1)) != target_budget:
                            raise ValueError("target Dense budget metadata mismatch")
                        if identity in eligible:
                            if dense_generation.get("execution_mode") != "generated_at_configured_budget":
                                raise ValueError("eligible capped artifact was not regenerated at target budget")
                            source_artifact = Path(eligible[identity]["source_artifact"])
                            if not source_artifact.is_file():
                                raise FileNotFoundError(source_artifact)
                            source_value = torch.load(
                                source_artifact,
                                map_location="cpu",
                                weights_only=False,
                                mmap=True,
                            )
                            source_tokens = list(source_value["dense"]["tokens"])
                            target_tokens = list(value["dense"]["tokens"])
                            if (
                                not bool(source_value["dense"]["reached_max_tokens"])
                                or len(source_tokens) != source_budget
                                or target_tokens[:source_budget] != source_tokens
                            ):
                                raise ValueError("32K extension does not preserve the exact 13K token prefix")
                            extension_counts["generated_at_32768"] += 1
                            extension_counts["prefix_identity_verified"] += 1
                        else:
                            if dense_generation.get("execution_mode") != "reused_noncapped_source_trajectory":
                                raise ValueError("non-eligible artifact was regenerated instead of reused")
                            if (
                                int(dense_generation.get("source_requested_max_new_tokens", -1))
                                != source_budget
                                or bool(value["dense"]["reached_max_tokens"])
                                or len(value["dense"]["tokens"]) >= source_budget
                            ):
                                raise ValueError("invalid non-capped Dense reuse metadata")
                            extension_counts["reused_noncapped"] += 1
                    decoding = value.get("forced_answer_decoding", {})
                    if (
                        decoding.get("strategy") != "greedy_argmax"
                        or decoding.get("do_sample") is not False
                        or int(decoding.get("max_new_tokens", -1))
                        != int(config["generation"]["force_answer_max_new_tokens"])
                    ):
                        raise ValueError("forced-answer decoding mismatch")
                    hidden = value["hidden"]
                    rows = value["rows"]
                    if hidden.shape != (len(rows), 1, 3584):
                        raise ValueError(f"hidden shape {tuple(hidden.shape)}")
                    row["problems_with_checkpoints"] += int(bool(rows))
                    row["dense_fallback_no_checkpoint"] += int(not rows)
                    row["checkpoints"] += len(rows)
                    row["dense_correct"] += int(value["dense"]["success"])
                    expected_dense_success = success(
                        dataset, expected_gold, value["dense"].get("prediction")
                    )
                    stored_dense_success = bool(value["dense"].get("success"))
                    if stored_dense_success != expected_dense_success:
                        expected_dense_success = isolated_success(
                            dataset, expected_gold, value["dense"].get("prediction")
                        )
                    if stored_dense_success != expected_dense_success:
                        raise ValueError("Dense success does not match stored prediction/gold")
                    row["dense_reached_max"] += int(value["dense"]["reached_max_tokens"])
                    for checkpoint in rows:
                        expected_current_success = success(
                            dataset, expected_gold, checkpoint.get("current_prediction")
                        )
                        stored_current_success = bool(checkpoint.get("current_success"))
                        if stored_current_success != expected_current_success:
                            expected_current_success = isolated_success(
                                dataset,
                                expected_gold,
                                checkpoint.get("current_prediction"),
                            )
                        derived = {
                            "dense_success": expected_dense_success,
                            "current_success": expected_current_success,
                            "correction": (not expected_current_success)
                            and expected_dense_success,
                            "damage": expected_current_success
                            and (not expected_dense_success),
                        }
                        for field, expected_value in derived.items():
                            if bool(checkpoint.get(field)) != bool(expected_value):
                                raise ValueError(
                                    f"checkpoint {checkpoint.get('checkpoint')} invalid {field}"
                                )
                        row["forced_answer_missing"] += int(checkpoint["current_prediction"] is None)
                        row["forced_answer_at_48_limit"] += int(checkpoint["branch_tokens"] >= 48)
                except Exception as error:
                    errors.append(f"{path}: {type(error).__name__}: {error}")
            row["dense_accuracy"] = row["dense_correct"] / len(paths) if paths else None
            row["mean_checkpoints"] = row["checkpoints"] / len(paths) if paths else None
            summary["splits"][f"{dataset}/{split}"] = row
    if extension_enabled:
        expected_total = sum(item["artifacts"] for item in summary["splits"].values())
        extension_passed = (
            extension_counts["generated_at_32768"] == extension_counts["eligible"]
            and extension_counts["prefix_identity_verified"] == extension_counts["eligible"]
            and extension_counts["reused_noncapped"]
            == expected_total - extension_counts["eligible"]
        )
        summary["selective_dense_extension"] = {
            **extension_counts,
            "source_dense_max_new_tokens": int(extension["source_dense_max_new_tokens"]),
            "target_dense_max_new_tokens": int(extension["target_dense_max_new_tokens"]),
            "passed": extension_passed,
        }
        if not extension_passed:
            errors.append("selective Dense extension count/prefix audit failed")
    if errors:
        summary["status"] = "failed"
        summary["errors"] = errors[:100]
    target = Path(config["output_root"]) / "COLLECTION_AUDIT.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

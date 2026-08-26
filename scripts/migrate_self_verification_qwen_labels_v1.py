#!/usr/bin/env python3
"""Recover Qwen judge labels from superseded Self-verification caches.

Only artifacts whose answer-chunk merge structure is unchanged are migrated.  Cases
where a Qwen null correctness changes nearest-answer merging are deliberately left
for full recollection.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import collect_literature_method_data_v1 as common
import collect_self_verification_v1 as self_verification

ROOT = common.ROOT

import torch

from src.final_paper_inference import atomic_torch_save
from src.utils import load_yaml


def destination(root: Path, dataset: str, schedule: str, split: str, name: str) -> Path:
    return root / dataset / "self_verification" / schedule / "cache" / split / name


def migrate_pair(
    native_path: Path,
    paragraph_path: Path,
    result_root: Path,
    dataset: str,
    config: dict,
    fingerprint: str,
) -> str:
    native = torch.load(native_path, map_location="cpu", weights_only=False)
    paragraph = torch.load(paragraph_path, map_location="cpu", weights_only=False)
    split = str(native["split"])
    problem_id = str(native["problem_id"])
    outputs = {
        "native": destination(result_root, dataset, "native", split, native_path.name),
        "paragraph": destination(result_root, dataset, "paragraph", split, native_path.name),
    }
    if all(self_verification.valid(path, fingerprint, schedule, problem_id) for schedule, path in outputs.items()):
        return "already_complete"
    if any(path.exists() for path in outputs.values()):
        raise RuntimeError(f"partial or incompatible destination for {problem_id}: {outputs}")

    judgments = []
    for row in native["chunk_judgments"]:
        if "labeler_reported_correctness" not in row:
            return "missing_original_qwen_label"
        updated = dict(row)
        updated["correctness"] = row["labeler_reported_correctness"]
        judgments.append(updated)
    judgments = self_verification.attach_frozen_evaluation_correctness(
        judgments, dataset, {"gold_answer": native["gold_answer"], "record": native["record"]}
    )
    merged = self_verification.merge_answerless(native["preliminary_chunks"], judgments)
    old_endpoints = [int(row["checkpoint"]) for row in native["rows"]]
    new_endpoints = [int(row["end"]) for row in merged]
    if old_endpoints != new_endpoints:
        return "merge_structure_changed"

    migrated = {}
    for schedule, source in (("native", native), ("paragraph", paragraph)):
        artifact = dict(source)
        rows = []
        for row in source["rows"]:
            updated = dict(row)
            chunk = merged[int(row["merged_chunk_index"])]
            updated["probe_label"] = bool(chunk["probe_label"])
            if schedule == "native":
                updated["evaluation_success"] = bool(chunk["evaluation_success"])
                updated["current_success"] = bool(chunk["evaluation_success"])
            else:
                updated["upcoming_evaluation_success"] = bool(chunk["evaluation_success"])
            rows.append(updated)
        artifact["rows"] = rows
        artifact["protocol_fingerprint"] = fingerprint
        artifact["chunk_judgments"] = judgments
        artifact["merged_chunks"] = merged
        artifact["method_config"] = config["methods"]["self_verification"]
        artifact["labeler_audit"] = {
            **artifact["labeler_audit"],
            "supervision_label_source": "restored_original_qwen3_4b_judge_output",
        }
        artifact["collection"] = {
            **artifact.get("collection", {}),
            "migration": "restore_labeler_reported_correctness_without_model_reexecution",
            "migration_host": socket.gethostname(),
            "migration_at": datetime.now(timezone.utc).isoformat(),
            "superseded_source": str(source.get("source_artifact", native_path)),
        }
        migrated[schedule] = artifact
    for schedule, artifact in migrated.items():
        path = outputs[schedule]
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(artifact, path)
        if not self_verification.valid(path, fingerprint, schedule, problem_id):
            raise RuntimeError(f"post-write validation failed: {path}")
    return "migrated"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    source_root = Path(args.source_root)
    if not source_root.is_absolute():
        source_root = ROOT / source_root
    config = load_yaml(config_path)
    result_root = ROOT / config["output_root"]
    fingerprint = common.canonical_fingerprint(config, args.dataset, "self_verification")
    counts: dict[str, int] = {}
    native_root = source_root / "native" / "cache"
    for native_path in sorted(native_root.glob("*/*.pt")):
        paragraph_path = source_root / "paragraph" / "cache" / native_path.parent.name / native_path.name
        if not paragraph_path.is_file():
            raise RuntimeError(f"missing paragraph pair: {paragraph_path}")
        status = migrate_pair(native_path, paragraph_path, result_root, args.dataset, config, fingerprint)
        counts[status] = counts.get(status, 0) + 1
    report = {
        "status": "complete",
        "dataset": args.dataset,
        "fingerprint": fingerprint,
        "source_root": str(source_root),
        "counts": counts,
    }
    report_path = result_root / f"SELF_VERIFICATION_QWEN_LABEL_MIGRATION_{args.dataset}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

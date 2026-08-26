#!/usr/bin/env python3
"""Migrate token-identical non-capped fixed-frontier artifacts to a clean 13K protocol."""
from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))

from collect_deepseek7b_fixed_budget_frontier_v1 import (  # noqa: E402
    dense_token_fingerprint,
    fixed_fingerprint,
)
from deepseek7b_protocol_v1 import (  # noqa: E402
    atomic_torch_save,
    canonical_fingerprint as source_fingerprint,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def valid_destination(
    path: Path,
    *,
    fingerprint: str,
    source_protocol_fingerprint: str,
    problem_id: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        return (
            value.get("status") == "complete"
            and value.get("protocol_fingerprint") == fingerprint
            and value.get("source_protocol_fingerprint") == source_protocol_fingerprint
            and str(value.get("problem_id")) == problem_id
            and value.get("collection", {}).get("execution_mode")
            == "verified_metadata_migration_from_token_identical_noncapped_v3_source"
        )
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--old-fixed-root", type=Path, required=True)
    parser.add_argument("--v3-source-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    source_config = load_yaml(Path(config["source"]["config"]))
    expected_source_fingerprint = source_fingerprint(source_config)
    v2_root = Path(config["source"]["output_root"]) / config["source"]["cache_subdirectory"]
    output_root = Path(config["output_root"])
    v3_root = args.v3_source_root

    migrated = skipped = 0
    counts: dict[str, int] = {}
    missing: list[dict[str, Any]] = []
    checks = {
        "v2_v3_dense_tokens_exact": 0,
        "v2_v3_dense_prediction_exact": 0,
        "v2_v3_dense_success_exact": 0,
        "v2_old_fixed_token_fingerprint_exact": 0,
        "rows_preserved": 0,
    }

    for dataset, dataset_config in config["datasets"].items():
        expected = int(dataset_config["heldout"])
        fixed_fingerprint_value = fixed_fingerprint(config, dataset)
        dataset_migrated = 0
        old_paths = sorted((args.old_fixed_root / dataset / "heldout").glob("sample_*.pt"))
        old_by_id = {path.name.removeprefix("sample_").removesuffix(".pt"): path for path in old_paths}

        for v2_path in sorted((v2_root / dataset / "heldout").glob("sample_*.pt")):
            problem_id = v2_path.name.removeprefix("sample_").removesuffix(".pt")
            old_path = old_by_id.get(problem_id)
            if old_path is None:
                v2_value = torch.load(v2_path, map_location="cpu", weights_only=False, mmap=True)
                missing.append(
                    {
                        "dataset": dataset,
                        "problem_id": problem_id,
                        "source_artifact": str(v2_path.resolve()),
                        "source_dense_tokens": len(v2_value["dense"]["tokens"]),
                        "source_reached_max_tokens": bool(v2_value["dense"]["reached_max_tokens"]),
                    }
                )
                continue

            destination = output_root / dataset / "heldout" / v2_path.name
            if args.resume and valid_destination(
                destination,
                fingerprint=fixed_fingerprint_value,
                source_protocol_fingerprint=expected_source_fingerprint,
                problem_id=problem_id,
            ):
                skipped += 1
                dataset_migrated += 1
                continue
            if destination.exists():
                raise RuntimeError(f"refusing to overwrite incompatible destination: {destination}")

            old = torch.load(old_path, map_location="cpu", weights_only=False, mmap=True)
            v2 = torch.load(v2_path, map_location="cpu", weights_only=False, mmap=True)
            v3_path = v3_root / dataset / "heldout" / v2_path.name
            if not v3_path.is_file():
                raise RuntimeError(f"missing v3 comparison source: {v3_path}")
            v3 = torch.load(v3_path, map_location="cpu", weights_only=False, mmap=True)

            for name, value in (("v2", v2), ("v3", v3), ("old_fixed", old)):
                if str(value.get("problem_id")) != problem_id:
                    raise RuntimeError(f"{dataset}/{problem_id}: {name} problem ID mismatch")
            if v2.get("protocol_fingerprint") != expected_source_fingerprint:
                raise RuntimeError(f"{dataset}/{problem_id}: v2 source fingerprint mismatch")

            v2_tokens = [int(token) for token in v2["dense"]["tokens"]]
            v3_tokens = [int(token) for token in v3["dense"]["tokens"]]
            if v2_tokens != v3_tokens:
                raise RuntimeError(f"{dataset}/{problem_id}: v2/v3 Dense token mismatch")
            checks["v2_v3_dense_tokens_exact"] += 1
            if v2["dense"]["prediction"] != v3["dense"]["prediction"]:
                raise RuntimeError(f"{dataset}/{problem_id}: v2/v3 Dense prediction mismatch")
            checks["v2_v3_dense_prediction_exact"] += 1
            if bool(v2["dense"]["success"]) != bool(v3["dense"]["success"]):
                raise RuntimeError(f"{dataset}/{problem_id}: v2/v3 Dense success mismatch")
            checks["v2_v3_dense_success_exact"] += 1
            if dense_token_fingerprint(v2_tokens) != old["source_dense_token_fingerprint"]:
                raise RuntimeError(f"{dataset}/{problem_id}: old fixed source-token mismatch")
            checks["v2_old_fixed_token_fingerprint_exact"] += 1
            if len(old.get("rows", [])) != len(config["budget"]["retained_fractions"]):
                raise RuntimeError(f"{dataset}/{problem_id}: old fixed row count mismatch")
            checks["rows_preserved"] += len(old["rows"])

            migrated_value = copy.deepcopy(old)
            migrated_value["protocol_id"] = config["protocol_id"]
            migrated_value["protocol_fingerprint"] = fixed_fingerprint_value
            migrated_value["source_artifact"] = str(v2_path.resolve())
            migrated_value["source_protocol_id"] = v2["protocol_id"]
            migrated_value["source_protocol_fingerprint"] = v2["protocol_fingerprint"]
            migrated_value["source_dense_token_fingerprint"] = dense_token_fingerprint(v2_tokens)
            migrated_value["dense"] = {
                "tokens": len(v2_tokens),
                "prediction": v2["dense"]["prediction"],
                "success": bool(v2["dense"]["success"]),
                "reached_max_tokens": bool(v2["dense"]["reached_max_tokens"]),
                "requested_max_new_tokens": int(
                    v2.get("dense_generation", {}).get("requested_max_new_tokens", 13000)
                ),
            }
            migrated_value["collection"] = {
                **migrated_value.get("collection", {}),
                "execution_mode": "verified_metadata_migration_from_token_identical_noncapped_v3_source",
                "migrated_at": datetime.now(timezone.utc).isoformat(),
                "old_fixed_artifact": str(old_path.resolve()),
                "v3_comparison_source": str(v3_path.resolve()),
                "v2_v3_dense_tokens_exact": True,
                "v2_v3_dense_prediction_exact": True,
                "v2_v3_dense_success_exact": True,
                "fixed_rows_reused_without_change": True,
            }
            atomic_torch_save(migrated_value, destination)
            migrated += 1
            dataset_migrated += 1

        if dataset_migrated != len(old_paths):
            raise RuntimeError(
                f"{dataset}: expected to migrate/skip {len(old_paths)}, got {dataset_migrated}"
            )
        if len(old_paths) + sum(item["dataset"] == dataset for item in missing) != expected:
            raise RuntimeError(f"{dataset}: old+missing coverage does not equal {expected}")
        counts[dataset] = dataset_migrated

    if len(missing) != 42:
        raise RuntimeError(f"expected exactly 42 missing fixed artifacts, found {len(missing)}")
    if not all(item["source_dense_tokens"] == 13000 for item in missing):
        raise RuntimeError("all 42 missing sources must be capped at exactly 13000 tokens")
    if not all(item["source_reached_max_tokens"] for item in missing):
        raise RuntimeError("all 42 missing sources must have reached the 13K cap")

    audit = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config["protocol_id"],
        "source_protocol_id": source_config["protocol_id"],
        "source_protocol_fingerprint": expected_source_fingerprint,
        "migrated": migrated,
        "skipped_existing_valid": skipped,
        "verified_migrated_total": sum(counts.values()),
        "counts_by_dataset": counts,
        "remaining_direct_13k_collection_count": len(missing),
        "remaining_by_dataset": {
            dataset: sum(item["dataset"] == dataset for item in missing)
            for dataset in config["datasets"]
        },
        "checks": checks,
        "remaining": missing,
    }
    atomic_json(audit, output_root / "MIGRATION_AUDIT.json")
    atomic_json(
        {
            "status": "frozen",
            "dense_reference": "v2_13k_complete_or_capped_trajectory",
            "all_heldout_samples_use_13k_protocol": True,
            "migrated_token_identical_noncapped": sum(counts.values()),
            "directly_collected_capped": len(missing),
            "capped_source_token_length": 13000,
            "note": "The 42 capped samples are evaluated relative to their frozen 13K trajectories, independently of the ongoing 32K extension experiment.",
        },
        output_root / "SOURCE_POLICY.json",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

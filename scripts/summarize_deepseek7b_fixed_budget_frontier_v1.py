#!/usr/bin/env python3
"""Audit and summarize the DeepSeek fixed relative-budget frontier."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from deepseek7b_protocol_v1 import canonical_fingerprint as source_fingerprint  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def fixed_fingerprint(config: dict[str, Any], dataset: str) -> str:
    payload = {
        "protocol_id": config["protocol_id"],
        "model": config["model"],
        "source": config["source"],
        "generation": config["generation"],
        "budget": config["budget"],
        "dataset": dataset,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def dense_token_fingerprint(tokens: list[int]) -> str:
    return hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def interval(values: list[float]) -> tuple[float, float]:
    low, high = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    source_config = load_yaml(Path(config["source"]["config"]))
    expected_source_fingerprint = source_fingerprint(source_config)
    output_root = Path(config["output_root"])
    fractions = [float(value) for value in config["budget"]["retained_fractions"]]
    replicates = int(config["reporting"]["bootstrap_replicates"])
    result_rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_protocol_id": source_config["protocol_id"],
        "source_protocol_fingerprint": expected_source_fingerprint,
        "datasets": {},
    }

    for dataset_index, (dataset, dataset_config) in enumerate(config["datasets"].items()):
        expected = int(dataset_config["heldout"])
        paths = sorted((output_root / dataset / "heldout").glob("sample_*.pt"))
        if len(paths) != expected:
            raise RuntimeError(f"{dataset}: expected {expected} artifacts, found {len(paths)}")
        artifacts = [
            torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            for path in paths
        ]
        ids = [str(value["problem_id"]) for value in artifacts]
        if len(set(ids)) != expected:
            raise RuntimeError(f"{dataset}: duplicate problem IDs")
        expected_fingerprint = fixed_fingerprint(config, dataset)
        source_checks = 0
        for value in artifacts:
            if (
                value.get("status") != "complete"
                or value.get("dataset") != dataset
                or value.get("split") != "heldout"
                or value.get("protocol_fingerprint") != expected_fingerprint
                or value.get("source_protocol_fingerprint") != expected_source_fingerprint
            ):
                raise RuntimeError(f"{dataset}: invalid artifact {value.get('problem_id')}")
            observed = [float(row["retained_fraction"]) for row in value["rows"]]
            if observed != fractions:
                raise RuntimeError(f"{dataset}: budget grid mismatch for {value['problem_id']}")
            source_path = Path(value["source_artifact"])
            if not source_path.is_file():
                raise RuntimeError(f"{dataset}: missing source artifact {source_path}")
            source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
            source_tokens = [int(token) for token in source["dense"]["tokens"]]
            if (
                str(source["problem_id"]) != str(value["problem_id"])
                or source["protocol_fingerprint"] != expected_source_fingerprint
                or dense_token_fingerprint(source_tokens)
                != value["source_dense_token_fingerprint"]
                or len(source_tokens) != int(value["dense"]["tokens"])
                or bool(source["dense"]["success"]) != bool(value["dense"]["success"])
            ):
                raise RuntimeError(f"{dataset}: source mismatch for {value['problem_id']}")
            source_checks += 1

        dense_correct = np.asarray([bool(value["dense"]["success"]) for value in artifacts])
        dense_tokens = np.asarray(
            [int(value["dense"]["tokens"]) for value in artifacts], dtype=np.float64
        )
        rng = np.random.default_rng(int(config["seed"]) + dataset_index)
        bootstrap = [rng.integers(0, expected, size=expected) for _ in range(replicates)]
        for row_index, fraction in enumerate(fractions):
            rows = [value["rows"][row_index] for value in artifacts]
            current_correct = np.asarray([bool(row["current_success"]) for row in rows])
            total_tokens = np.asarray(
                [int(row["stop_total_tokens"]) for row in rows], dtype=np.float64
            )
            lost = np.asarray([bool(row["lost_correct"]) for row in rows])
            helped = np.asarray([bool(row["helped"]) for row in rows])
            truncated = np.asarray(
                [bool(row.get("forced_answer_truncated", False)) for row in rows]
            )
            after_close = np.asarray(
                [bool(row.get("checkpoint_after_think_close", False)) for row in rows]
            )
            delta_samples = []
            reduction_samples = []
            for indices in bootstrap:
                delta_samples.append(
                    100.0
                    * float(current_correct[indices].mean() - dense_correct[indices].mean())
                )
                reduction_samples.append(
                    1.0 - float(total_tokens[indices].mean() / dense_tokens[indices].mean())
                )
            delta_ci = interval(delta_samples)
            reduction_ci = interval(reduction_samples)
            result_rows.append(
                {
                    "dataset": dataset,
                    "retained_fraction": fraction,
                    "target_reasoning_saving_fraction": 1.0 - fraction,
                    "n": expected,
                    "dense_accuracy": float(dense_correct.mean()),
                    "accuracy": float(current_correct.mean()),
                    "accuracy_delta_pp": 100.0
                    * float(current_correct.mean() - dense_correct.mean()),
                    "accuracy_delta_pp_ci95_low": delta_ci[0],
                    "accuracy_delta_pp_ci95_high": delta_ci[1],
                    "mean_dense_tokens": float(dense_tokens.mean()),
                    "mean_total_tokens": float(total_tokens.mean()),
                    "total_token_reduction": 1.0
                    - float(total_tokens.mean() / dense_tokens.mean()),
                    "total_token_reduction_ci95_low": reduction_ci[0],
                    "total_token_reduction_ci95_high": reduction_ci[1],
                    "lost_correct": int(lost.sum()),
                    "helped": int(helped.sum()),
                    "forced_answer_truncated": int(truncated.sum()),
                    "checkpoints_after_think_close": int(after_close.sum()),
                    "dense_fallbacks": int(
                        sum(bool(row["dense_fallback"]) for row in rows)
                    ),
                }
            )
        audit["datasets"][dataset] = {
            "artifacts": expected,
            "unique_problem_ids": len(set(ids)),
            "source_artifacts_reverified": source_checks,
            "protocol_fingerprint": expected_fingerprint,
            "budget_rows_per_artifact": len(fractions),
            "split": "heldout",
            "selection_uses_heldout": False,
        }

    fields = list(result_rows[0])
    csv_path = output_root / "RESULTS.csv"
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)
    os.replace(temporary_csv, csv_path)
    atomic_json(result_rows, output_root / "RESULTS.json")
    atomic_json(audit, output_root / "AUDIT.json")

    lines = [
        "# DeepSeek-R1-Distill-Qwen-7B fixed relative-budget frontier",
        "",
        "Each retained fraction is applied independently to every problem's complete frozen Dense response. Total-token cost excludes prompt tokens and includes the forced-answer suffix plus the generated short answer.",
        "",
        "| Dataset | Retained Dense response | Target reasoning saving | Accuracy | Delta accuracy (pp) | Actual total-token reduction | Lost | Helped | Truncated forced answers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            f"| {row['dataset']} | {100 * row['retained_fraction']:.0f}% | "
            f"{100 * row['target_reasoning_saving_fraction']:.0f}% | "
            f"{100 * row['accuracy']:.2f}% | {row['accuracy_delta_pp']:+.2f} | "
            f"{100 * row['total_token_reduction']:.2f}% | {row['lost_correct']} | "
            f"{row['helped']} | {row['forced_answer_truncated']} |"
        )
    (output_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(
        {
            "status": "complete",
            "protocol_id": config["protocol_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "result_rows": len(result_rows),
            "expected_result_rows": len(config["datasets"]) * len(fractions),
            "artifacts": ["RESULTS.csv", "RESULTS.json", "RESULTS.md", "AUDIT.json"],
        },
        output_root / "EXPERIMENT_COMPLETE.json",
    )
    print(json.dumps({"status": "complete", "rows": len(result_rows)}, indent=2))


if __name__ == "__main__":
    main()

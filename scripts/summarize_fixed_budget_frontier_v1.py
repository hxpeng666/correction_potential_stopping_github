#!/usr/bin/env python3
"""Audit and summarize the fixed relative-budget held-out frontier."""
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

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "src").is_dir():
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.utils import load_yaml


def canonical_fingerprint(config: dict[str, Any], dataset: str) -> str:
    payload = {
        "protocol_id": config["protocol_id"],
        "model": config["model"],
        "generation": config["generation"],
        "budget": config["budget"],
        "dataset": dataset,
        "source_root": config["datasets"][dataset]["source_root"],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def bootstrap_indices(artifacts: list[dict[str, Any]], dataset: str, rng, replicates: int):
    n = len(artifacts)
    if dataset != "mmlu_pro":
        for _ in range(replicates):
            yield rng.integers(0, n, size=n)
        return
    categories: dict[str, list[int]] = {}
    for index, artifact in enumerate(artifacts):
        category = str(artifact.get("record", {}).get("category", "unknown"))
        categories.setdefault(category, []).append(index)
    groups = [np.asarray(indices, dtype=np.int64) for _, indices in sorted(categories.items())]
    for _ in range(replicates):
        yield np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])


def interval(values: list[float]) -> list[float]:
    return [float(x) for x in np.percentile(np.asarray(values), [2.5, 97.5])]


def aes(accuracy: float, tokens: float, dense_accuracy: float, dense_tokens: float) -> float:
    saving = (dense_tokens - tokens) / dense_tokens
    if accuracy >= dense_accuracy:
        return saving + 3.0 * (accuracy - dense_accuracy) / dense_accuracy
    return saving - 5.0 * (dense_accuracy - accuracy) / dense_accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    output_root = ROOT / config["output_root"]
    fractions = [float(x) for x in config["budget"]["retained_fractions"]]
    replicates = int(config["reporting"]["bootstrap_replicates"])
    result_rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
    }

    for dataset, dataset_config in config["datasets"].items():
        paths = sorted((output_root / dataset / "heldout").glob("sample_*.pt"))
        expected = int(dataset_config["heldout"])
        if len(paths) != expected:
            raise RuntimeError(f"{dataset}: expected {expected} artifacts, found {len(paths)}")
        artifacts = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
        ids = [str(value["problem_id"]) for value in artifacts]
        if len(set(ids)) != expected:
            raise RuntimeError(f"{dataset}: duplicate problem IDs")
        fingerprints = sorted(set(str(value["protocol_fingerprint"]) for value in artifacts))
        expected_fingerprint = canonical_fingerprint(config, dataset)
        if fingerprints != [expected_fingerprint]:
            raise RuntimeError(f"{dataset}: mixed protocol fingerprints: {fingerprints}")
        for value in artifacts:
            if value.get("status") != "complete" or value.get("split") != "heldout":
                raise RuntimeError(f"{dataset}: invalid artifact {value.get('problem_id')}")
            observed = [float(row["retained_fraction"]) for row in value["rows"]]
            if observed != fractions:
                raise RuntimeError(f"{dataset}: budget grid mismatch for {value['problem_id']}")

        dense_correct = np.asarray([bool(value["dense"]["success"]) for value in artifacts])
        dense_tokens = np.asarray([int(value["dense"]["reasoning_tokens"]) for value in artifacts], dtype=np.float64)
        rng = np.random.default_rng(int(config["seed"]) + (0 if dataset == "gsm8k" else 1))
        bootstrap = list(bootstrap_indices(artifacts, dataset, rng, replicates))
        for row_index, fraction in enumerate(fractions):
            rows = [value["rows"][row_index] for value in artifacts]
            current_correct = np.asarray([bool(row["current_success"]) for row in rows])
            total_tokens = np.asarray([int(row["stop_total_tokens"]) for row in rows], dtype=np.float64)
            lost = np.asarray([bool(row["lost_correct"]) for row in rows])
            helped = np.asarray([bool(row["helped"]) for row in rows])
            truncation = np.asarray([bool(row.get("forced_answer_truncated", False)) for row in rows])
            after_close = np.asarray([bool(row.get("checkpoint_after_think_close", False)) for row in rows])
            delta_samples: list[float] = []
            reduction_samples: list[float] = []
            for indices in bootstrap:
                delta_samples.append(100.0 * float((current_correct[indices].mean() - dense_correct[indices].mean())))
                reduction_samples.append(1.0 - float(total_tokens[indices].mean() / dense_tokens[indices].mean()))
            result_rows.append({
                "dataset": dataset,
                "retained_fraction": fraction,
                "target_reasoning_saving_fraction": 1.0 - fraction,
                "n": expected,
                "dense_accuracy": float(dense_correct.mean()),
                "accuracy": float(current_correct.mean()),
                "accuracy_delta_pp": 100.0 * float(current_correct.mean() - dense_correct.mean()),
                "accuracy_delta_pp_ci95_low": interval(delta_samples)[0],
                "accuracy_delta_pp_ci95_high": interval(delta_samples)[1],
                "mean_dense_tokens": float(dense_tokens.mean()),
                "mean_total_tokens": float(total_tokens.mean()),
                "total_token_reduction": 1.0 - float(total_tokens.mean() / dense_tokens.mean()),
                "total_token_reduction_ci95_low": interval(reduction_samples)[0],
                "total_token_reduction_ci95_high": interval(reduction_samples)[1],
                "lost_correct": int(lost.sum()),
                "helped": int(helped.sum()),
                "forced_answer_truncated": int(truncation.sum()),
                "checkpoints_after_think_close": int(after_close.sum()),
                "dense_fallbacks": int(sum(bool(row["dense_fallback"]) for row in rows)),
                "aes": aes(
                    float(current_correct.mean()), float(total_tokens.mean()),
                    float(dense_correct.mean()), float(dense_tokens.mean()),
                ),
            })
        audit["datasets"][dataset] = {
            "artifacts": expected,
            "unique_problem_ids": len(set(ids)),
            "protocol_fingerprint": fingerprints[0],
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
        "# Fixed relative-budget frontier",
        "",
        "All rows are frozen held-out results. The retained fraction is applied independently to each problem's complete Dense generated response. Total-token cost includes the forced-answer suffix and generated answer.",
        "",
        "| Dataset | Retained Dense response | Target Dense response saving | Acc | Delta Acc (pp) | Actual total-token reduction | Lost | Helped | AES |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            f"| {row['dataset']} | {100*row['retained_fraction']:.0f}% | "
            f"{100*row['target_reasoning_saving_fraction']:.0f}% | {100*row['accuracy']:.2f}% | "
            f"{row['accuracy_delta_pp']:+.2f} | {100*row['total_token_reduction']:.2f}% | "
            f"{row['lost_correct']} | {row['helped']} | {row['aes']:.4f} |"
        )
    (output_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json({
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_rows": len(result_rows),
        "artifacts": ["RESULTS.csv", "RESULTS.json", "RESULTS.md", "AUDIT.json"],
    }, output_root / "EXPERIMENT_COMPLETE.json")
    print(json.dumps({"status": "complete", "rows": len(result_rows)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create an isolated per-sample queue from the frozen selected common cache."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from greedy_forced_common_v1 import (
    artifact_valid,
    atomic_json,
    ensure_task,
    load_config,
    output_path,
    protocol_fingerprint,
    queue_counts,
    recover_stale_claims,
    resolve,
    source_split_path,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_paper_greedy_forced_v1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    fingerprint = protocol_fingerprint(config)
    recovered = recover_stale_claims(config)
    created = existing = 0
    observed: dict[str, dict[str, int]] = {}
    seen: set[tuple[str, str, str]] = set()
    for dataset, dataset_config in config["datasets"].items():
        observed[dataset] = {}
        for split, expected in dataset_config["expected_counts"].items():
            sources = sorted(source_split_path(dataset_config, split).glob("sample_*.pt"))
            observed[dataset][split] = len(sources)
            if len(sources) != int(expected):
                raise RuntimeError(
                    f"source count mismatch {dataset}/{split}: {len(sources)} != {expected}"
                )
            for source in sources:
                problem_id = source.stem.removeprefix("sample_")
                key = (dataset, split, problem_id)
                if key in seen:
                    raise RuntimeError(f"duplicate source sample: {key}")
                seen.add(key)
                destination = output_path(config, dataset, split, problem_id)
                if artifact_valid(destination, fingerprint, problem_id):
                    existing += 1
                    continue
                payload = {
                    "dataset": dataset,
                    "split": split,
                    "problem_id": problem_id,
                    "source_path": str(source.resolve()),
                    "destination": str(destination),
                    "source_protocol_fingerprint": dataset_config[
                        "source_protocol_fingerprint"
                    ],
                    "protocol_fingerprint": fingerprint,
                }
                created += int(ensure_task(config, payload))
    manifest = {
        "status": "prepared",
        "protocol_id": config["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_samples": len(seen),
        "observed_counts": observed,
        "tasks_created_now": created,
        "artifacts_already_complete": existing,
        "stale_claims_recovered": recovered,
        "queue": queue_counts(config),
    }
    atomic_json(manifest, resolve(config["output_root"]) / "PREPARE_MANIFEST.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

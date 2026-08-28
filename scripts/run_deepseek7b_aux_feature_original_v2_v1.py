#!/usr/bin/env python3
"""Screen pooling/readout variants on the uncensored original-v2 protocol."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def specs() -> list[dict[str, Any]]:
    values = []
    for representation in ("last4_mean", "paragraph_mean", "prefix_mean"):
        values.append(
            {
                "label": f"pool_{representation}",
                "representation": representation,
                "feature": "hidden_scalars",
                "readout": "full",
                "architecture": "standard",
                "pca": None,
            }
        )
        # Preserve the complete historical feature-construction axis: each
        # pooled representation was also screened after PCA, both without and
        # with the one-step readout.  These 3 x 4 x 2 = 24 variants are part of
        # the frozen 149-candidate/domain protocol and must not be dropped.
        for dimension in (32, 64, 128, 256):
            values.append(
                {
                    "label": f"pool_{representation}_pca{dimension}_compact",
                    "representation": representation,
                    "feature": "pca_hidden_scalars",
                    "readout": "full",
                    "architecture": "compact",
                    "pca": dimension,
                }
            )
            values.append(
                {
                    "label": (
                        f"pool_{representation}_pca{dimension}_"
                        "one_step_full_compact"
                    ),
                    "representation": representation,
                    "feature": "pca_hidden_scalars_one_step",
                    "readout": "full",
                    "architecture": "compact",
                    "pca": dimension,
                }
            )
    for readout in ("distribution", "stability", "full"):
        for architecture in ("linear", "compact"):
            values.append(
                {
                    "label": f"one_step_{readout}_{architecture}",
                    "representation": "last",
                    "feature": "one_step_only",
                    "readout": readout,
                    "architecture": architecture,
                    "pca": None,
                }
            )
        values.append(
            {
                "label": f"scalars_one_step_{readout}_compact",
                "representation": "last",
                "feature": "scalars_one_step",
                "readout": readout,
                "architecture": "compact",
                "pca": None,
            }
        )
        values.append(
            {
                "label": f"hidden_one_step_{readout}_standard",
                "representation": "last",
                "feature": "hidden_scalars_one_step",
                "readout": readout,
                "architecture": "standard",
                "pca": None,
            }
        )
    for representation in ("last4_mean", "paragraph_mean", "prefix_mean"):
        values.append(
            {
                "label": f"pool_{representation}_one_step_full",
                "representation": representation,
                "feature": "hidden_scalars_one_step",
                "readout": "full",
                "architecture": "standard",
                "pca": None,
            }
        )
    for dimension in (64, 128, 256):
        for architecture in ("linear", "compact"):
            values.append(
                {
                    "label": f"pca{dimension}_one_step_full_{architecture}",
                    "representation": "last",
                    "feature": "pca_hidden_scalars_one_step",
                    "readout": "full",
                    "architecture": architecture,
                    "pca": dimension,
                }
            )
    labels = [value["label"] for value in values]
    if len(labels) != len(set(labels)):
        raise AssertionError("duplicate auxiliary feature labels")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--aux-root", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    if args.cpu_threads != 1:
        raise ValueError("strict reproducibility freezes --cpu-threads 1")
    if args.parallel != 1:
        raise ValueError(
            "strict reproducibility runs one training process per GPU; "
            "use independent committed runners to distribute across GPUs"
        )
    sys.path.insert(0, str(PROJECT))
    from src.reproducibility import (
        code_provenance,
        deterministic_subprocess_environment,
    )

    environment = deterministic_subprocess_environment(seed=0)
    code_identity = code_provenance(
        PROJECT,
        (
            "scripts/run_deepseek7b_aux_feature_original_v2_v1.py",
            "scripts/train_deepseek7b_method_exploration_v1.py",
            "src/deepseek7b_method_exploration_v1.py",
            "src/reproducibility.py",
        ),
    )
    output_root = args.output_root.resolve()
    source_dataset = "gsm8k" if args.dataset == "gsm8k" else "math"
    raw_root = args.source_root.resolve() / source_dataset
    auxiliary_root = args.aux_root.resolve() / source_dataset
    audit_path = auxiliary_root / "probe_train" / "AUDIT.json"
    manifest_path = output_root / f"AUX_FEATURE_SCREEN_{args.dataset.upper()}_MANIFEST.json"
    state = {
        "status": "waiting_for_auxiliary_train",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scientific_protocol": "uncensored_original_v2",
        "right_censoring": False,
        "reference_point_loss": "legacy_weighted",
        "dataset": args.dataset,
        "gpu": args.gpu,
        "auxiliary_audit": str(audit_path),
        "tasks": len(specs()),
        "completed": [],
        "failed": [],
        "code_identity": code_identity,
        "source_root": str(args.source_root.resolve()),
        "aux_root": str(args.aux_root.resolve()),
    }
    atomic_json(state, manifest_path)
    while True:
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text())
            if audit.get("status") == "complete":
                break
        time.sleep(args.poll_seconds)
    state["status"] = "running"
    state["auxiliary_audit_payload"] = audit
    atomic_json(state, manifest_path)

    def run(spec: dict[str, Any]):
        output = output_root / "screen" / args.dataset / "aux_feature" / spec["label"]
        command = [
            str(PYTHON),
            "scripts/train_deepseek7b_method_exploration_v1.py",
            "--dataset",
            args.dataset,
            "--config",
            str(args.config),
            "--raw-root",
            str(raw_root),
            "--aux-raw-root",
            str(auxiliary_root),
            "--output",
            str(output),
            "--gpu",
            str(args.gpu),
            "--layer",
            "16",
            "--representation-kind",
            spec["representation"],
            "--feature-kind",
            spec["feature"],
            "--readout-kind",
            spec["readout"],
            "--probe-architecture",
            spec["architecture"],
            "--point-loss",
            "legacy_weighted",
            "--trajectory-scope",
            "all_dangerous",
            "--trajectory-aggregation",
            "normalized_softmin",
            "--beta",
            "0.5",
            "--rho",
            "1.0",
            "--lambda-protect",
            "1.0",
            "--lambda-separation",
            "0.0",
            "--gamma",
            "0.5",
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--cpu-threads",
            str(args.cpu_threads),
            "--screen-only",
            "--resume",
        ]
        if spec["pca"] is not None:
            command += ["--pca-dim", str(spec["pca"])]
        log = output_root / "logs" / f"aux_feature_{args.dataset}_{spec['label']}.log"
        with log.open("a", encoding="utf-8") as handle:
            handle.write("COMMAND " + json.dumps(command) + "\n")
            handle.flush()
            result = subprocess.run(
                command,
                cwd=PROJECT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return {
            "label": spec["label"],
            "output": str(output),
            "log": str(log),
            "returncode": result.returncode,
        }

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run, spec) for spec in specs()]
        for future in as_completed(futures):
            result = future.result()
            state["completed" if result["returncode"] == 0 else "failed"].append(result)
            atomic_json(state, manifest_path)
    state["status"] = "complete" if not state["failed"] else "failed"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(state, manifest_path)
    if state["failed"]:
        raise SystemExit(f"{len(state['failed'])} auxiliary feature screens failed")


if __name__ == "__main__":
    main()

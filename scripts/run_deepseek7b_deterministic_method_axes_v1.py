#!/usr/bin/env python3
"""Rerun the three DeepSeek method-axis studies from one clean commit."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reproducibility import code_provenance, deterministic_subprocess_environment


ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str], log: Path, environment: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def smoke_command(
    *, config: Path, data_root: Path, output: Path, gpu: int
) -> list[str]:
    return [
        sys.executable,
        "scripts/train_deepseek7b_method_exploration_v1.py",
        "--dataset", "gsm8k",
        "--config", str(config),
        "--raw-root", str(data_root / "gsm8k"),
        "--output", str(output),
        "--gpu", str(gpu),
        "--layer", "16",
        "--representation-kind", "last",
        "--feature-kind", "hidden_scalars",
        "--readout-kind", "full",
        "--probe-architecture", "standard",
        "--point-loss", "legacy_weighted",
        "--trajectory-scope", "all_dangerous",
        "--trajectory-aggregation", "normalized_softmin",
        "--beta", "0.5",
        "--rho", "1.0",
        "--lambda-protect", "1.0",
        "--lambda-separation", "0.0",
        "--gamma", "0.5",
        "--epochs", "24",
        "--patience", "6",
        "--cpu-threads", "1",
        "--screen-only",
        "--resume",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--aux-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    config = args.config.resolve()
    data_root = args.data_root.resolve()
    aux_root = args.aux_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    environment = deterministic_subprocess_environment(seed=0)
    identity = code_provenance(
        ROOT,
        (
            "scripts/run_deepseek7b_deterministic_method_axes_v1.py",
            "scripts/run_deepseek7b_method_axes_original_v2_v1.py",
            "scripts/run_deepseek7b_aux_feature_original_v2_v1.py",
            "scripts/train_deepseek7b_method_exploration_v1.py",
            "scripts/score_deepseek7b_axis_external_original_v2_v1.py",
            "scripts/calibrate_deepseek7b_axis_ltt_original_v2_v1.py",
            "scripts/recalibrate_deepseek7b_method_exploration_ltt_v1.py",
            "scripts/audit_deterministic_probe_pair_v1.py",
            "src/deepseek7b_method_exploration_v1.py",
            "src/reproducibility.py",
        ),
    )
    state = {
        "status": "determinism_gate",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "code_identity": identity,
        "config": str(config),
        "data_root": str(data_root),
        "aux_root": str(aux_root),
        "output_root": str(output),
        "gpu": args.gpu,
        "execution": "single process per GPU; strict deterministic algorithms",
        "completed": [],
    }
    atomic_json(state, output / "RUN_MANIFEST.json")

    gate = output / "determinism_gate"
    for name in ("run1", "run2"):
        run(
            smoke_command(
                config=config,
                data_root=data_root,
                output=gate / name,
                gpu=args.gpu,
            ),
            logs / f"determinism_{name}.log",
            environment,
        )
    run(
        [
            sys.executable,
            "scripts/audit_deterministic_probe_pair_v1.py",
            "--left", str(gate / "run1"),
            "--right", str(gate / "run2"),
            "--output", str(gate / "AUDIT.json"),
        ],
        logs / "determinism_audit.log",
        environment,
    )
    state["completed"].append("determinism_gate")
    state["status"] = "training_101_core_axes"
    atomic_json(state, output / "RUN_MANIFEST.json")

    run(
        [
            sys.executable,
            "scripts/run_deepseek7b_method_axes_original_v2_v1.py",
            "--datasets", "gsm8k", "math",
            "--axes", "weight", "robust", "reach", "feature",
            "--config", str(config),
            "--output-root", str(output),
            "--source-root", str(data_root),
            "--gpu", str(args.gpu),
            "--parallel", "1",
            "--cpu-threads", "1",
        ],
        logs / "core_axes.log",
        environment,
    )
    state["completed"].append("101_core_axes_per_dataset")
    state["status"] = "training_48_aux_axes"
    atomic_json(state, output / "RUN_MANIFEST.json")
    for dataset in ("gsm8k", "math"):
        run(
            [
                sys.executable,
                "scripts/run_deepseek7b_aux_feature_original_v2_v1.py",
                "--dataset", dataset,
                "--config", str(config),
                "--output-root", str(output),
                "--source-root", str(data_root),
                "--aux-root", str(aux_root),
                "--gpu", str(args.gpu),
                "--parallel", "1",
                "--cpu-threads", "1",
            ],
            logs / f"aux_axes_{dataset}.log",
            environment,
        )
        state["completed"].append(f"48_aux_axes/{dataset}")
        atomic_json(state, output / "RUN_MANIFEST.json")

    state["status"] = "external_scoring"
    atomic_json(state, output / "RUN_MANIFEST.json")
    external = output / "external_scores"
    for dataset in ("gsm8k", "math"):
        run(
            [
                sys.executable,
                "scripts/score_deepseek7b_axis_external_original_v2_v1.py",
                "--dataset", dataset,
                "--source", str(output),
                "--output", str(external),
                "--data-root", str(data_root),
                "--aux-root", str(aux_root),
                "--gpu", str(args.gpu),
            ],
            logs / f"external_scores_{dataset}.log",
            environment,
        )
        state["completed"].append(f"external_scores/{dataset}")
        atomic_json(state, output / "RUN_MANIFEST.json")

    state["status"] = "trajectory_envelope_ltt"
    atomic_json(state, output / "RUN_MANIFEST.json")
    run(
        [
            sys.executable,
            "scripts/calibrate_deepseek7b_axis_ltt_original_v2_v1.py",
            "--source", str(output),
            "--external-scores", str(external),
            "--output", str(output / "ltt"),
            "--data-root", str(data_root),
            "--alphas", *[str(alpha) for alpha in ALPHAS],
            "--main-alpha", "0.03",
            "--delta", "0.05",
            "--grid-size", "101",
        ],
        logs / "trajectory_envelope_ltt.log",
        environment,
    )
    state["completed"].append("trajectory_envelope_ltt")
    state["status"] = "complete"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(state, output / "RUN_MANIFEST.json")
    atomic_json(
        {
            "status": "complete",
            "completed_at": state["completed_at"],
            "git_commit": identity["git"]["commit"],
            "models_per_dataset": 149,
            "experiments": [
                "feature_construction",
                "bce_weight",
                "first_hit_trajectory",
            ],
            "alphas": list(ALPHAS),
        },
        output / "EXPERIMENT_COMPLETE.json",
    )


if __name__ == "__main__":
    main()

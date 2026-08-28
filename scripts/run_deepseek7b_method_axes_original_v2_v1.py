#!/usr/bin/env python3
"""Launch controlled method axes on the frozen, uncensored DeepSeek-7B v2 cache.

The common reference is deliberately the historical method used by the main
five-ablation table: h + six scalar features, legacy weighted point BCE, and
normalized trajectory soft-min.  Each non-weight axis changes only its named
factor.  The weight axis still enumerates all four predeclared point losses.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
TRAIN = "scripts/train_deepseek7b_method_exploration_v1.py"


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def base_spec() -> dict[str, Any]:
    return {
        "representation_kind": "last",
        "feature_kind": "hidden_scalars",
        "probe_architecture": "standard",
        "point_loss": "legacy_weighted",
        "stratified_problem_batches": False,
        "trajectory_scope": "all_dangerous",
        "trajectory_aggregation": "normalized_softmin",
        "beta": 0.5,
        "rho": 1.0,
        "lambda_protect": 1.0,
        "lambda_separation": 0.0,
        "gamma": 0.5,
        "pca_dim": None,
    }


def with_updates(axis: str, label: str, **updates: Any) -> dict[str, Any]:
    value = base_spec()
    value.update(updates)
    value["axis"] = axis
    value["label"] = label
    return value


def weight_specs() -> list[dict[str, Any]]:
    return [
        with_updates(
            "weight",
            "legacy_weighted",
            point_loss="legacy_weighted",
            stratified_problem_batches=False,
        ),
        with_updates(
            "weight",
            "checkpoint_proper",
            point_loss="checkpoint_proper",
            stratified_problem_batches=False,
        ),
        with_updates(
            "weight",
            "problem_balanced_random",
            point_loss="problem_balanced_proper",
            stratified_problem_batches=False,
        ),
        with_updates(
            "weight",
            "problem_balanced_stratified",
            point_loss="problem_balanced_proper",
            stratified_problem_batches=True,
        ),
    ]


def robust_specs() -> list[dict[str, Any]]:
    values = [
        with_updates(
            "robust",
            "no_trajectory",
            trajectory_aggregation="none",
            lambda_protect=0.0,
        )
    ]
    for aggregation, short in (
        ("hard_min", "hardmin"),
        ("normalized_softmin", "softmin"),
    ):
        for weight in (0.25, 0.5, 1.0):
            values.append(
                with_updates(
                    "robust",
                    f"{short}_lp{weight:g}",
                    trajectory_aggregation=aggregation,
                    lambda_protect=weight,
                )
            )
    for aggregation, short in (
        ("bottomk_mean", "bottomk"),
        ("lower_tail_cvar", "cvar"),
    ):
        for rho in (0.25, 0.5, 1.0):
            for weight in (0.25, 0.5, 1.0):
                values.append(
                    with_updates(
                        "robust",
                        f"{short}_rho{rho:g}_lp{weight:g}",
                        trajectory_aggregation=aggregation,
                        rho=rho,
                        lambda_protect=weight,
                    )
                )
    return values


def reach_specs() -> list[dict[str, Any]]:
    values = []
    for aggregation, short, rhos in (
        ("normalized_softmin", "softmin", (1.0,)),
        ("lower_tail_cvar", "cvar", (0.25, 0.5, 1.0)),
    ):
        for rho in rhos:
            for protect in (0.25, 0.5, 1.0):
                for separation in (0.0, 0.25, 0.5):
                    gammas = (0.5,) if separation == 0 else (0.5, 1.0)
                    for gamma in gammas:
                        values.append(
                            with_updates(
                                "reach",
                                (
                                    f"{short}_rho{rho:g}_lp{protect:g}_"
                                    f"ls{separation:g}_g{gamma:g}"
                                ),
                                trajectory_scope="reachability_earliest_safe",
                                trajectory_aggregation=aggregation,
                                rho=rho,
                                lambda_protect=protect,
                                lambda_separation=separation,
                                gamma=gamma,
                            )
                        )
    return values


def feature_specs() -> list[dict[str, Any]]:
    values = [
        with_updates("feature", "hidden_only_standard", feature_kind="hidden_only"),
        with_updates("feature", "hidden_scalars_standard", feature_kind="hidden_scalars"),
        with_updates(
            "feature",
            "scalars_only_linear",
            feature_kind="scalars_only",
            probe_architecture="linear",
        ),
        with_updates(
            "feature",
            "scalars_only_compact",
            feature_kind="scalars_only",
            probe_architecture="compact",
        ),
    ]
    for dimension in (32, 64, 128, 256):
        for architecture in ("linear", "compact"):
            values.append(
                with_updates(
                    "feature",
                    f"pca{dimension}_{architecture}",
                    feature_kind="pca_hidden_scalars",
                    pca_dim=dimension,
                    probe_architecture=architecture,
                )
            )
    return values


SPEC_BUILDERS = {
    "weight": weight_specs,
    "robust": robust_specs,
    "reach": reach_specs,
    "feature": feature_specs,
}


def command_for(
    spec: dict[str, Any],
    dataset: str,
    config: Path,
    output_root: Path,
    source_root: Path,
    gpu: int,
    epochs: int,
    patience: int,
    cpu_threads: int,
) -> tuple[list[str], Path]:
    source_dataset = "gsm8k" if dataset == "gsm8k" else "math"
    raw_root = source_root / source_dataset
    output = output_root / "screen" / dataset / spec["axis"] / spec["label"]
    command = [
        str(PYTHON),
        TRAIN,
        "--dataset",
        dataset,
        "--config",
        str(config),
        "--raw-root",
        str(raw_root),
        "--output",
        str(output),
        "--gpu",
        str(gpu),
        "--layer",
        "16",
        "--representation-kind",
        str(spec["representation_kind"]),
        "--feature-kind",
        str(spec["feature_kind"]),
        "--probe-architecture",
        str(spec["probe_architecture"]),
        "--point-loss",
        str(spec["point_loss"]),
        "--trajectory-scope",
        str(spec["trajectory_scope"]),
        "--trajectory-aggregation",
        str(spec["trajectory_aggregation"]),
        "--beta",
        str(spec["beta"]),
        "--rho",
        str(spec["rho"]),
        "--lambda-protect",
        str(spec["lambda_protect"]),
        "--lambda-separation",
        str(spec["lambda_separation"]),
        "--gamma",
        str(spec["gamma"]),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--cpu-threads",
        str(cpu_threads),
        "--screen-only",
        "--resume",
    ]
    if spec["stratified_problem_batches"]:
        command.append("--stratified-problem-batches")
    if spec["pca_dim"] is not None:
        command += ["--pca-dim", str(spec["pca_dim"])]
    return command, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--axes", nargs="+", choices=tuple(SPEC_BUILDERS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--gpu", type=int, default=-1,
        help="probe device; -1 uses CPU so frozen-LLM feature collection can run concurrently",
    )
    parser.add_argument(
        "--gpus",
        help="optional comma-separated probe GPUs assigned round-robin",
    )
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument(
        "--per-gpu-parallel",
        type=int,
        default=1,
        help="maximum independent deterministic probe processes sharing one GPU",
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.cpu_threads != 1:
        raise ValueError("strict reproducibility freezes --cpu-threads 1")
    if not 1 <= args.per_gpu_parallel <= 8:
        raise ValueError("--per-gpu-parallel must be in [1, 8]")
    if args.per_gpu_parallel > args.parallel:
        raise ValueError("--per-gpu-parallel cannot exceed --parallel")
    sys.path.insert(0, str(PROJECT))
    from src.reproducibility import (
        code_provenance,
        deterministic_subprocess_environment,
    )

    environment = deterministic_subprocess_environment(seed=0)
    code_identity = code_provenance(
        PROJECT,
        (
            "scripts/run_deepseek7b_method_axes_original_v2_v1.py",
            "scripts/train_deepseek7b_method_exploration_v1.py",
            "src/deepseek7b_method_exploration_v1.py",
            "src/reproducibility.py",
        ),
    )
    specs = []
    for axis in args.axes:
        specs.extend(SPEC_BUILDERS[axis]())
    probe_gpus = (
        [int(value) for value in args.gpus.split(",")]
        if args.gpus is not None
        else [args.gpu]
    )
    if not probe_gpus:
        raise ValueError("empty probe GPU list")
    tasks = []
    # Interleave datasets so GSM8K and MATH evidence advances concurrently.
    for spec in specs:
        for dataset in args.datasets:
            assigned_gpu = probe_gpus[len(tasks) % len(probe_gpus)]
            command, output = command_for(
                spec,
                dataset,
                args.config,
                args.output_root,
                args.source_root.resolve(),
                assigned_gpu,
                args.epochs,
                args.patience,
                args.cpu_threads,
            )
            tasks.append((dataset, spec, command, output))
    manifest_path = args.output_root / "SCREEN_RUN_MANIFEST.json"
    lock = threading.Lock()
    state = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scientific_protocol": "uncensored_original_v2",
        "reference_method": {
            "feature": "hidden_scalars",
            "point_loss": "legacy_weighted",
            "trajectory": "normalized_softmin_beta0.5_lambda1",
        },
        "right_censoring": False,
        "datasets": args.datasets,
        "axes": args.axes,
        "parallel": args.parallel,
        "per_gpu_parallel": args.per_gpu_parallel,
        "probe_gpus": probe_gpus,
        "epochs": args.epochs,
        "patience": args.patience,
        "tasks": len(tasks),
        "completed": [],
        "failed": [],
        "code_identity": code_identity,
        "source_root": str(args.source_root.resolve()),
    }
    atomic_json(state, manifest_path)

    # Every child is an isolated process with a fixed RNG seed and deterministic
    # CUDA algorithms.  A bounded semaphore lets several such processes share a
    # mostly-idle GPU without changing the numerical protocol.  The formal
    # parent runner certifies serial-vs-concurrent bitwise parity before using a
    # value greater than one.
    gpu_slots = {
        gpu: threading.BoundedSemaphore(args.per_gpu_parallel)
        for gpu in probe_gpus
    }

    def run_task(item):
        dataset, spec, command, output = item
        log = args.output_root / "logs" / f"screen_{dataset}_{spec['axis']}_{spec['label']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        assigned_gpu = int(command[command.index("--gpu") + 1])
        with gpu_slots[assigned_gpu]:
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
        return dataset, spec, output, log, result.returncode

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_task, item): item for item in tasks}
        for future in as_completed(futures):
            dataset, spec, output, log, returncode = future.result()
            record = {
                "dataset": dataset,
                "axis": spec["axis"],
                "label": spec["label"],
                "output": str(output),
                "log": str(log),
                "returncode": returncode,
            }
            with lock:
                state["completed" if returncode == 0 else "failed"].append(record)
                atomic_json(state, manifest_path)
    state["status"] = "complete" if not state["failed"] else "failed"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(state, manifest_path)
    if state["failed"]:
        raise SystemExit(f"{len(state['failed'])} screening tasks failed")


if __name__ == "__main__":
    main()

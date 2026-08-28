#!/usr/bin/env python3
"""Run paired original/forced grader LR sweeps on the two locked A100s."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from summarize_deepseek7b_probe_lr_sweep_v1 import (
    DATASETS,
    GRADERS,
    LEARNING_RATES,
    METHODS,
    lr_tag,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_free_mib() -> dict[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        index, free = line.split(",")
        result[int(index.strip())] = int(free.strip())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--forced-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=3500)
    parser.add_argument("--poll-seconds", type=int, default=5)
    args = parser.parse_args()

    graders = {
        "original_13k_parser": args.original_root,
        "forced_answer_at_cap": args.forced_root,
    }
    if tuple(graders) != GRADERS:
        raise RuntimeError("grader enumeration drift")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=args.repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise RuntimeError("formal LR sweep requires a clean git worktree")

    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "logs"
    log_root.mkdir(exist_ok=True)
    manifest_path = args.output_root / "RUN_MANIFEST.json"
    rounds = [
        (dataset, method, learning_rate)
        for method in ("bce_trajectory", "bce")
        for dataset in DATASETS
        for learning_rate in LEARNING_RATES
    ]
    manifest = {
        "status": "running",
        "started_at": utc_now(),
        "commit": commit,
        "formal_hardware_class": "NVIDIA A100 80GB PCIe",
        "gpu_policy": "one probe per A100; grader assignment alternates by round",
        "minimum_free_mib": args.minimum_free_mib,
        "selection_rule": "validation_objective",
        "calibration_used": False,
        "heldout_used": False,
        "empirical_B_used": False,
        "rounds": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    for round_index, (dataset, method, learning_rate) in enumerate(rounds):
        outputs = {
            grader: args.output_root
            / grader
            / dataset
            / method
            / f"lr_{lr_tag(learning_rate)}"
            / "seed_0"
            for grader in GRADERS
        }
        if all((output / "phase.complete").is_file() for output in outputs.values()):
            manifest["rounds"].append(
                {
                    "dataset": dataset,
                    "method": method,
                    "learning_rate": learning_rate,
                    "status": "skipped_complete",
                }
            )
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            continue

        while True:
            free = gpu_free_mib()
            if free.get(0, 0) >= args.minimum_free_mib and free.get(
                1, 0
            ) >= args.minimum_free_mib:
                break
            time.sleep(args.poll_seconds)

        assignment = (
            {"original_13k_parser": 0, "forced_answer_at_cap": 1}
            if round_index % 2 == 0
            else {"original_13k_parser": 1, "forced_answer_at_cap": 0}
        )
        processes: dict[str, tuple[subprocess.Popen[bytes], object, Path]] = {}
        started = utc_now()
        for grader in GRADERS:
            output = outputs[grader]
            output.parent.mkdir(parents=True, exist_ok=True)
            log_path = log_root / (
                f"{grader}_{dataset}_{method}_lr_{lr_tag(learning_rate)}_seed0.log"
            )
            point_args = (
                [
                    "--trajectory-aggregation",
                    "normalized_softmin",
                    "--lambda-protect",
                    "1.0",
                    "--beta",
                    "0.5",
                ]
                if method == "bce_trajectory"
                else [
                    "--trajectory-aggregation",
                    "none",
                    "--lambda-protect",
                    "0.0",
                ]
            )
            command = [
                str(args.python),
                "scripts/train_deepseek7b_method_exploration_v1.py",
                "--dataset",
                dataset,
                "--config",
                str(args.config),
                "--raw-root",
                str(graders[grader] / dataset),
                "--output",
                str(output),
                "--gpu",
                str(assignment[grader]),
                "--layer",
                "16",
                "--schedule",
                "sentence",
                "--representation-kind",
                "last",
                "--feature-kind",
                "hidden_scalars",
                "--probe-architecture",
                "standard",
                "--point-loss",
                "checkpoint_proper",
                "--trajectory-scope",
                "all_dangerous",
                "--rho",
                "1.0",
                "--lambda-separation",
                "0.0",
                "--gamma",
                "0.5",
                "--epochs",
                "48",
                "--patience",
                "48",
                "--screen-only",
                "--selection-rule",
                "validation_objective",
                "--learning-rate",
                str(learning_rate),
                "--weight-decay",
                "0.001",
                "--seed",
                "0",
                "--split-seed",
                "0",
                "--resume",
                *point_args,
            ]
            handle = log_path.open("ab")
            process = subprocess.Popen(
                command,
                cwd=args.repo,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes[grader] = (process, handle, log_path)

        failures = []
        for grader, (process, handle, log_path) in processes.items():
            returncode = process.wait()
            handle.close()
            if returncode != 0:
                failures.append(
                    {"grader": grader, "returncode": returncode, "log": str(log_path)}
                )
        record = {
            "dataset": dataset,
            "method": method,
            "learning_rate": learning_rate,
            "started_at": started,
            "ended_at": utc_now(),
            "gpu_assignment": assignment,
            "status": "failed" if failures else "complete",
            "failures": failures,
        }
        manifest["rounds"].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if failures:
            manifest["status"] = "failed"
            manifest["ended_at"] = utc_now()
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            raise SystemExit(json.dumps(failures))

    subprocess.run(
        [
            str(args.python),
            "scripts/summarize_deepseek7b_probe_lr_sweep_v1.py",
            "--output-root",
            str(args.output_root),
        ],
        cwd=args.repo,
        env=env,
        check=True,
    )
    manifest["status"] = "complete"
    manifest["ended_at"] = utc_now()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

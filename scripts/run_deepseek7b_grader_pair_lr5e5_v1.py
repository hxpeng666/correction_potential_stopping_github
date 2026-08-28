#!/usr/bin/env python3
"""Run the frozen-LR original-vs-forced-cap grader evaluation on two A100s."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GRADERS = ("original_grader", "forced_cap_grader")
ROUNDS = (
    ("gsm8k", "bce_traj"),
    ("gsm8k", "bce"),
    ("math", "bce_traj"),
    ("math", "bce"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def free_mib() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        int(line.split(",")[0].strip()): int(line.split(",")[1].strip())
        for line in result.stdout.splitlines()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--forced-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=3000)
    parser.add_argument("--poll-seconds", type=int, default=5)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_root.resolve()
    roots = {
        "original_grader": args.original_root.resolve(),
        "forced_cap_grader": args.forced_root.resolve(),
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("formal grader-pair run requires a clean worktree")
    output.mkdir(parents=True, exist_ok=True)
    log_root = output / "logs"
    log_root.mkdir(exist_ok=True)
    manifest = {
        "status": "training",
        "started_at": now(),
        "git_commit": commit,
        "learning_rate": 5e-5,
        "training_seed": 0,
        "split_seed": 0,
        "maximum_epochs": 24,
        "patience": 6,
        "model_selection": "minimum_internal_validation_objective",
        "deployment_calibration": "trajectory_envelope_ltt_only",
        "fixed_empirical_B_used": False,
        "hardware": "two NVIDIA A100 80GB PCIe; grader assignment alternates by round",
        "completed": [],
    }
    manifest_path = output / "RUN_MANIFEST.json"
    atomic_json(manifest, manifest_path)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    for round_index, (dataset, method) in enumerate(ROUNDS):
        assignment = (
            {"original_grader": 0, "forced_cap_grader": 1}
            if round_index % 2 == 0
            else {"original_grader": 1, "forced_cap_grader": 0}
        )
        outputs = {
            grader: output / "conditions" / grader / "probes" / dataset / method
            for grader in GRADERS
        }
        if all((path / "phase.complete").is_file() for path in outputs.values()):
            manifest["completed"].append(
                {"dataset": dataset, "method": method, "status": "skipped_complete"}
            )
            atomic_json(manifest, manifest_path)
            continue
        while True:
            available = free_mib()
            if all(available.get(gpu, 0) >= args.minimum_free_mib for gpu in (0, 1)):
                break
            time.sleep(args.poll_seconds)
        running = {}
        for grader in GRADERS:
            grader_root = roots[grader]
            heldout = grader_root / ("gsm8k" if dataset == "gsm8k" else "math500")
            command = [
                str(args.python),
                "scripts/train_deepseek7b_method_exploration_v1.py",
                "--dataset", dataset,
                "--config", str(args.config.resolve()),
                "--raw-root", str(grader_root / dataset),
                "--heldout-root", str(heldout),
                "--output", str(outputs[grader]),
                "--gpu", str(assignment[grader]),
                "--layer", "16",
                "--schedule", "sentence",
                "--representation-kind", "last",
                "--feature-kind", "hidden_scalars",
                "--probe-architecture", "standard",
                "--point-loss", "checkpoint_proper",
                "--trajectory-scope", "all_dangerous",
                "--rho", "1.0",
                "--lambda-separation", "0.0",
                "--gamma", "0.5",
                "--epochs", "24",
                "--patience", "6",
                "--selection-rule", "validation_objective",
                "--deployment-calibration", "scores_only",
                "--learning-rate", "0.00005",
                "--weight-decay", "0.001",
                "--seed", "0",
                "--split-seed", "0",
                "--resume",
            ]
            if dataset == "math":
                command.extend(["--ood-root", str(grader_root / "aime")])
            if method == "bce_traj":
                command.extend(
                    ["--trajectory-aggregation", "normalized_softmin", "--lambda-protect", "1.0", "--beta", "0.5"]
                )
            else:
                command.extend(
                    ["--trajectory-aggregation", "none", "--lambda-protect", "0.0"]
                )
            log_path = log_root / f"train_{grader}_{dataset}_{method}.log"
            handle = log_path.open("ab")
            process = subprocess.Popen(
                command, cwd=repo, env=environment, stdout=handle, stderr=subprocess.STDOUT
            )
            running[grader] = (process, handle, log_path)
        failures = []
        for grader, (process, handle, log_path) in running.items():
            return_code = process.wait()
            handle.close()
            if return_code:
                failures.append({"grader": grader, "code": return_code, "log": str(log_path)})
        manifest["completed"].append(
            {
                "dataset": dataset,
                "method": method,
                "gpu_assignment": assignment,
                "status": "failed" if failures else "complete",
                "failures": failures,
            }
        )
        atomic_json(manifest, manifest_path)
        if failures:
            manifest["status"] = "failed"
            atomic_json(manifest, manifest_path)
            raise SystemExit(json.dumps(failures))

    manifest["status"] = "ltt_evaluation"
    atomic_json(manifest, manifest_path)
    ltt_output = output.parent / f"{output.name}_ltt"
    subprocess.run(
        [
            str(args.python),
            "scripts/evaluate_deepseek7b_grader_pair_lr5e5_ltt_v1.py",
            "--experiment-root", str(output),
            "--original-data-root", str(roots["original_grader"]),
            "--forced-data-root", str(roots["forced_cap_grader"]),
            "--output", str(ltt_output),
            "--delta", "0.05",
            "--grid-size", "101",
        ],
        cwd=repo,
        env=environment,
        check=True,
        stdout=(log_root / "ltt.log").open("ab"),
        stderr=subprocess.STDOUT,
    )
    manifest["status"] = "complete"
    manifest["completed_at"] = now()
    manifest["ltt_output"] = str(ltt_output)
    atomic_json(manifest, manifest_path)
    atomic_json(
        {
            "status": "complete",
            "completed_at": manifest["completed_at"],
            "git_commit": commit,
            "ltt_audit": str(ltt_output / "AUDIT.json"),
        },
        output / "EXPERIMENT_COMPLETE.json",
    )


if __name__ == "__main__":
    main()

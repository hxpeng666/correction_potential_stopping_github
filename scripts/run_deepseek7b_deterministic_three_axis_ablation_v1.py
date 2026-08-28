#!/usr/bin/env python3
"""Run the committed deterministic three-axis DeepSeek-7B ablation.

The primary method is checkpoint-proper BCE with last hidden plus six scalar
features and normalized soft-min trajectory protection.  Every ablation changes
exactly one of feature construction, checkpoint BCE weighting, or trajectory
protection.  One isolated probe process is used per certified A100.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def primary() -> dict[str, Any]:
    return {
        "representation_kind": "last",
        "feature_kind": "hidden_scalars",
        "readout_kind": "full",
        "pca_dim": None,
        "probe_architecture": "standard",
        "point_loss": "checkpoint_proper",
        "stratified_problem_batches": False,
        "trajectory_scope": "all_dangerous",
        "trajectory_aggregation": "normalized_softmin",
        "beta": 0.5,
        "rho": 1.0,
        "lambda_protect": 1.0,
        "lambda_separation": 0.0,
        "gamma": 0.5,
        "uses_auxiliary": False,
        "online_readout_token_cost": 0,
    }


def variant(identifier: str, axis: str, **updates: Any) -> dict[str, Any]:
    value = primary()
    value.update(updates)
    value.update({"id": identifier, "axis": axis})
    return value


def variants() -> list[dict[str, Any]]:
    return [
        variant("primary", "primary"),
        # Feature construction: all loss and calibration settings stay primary.
        variant(
            "feature_last4_mean", "feature",
            representation_kind="last4_mean", uses_auxiliary=True,
        ),
        variant(
            "feature_paragraph_mean", "feature",
            representation_kind="paragraph_mean", uses_auxiliary=True,
        ),
        variant(
            "feature_prefix_mean", "feature",
            representation_kind="prefix_mean", uses_auxiliary=True,
        ),
        variant(
            "feature_pca32_linear", "feature",
            feature_kind="pca_hidden_scalars", pca_dim=32,
            probe_architecture="linear",
        ),
        variant(
            "feature_one_step_full", "feature",
            feature_kind="hidden_scalars_one_step", readout_kind="full",
            uses_auxiliary=True, online_readout_token_cost=6,
        ),
        variant(
            "feature_pool_paragraph_pca256", "feature",
            representation_kind="paragraph_mean",
            feature_kind="pca_hidden_scalars", pca_dim=256,
            probe_architecture="compact", uses_auxiliary=True,
        ),
        variant(
            "feature_pool_last4_pca128_one_step", "feature",
            representation_kind="last4_mean",
            feature_kind="pca_hidden_scalars_one_step", readout_kind="full",
            pca_dim=128, probe_architecture="compact", uses_auxiliary=True,
            online_readout_token_cost=6,
        ),
        # Point BCE weighting: feature and trajectory terms stay primary.
        variant("weight_legacy_remaining", "bce_weight", point_loss="legacy_weighted"),
        variant(
            "weight_problem_balanced_random", "bce_weight",
            point_loss="problem_balanced_proper",
        ),
        variant(
            "weight_problem_balanced_stratified", "bce_weight",
            point_loss="problem_balanced_proper", stratified_problem_batches=True,
        ),
        # Trajectory protection / first-hit: feature and point BCE stay primary.
        variant(
            "trajectory_none", "first_hit_trajectory",
            trajectory_aggregation="none", lambda_protect=0.0,
        ),
        variant(
            "trajectory_hard_min", "first_hit_trajectory",
            trajectory_aggregation="hard_min", lambda_protect=0.5,
        ),
        variant(
            "trajectory_bottomk", "first_hit_trajectory",
            trajectory_aggregation="bottomk_mean", rho=0.5,
            lambda_protect=0.25,
        ),
        variant(
            "trajectory_cvar", "first_hit_trajectory",
            trajectory_aggregation="lower_tail_cvar", rho=0.5,
            lambda_protect=0.5,
        ),
        variant(
            "first_hit_earliest_safe_cvar", "first_hit_trajectory",
            trajectory_scope="reachability_earliest_safe",
            trajectory_aggregation="lower_tail_cvar", rho=0.25,
            lambda_protect=0.25,
        ),
    ]


def command_for(
    *,
    python: Path,
    config: Path,
    data_root: Path,
    auxiliary_root: Path,
    output: Path,
    dataset: str,
    spec: dict[str, Any],
    gpu: int,
) -> list[str]:
    raw_dataset = "gsm8k" if dataset == "gsm8k" else "math"
    heldout_dataset = "gsm8k" if dataset == "gsm8k" else "math500"
    command = [
        str(python),
        "scripts/train_deepseek7b_method_exploration_v1.py",
        "--dataset", dataset,
        "--config", str(config),
        "--raw-root", str(data_root / raw_dataset),
        "--heldout-root", str(data_root / heldout_dataset),
        "--output", str(output),
        "--gpu", str(gpu),
        "--layer", "16",
        # The frozen artifacts use the historical `sentence` compatibility tag
        # for the scientific paragraph-boundary checkpoints.
        "--schedule", "sentence",
        "--representation-kind", str(spec["representation_kind"]),
        "--feature-kind", str(spec["feature_kind"]),
        "--readout-kind", str(spec["readout_kind"]),
        "--probe-architecture", str(spec["probe_architecture"]),
        "--point-loss", str(spec["point_loss"]),
        "--trajectory-scope", str(spec["trajectory_scope"]),
        "--trajectory-aggregation", str(spec["trajectory_aggregation"]),
        "--beta", str(spec["beta"]),
        "--rho", str(spec["rho"]),
        "--lambda-protect", str(spec["lambda_protect"]),
        "--lambda-separation", str(spec["lambda_separation"]),
        "--gamma", str(spec["gamma"]),
        "--epochs", "24",
        "--patience", "6",
        "--selection-rule", "validation_objective",
        "--deployment-calibration", "scores_only",
        "--learning-rate", "0.00005",
        "--weight-decay", "0.001",
        "--seed", "0",
        "--split-seed", "0",
        "--cpu-threads", "1",
        "--resume",
    ]
    if dataset == "math":
        command.extend(["--ood-root", str(data_root / "aime")])
    if spec["pca_dim"] is not None:
        command.extend(["--pca-dim", str(spec["pca_dim"])])
    if spec["stratified_problem_batches"]:
        command.append("--stratified-problem-batches")
    if spec["uses_auxiliary"]:
        command.extend(
            [
                "--aux-raw-root", str(auxiliary_root / raw_dataset),
                "--aux-heldout-root", str(
                    auxiliary_root / ("gsm8k_heldout" if dataset == "gsm8k" else "math500")
                ),
            ]
        )
        if dataset == "math":
            command.extend(["--aux-ood-root", str(auxiliary_root / "aime")])
    return command


def run_pair(
    jobs: list[tuple[list[str], Path]], *, repo: Path, environment: dict[str, str]
) -> list[dict[str, Any]]:
    def one(command: list[str], log: Path) -> dict[str, Any]:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as handle:
            handle.write(("COMMAND " + json.dumps(command) + "\n").encode())
            handle.flush()
            completed = subprocess.run(
                command, cwd=repo, env=environment,
                stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
        return {"command": command, "log": str(log), "returncode": completed.returncode}

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(one, command, log) for command, log in jobs]
        return [future.result() for future in as_completed(futures)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--auxiliary-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = args.config.resolve()
    data_root = args.data_root.resolve()
    auxiliary_root = args.auxiliary_root.resolve()
    output = args.output_root.resolve()
    python = args.python.resolve()

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("formal ablation requires a clean committed worktree")
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    if config_payload["probe"]["learning_rate"] != 5e-5:
        raise AssertionError("unexpected frozen learning rate")
    if config_payload["calibration"]["empirical_budget_B_used"] is not False:
        raise AssertionError("formal ablation must not use empirical B")

    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    registry = variants()
    atomic_json({item["id"]: item for item in registry}, output / "VARIANTS.json")
    manifest = {
        "status": "determinism_gate",
        "started_at": now(),
        "git_commit": commit,
        "config": str(config),
        "data_root": str(data_root),
        "auxiliary_root": str(auxiliary_root),
        "learning_rate": 5e-5,
        "training_seed": 0,
        "split_seed": 0,
        "maximum_epochs": 24,
        "patience": 6,
        "model_selection": "minimum_internal_validation_objective",
        "deployment_calibration": "trajectory_envelope_ltt_only",
        "calibration_objective": "total_generated_token_reduction",
        "fixed_empirical_B_used": False,
        "hardware": "one isolated probe process on each of certified A100 GPU0/GPU1",
        "variants": [item["id"] for item in registry],
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

    # Cross-A100 bitwise gate on the exact primary method and current data view.
    gate_spec = registry[0]
    for dataset in ("gsm8k", "math"):
        jobs = []
        for gpu in (0, 1):
            gate_output = output / "determinism_gate" / dataset / f"gpu{gpu}"
            jobs.append(
                (
                    command_for(
                        python=python, config=config, data_root=data_root,
                        auxiliary_root=auxiliary_root, output=gate_output,
                        dataset=dataset, spec=gate_spec, gpu=gpu,
                    ),
                    logs / f"gate_{dataset}_gpu{gpu}.log",
                )
            )
        results = run_pair(jobs, repo=repo, environment=environment)
        if any(result["returncode"] for result in results):
            manifest["status"] = "failed"
            manifest["failure"] = {"stage": "gate_training", "results": results}
            atomic_json(manifest, manifest_path)
            raise SystemExit(json.dumps(results))
        audit_path = output / "determinism_gate" / dataset / "AUDIT.json"
        audit_log = logs / f"gate_{dataset}_audit.log"
        audit_command = [
            str(python), "scripts/audit_deterministic_probe_pair_v1.py",
            "--left", str(output / "determinism_gate" / dataset / "gpu0"),
            "--right", str(output / "determinism_gate" / dataset / "gpu1"),
            "--output", str(audit_path),
        ]
        audit_result = run_pair(
            [(audit_command, audit_log)], repo=repo, environment=environment
        )[0]
        if audit_result["returncode"]:
            manifest["status"] = "failed"
            manifest["failure"] = {"stage": "gate_audit", "result": audit_result}
            atomic_json(manifest, manifest_path)
            raise SystemExit(json.dumps(audit_result))
        # Reuse one certified numerical primary output in the ablation registry.
        primary_output = output / "probes" / "primary" / dataset
        primary_output.parent.mkdir(parents=True, exist_ok=True)
        if not primary_output.exists():
            os.symlink(
                os.path.relpath(
                    output / "determinism_gate" / dataset / "gpu0",
                    primary_output.parent,
                ),
                primary_output,
            )
        manifest["completed"].append({"stage": "determinism_gate", "dataset": dataset})
        atomic_json(manifest, manifest_path)

    manifest["status"] = "training_ablation_variants"
    atomic_json(manifest, manifest_path)
    for index, spec in enumerate(registry[1:], start=1):
        # Alternate dataset-to-card assignment to remove any systematic card/axis pairing.
        assignment = (
            {"gsm8k": 0, "math": 1} if index % 2 else {"gsm8k": 1, "math": 0}
        )
        jobs = []
        for dataset in ("gsm8k", "math"):
            probe_output = output / "probes" / spec["id"] / dataset
            jobs.append(
                (
                    command_for(
                        python=python, config=config, data_root=data_root,
                        auxiliary_root=auxiliary_root, output=probe_output,
                        dataset=dataset, spec=spec, gpu=assignment[dataset],
                    ),
                    logs / f"train_{spec['id']}_{dataset}.log",
                )
            )
        results = run_pair(jobs, repo=repo, environment=environment)
        record = {
            "stage": "training",
            "variant": spec["id"],
            "gpu_assignment": assignment,
            "results": results,
        }
        manifest["completed"].append(record)
        atomic_json(manifest, manifest_path)
        if any(result["returncode"] for result in results):
            manifest["status"] = "failed"
            manifest["failure"] = record
            atomic_json(manifest, manifest_path)
            raise SystemExit(json.dumps(record))

    manifest["status"] = "trajectory_envelope_ltt"
    atomic_json(manifest, manifest_path)
    evaluate_command = [
        str(python), "scripts/evaluate_deepseek7b_deterministic_three_axis_ablation_v1.py",
        "--experiment-root", str(output),
        "--data-root", str(data_root),
        "--output", str(output / "ltt"),
        "--delta", "0.05",
        "--grid-size", "101",
    ]
    evaluation = run_pair(
        [(evaluate_command, logs / "ltt.log")], repo=repo, environment=environment
    )[0]
    if evaluation["returncode"]:
        manifest["status"] = "failed"
        manifest["failure"] = {"stage": "ltt", "result": evaluation}
        atomic_json(manifest, manifest_path)
        raise SystemExit(json.dumps(evaluation))
    manifest["status"] = "complete"
    manifest["completed_at"] = now()
    manifest["ltt_output"] = str(output / "ltt")
    atomic_json(manifest, manifest_path)
    atomic_json(
        {
            "status": "complete",
            "completed_at": manifest["completed_at"],
            "git_commit": commit,
            "unique_variants": len(registry),
            "training_runs": len(registry) * 2 + 2,
            "result_rows": 324,
            "ltt_audit": str(output / "ltt" / "AUDIT.json"),
        },
        output / "EXPERIMENT_COMPLETE.json",
    )


if __name__ == "__main__":
    main()

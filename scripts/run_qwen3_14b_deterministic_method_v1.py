#!/usr/bin/env python3
"""Train, calibrate, and evaluate the deterministic Qwen3-14B five-method suite."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


METHODS = ("correctness", "consistency", "last_switch", "bce", "bce_traj")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_jobs(jobs, repo: Path, environment: dict[str, str]):
    def one(command: list[str], log: Path):
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


def exploration_command(python, config, data, output, dataset, gpu, trajectory):
    command = [
        str(python), "scripts/train_deepseek7b_method_exploration_v1.py",
        "--dataset", dataset,
        "--config", str(config),
        "--raw-root", str(data / ("gsm8k" if dataset == "gsm8k" else "math")),
        "--heldout-root", str(data / ("gsm8k" if dataset == "gsm8k" else "math500")),
        "--output", str(output), "--gpu", str(gpu), "--layer", "20",
        "--schedule", "paragraph", "--representation-kind", "last",
        "--feature-kind", "hidden_scalars", "--readout-kind", "full",
        "--probe-architecture", "standard", "--point-loss", "checkpoint_proper",
        "--trajectory-scope", "all_dangerous",
        "--trajectory-aggregation", "normalized_softmin" if trajectory else "none",
        "--beta", "0.5", "--rho", "1.0",
        "--lambda-protect", "1.0" if trajectory else "0.0",
        "--lambda-separation", "0.0", "--gamma", "0.5",
        "--epochs", "24", "--patience", "6", "--batch-problems", "24",
        "--learning-rate", "0.00005", "--weight-decay", "0.001",
        "--seed", "0", "--split-seed", "0", "--cpu-threads", "1",
        "--selection-rule", "validation_objective",
        "--deployment-calibration", "scores_only", "--resume",
    ]
    if dataset == "math":
        command.extend(["--ood-root", str(data / "aime")])
    return command


def baseline_command(python, config, data, output, dataset, gpu, method):
    return [
        str(python), "scripts/train_deepseek7b_ablation_v1.py",
        "--dataset", dataset, "--config", str(config),
        "--raw-root", str(data / ("gsm8k" if dataset == "gsm8k" else "math")),
        "--heldout-root", str(data / ("gsm8k" if dataset == "gsm8k" else "math500")),
        "--output", str(output), "--method", method, "--seed", "0", "--gpu", str(gpu),
        "--schedule", "paragraph", "--actual-schedule-label", "paragraph",
        "--layer", "20", "--feature-kind", "full_no_delta", "--loss", "bce",
        "--trajectory-aggregation", "normalized_softmin", "--trajectory-beta", "0.5",
        "--trajectory-weight", "1.0", "--epochs", "24",
        "--selection-rule", "label_ap", "--resume",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    repo, config, data = args.repo.resolve(), args.config.resolve(), args.data_root.resolve()
    output, python = args.output_root.resolve(), args.python.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("formal Qwen3-14B method run requires a clean committed repository")
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    if cfg["probe"]["learning_rate"] != 5e-5 or cfg["features"]["primary_width"] != 5126:
        raise AssertionError("unexpected frozen Qwen3-14B probe configuration")
    if cfg["calibration"]["empirical_budget_B_used"] is not False:
        raise AssertionError("fixed empirical B is forbidden in the main result")
    collection = json.loads(args.collection_manifest.read_text(encoding="utf-8"))
    if collection.get("status") != "complete":
        raise RuntimeError("collection manifest is incomplete")
    if collection.get("protocol_fingerprint") != cfg["data"]["collection_protocol_fingerprint"]:
        raise AssertionError("collection protocol fingerprint differs from the frozen config")
    if collection["collection"]["actual_artifacts"] != 5449:
        raise AssertionError("expected exactly 5449 collection artifacts")
    if collection["collection"]["checkpoint_count"] != 276216:
        raise AssertionError("unexpected checkpoint count")

    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    manifest = {
        "status": "determinism_gate", "started_at": now(), "git_commit": commit,
        "config": str(config), "data_root": str(data),
        "collection_manifest": str(args.collection_manifest.resolve()),
        "collection_protocol_fingerprint": collection["protocol_fingerprint"],
        "methods": list(METHODS), "learning_rate": 5e-5,
        "training_seed": 0, "split_seed": 0, "max_epochs": 24, "patience": 6,
        "controlled_target_model_selection": "maximum_internal_validation_label_ap",
        "correction_probe_model_selection": "minimum_internal_validation_objective",
        "primary_probe": {
            "features": "last hidden + six scalars", "point_loss": "checkpoint_proper",
            "trajectory": "normalized_softmin", "beta": 0.5, "lambda": 1.0,
        },
        "calibration": {
            "method": "trajectory_envelope_ltt", "alphas": cfg["calibration"]["ltt_alphas"],
            "delta": 0.05, "candidate_source": "probe_train_only",
            "certification_source": "calibration_only", "fixed_empirical_B_used": False,
            "objective": "reasoning_only_token_reduction_to_match_deepseek7b_final",
        },
        "completed": [],
    }
    manifest_path = output / "RUN_MANIFEST.json"
    atomic_json(manifest, manifest_path)
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
    })

    # Exact cross-A100 gate for the primary BCE+trajectory probe.
    for dataset in ("gsm8k", "math"):
        jobs = []
        for gpu in (0, 1):
            gate_output = output / "determinism_gate" / dataset / f"gpu{gpu}"
            jobs.append((
                exploration_command(python, config, data, gate_output, dataset, gpu, True),
                logs / f"gate_{dataset}_gpu{gpu}.log",
            ))
        results = run_jobs(jobs, repo, environment)
        if any(row["returncode"] for row in results):
            manifest.update(status="failed", failure={"stage": "gate_training", "dataset": dataset, "results": results})
            atomic_json(manifest, manifest_path)
            raise SystemExit(1)
        audit = output / "determinism_gate" / dataset / "AUDIT.json"
        audit_command = [
            str(python), "scripts/audit_deterministic_probe_pair_v1.py",
            "--left", str(output / "determinism_gate" / dataset / "gpu0"),
            "--right", str(output / "determinism_gate" / dataset / "gpu1"),
            "--output", str(audit),
        ]
        result = run_jobs([(audit_command, logs / f"gate_{dataset}_audit.log")], repo, environment)[0]
        if result["returncode"]:
            manifest.update(status="failed", failure={"stage": "gate_audit", "dataset": dataset, "result": result})
            atomic_json(manifest, manifest_path)
            raise SystemExit(1)
        target = output / "probes" / ("gsm8k" if dataset == "gsm8k" else "math500") / "bce_traj"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.symlink(os.path.relpath(output / "determinism_gate" / dataset / "gpu0", target.parent), target)
        manifest["completed"].append({"stage": "determinism_gate", "dataset": dataset})
        atomic_json(manifest, manifest_path)

    manifest["status"] = "training"
    atomic_json(manifest, manifest_path)
    for index, method in enumerate(("bce", "correctness", "consistency", "last_switch")):
        assignment = {"gsm8k": index % 2, "math": 1 - (index % 2)}
        jobs = []
        for dataset in ("gsm8k", "math"):
            destination = output / "probes" / ("gsm8k" if dataset == "gsm8k" else "math500") / method
            command = (
                exploration_command(python, config, data, destination, dataset, assignment[dataset], False)
                if method == "bce"
                else baseline_command(python, config, data, destination, dataset, assignment[dataset], method)
            )
            jobs.append((command, logs / f"train_{method}_{dataset}.log"))
        results = run_jobs(jobs, repo, environment)
        record = {"stage": "training", "method": method, "gpu_assignment": assignment, "results": results}
        manifest["completed"].append(record)
        atomic_json(manifest, manifest_path)
        if any(row["returncode"] for row in results):
            manifest.update(status="failed", failure=record)
            atomic_json(manifest, manifest_path)
            raise SystemExit(1)

    manifest["status"] = "aime_shared_probe_scoring"
    atomic_json(manifest, manifest_path)
    controlled = ("correctness", "consistency", "last_switch")
    jobs = []
    for index, method in enumerate(controlled):
        source = output / "probes" / "math500" / method
        destination = output / "probes" / "aime" / method
        command = [
            str(python), "scripts/evaluate_deepseek7b_ood_v2.py", "--dataset", "aime",
            "--source-probe", str(source), "--heldout-root", str(data / "aime"),
            "--output", str(destination), "--gpu", str(index % 2),
            "--runtime-lock", str(repo / "configs/runtime_a100_torch271_cuda126_v1.json"), "--resume",
        ]
        jobs.append((command, logs / f"aime_{method}.log"))
    # Two at a time preserves the one-process-per-A100 deterministic execution rule.
    results = []
    for left in range(0, len(jobs), 2):
        batch = jobs[left:left + 2]
        if len(batch) == 1:
            batch[0][0][batch[0][0].index("--gpu") + 1] = "0"
        results.extend(run_jobs(batch, repo, environment))
    if any(row["returncode"] for row in results):
        manifest.update(status="failed", failure={"stage": "aime_scoring", "results": results})
        atomic_json(manifest, manifest_path)
        raise SystemExit(1)
    manifest["completed"].append({"stage": "aime_shared_probe_scoring", "results": results})

    manifest["status"] = "trajectory_envelope_ltt"
    atomic_json(manifest, manifest_path)
    evaluate = [
        str(python), "scripts/evaluate_qwen3_14b_five_method_ltt_v1.py",
        "--probe-root", str(output), "--data-root", str(data),
        "--output", str(output / "ltt"), "--delta", "0.05", "--grid-size", "101",
    ]
    result = run_jobs([(evaluate, logs / "ltt.log")], repo, environment)[0]
    if result["returncode"]:
        manifest.update(status="failed", failure={"stage": "ltt", "result": result})
        atomic_json(manifest, manifest_path)
        raise SystemExit(1)
    manifest.update(status="complete", completed_at=now(), ltt_output=str(output / "ltt"))
    atomic_json(manifest, manifest_path)
    atomic_json(
        {
            "status": "complete", "completed_at": manifest["completed_at"],
            "git_commit": commit, "methods": 5, "trained_probes": 10,
            "ltt_rows": 90, "ltt_audit": str(output / "ltt" / "AUDIT.json"),
        }, output / "EXPERIMENT_COMPLETE.json"
    )


if __name__ == "__main__":
    main()

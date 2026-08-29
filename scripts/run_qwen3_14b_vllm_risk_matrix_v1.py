#!/usr/bin/env python3
"""Run fail-closed cache/batch risk and efficiency gates on two A100s."""
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

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_vllm_risk_pair_v1 import metrics
from run_qwen3_14b_vllm_full_v1 import (
    collector_command,
    deterministic_environment,
    gate_artifact,
    run_logged,
)
from src.reproducibility import code_provenance, sha256_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def start_logged(
    command: list[str], log: Path, environment: dict[str, str]
) -> tuple[subprocess.Popen, Any, float]:
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    handle.write("COMMAND " + json.dumps(command) + "\n")
    handle.flush()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return process, handle, time.perf_counter()


def finish_logged(value: tuple[subprocess.Popen, Any, float]) -> tuple[int, float]:
    process, handle, started = value
    code = process.wait()
    elapsed = time.perf_counter() - started
    handle.close()
    return code, elapsed


def run_worker(
    python: Path,
    config: Path,
    prepared_root: Path,
    model_path: Path,
    root: Path,
    gpu: int,
    worker: str,
    profile: str,
    problem_ids: list[str],
    task_order: str,
    log: Path,
    environment_config: dict[str, Any],
) -> tuple[int, float]:
    command = collector_command(
        python,
        config,
        prepared_root,
        model_path,
        root,
        gpu,
        0.45,
        worker,
        0,
        1,
        profile,
        problem_ids=problem_ids,
        task_order=task_order,
    )
    started = start_logged(command, log, deterministic_environment(environment_config, gpu))
    return finish_logged(started)


def audit_exact(
    python: Path,
    config: dict[str, Any],
    left: Path,
    right: Path,
    output: Path,
    gpu_mode: str,
) -> bool:
    code = run_logged(
        [
            str(python),
            "scripts/audit_deterministic_collection_pair_v1.py",
            "--left",
            str(left),
            "--right",
            str(right),
            "--gpu-mode",
            gpu_mode,
            "--output",
            str(output),
        ],
        output.with_suffix(".log"),
        deterministic_environment(config),
    )
    return code == 0


def audit_risk(
    python: Path,
    config: dict[str, Any],
    left: Path,
    right: Path,
    output: Path,
) -> dict[str, Any]:
    code = run_logged(
        [
            str(python),
            "scripts/audit_vllm_risk_pair_v1.py",
            "--left",
            str(left),
            "--right",
            str(right),
            "--output",
            str(output),
        ],
        output.with_suffix(".log"),
        deterministic_environment(config),
    )
    if code != 0:
        raise RuntimeError(f"risk audit crashed: {output}")
    return json.loads(output.read_text(encoding="utf-8"))


def artifact_metrics(root: Path, problems: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        metrics(torch.load(gate_artifact(root, problem), map_location="cpu", weights_only=False))
        for problem in problems
    ]
    sums = {
        name: sum(float(value[name]) for value in values)
        for name in (
            "dense_wall_ms",
            "hidden_wall_ms",
            "branch_wall_ms",
            "branch_cached_tokens",
            "branch_context_tokens",
        )
    }
    sums["branch_cache_fraction"] = (
        sums["branch_cached_tokens"] / sums["branch_context_tokens"]
        if sums["branch_context_tokens"]
        else 0.0
    )
    return {"problems": values, "sum": sums}


def phase_speedup(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float | None]:
    result = {}
    for phase in ("dense", "hidden", "branch"):
        key = f"{phase}_wall_ms"
        denominator = float(candidate["sum"][key])
        result[phase] = float(baseline["sum"][key]) / denominator if denominator else None
    baseline_total = sum(float(baseline["sum"][f"{phase}_wall_ms"]) for phase in ("dense", "hidden", "branch"))
    candidate_total = sum(float(candidate["sum"][f"{phase}_wall_ms"]) for phase in ("dense", "hidden", "branch"))
    result["measured_phase_total"] = baseline_total / candidate_total if candidate_total else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    profiles = [str(value) for value in config["vllm_risk_gate"]["candidate_profiles"]]
    problems = [dict(value) for value in config["determinism_gate"]["problems"]]
    problem_ids = [str(value["problem_id"]) for value in problems]
    output = args.output_root.resolve()
    identity = code_provenance(
        ROOT,
        (
            "configs/qwen3_14b_deterministic_ood13k_vllm_full_v1.yaml",
            "scripts/run_qwen3_14b_vllm_risk_matrix_v1.py",
            "scripts/run_qwen3_14b_vllm_worker_v1.py",
            "scripts/collect_qwen3_14b_vllm_full_v1.py",
            "scripts/audit_deterministic_collection_pair_v1.py",
            "scripts/audit_vllm_risk_pair_v1.py",
            "src/reproducibility.py",
        ),
    )
    invocation = {
        "config": str(args.config.resolve()),
        "prepared_root": str(args.prepared_root.resolve()),
        "model_path": str(args.model_path.resolve()),
        "output_root": str(output),
        "python": str(args.python.resolve()),
        "profiles": profiles,
        "problem_ids": problem_ids,
        "git_commit": identity["git"]["commit"],
    }
    manifest_path = output / "RISK_GATE_MANIFEST.json"
    if output.exists() and not args.resume:
        raise RuntimeError(f"risk output already exists: {output}")
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("invocation_fingerprint") != sha256_json(invocation):
            raise RuntimeError("risk output invocation mismatch")
    output.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "running",
        "started_at": utc_now(),
        "invocation": invocation,
        "invocation_fingerprint": sha256_json(invocation),
        "code_identity": identity,
        "runs": {},
    }
    atomic_json(state, manifest_path)
    try:
        for profile in profiles:
            profile_root = output / "profiles" / profile
            repeat0_root = profile_root / "same_gpu_repeat0"
            code, elapsed = run_worker(
                args.python,
                args.config,
                args.prepared_root,
                args.model_path,
                repeat0_root,
                1,
                f"risk_{profile}_repeat0",
                profile,
                problem_ids,
                "canonical",
                output / "logs" / f"{profile}_repeat0.log",
                config,
            )
            state["runs"][f"{profile}/repeat0"] = {"returncode": code, "elapsed_seconds": elapsed}
            atomic_json(state, manifest_path)
            if code != 0:
                raise RuntimeError(f"profile crashed: {profile}/repeat0")
            repeat1_command = collector_command(
                args.python,
                args.config,
                args.prepared_root,
                args.model_path,
                profile_root / "same_gpu_repeat1",
                1,
                0.45,
                f"risk_{profile}_repeat1",
                0,
                1,
                profile,
                problem_ids=problem_ids,
            )
            cross_command = collector_command(
                args.python,
                args.config,
                args.prepared_root,
                args.model_path,
                profile_root / "cross_gpu0",
                0,
                0.45,
                f"risk_{profile}_cross",
                0,
                1,
                profile,
                problem_ids=problem_ids,
            )
            running = {
                "repeat1": start_logged(
                    repeat1_command,
                    output / "logs" / f"{profile}_repeat1.log",
                    deterministic_environment(config, 1),
                ),
                "cross": start_logged(
                    cross_command,
                    output / "logs" / f"{profile}_cross.log",
                    deterministic_environment(config, 0),
                ),
            }
            for name, process in running.items():
                code, elapsed = finish_logged(process)
                state["runs"][f"{profile}/{name}"] = {
                    "returncode": code,
                    "elapsed_seconds": elapsed,
                }
                atomic_json(state, manifest_path)
                if code != 0:
                    raise RuntimeError(f"profile crashed: {profile}/{name}")

        composition_profile = "full_apc_b2"
        composition_root = output / "composition" / composition_profile
        composition_runs = (
            ("paired_reverse", problem_ids, "reverse"),
            (f"solo_{problem_ids[0]}", [problem_ids[0]], "canonical"),
            (f"solo_{problem_ids[1]}", [problem_ids[1]], "canonical"),
        )
        for name, local_ids, order in composition_runs:
            code, elapsed = run_worker(
                args.python,
                args.config,
                args.prepared_root,
                args.model_path,
                composition_root / name,
                1,
                f"risk_composition_{name}",
                composition_profile,
                local_ids,
                order,
                output / "logs" / f"composition_{name}.log",
                config,
            )
            state["runs"][f"composition/{name}"] = {
                "returncode": code,
                "elapsed_seconds": elapsed,
            }
            atomic_json(state, manifest_path)
            if code != 0:
                raise RuntimeError(f"composition run crashed: {name}")

        results: dict[str, Any] = {}
        baseline_root = output / "profiles" / "baseline_b1" / "same_gpu_repeat0"
        baseline_metrics = artifact_metrics(baseline_root, problems)
        for profile in profiles:
            root = output / "profiles" / profile
            profile_result: dict[str, Any] = {
                "same_gpu_exact": {},
                "cross_gpu_exact": {},
                "vs_baseline": {},
                "metrics": artifact_metrics(root / "same_gpu_repeat0", problems),
            }
            for problem in problems:
                problem_id = str(problem["problem_id"])
                reference = gate_artifact(root / "same_gpu_repeat0", problem)
                profile_result["same_gpu_exact"][problem_id] = audit_exact(
                    args.python,
                    config,
                    reference,
                    gate_artifact(root / "same_gpu_repeat1", problem),
                    output / "audits" / f"{profile}_{problem_id}_same.json",
                    "same",
                )
                profile_result["cross_gpu_exact"][problem_id] = audit_exact(
                    args.python,
                    config,
                    reference,
                    gate_artifact(root / "cross_gpu0", problem),
                    output / "audits" / f"{profile}_{problem_id}_cross.json",
                    "distinct",
                )
                profile_result["vs_baseline"][problem_id] = audit_risk(
                    args.python,
                    config,
                    gate_artifact(baseline_root, problem),
                    reference,
                    output / "audits" / f"{profile}_{problem_id}_vs_baseline.json",
                )
            profile_result["speedup_vs_baseline"] = phase_speedup(
                baseline_metrics, profile_result["metrics"]
            )
            results[profile] = profile_result

        composition: dict[str, Any] = {}
        paired_root = output / "profiles" / composition_profile / "same_gpu_repeat0"
        for problem in problems:
            problem_id = str(problem["problem_id"])
            reference = gate_artifact(paired_root, problem)
            composition[f"{problem_id}/paired_reverse"] = audit_risk(
                args.python,
                config,
                reference,
                gate_artifact(composition_root / "paired_reverse", problem),
                output / "audits" / f"composition_{problem_id}_reverse.json",
            )
            composition[f"{problem_id}/solo"] = audit_risk(
                args.python,
                config,
                reference,
                gate_artifact(composition_root / f"solo_{problem_id}", problem),
                output / "audits" / f"composition_{problem_id}_solo.json",
            )

        accepted = []
        for profile, value in results.items():
            exact = (
                all(value["same_gpu_exact"].values())
                and all(value["cross_gpu_exact"].values())
                and all(
                    item["all_scientific_exact"]
                    for item in value["vs_baseline"].values()
                )
            )
            if profile == composition_profile:
                exact = exact and all(
                    item["all_scientific_exact"] for item in composition.values()
                )
            value["accepted"] = bool(exact)
            if exact:
                accepted.append(profile)
        minimum = float(config["vllm_risk_gate"]["minimum_material_speedup"])
        ranked = sorted(
            accepted,
            key=lambda name: float(
                results[name]["speedup_vs_baseline"]["measured_phase_total"] or 0.0
            ),
            reverse=True,
        )
        recommendation = ranked[0] if ranked else None
        if (
            recommendation is not None
            and recommendation != "baseline_b1"
            and float(
                results[recommendation]["speedup_vs_baseline"]["measured_phase_total"]
                or 0.0
            )
            < minimum
        ):
            recommendation = "baseline_b1"
        report = {
            "status": "complete",
            "all_runs_completed": True,
            "hard_gate_policy": config["vllm_risk_gate"],
            "results": results,
            "batch_composition": composition,
            "accepted_profiles": accepted,
            "recommended_profile": recommendation,
            "created_at": utc_now(),
        }
        atomic_json(report, output / "RISK_MATRIX.json")
        state["status"] = "complete"
        state["completed_at"] = utc_now()
        state["recommended_profile"] = recommendation
        atomic_json(state, manifest_path)
        print(json.dumps(report, indent=2))
    except Exception as error:
        state.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": utc_now(),
            }
        )
        atomic_json(state, manifest_path)
        raise


if __name__ == "__main__":
    main()

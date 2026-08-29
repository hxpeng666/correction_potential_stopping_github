#!/usr/bin/env python3
"""Gate, launch and audit the isolated full-vLLM Qwen3-14B collection."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SELF_REPRODUCIBILITY_POLICY = "same_profile_same_and_cross_gpu_exact_v1"

from src.reproducibility import (
    code_provenance,
    deterministic_subprocess_environment,
    sha256_file,
    sha256_json,
)

FORMAL_WORKERS = (
    (0, 0.45, "formal_gpu0_replica0", 0, 3),
    (1, 0.45, "formal_gpu1_replica0", 1, 3),
    (1, 0.45, "formal_gpu1_replica1", 2, 3),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def deterministic_environment(config: dict[str, Any], physical_gpu: int | None = None) -> dict[str, str]:
    environment = deterministic_subprocess_environment(seed=0)
    environment.update(
        {
            str(key): str(value)
            for key, value in config["reproducibility"]["required_environment"].items()
        }
    )
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["VLLM_NO_USAGE_STATS"] = "1"
    if physical_gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    return environment


def collector_command(
    python: Path,
    config: Path,
    prepared_root: Path,
    model_path: Path,
    output_root: Path,
    physical_gpu: int,
    memory: float,
    worker: str,
    shard: int,
    num_shards: int,
    profile: str,
    *,
    problem_ids: list[str] | None = None,
    task_order: str = "canonical",
) -> list[str]:
    value = [
        str(python),
        "scripts/run_qwen3_14b_vllm_worker_v1.py",
        "--config",
        str(config),
        "--prepared-root",
        str(prepared_root),
        "--model-path",
        str(model_path),
        "--output-root",
        str(output_root),
        "--physical-gpu",
        str(physical_gpu),
        "--gpu-memory-utilization",
        str(memory),
        "--worker-id",
        worker,
        "--shard-index",
        str(shard),
        "--num-shards",
        str(num_shards),
        "--profile",
        profile,
        "--task-order",
        task_order,
        "--resume",
    ]
    for problem_id in problem_ids or []:
        value.extend(("--problem-id", problem_id))
    return value


def run_logged(
    value: list[str], log: Path, environment: dict[str, str]
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(value) + "\n")
        handle.flush()
        return subprocess.run(
            value,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        ).returncode


def gate_artifact(root: Path, problem: dict[str, Any]) -> Path:
    return (
        root
        / "cache"
        / str(problem["dataset"])
        / str(problem["split"])
        / f"sample_{problem['problem_id']}.pt"
    )


def validate_risk_gate(
    config: dict[str, Any], risk_gate: dict[str, Any], profile: str
) -> dict[str, Any]:
    profile_result = risk_gate.get("results", {}).get(profile, {})
    expected_gate_problem_ids = {
        str(value["problem_id"]) for value in config["determinism_gate"]["problems"]
    }
    same_gpu_exact = profile_result.get("same_gpu_exact", {})
    cross_gpu_exact = profile_result.get("cross_gpu_exact", {})
    if (
        risk_gate.get("status") != "complete"
        or profile not in risk_gate.get("accepted_profiles", [])
        or risk_gate.get("recommended_profile") != profile
        or risk_gate.get("acceptance_policy", {}).get("name")
        != SELF_REPRODUCIBILITY_POLICY
        or profile_result.get("self_reproducibility_accepted") is not True
        or set(same_gpu_exact) != expected_gate_problem_ids
        or not all(same_gpu_exact.values())
        or set(cross_gpu_exact) != expected_gate_problem_ids
        or not all(cross_gpu_exact.values())
    ):
        raise RuntimeError(
            f"profile {profile} is not the completed self-reproducible risk-matrix recommendation"
        )
    return profile_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--transformers-reference-artifact", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--risk-gate-audit", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    risk_gate = json.loads(args.risk_gate_audit.read_text(encoding="utf-8"))
    profile_result = validate_risk_gate(config, risk_gate, args.profile)
    output = args.output_root.resolve()
    identity = code_provenance(
        ROOT,
        (
            "configs/qwen3_14b_deterministic_ood13k_vllm_full_v1.yaml",
            "configs/qwen3_14b_vllm_full_v1_requirements.txt",
            "scripts/run_qwen3_14b_vllm_full_v1.py",
            "scripts/run_qwen3_14b_vllm_worker_v1.py",
            "scripts/collect_qwen3_14b_vllm_full_v1.py",
            "scripts/audit_qwen3_14b_deterministic_ood_v1.py",
            "scripts/audit_deterministic_collection_pair_v1.py",
            "scripts/audit_vllm_protocol_alignment_v1.py",
            "src/reproducibility.py",
        ),
    )
    invocation = {
        "config": str(args.config.resolve()),
        "prepared_root": str(args.prepared_root.resolve()),
        "model_path": str(args.model_path.resolve()),
        "output_root": str(output),
        "python": str(args.python.resolve()),
        "transformers_reference_artifact": str(
            args.transformers_reference_artifact.resolve()
        ),
        "git_commit": identity["git"]["commit"],
        "profile": args.profile,
        "risk_gate_audit": str(args.risk_gate_audit.resolve()),
        "risk_gate_audit_sha256": sha256_file(args.risk_gate_audit),
    }
    invocation_fingerprint = sha256_json(invocation)
    manifest_path = output / "RUN_MANIFEST.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("invocation_fingerprint") != invocation_fingerprint:
            raise RuntimeError("refusing to reuse output root for a different invocation")
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(f"{os.getpid()}\n", output / "supervisor.pid")
    state = {
        "status": "determinism_gate",
        "started_at": utc_now(),
        "invocation": invocation,
        "invocation_fingerprint": invocation_fingerprint,
        "code_identity": identity,
        "risk_gate": {
            "path": str(args.risk_gate_audit.resolve()),
            "sha256": sha256_file(args.risk_gate_audit),
            "recommended_profile": risk_gate["recommended_profile"],
            "accepted_profiles": risk_gate["accepted_profiles"],
            "acceptance_policy": risk_gate["acceptance_policy"],
            "baseline_equivalent": profile_result.get("baseline_equivalent"),
        },
        "formal_workers": [
            {
                "physical_gpu": gpu,
                "gpu_memory_utilization": memory,
                "worker": worker,
                "shard_index": shard,
                "num_shards": total,
            }
            for gpu, memory, worker, shard, total in FORMAL_WORKERS
        ],
        "stages": [],
    }
    atomic_json(state, manifest_path)
    gate_problems = [dict(value) for value in config["determinism_gate"]["problems"]]
    gate_problem_ids = [str(value["problem_id"]) for value in gate_problems]
    primary_problem_id = str(config["determinism_gate"]["primary_problem_id"])
    primary_problem = next(
        value for value in gate_problems if str(value["problem_id"]) == primary_problem_id
    )
    gate_specs = (
        (0, "same_gpu_repeat0"),
        (0, "same_gpu_repeat1"),
        (1, "cross_gpu1"),
    )
    for physical_gpu, name in gate_specs:
        gate_root = output / "determinism_gate" / name
        command = collector_command(
            args.python,
            args.config,
            args.prepared_root,
            args.model_path,
            gate_root,
            physical_gpu,
            0.45,
            f"gate_{name}",
            0,
            1,
            args.profile,
            problem_ids=gate_problem_ids,
        )
        code = run_logged(
            command,
            output / "logs" / f"gate_{name}.log",
            deterministic_environment(config, physical_gpu),
        )
        if code != 0:
            state.update(
                {
                    "status": "failed",
                    "stage": "determinism_gate",
                    "failed_gate": name,
                    "returncode": code,
                }
            )
            atomic_json(state, manifest_path)
            raise SystemExit(2)

    same_gate_payloads = []
    cross_gate_payloads = []
    for problem in gate_problems:
        problem_id = str(problem["problem_id"])
        left = gate_artifact(output / "determinism_gate" / "same_gpu_repeat0", problem)
        comparisons = (
            (
                gate_artifact(output / "determinism_gate" / "same_gpu_repeat1", problem),
                output / f"SAME_GPU_REPEAT_GATE_{problem_id}.json",
                "same",
                f"same_gpu_gate_audit_{problem_id}.log",
                same_gate_payloads,
            ),
            (
                gate_artifact(output / "determinism_gate" / "cross_gpu1", problem),
                output / f"CROSS_GPU_GATE_{problem_id}.json",
                "distinct",
                f"cross_gpu_gate_audit_{problem_id}.log",
                cross_gate_payloads,
            ),
        )
        for right, destination, mode, log_name, payloads in comparisons:
            command = [
                str(args.python),
                "scripts/audit_deterministic_collection_pair_v1.py",
                "--left",
                str(left),
                "--right",
                str(right),
                "--gpu-mode",
                mode,
                "--output",
                str(destination),
            ]
            if run_logged(
                command,
                output / "logs" / log_name,
                deterministic_environment(config),
            ) != 0:
                state.update({"status": "failed", "stage": log_name})
                atomic_json(state, manifest_path)
                raise SystemExit(2)
            payloads.append(json.loads(destination.read_text(encoding="utf-8")))

    primary_left = gate_artifact(
        output / "determinism_gate" / "same_gpu_repeat0", primary_problem
    )

    alignment = output / "PROTOCOL_ALIGNMENT_GATE.json"
    alignment_command = [
        str(args.python),
        "scripts/audit_vllm_protocol_alignment_v1.py",
        "--transformers",
        str(args.transformers_reference_artifact),
        "--vllm",
        str(primary_left),
        "--output",
        str(alignment),
    ]
    if run_logged(
        alignment_command,
        output / "logs" / "protocol_alignment_gate.log",
        deterministic_environment(config),
    ) != 0:
        state.update({"status": "failed", "stage": "protocol_alignment_gate"})
        atomic_json(state, manifest_path)
        raise SystemExit(2)
    combined = {
        "status": "complete",
        "all_exact": True,
        "same_gpu_repeat": same_gate_payloads,
        "cross_gpu": cross_gate_payloads,
        "protocol_alignment": json.loads(alignment.read_text(encoding="utf-8")),
        "risk_matrix_sha256": sha256_file(args.risk_gate_audit),
        "risk_matrix_recommended_profile": risk_gate["recommended_profile"],
        "created_at": utc_now(),
    }
    atomic_json(combined, output / "DETERMINISM_GATE.json")
    state["stages"].append("same_gpu_repeat_exact_gate")
    state["stages"].append("cross_gpu_exact_gate")
    state["stages"].append("non_engine_protocol_alignment_gate")
    state["status"] = "collecting"
    atomic_json(state, manifest_path)

    formal_processes = []
    for gpu, memory, worker, shard, total in FORMAL_WORKERS:
        command = collector_command(
            args.python,
            args.config,
            args.prepared_root,
            args.model_path,
            output,
            gpu,
            memory,
            worker,
            shard,
            total,
            args.profile,
        )
        log = output / "logs" / f"{worker}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("a", encoding="utf-8")
        handle.write("COMMAND " + json.dumps(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=deterministic_environment(config, gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        formal_processes.append((process, handle))
    returncodes = []
    for process, handle in formal_processes:
        returncodes.append(process.wait())
        handle.close()
    if returncodes != [0] * len(FORMAL_WORKERS):
        state.update(
            {"status": "failed", "stage": "collection", "returncodes": returncodes}
        )
        atomic_json(state, manifest_path)
        raise SystemExit(2)
    state["stages"].append("two_gpu_collection")
    state["status"] = "auditing"
    atomic_json(state, manifest_path)

    collection_audit = [
        str(args.python),
        "scripts/audit_qwen3_14b_deterministic_ood_v1.py",
        "--config",
        str(args.config),
        "--prepared-root",
        str(args.prepared_root),
        "--output-root",
        str(output),
        "--gate-audit",
        str(output / "DETERMINISM_GATE.json"),
        "--profile",
        args.profile,
    ]
    if run_logged(
        collection_audit,
        output / "logs" / "audit.log",
        deterministic_environment(config),
    ) != 0:
        state.update({"status": "failed", "stage": "collection_audit"})
        atomic_json(state, manifest_path)
        raise SystemExit(2)
    state["stages"].append("collection_audit")
    state["status"] = "complete"
    state["completed_at"] = utc_now()
    atomic_json(state, manifest_path)
    atomic_json(
        {
            "status": "complete",
            "completed_at": state["completed_at"],
            "git_commit": identity["git"]["commit"],
            "audit": str((output / "COLLECTION_AUDIT.json").resolve()),
        },
        output / "EXPERIMENT_COMPLETE.json",
    )


if __name__ == "__main__":
    main()

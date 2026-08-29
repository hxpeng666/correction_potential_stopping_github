#!/usr/bin/env python3
"""Gate, launch and audit the isolated full-vLLM Qwen3-14B collection."""
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


def process_belongs_to_roots(pid: int, roots: set[int]) -> bool:
    """Return whether pid is one of roots or a live descendant of one."""
    seen: set[int] = set()
    current = pid
    while current > 1 and current not in seen:
        if current in roots:
            return True
        seen.add(current)
        status = Path(f"/proc/{current}/status")
        try:
            parent_line = next(
                line for line in status.read_text(encoding="utf-8").splitlines()
                if line.startswith("PPid:")
            )
            current = int(parent_line.split()[1])
        except (FileNotFoundError, PermissionError, StopIteration, ValueError):
            return False
    return current in roots


def worker_capacity(
    total_mib: int,
    external_used_mib: int,
    required_slot_mib: int,
    maximum: int,
) -> int:
    if min(total_mib, external_used_mib, required_slot_mib, maximum) < 0:
        raise ValueError("GPU capacity values must be non-negative")
    if required_slot_mib == 0:
        raise ValueError("required_slot_mib must be positive")
    return max(0, min(maximum, (total_mib - external_used_mib) // required_slot_mib))


def gpu_resource_snapshot(
    physical_gpus: list[int],
    active_worker_roots: set[int],
    required_slot_mib: int,
    maximum_workers_per_gpu: int,
) -> dict[int, dict[str, Any]]:
    gpu_output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    snapshot: dict[int, dict[str, Any]] = {}
    uuid_to_gpu: dict[str, int] = {}
    for line in gpu_output.splitlines():
        index_text, uuid, total_text, free_text = [item.strip() for item in line.split(",")]
        gpu = int(index_text)
        if gpu not in physical_gpus:
            continue
        snapshot[gpu] = {
            "total_mib": int(total_text),
            "free_mib": int(free_text),
            "external_processes": [],
            "task_processes": [],
        }
        uuid_to_gpu[uuid] = gpu
    app_output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in app_output.splitlines():
        uuid, pid_text, used_text, process_name = [
            item.strip() for item in line.split(",", 3)
        ]
        gpu = uuid_to_gpu.get(uuid)
        if gpu is None:
            continue
        process = {
            "pid": int(pid_text),
            "used_memory_mib": int(used_text),
            "process_name": process_name,
        }
        key = (
            "task_processes"
            if process_belongs_to_roots(int(pid_text), active_worker_roots)
            else "external_processes"
        )
        snapshot[gpu][key].append(process)
    for value in snapshot.values():
        external_used = sum(
            int(process["used_memory_mib"])
            for process in value["external_processes"]
        )
        value["external_used_mib"] = external_used
        value["worker_capacity"] = worker_capacity(
            int(value["total_mib"]),
            external_used,
            required_slot_mib,
            maximum_workers_per_gpu,
        )
    if set(snapshot) != set(physical_gpus):
        raise RuntimeError(f"missing requested GPU telemetry: {snapshot.keys()}")
    return snapshot


def scheduler_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(config["formal_scheduler"])
    physical_gpus = [int(value) for value in settings["physical_gpus"]]
    if physical_gpus != [0, 1]:
        raise RuntimeError(f"formal physical GPUs must remain [0, 1]: {physical_gpus}")
    if int(settings["num_shards"]) != 4:
        raise RuntimeError("dynamic formal scheduler requires exactly four logical shards")
    if int(settings["max_workers_per_gpu"]) != 2:
        raise RuntimeError("dynamic formal scheduler requires a two-worker per-GPU cap")
    if int(settings["required_memory_mib_per_worker_slot"]) < 40960:
        raise RuntimeError("worker slot reservation must be at least 40960 MiB")
    if float(settings["worker_gpu_memory_utilization"]) != 0.45:
        raise RuntimeError("formal worker memory utilization must remain 0.45")
    if [int(value) for value in settings["same_gpu_gate_preference"]] != [1, 0]:
        raise RuntimeError("same-GPU gate preference must remain [1, 0]")
    if settings.get("allow_collection_before_cross_gpu_gate") is not True:
        raise RuntimeError("adaptive collection must be allowed before the cross-GPU gate")
    return settings


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
    scheduler = scheduler_settings(config)
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
        "formal_scheduler": scheduler,
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
                "physical_gpu": None,
                "gpu_memory_utilization": float(
                    scheduler["worker_gpu_memory_utilization"]
                ),
                "worker": f"formal_shard{shard}",
                "shard_index": shard,
                "num_shards": int(scheduler["num_shards"]),
                "status": "pending",
            }
            for shard in range(int(scheduler["num_shards"]))
        ],
        "scheduler": {
            "settings": scheduler,
            "last_gpu_snapshot": None,
            "poll_count": 0,
        },
        "stages": [],
    }
    atomic_json(state, manifest_path)
    gate_problems = [dict(value) for value in config["determinism_gate"]["problems"]]
    gate_problem_ids = [str(value["problem_id"]) for value in gate_problems]
    primary_problem_id = str(config["determinism_gate"]["primary_problem_id"])
    primary_problem = next(
        value for value in gate_problems if str(value["problem_id"]) == primary_problem_id
    )
    physical_gpus = [int(value) for value in scheduler["physical_gpus"]]
    primary_gpu: int | None = None
    while primary_gpu is None:
        snapshot = gpu_resource_snapshot(
            physical_gpus,
            set(),
            int(scheduler["required_memory_mib_per_worker_slot"]),
            1,
        )
        state["scheduler"]["last_gpu_snapshot"] = snapshot
        state["scheduler"]["poll_count"] += 1
        primary_gpu = next(
            (
                int(gpu)
                for gpu in scheduler["same_gpu_gate_preference"]
                if int(snapshot[int(gpu)]["worker_capacity"]) >= 1
            ),
            None,
        )
        if primary_gpu is None:
            state["status"] = "determinism_gate_waiting_for_gpu_memory"
            state["waiting_gate"] = "same_gpu_repeat0"
            atomic_json(state, manifest_path)
            time.sleep(int(scheduler["poll_seconds"]))
    cross_gpu = next(gpu for gpu in physical_gpus if gpu != primary_gpu)
    state["gate_placement"] = {
        "same_gpu": primary_gpu,
        "cross_gpu": cross_gpu,
        "collection_before_cross_gpu_gate": True,
    }
    for name in ("same_gpu_repeat0", "same_gpu_repeat1"):
        while True:
            snapshot = gpu_resource_snapshot(
                [primary_gpu],
                set(),
                int(scheduler["required_memory_mib_per_worker_slot"]),
                1,
            )
            state["scheduler"]["last_gpu_snapshot"] = snapshot
            state["scheduler"]["poll_count"] += 1
            if int(snapshot[primary_gpu]["worker_capacity"]) >= 1:
                state["status"] = "determinism_gate"
                atomic_json(state, manifest_path)
                break
            state["status"] = "determinism_gate_waiting_for_gpu_memory"
            state["waiting_gate"] = name
            atomic_json(state, manifest_path)
            time.sleep(int(scheduler["poll_seconds"]))
        state.pop("waiting_gate", None)
        gate_root = output / "determinism_gate" / name
        command = collector_command(
            args.python,
            args.config,
            args.prepared_root,
            args.model_path,
            gate_root,
            primary_gpu,
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
            deterministic_environment(config, primary_gpu),
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
    for problem in gate_problems:
        problem_id = str(problem["problem_id"])
        left = gate_artifact(output / "determinism_gate" / "same_gpu_repeat0", problem)
        destination = output / f"SAME_GPU_REPEAT_GATE_{problem_id}.json"
        command = [
            str(args.python),
            "scripts/audit_deterministic_collection_pair_v1.py",
            "--left",
            str(left),
            "--right",
            str(gate_artifact(output / "determinism_gate" / "same_gpu_repeat1", problem)),
            "--gpu-mode",
            "same",
            "--output",
            str(destination),
        ]
        log_name = f"same_gpu_gate_audit_{problem_id}.log"
        if run_logged(
            command,
            output / "logs" / log_name,
            deterministic_environment(config),
        ) != 0:
            state.update({"status": "failed", "stage": log_name})
            atomic_json(state, manifest_path)
            raise SystemExit(2)
        same_gate_payloads.append(json.loads(destination.read_text(encoding="utf-8")))

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
    same_only = {
        "status": "same_gpu_complete_cross_gpu_pending",
        "same_gpu_exact": True,
        "same_gpu_repeat": same_gate_payloads,
        "protocol_alignment": json.loads(alignment.read_text(encoding="utf-8")),
        "risk_matrix_sha256": sha256_file(args.risk_gate_audit),
        "risk_matrix_recommended_profile": risk_gate["recommended_profile"],
        "created_at": utc_now(),
    }
    atomic_json(same_only, output / "DETERMINISM_GATE_SAME_GPU.json")
    state["stages"].append("same_gpu_repeat_exact_gate")
    state["stages"].append("non_engine_protocol_alignment_gate")
    state["status"] = "collecting_pending_cross_gpu_gate"
    atomic_json(state, manifest_path)

    pending = list(range(int(scheduler["num_shards"])))
    active: dict[int, tuple[subprocess.Popen, Any, int]] = {}
    cross_process: subprocess.Popen | None = None
    cross_handle: Any | None = None
    cross_complete = False
    cross_gate_payloads: list[dict[str, Any]] = []
    failed = False
    worker_records = {
        int(value["shard_index"]): value for value in state["formal_workers"]
    }
    state["cross_gpu_gate"] = {
        "physical_gpu": cross_gpu,
        "status": "pending_memory",
    }
    while pending or active or not cross_complete:
        for shard, (process, handle, gpu) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            del active[shard]
            record = worker_records[shard]
            record["returncode"] = returncode
            record["completed_at"] = utc_now()
            record["status"] = "complete" if returncode == 0 else "failed"
            if returncode != 0:
                failed = True
                state.update(
                    {
                        "status": "failed",
                        "stage": "collection",
                        "failed_worker": record["worker"],
                        "returncode": returncode,
                    }
                )
        if cross_process is not None and cross_process.poll() is not None:
            cross_returncode = int(cross_process.returncode)
            if cross_handle is not None:
                cross_handle.close()
            cross_process = None
            cross_handle = None
            state["cross_gpu_gate"].update(
                {
                    "returncode": cross_returncode,
                    "worker_completed_at": utc_now(),
                }
            )
            if cross_returncode != 0:
                failed = True
                state.update(
                    {
                        "status": "failed",
                        "stage": "cross_gpu_gate",
                        "returncode": cross_returncode,
                    }
                )
                state["cross_gpu_gate"]["status"] = "failed"
            else:
                cross_gate_payloads = []
                for problem in gate_problems:
                    problem_id = str(problem["problem_id"])
                    destination = output / f"CROSS_GPU_GATE_{problem_id}.json"
                    command = [
                        str(args.python),
                        "scripts/audit_deterministic_collection_pair_v1.py",
                        "--left",
                        str(
                            gate_artifact(
                                output / "determinism_gate" / "same_gpu_repeat0",
                                problem,
                            )
                        ),
                        "--right",
                        str(
                            gate_artifact(
                                output
                                / "determinism_gate"
                                / f"cross_gpu{cross_gpu}",
                                problem,
                            )
                        ),
                        "--gpu-mode",
                        "distinct",
                        "--output",
                        str(destination),
                    ]
                    log_name = f"cross_gpu_gate_audit_{problem_id}.log"
                    if run_logged(
                        command,
                        output / "logs" / log_name,
                        deterministic_environment(config),
                    ) != 0:
                        failed = True
                        state.update({"status": "failed", "stage": log_name})
                        state["cross_gpu_gate"]["status"] = "failed"
                        break
                    cross_gate_payloads.append(
                        json.loads(destination.read_text(encoding="utf-8"))
                    )
                if not failed:
                    cross_complete = True
                    state["cross_gpu_gate"]["status"] = "complete"
                    state["cross_gpu_gate"]["completed_at"] = utc_now()
                    state["stages"].append("cross_gpu_exact_gate")
                    state["status"] = "collecting"
                    combined = {
                        "status": "complete",
                        "all_exact": True,
                        "gate_placement": state["gate_placement"],
                        "same_gpu_repeat": same_gate_payloads,
                        "cross_gpu": cross_gate_payloads,
                        "protocol_alignment": json.loads(
                            alignment.read_text(encoding="utf-8")
                        ),
                        "risk_matrix_sha256": sha256_file(args.risk_gate_audit),
                        "risk_matrix_recommended_profile": risk_gate[
                            "recommended_profile"
                        ],
                        "created_at": utc_now(),
                    }
                    atomic_json(combined, output / "DETERMINISM_GATE.json")
        roots = {process.pid for process, _, _ in active.values()}
        if cross_process is not None:
            roots.add(cross_process.pid)
        snapshot = gpu_resource_snapshot(
            physical_gpus,
            roots,
            int(scheduler["required_memory_mib_per_worker_slot"]),
            int(scheduler["max_workers_per_gpu"]),
        )
        active_per_gpu = {
            gpu: sum(1 for _, _, local_gpu in active.values() if local_gpu == gpu)
            for gpu in physical_gpus
        }
        if cross_process is not None:
            active_per_gpu[cross_gpu] += 1
        state["scheduler"]["last_gpu_snapshot"] = snapshot
        state["scheduler"]["poll_count"] += 1
        if (
            not failed
            and not cross_complete
            and cross_process is None
            and active_per_gpu[cross_gpu] < int(snapshot[cross_gpu]["worker_capacity"])
        ):
            cross_root = output / "determinism_gate" / f"cross_gpu{cross_gpu}"
            command = collector_command(
                args.python,
                args.config,
                args.prepared_root,
                args.model_path,
                cross_root,
                cross_gpu,
                float(scheduler["worker_gpu_memory_utilization"]),
                f"gate_cross_gpu{cross_gpu}",
                0,
                1,
                args.profile,
                problem_ids=gate_problem_ids,
            )
            log = output / "logs" / f"gate_cross_gpu{cross_gpu}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cross_handle = log.open("a", encoding="utf-8")
            cross_handle.write("COMMAND " + json.dumps(command) + "\n")
            cross_handle.flush()
            cross_process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=deterministic_environment(config, cross_gpu),
                stdout=cross_handle,
                stderr=subprocess.STDOUT,
            )
            active_per_gpu[cross_gpu] += 1
            state["cross_gpu_gate"].update(
                {
                    "status": "running",
                    "pid": cross_process.pid,
                    "started_at": utc_now(),
                }
            )
        if not failed and pending:
            allowed_formal_gpus = {primary_gpu}
            if cross_complete:
                allowed_formal_gpus.add(cross_gpu)
            launched = True
            while pending and launched:
                launched = False
                for gpu in sorted(
                    allowed_formal_gpus,
                    key=lambda item: (active_per_gpu[item], item),
                ):
                    if not pending or active_per_gpu[gpu] >= int(
                        snapshot[gpu]["worker_capacity"]
                    ):
                        continue
                    shard = pending.pop(0)
                    worker = f"formal_shard{shard}"
                    command = collector_command(
                        args.python,
                        args.config,
                        args.prepared_root,
                        args.model_path,
                        output,
                        gpu,
                        float(scheduler["worker_gpu_memory_utilization"]),
                        worker,
                        shard,
                        int(scheduler["num_shards"]),
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
                    active[shard] = (process, handle, gpu)
                    active_per_gpu[gpu] += 1
                    record = worker_records[shard]
                    record.update(
                        {
                            "physical_gpu": gpu,
                            "pid": process.pid,
                            "status": "running",
                            "started_at": utc_now(),
                        }
                    )
                    launched = True
        atomic_json(state, manifest_path)
        if failed and not active and cross_process is None:
            raise SystemExit(2)
        if pending or active or not cross_complete:
            time.sleep(int(scheduler["poll_seconds"]))
    state["stages"].append("adaptive_two_gpu_collection")
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

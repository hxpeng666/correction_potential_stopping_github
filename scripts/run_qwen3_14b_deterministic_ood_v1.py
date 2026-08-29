#!/usr/bin/env python3
"""Gate, launch and audit the two-A100 Qwen3-14B collection."""
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

from src.reproducibility import code_provenance, deterministic_subprocess_environment, sha256_json


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def command(
    python: Path,
    config: Path,
    prepared_root: Path,
    model_path: Path,
    output_root: Path,
    gpu: int,
    worker: str,
    shard: int,
    num_shards: int,
    *,
    problem_id: str | None = None,
) -> list[str]:
    value = [
        str(python),
        "scripts/collect_qwen3_14b_deterministic_ood_v1.py",
        "--config", str(config),
        "--prepared-root", str(prepared_root),
        "--model-path", str(model_path),
        "--output-root", str(output_root),
        "--gpu", str(gpu),
        "--worker-id", worker,
        "--shard-index", str(shard),
        "--num-shards", str(num_shards),
        "--resume",
    ]
    if problem_id is not None:
        value.extend(("--problem-id", problem_id))
    return value


def run_logged(value: list[str], log: Path, environment: dict[str, str]) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(value) + "\n")
        handle.flush()
        return subprocess.run(value, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT).returncode


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
    output = args.output_root.resolve()
    identity = code_provenance(
        ROOT,
        (
            "configs/qwen3_14b_deterministic_ood13k_v1.yaml",
            "scripts/run_qwen3_14b_deterministic_ood_v1.py",
            "scripts/collect_qwen3_14b_deterministic_ood_v1.py",
            "scripts/audit_qwen3_14b_deterministic_ood_v1.py",
            "scripts/audit_deterministic_collection_pair_v1.py",
            "src/reproducibility.py",
        ),
    )
    invocation = {
        "config": str(args.config.resolve()),
        "prepared_root": str(args.prepared_root.resolve()),
        "model_path": str(args.model_path.resolve()),
        "output_root": str(output),
        "python": str(args.python.resolve()),
        "git_commit": identity["git"]["commit"],
    }
    invocation_fingerprint = sha256_json(invocation)
    manifest_path = output / "RUN_MANIFEST.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("invocation_fingerprint") != invocation_fingerprint:
            raise RuntimeError("refusing to reuse output root for a different invocation")
    output.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "determinism_gate",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "invocation": invocation,
        "invocation_fingerprint": invocation_fingerprint,
        "code_identity": identity,
        "stages": [],
    }
    atomic_json(state, manifest_path)
    environment = deterministic_subprocess_environment(seed=0)

    gate_problem = str(config["determinism_gate"]["problem_id"])
    gate_processes = []
    for gpu in (0, 1):
        gate_root = output / "determinism_gate" / f"gpu{gpu}"
        gate_command = command(
            args.python, args.config, args.prepared_root, args.model_path,
            gate_root, gpu, f"gate_gpu{gpu}", 0, 1, problem_id=gate_problem,
        )
        log = output / "logs" / f"gate_gpu{gpu}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("a", encoding="utf-8")
        handle.write("COMMAND " + json.dumps(gate_command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            gate_command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT
        )
        gate_processes.append((process, handle))
    gate_codes = []
    for process, handle in gate_processes:
        gate_codes.append(process.wait())
        handle.close()
    if gate_codes != [0, 0]:
        state.update({"status": "failed", "stage": "determinism_gate", "returncodes": gate_codes})
        atomic_json(state, manifest_path)
        raise SystemExit(2)
    relative = Path("cache") / "math" / "probe_train" / f"sample_{gate_problem}.pt"
    gate_audit = output / "DETERMINISM_GATE.json"
    audit_command = [
        str(args.python), "scripts/audit_deterministic_collection_pair_v1.py",
        "--left", str(output / "determinism_gate" / "gpu0" / relative),
        "--right", str(output / "determinism_gate" / "gpu1" / relative),
        "--output", str(gate_audit),
    ]
    if run_logged(audit_command, output / "logs" / "gate_audit.log", environment) != 0:
        state.update({"status": "failed", "stage": "determinism_gate_audit"})
        atomic_json(state, manifest_path)
        raise SystemExit(2)
    state["stages"].append("cross_gpu_exact_gate")
    state["status"] = "collecting"
    atomic_json(state, manifest_path)

    formal_processes = []
    for gpu in (0, 1):
        worker = f"formal_gpu{gpu}"
        formal_command = command(
            args.python, args.config, args.prepared_root, args.model_path,
            output, gpu, worker, gpu, 2,
        )
        log = output / "logs" / f"{worker}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("a", encoding="utf-8")
        handle.write("COMMAND " + json.dumps(formal_command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            formal_command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT
        )
        formal_processes.append((process, handle))
    returncodes = []
    for process, handle in formal_processes:
        returncodes.append(process.wait())
        handle.close()
    if returncodes != [0, 0]:
        state.update({"status": "failed", "stage": "collection", "returncodes": returncodes})
        atomic_json(state, manifest_path)
        raise SystemExit(2)
    state["stages"].append("two_gpu_collection")
    state["status"] = "auditing"
    atomic_json(state, manifest_path)

    collection_audit = [
        str(args.python), "scripts/audit_qwen3_14b_deterministic_ood_v1.py",
        "--config", str(args.config),
        "--prepared-root", str(args.prepared_root),
        "--output-root", str(output),
        "--gate-audit", str(gate_audit),
    ]
    if run_logged(collection_audit, output / "logs" / "audit.log", environment) != 0:
        state.update({"status": "failed", "stage": "collection_audit"})
        atomic_json(state, manifest_path)
        raise SystemExit(2)
    state["stages"].append("collection_audit")
    state["status"] = "complete"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
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

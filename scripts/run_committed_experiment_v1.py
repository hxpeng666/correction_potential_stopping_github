#!/usr/bin/env python3
"""Launch a formal experiment only from a clean, identifiable Git commit.

This generic wrapper is the fallback for experiments without a dedicated
runner.  Dedicated runners should enforce the same contract internally.
"""
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

from src.reproducibility import (
    code_provenance,
    deterministic_subprocess_environment,
    sha256_file,
    sha256_json,
)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to execute, conventionally after --",
    )
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("missing experiment command")

    identity = code_provenance(
        ROOT,
        ("scripts/run_committed_experiment_v1.py", "src/reproducibility.py"),
    )
    configs = {
        str(path.resolve()): sha256_file(path.resolve()) for path in args.config
    }
    invocation = {
        "name": args.name,
        "git_commit": identity["git"]["commit"],
        "command": command,
        "configs": configs,
    }
    fingerprint = sha256_json(invocation)
    destination = args.output_root.resolve()
    manifest = destination / "RUN_MANIFEST.json"
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("invocation_fingerprint") != fingerprint:
            raise RuntimeError(
                f"refusing to reuse {destination} for a different committed invocation"
            )
    destination.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "invocation_fingerprint": fingerprint,
        "invocation": invocation,
        "code_identity": identity,
        "determinism_environment": deterministic_subprocess_environment(seed=0),
    }
    # Do not serialize the full inherited environment, which may contain
    # credentials.  Record only the frozen scientific variables.
    state["determinism_environment"] = {
        key: state["determinism_environment"][key]
        for key in (
            "PYTHONHASHSEED",
            "CUBLAS_WORKSPACE_CONFIG",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "TOKENIZERS_PARALLELISM",
        )
    }
    atomic_json(state, manifest)
    environment = deterministic_subprocess_environment(seed=0)
    result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    state["returncode"] = int(result.returncode)
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    state["status"] = "complete" if result.returncode == 0 else "failed"
    atomic_json(state, manifest)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

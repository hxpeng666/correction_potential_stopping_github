#!/usr/bin/env python3
"""Run the three full-vLLM collector phases in isolated Python processes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "scripts" / "collect_qwen3_14b_vllm_full_v1.py"
PHASES = ("dense", "hidden", "branches")


def main() -> None:
    if "--phase" in sys.argv[1:]:
        raise SystemExit("the phase is controlled by the isolated worker wrapper")
    started = time.time()
    for phase in PHASES:
        command = [sys.executable, str(COLLECTOR), *sys.argv[1:], "--phase", phase]
        print(
            json.dumps(
                {
                    "status": "starting_isolated_phase",
                    "phase": phase,
                    "pid": os.getpid(),
                    "command": command,
                }
            ),
            flush=True,
        )
        completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
    print(
        json.dumps(
            {
                "status": "isolated_worker_complete",
                "phases": list(PHASES),
                "elapsed_seconds": time.time() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the frozen 6 schedule x 4 target GSM8K probe matrix on a GPU pool."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = ("sentence", "fixed_budget", "prefix_stride", "lynx_cue", "paragraph", "hybrid")
TARGETS = (("correctness", "correctness"), ("consistency", "consistency"), ("last_switch", "last_switch"), ("correction_bce", "correction"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), default="gsm8k")
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    # Avoid importing the ML stack in the lightweight controller.
    import yaml
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = ROOT / config["output_root"]
    audit = json.loads((output_root / "cache_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise RuntimeError("cache audit is not complete")
    if args.dataset == "gsm8k":
        job_values = [(schedule, label, method) for schedule in SCHEDULES for label, method in TARGETS]
    else:
        method_by_label = dict(TARGETS)
        job_values = [
            (str(row["schedule"]), str(row["target"]), method_by_label[str(row["target"])])
            for row in config["comparison"]["mmlu_pro_combinations"]
        ]
    jobs = deque(job_values)
    # Treat each comma-separated entry as an independent launch slot.  Repeated
    # physical GPU ids intentionally allow several lightweight probes to share
    # a large-memory GPU (for example ``--gpus 1,1,1,2,3``).
    gpu_pool = deque(
        enumerate(int(value) for value in args.gpus.split(",") if value.strip())
    )
    running: dict[int, tuple[subprocess.Popen, object, int, str, str]] = {}
    failures: list[dict[str, object]] = []
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while jobs or running:
        while jobs and gpu_pool:
            slot, gpu = gpu_pool.popleft()
            schedule, label, method = jobs.popleft()
            destination = output_root / "probes" / schedule / label
            log_path = logs / f"probe_{schedule}_{label}.log"
            handle = log_path.open("a", encoding="utf-8")
            command = [
                args.python, "scripts/train_controlled_label_2566_v3.py",
                "--dataset", args.dataset, "--config", str(config_path),
                "--raw-root", str(output_root / "cache" / schedule),
                "--output", str(destination), "--method", method,
                "--seed", "0", "--gpu", "0", "--schedule", "sentence",
                "--actual-schedule-label", schedule,
                "--layer", "20", "--feature-kind", "full_no_delta", "--loss", "bce", "--resume",
            ]
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            running[slot] = (process, handle, gpu, schedule, label)
            print(json.dumps({"status": "launched", "slot": slot, "gpu": gpu, "pid": process.pid, "schedule": schedule, "target": label}), flush=True)
        time.sleep(args.poll_seconds)
        for slot, (process, handle, gpu, schedule, label) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del running[slot]
            gpu_pool.append((slot, gpu))
            event = {"status": "complete" if code == 0 else "failed", "slot": slot, "gpu": gpu, "returncode": code, "schedule": schedule, "target": label}
            print(json.dumps(event), flush=True)
            if code != 0:
                failures.append(event)
    summary = {"status": "complete" if not failures else "failed", "dataset": args.dataset, "jobs": len(job_values), "job_values": job_values, "failures": failures}
    (output_root / "probe_matrix_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

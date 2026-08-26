#!/usr/bin/env python3
"""Run GSM8K-frozen normalized-trajectory candidates on MMLU-Pro."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = ("paragraph", "sentence", "lynx_cue")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_root = ROOT / config["output_root"]
    audit = json.loads((source_root / "cache_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise RuntimeError("MMLU-Pro source cache audit is not complete")
    if any(schedule not in audit["schedules"] for schedule in SCHEDULES):
        raise RuntimeError("one or more frozen candidate schedules lack MMLU-Pro cache")

    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = deque(SCHEDULES)
    gpu_pool = deque(
        enumerate(int(value) for value in args.gpus.split(",") if value.strip())
    )
    running: dict[int, tuple[subprocess.Popen, object, int, str]] = {}
    failures: list[dict[str, object]] = []
    while jobs or running:
        while jobs and gpu_pool:
            slot, gpu = gpu_pool.popleft()
            schedule = jobs.popleft()
            destination = output_root / "probes" / schedule / "correction_trajectory_normalized"
            log_path = logs / f"probe_{schedule}_correction_trajectory_normalized.log"
            handle = log_path.open("a", encoding="utf-8")
            command = [
                args.python,
                "scripts/train_controlled_label_normalized_trajectory_v1.py",
                "--dataset", "mmlu_pro",
                "--config", str(config_path),
                "--raw-root", str(source_root / "cache" / schedule),
                "--output", str(destination),
                "--method", "correction",
                "--seed", "0",
                "--gpu", "0",
                "--schedule", "sentence",
                "--actual-schedule-label", schedule,
                "--layer", "20",
                "--feature-kind", "full_no_delta",
                "--loss", "bce_traj",
                "--trajectory-aggregation", "normalized_softmin",
                "--trajectory-beta", "0.5",
                "--trajectory-weight", "1",
                "--resume",
            ]
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[slot] = (process, handle, gpu, schedule)
            print(json.dumps({
                "status": "launched", "slot": slot, "gpu": gpu,
                "pid": process.pid, "schedule": schedule,
            }), flush=True)
        time.sleep(args.poll_seconds)
        for slot, (process, handle, gpu, schedule) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            del running[slot]
            gpu_pool.append((slot, gpu))
            event = {
                "status": "complete" if returncode == 0 else "failed",
                "slot": slot, "gpu": gpu, "returncode": returncode,
                "schedule": schedule,
            }
            print(json.dumps(event), flush=True)
            if returncode != 0:
                failures.append(event)
    summary = {
        "status": "complete" if not failures else "failed",
        "dataset": "mmlu_pro",
        "target": "correction_trajectory_normalized",
        "trajectory_aggregation": "normalized_softmin",
        "trajectory_beta": 0.5,
        "trajectory_weight": 1.0,
        "selection_frozen_on": "GSM8K calibration only",
        "schedules": list(SCHEDULES),
        "failures": failures,
    }
    (output_root / "probe_matrix_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

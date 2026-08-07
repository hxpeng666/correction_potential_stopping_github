from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    with value.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def atomic_json(payload: Any, path: str | Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                     prefix=target.name, suffix=".tmp", delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=json_default)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(target)
    return target


def json_default(value: Any):
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def add_common_args(parser: argparse.ArgumentParser, default_config: str) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int)
    return parser


def command_record(script: str, config: dict[str, Any], args: argparse.Namespace) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = Path(script).stem
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "argv": vars(args),
        "config": config,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    return atomic_json(payload, ROOT / "logs" / f"{name}_{stamp}_{os.getpid()}.json")


def completed(path: str | Path, resume: bool) -> bool:
    target = Path(path)
    if not target.is_absolute(): target = ROOT / target
    if not resume or not target.exists(): return False
    try: return json.loads(target.read_text()).get("status") in {"complete", "ok", "PASS", "SKIPPED"}
    except (json.JSONDecodeError, OSError): return False


def skipped(path: str | Path, phase: str, reason: str, upstream: Any = None) -> dict[str, Any]:
    payload = {"status": "SKIPPED", "phase": phase, "reason": reason,
               "upstream": upstream, "timestamp": datetime.now(timezone.utc).isoformat()}
    atomic_json(payload, path)
    return payload


def gpu_snapshot(index: int | None = None) -> dict[str, Any]:
    result = {"available": torch.cuda.is_available(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if not torch.cuda.is_available(): return result
    device = torch.cuda.current_device() if index is None else index
    props = torch.cuda.get_device_properties(device)
    result.update({"logical_id": device, "name": props.name, "total_memory_bytes": props.total_memory,
                   "allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device)})
    return result


def gpu_telemetry(physical_index: int) -> dict[str, Any]:
    try:
        fields = "temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu"
        line = subprocess.check_output(
            ["nvidia-smi", f"--id={physical_index}", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip().splitlines()[0]
        values = [value.strip() for value in line.split(",")]
        return dict(zip(("temperature_c", "power_w", "memory_used_mib", "memory_total_mib", "utilization_percent"), values))
    except Exception as error:
        return {"telemetry_error": f"{type(error).__name__}: {error}"}

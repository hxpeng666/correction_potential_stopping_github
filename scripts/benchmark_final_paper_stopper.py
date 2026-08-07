#!/usr/bin/env python3
"""测量初始化后的 5126→384→96→1 检查点模块开销。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.final_paper_probe import FinalPaperProbe
from src.final_paper_protocol import BOUNDARY
from src.utils import atomic_json, seed_everything


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed_everything(20260803)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model = FinalPaperProbe(5126).to(device).eval()
    current = torch.randn(2560, device=device, dtype=torch.float32)
    previous = torch.randn(2560, device=device, dtype=torch.float32)
    mean = torch.zeros(5126, device=device)
    scale = torch.ones(5126, device=device)
    decoded_tail = "The intermediate result is 42."
    previous_checkpoint = 240

    def checkpoint() -> torch.Tensor:
        is_scheduled = (
            64 <= 256 <= 768
            and 256 - previous_checkpoint >= 8
            and any(match.end() == len(decoded_tail) for match in BOUNDARY.finditer(decoded_tail))
        )
        if not is_scheduled:
            raise RuntimeError("benchmark schedule fixture is invalid")
        delta = current - previous
        delta_norm = torch.linalg.vector_norm(delta).reshape(1)
        current_norm = torch.linalg.vector_norm(current).reshape(1)
        cosine = (current @ delta / (current_norm * delta_norm).clamp_min(1e-6)).reshape(1)
        position = torch.tensor([256.0], device=device)
        feature = torch.cat(
            [
                current,
                delta,
                position,
                torch.log1p(position),
                torch.tensor([16.0, 1.2], device=device),
                delta_norm,
                cosine,
            ]
        )
        score = torch.sigmoid(model(((feature - mean) / scale)[None, :]))
        return score <= 0.5

    with torch.inference_mode():
        for _ in range(args.warmup):
            checkpoint()
        torch.cuda.synchronize()
        gpu_ms: list[float] = []
        wall_ms: list[float] = []
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            start.record()
            checkpoint()
            end.record()
            end.synchronize()
            wall_ms.append(1000.0 * (time.perf_counter() - wall_start))
            gpu_ms.append(float(start.elapsed_time(end)))
    payload = {
        "status": "complete",
        "seed": 20260803,
        "device": torch.cuda.get_device_name(args.gpu),
        "architecture": [5126, 384, 96, 1],
        "initialized_weights_valid_for_timing": True,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "cuda_event": summary(gpu_ms),
        "synchronized_wall": summary(wall_ms),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    atomic_json(payload, output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

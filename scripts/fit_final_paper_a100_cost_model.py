#!/usr/bin/env python3
"""使用仅用于计时的样本拟合并验证单请求回放成本模型。

只使用明确标记为 ``timing_valid=true`` 的产物，且绝不扫描留出任务划分。
脚本执行确定性的样本级 80/20 拟合—验证划分，并报告预填充、令牌解码、
完整推理总时间和检查点前缀时间的误差。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.utils import atomic_json


def design(context: np.ndarray) -> np.ndarray:
    scaled = context.astype(np.float64) / 4096.0
    return np.column_stack([np.ones_like(scaled), scaled, scaled * scaled])


def fit_observations(values: list[tuple[float, float]]) -> dict[str, Any]:
    if not values:
        raise ValueError("no timing observations")
    x = np.asarray([row[0] for row in values], dtype=np.float64)
    y = np.asarray([row[1] for row in values], dtype=np.float64)
    matrix = design(x)
    ridge = np.eye(matrix.shape[1]) * 1e-8
    coefficients = np.linalg.solve(matrix.T @ matrix + ridge, matrix.T @ y)
    predicted = np.maximum(matrix @ coefficients, 0.001)
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = (
        1.0 - float(np.sum((y - predicted) ** 2)) / denominator
        if denominator > 0
        else 1.0
    )
    return {
        "coefficients": coefficients.tolist(),
        "context_scale": 4096.0,
        "minimum_prediction_ms": 0.001,
        "observations": len(values),
        "context_min": float(x.min()),
        "context_max": float(x.max()),
        "target_mean_ms": float(y.mean()),
        "fit_r2": r2,
    }


def predict(model: dict[str, Any], context: int | float) -> float:
    value = float(context) / float(model["context_scale"])
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    result = float(coefficients @ np.asarray([1.0, value, value * value]))
    return max(result, float(model["minimum_prediction_ms"]))


def autoregressive_cost(model: dict[str, Any], context: int, steps: int) -> float:
    return float(
        sum(predict(model, context + offset) for offset in range(max(0, steps)))
    )


def error_summary(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    if not actual:
        return {"observations": 0}
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    absolute = np.abs(p - y)
    relative = absolute / np.maximum(np.abs(y), 1e-6)
    return {
        "observations": len(y),
        "mae_ms": float(absolute.mean()),
        "mape_percent": float(100.0 * relative.mean()),
        "p95_relative_error_percent": float(100.0 * np.percentile(relative, 95)),
        "bias_ms": float((p - y).mean()),
    }


def fold(problem_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{problem_id}".encode()).digest()
    return "validation" if int.from_bytes(digest[:8], "big") % 5 == 0 else "fit"


def parse_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use DATASET=PATH, e.g. gsm8k=results/cache/gsm8k")
    dataset, raw = value.split("=", 1)
    if dataset not in {"gsm8k", "mmlu"}:
        raise argparse.ArgumentTypeError(f"unknown dataset {dataset}")
    path = Path(raw)
    return dataset, path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help="每个时间缓存传入一次；只扫描探针训练集和校准集。",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    sample_artifacts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded = defaultdict(int)
    for raw_root in args.cache_root:
        dataset, cache_root = parse_root(raw_root)
        for split in ("probe_train", "calibration"):
            for path in sorted((cache_root / "dense" / split).glob("sample_*.pt")):
                artifact = torch.load(path, map_location="cpu", weights_only=False)
                if artifact.get("status") != "complete":
                    excluded["incomplete"] += 1
                    continue
                if not bool(artifact.get("timing_valid", False)):
                    excluded["timing_invalid"] += 1
                    continue
                dense = artifact["dense"]
                times = [float(value) for value in dense.get("decode_cuda_ms", [])]
                if (
                    not np.isfinite(float(dense.get("prefill_cuda_ms", np.nan)))
                    or not times
                    or not np.isfinite(times).all()
                ):
                    excluded["missing_timing_fields"] += 1
                    continue
                artifact["_dataset_for_timing"] = dataset
                sample_artifacts[str(artifact["collection_device"])].append(artifact)
    if not sample_artifacts:
        raise FileNotFoundError("no complete timing_valid Dense artifacts were found")

    models: dict[str, Any] = {}
    for device, artifacts in sorted(sample_artifacts.items()):
        fit_samples = [
            value
            for value in artifacts
            if fold(str(value["problem_id"]), args.seed) == "fit"
        ]
        validation_samples = [
            value
            for value in artifacts
            if fold(str(value["problem_id"]), args.seed) == "validation"
        ]
        if not fit_samples or not validation_samples:
            raise ValueError(
                f"device {device} needs nonempty fit and validation samples; "
                f"found {len(fit_samples)}/{len(validation_samples)}"
            )
        prefill_observations: list[tuple[float, float]] = []
        decode_observations: list[tuple[float, float]] = []
        for artifact in fit_samples:
            prompt = int(artifact["prompt_tokens"])
            dense = artifact["dense"]
            prefill_observations.append((prompt, float(dense["prefill_cuda_ms"])))
            times = [float(value) for value in dense["decode_cuda_ms"]]
            stride = max(1, len(times) // 256)
            decode_observations.extend(
                (prompt + index + 1, times[index])
                for index in range(0, len(times), stride)
            )
        prefill_model = fit_observations(prefill_observations)
        decode_model = fit_observations(decode_observations)

        prefill_actual: list[float] = []
        prefill_predicted: list[float] = []
        decode_actual: list[float] = []
        decode_predicted: list[float] = []
        dense_actual: list[float] = []
        dense_predicted: list[float] = []
        checkpoint_actual: list[float] = []
        checkpoint_predicted: list[float] = []
        validation_ids = []
        for artifact in validation_samples:
            validation_ids.append(str(artifact["problem_id"]))
            prompt = int(artifact["prompt_tokens"])
            dense = artifact["dense"]
            prefill_ms = float(dense["prefill_cuda_ms"])
            times = [float(value) for value in dense["decode_cuda_ms"]]
            prefill_actual.append(prefill_ms)
            prefill_predicted.append(predict(prefill_model, prompt))
            stride = max(1, len(times) // 256)
            for index in range(0, len(times), stride):
                decode_actual.append(times[index])
                decode_predicted.append(predict(decode_model, prompt + index + 1))
            dense_actual.append(prefill_ms + sum(times))
            dense_predicted.append(
                predict(prefill_model, prompt)
                + autoregressive_cost(decode_model, prompt + 1, len(times))
            )
            for checkpoint in artifact["schedules"]["fixed"]:
                steps = min(max(0, int(checkpoint) - 1), len(times))
                checkpoint_actual.append(prefill_ms + sum(times[:steps]))
                checkpoint_predicted.append(
                    predict(prefill_model, prompt)
                    + autoregressive_cost(decode_model, prompt + 1, steps)
                )
        validation = {
            "prefill": error_summary(prefill_actual, prefill_predicted),
            "decode_token": error_summary(decode_actual, decode_predicted),
            "dense_total": error_summary(dense_actual, dense_predicted),
            "checkpoint_prefix": error_summary(
                checkpoint_actual,
                checkpoint_predicted,
            ),
        }
        dense_gate = validation["dense_total"]
        checkpoint_gate = validation["checkpoint_prefix"]
        gate_pass = bool(
            dense_gate["mape_percent"] <= 5.0
            and dense_gate["p95_relative_error_percent"] <= 10.0
            and checkpoint_gate["mape_percent"] <= 5.0
            and checkpoint_gate["p95_relative_error_percent"] <= 10.0
        )
        models[device] = {
            "fit_samples": len(fit_samples),
            "validation_samples": len(validation_samples),
            "fit_problem_ids": sorted(str(value["problem_id"]) for value in fit_samples),
            "validation_problem_ids": sorted(validation_ids),
            "prefill": prefill_model,
            "decode_token": decode_model,
            "validation": validation,
            "validation_gate": {
                "mape_percent_max": 5.0,
                "p95_relative_error_percent_max": 10.0,
                "pass": gate_pass,
            },
        }

    devices = sorted(models)
    label = (
        f"{devices[0]} 单请求回放估计延迟"
        if len(devices) == 1
        else "按设备分别计算的单请求回放估计延迟"
    )
    payload = {
        "status": (
            "complete" if all(value["validation_gate"]["pass"] for value in models.values())
            else "validation_failed"
        ),
        "seed": args.seed,
        "sample_split": "deterministic SHA-256 80% fit / 20% validation",
        "scanned_task_splits": ["probe_train", "calibration"],
        "heldout_scanned": False,
        "timing_source": "single-request CUDA events after warm-up",
        "excluded_timing_sources": ["Direct/forced branch worker timing"],
        "excluded_counts": dict(excluded),
        "latency_label": label,
        "models_by_device": models,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    atomic_json(payload, output)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "complete":
        raise SystemExit(3)


if __name__ == "__main__":
    main()

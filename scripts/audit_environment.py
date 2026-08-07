#!/usr/bin/env python3
"""在收集前写出可移植的环境、模型与数据审计。"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import transformers

from src.final_paper_protocol import canonical_fingerprint
from src.qwen3_reasoning import inspect_qwen3
from src.utils import atomic_json, load_yaml


def nvidia_smi() -> dict:
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip().splitlines()
        return {"driver_versions": sorted(set(driver))}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/Qwen3-4B"))
    parser.add_argument(
        "--gsm8k-config",
        type=Path,
        default=Path("configs/final_paper_replay_v2_gsm8k_fp16.yaml"),
    )
    parser.add_argument(
        "--mmlu-config",
        type=Path,
        default=Path("configs/final_paper_replay_v2_mmlu_fp16.yaml"),
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path("results/final_paper_replay_v2/splits"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path
    splits_root = args.splits_root if args.splits_root.is_absolute() else ROOT / args.splits_root
    configs = {}
    manifests = {}
    for dataset, raw in (
        ("gsm8k", args.gsm8k_config),
        ("mmlu", args.mmlu_config),
    ):
        path = raw if raw.is_absolute() else ROOT / raw
        config = load_yaml(path)
        configs[dataset] = {
            "path": str(path),
            "fingerprint": canonical_fingerprint(config),
            "model": config["model"],
            "generation": config["generation"],
            "checkpoint_protocol": config["checkpoint_protocol"],
            "seed": config["seed"],
        }
        manifest_path = splits_root / f"{dataset}_split.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[dataset] = {
            "path": str(manifest_path),
            "fingerprint": manifest["fingerprint"],
            "dataset": manifest["dataset"],
            "hub_revision": manifest.get(
                "hub_revision", config["dataset"].get("hub_revision")
            ),
            "file_counts": {
                name: value["count"] for name, value in manifest["files"].items()
            },
        }
    gpus = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "logical_index": index,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "compute_capability": [props.major, props.minor],
            }
        )
    payload = {
        "status": "complete",
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_smi": nvidia_smi(),
        "gpus": gpus,
        "model": inspect_qwen3(model_path),
        "configs": configs,
        "data_manifests": manifests,
        "base_model_frozen": True,
        "quantization": False,
        "global_seed": 20260803,
        "generation_seed_key": [
            "global_seed",
            "dataset",
            "split",
            "sample_id",
            "checkpoint",
        ],
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    atomic_json(payload, output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

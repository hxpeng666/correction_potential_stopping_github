#!/usr/bin/env python3
"""Freeze exact code, data, model metadata, and runtime identities for the run."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy
import pandas
import sklearn
import torch
import transformers
import yaml

from deepseek7b_protocol_v1 import canonical_fingerprint


PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_root = Path(config["model"]["local_path"])
    prepared_root = Path(config["data"]["prepared_root"])
    files = [
        args.config,
        prepared_root / "MANIFEST.json",
        *sorted(prepared_root.glob("*/*.jsonl")),
        model_root / "config.json",
        model_root / "generation_config.json",
        model_root / "model.safetensors.index.json",
        PROJECT / "scripts/deepseek7b_protocol_v1.py",
        PROJECT / "scripts/prepare_deepseek7b_data_v1.py",
        PROJECT / "scripts/collect_deepseek7b_paragraph_v1.py",
        PROJECT / "scripts/train_deepseek7b_ablation_v1.py",
        PROJECT / "scripts/audit_deepseek7b_collection_v1.py",
        PROJECT / "scripts/summarize_deepseek7b_results_v1.py",
        PROJECT / "scripts/test_deepseek7b_probe_pipeline_v1.py",
        PROJECT / "scripts/freeze_deepseek7b_protocol_v1.py",
        PROJECT / "scripts/audit_deepseek7b_completion_v1.py",
        PROJECT / "scripts/migrate_deepseek7b_cache_v2.py",
        PROJECT / "scripts/migrate_deepseek7b_selective_budget_v3.py",
        PROJECT / "scripts/repair_deepseek7b_numeric_labels_v2.py",
        PROJECT / "scripts/evaluate_deepseek7b_ood_v2.py",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing protocol files: {missing}")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    payload = {
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config["protocol_id"],
        "config_fingerprint": canonical_fingerprint(config),
        "files": [identity(path) for path in files],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpus": gpu,
        },
        "math_verifier": {
            "implementation": "dataset-standard semantic equality adapted from Qwen2.5-Math official grader",
            "upstream": "https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/grader.py",
            "local_authoritative_file": str((PROJECT / "scripts/deepseek7b_protocol_v1.py").resolve()),
            "unit_test_log": str((Path(config["output_root"]) / "logs/math_verifier_test.log").resolve()),
        },
        "probe_contract_test": {
            "exit_file": str((Path(config["output_root"]) / "logs/probe_contract_test_after_verifier.exit").resolve()),
            "log": str((Path(config["output_root"]) / "logs/probe_contract_test.log").resolve()),
        },
    }
    target = Path(config["output_root"]) / "PROTOCOL_FREEZE.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({"status": "frozen", "target": str(target), "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pack per-problem token-pooling artifacts into representation-specific split files."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.final_paper_inference import atomic_torch_save
from src.legacy_empirical_probe_normalized_v1 import load_checkpoint_split
from src.utils import atomic_json


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def prompt_by_problem(directory: Path) -> dict[str, np.ndarray]:
    output = {}
    for path in sorted(directory.glob("sample_*.pt")):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        value = artifact.get("prompt_hidden_ln")
        if not torch.is_tensor(value) or tuple(value.shape) != (2560,):
            raise ValueError(f"invalid prompt hidden: {path}")
        output[str(artifact["problem_id"])] = value.float().numpy()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--schedule", default="sentence")
    args = parser.parse_args()
    manifest: dict[str, Any] = {
        "status": "complete",
        "schema": "packed_paragraph_representation_cache_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.raw_root.resolve()),
        "splits": {},
    }
    expected_names = None
    for split in ("probe_train", "calibration", "heldout"):
        source_dir = args.raw_root / split
        first = next(iter(sorted(source_dir.glob("sample_*.pt"))), None)
        if first is None:
            raise FileNotFoundError(source_dir)
        first_artifact = torch.load(first, map_location="cpu", weights_only=False)
        names = [str(value) for value in first_artifact["representation_names"]]
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(f"representation schema changed in {split}")
        frame, hidden, _, fallbacks = load_checkpoint_split(source_dir, args.schedule)
        if hidden.shape[1] != len(names):
            raise ValueError(f"hidden representation mismatch in {split}: {hidden.shape}")
        prompts = prompt_by_problem(source_dir)
        prompt_rows = np.stack(
            [prompts[str(problem_id)] for problem_id in frame.problem_id]
        ).astype(np.float16, copy=False)
        split_manifest = {
            "problems": int(frame.problem_id.nunique()) + len(fallbacks),
            "scorable_problems": int(frame.problem_id.nunique()),
            "fallback_problems": len(fallbacks),
            "rows": len(frame),
            "representations": {},
        }
        for index, name in enumerate(names):
            destination = args.output_root / name / f"{split}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "status": "complete",
                "schema": "packed_paragraph_representation_cache_v1",
                "split": split,
                "representation": name,
                "frame": frame,
                "hidden": torch.from_numpy(hidden[:, index:index + 1]).to(torch.float16),
                "prompt_hidden_ln": torch.from_numpy(prompt_rows),
                "capture_layers": [20],
                "fallbacks": fallbacks,
            }
            identity = {
                "split": split,
                "representation": name,
                "rows": len(frame),
                "problem_ids": frame.problem_id.astype(str).tolist(),
                "checkpoints": frame.checkpoint.astype(int).tolist(),
            }
            payload["packed_fingerprint"] = canonical(identity)
            atomic_torch_save(payload, destination)
            split_manifest["representations"][name] = {
                "path": str(destination.resolve()),
                "bytes": destination.stat().st_size,
                "packed_fingerprint": payload["packed_fingerprint"],
            }
        manifest["splits"][split] = split_manifest
        del frame, hidden, prompt_rows
    manifest["representation_names"] = expected_names
    manifest["manifest_fingerprint"] = canonical(manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest, args.output_root / "PACKED_MANIFEST.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

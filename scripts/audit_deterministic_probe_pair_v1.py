#!/usr/bin/env python3
"""Require exact equality for a deterministic negative-control probe pair."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reproducibility import code_provenance, sha256_file


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load(directory: Path) -> dict[str, Any]:
    report_path = directory / "probe.json"
    model_path = directory / "probe.pt"
    scores_path = directory / "scores.pt"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    scores = torch.load(scores_path, map_location="cpu", weights_only=False)
    return {
        "directory": str(directory.resolve()),
        "report": report,
        "model": model,
        "scores": scores,
        "files": {
            "probe.json": sha256_file(report_path),
            "probe.pt": sha256_file(model_path),
            "scores.pt": sha256_file(scores_path),
        },
    }


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_repro = left["report"]["reproducibility"]
    right_repro = right["report"]["reproducibility"]
    required_checks: dict[str, bool] = {
        "same_git_commit": (
            left_repro["code"]["git"]["commit"]
            == right_repro["code"]["git"]["commit"]
        ),
        "same_determinism_settings": left_repro["settings"] == right_repro["settings"],
        "same_input_identity": left_repro["input"] == right_repro["input"],
        "same_initial_state": (
            left_repro["initial_state_sha256"]
            == right_repro["initial_state_sha256"]
        ),
        "same_final_state_hash": (
            left_repro["final_state_sha256"]
            == right_repro["final_state_sha256"]
        ),
        "same_best_epoch": left["report"]["best_epoch"] == right["report"]["best_epoch"],
        "same_history": left["report"]["history"] == right["report"]["history"],
    }
    left_state = left["model"]["state_dict"]
    right_state = right["model"]["state_dict"]
    required_checks["state_dict_tensor_exact"] = (
        list(left_state) == list(right_state)
        and all(torch.equal(left_state[key], right_state[key]) for key in left_state)
    )
    left_scores = left["scores"]["scores"]
    right_scores = right["scores"]["scores"]
    required_checks["score_tensors_exact"] = (
        set(left_scores) == set(right_scores)
        and all(torch.equal(left_scores[key], right_scores[key]) for key in left_scores)
    )
    score_differences = {}
    for split in sorted(set(left_scores) & set(right_scores)):
        a = left_scores[split].numpy().astype(np.float64)
        b = right_scores[split].numpy().astype(np.float64)
        score_differences[split] = {
            "max_abs": float(np.max(np.abs(a - b))),
            "mean_abs": float(np.mean(np.abs(a - b))),
        }
    # Calibration and held-out policy metrics are intentionally not an
    # invariant of the grader comparison.  The forced-at-cap grader changes
    # Dense final correctness for capped calibration/test trajectories, so the
    # risk curve and selected threshold may legitimately change even when the
    # correctness probe itself is bitwise identical.  Keep this diagnostic in
    # the audit without letting it invalidate the training negative control.
    informational_checks = {
        "same_calibration": (
            left["report"]["calibration"] == right["report"]["calibration"]
        ),
    }
    all_required_exact = all(required_checks.values())
    return {
        "status": "complete" if all_required_exact else "failed",
        "all_exact": all_required_exact,
        "required_checks": required_checks,
        "informational_checks": informational_checks,
        "score_differences": score_differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = code_provenance(
        ROOT,
        (
            "scripts/audit_deterministic_probe_pair_v1.py",
            "src/reproducibility.py",
        ),
    )
    left = load(args.left)
    right = load(args.right)
    result = compare(left, right)
    result.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "left": {"directory": left["directory"], "files": left["files"]},
            "right": {"directory": right["directory"], "files": right["files"]},
            "audit_code_identity": identity,
        }
    )
    atomic_json(result, args.output)
    print(json.dumps(result, indent=2))
    if not result["all_exact"]:
        raise SystemExit("deterministic negative control failed")


if __name__ == "__main__":
    main()

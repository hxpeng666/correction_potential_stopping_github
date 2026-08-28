#!/usr/bin/env python3
"""Run deterministic suffix generation, exact gate, and exit attribution."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reproducibility import code_provenance, deterministic_subprocess_environment


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def command(python: Path, script: str, *arguments: str) -> list[str]:
    return [str(python), str(ROOT / script), *arguments]


def run_logged(cmd: list[str], log: Path, environment: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(cmd) + "\n")
        handle.flush()
        subprocess.run(cmd, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)


def only_artifact(root: Path) -> Path:
    paths = sorted(root.glob("*/*/*.pt"))
    if len(paths) != 1:
        raise AssertionError(f"expected one gate artifact in {root}, found {len(paths)}")
    return paths[0]


def branch_signature(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "problem_id": value["problem_id"],
        "records": [
            {
                "checkpoint": row["checkpoint"],
                "variants": {
                    label: {
                        "branch_token_ids": local["branch_token_ids"],
                        "prediction": local["prediction"],
                        "success": local["success"],
                        "complete_boxed": local["complete_boxed"],
                    }
                    for label, local in sorted(row["variants"].items())
                },
            }
            for row in value["records"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = code_provenance(
        ROOT,
        (
            "configs/deepseek7b_deterministic_exit_suffix_v1.yaml",
            "scripts/prepare_deepseek7b_deterministic_exit_suffix_v1.py",
            "scripts/collect_deepseek7b_deterministic_suffix_v1.py",
            "scripts/analyze_deepseek7b_deterministic_exit_suffix_v1.py",
            "scripts/run_deepseek7b_deterministic_exit_suffix_v1.py",
            "src/reproducibility.py",
        ),
    )
    environment = deterministic_subprocess_environment(seed=0)
    manifest_path = output / "SAMPLE_MANIFEST.json"
    run_manifest = {
        "status": "preparing",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "code_identity": provenance,
        "config": str(args.config.resolve()),
        "output_root": str(output),
        "gpu": args.gpu,
        "stages": [],
    }
    atomic_json(run_manifest, output / "RUN_MANIFEST.json")
    try:
        if not manifest_path.is_file():
            run_logged(
                command(
                    args.python,
                    "scripts/prepare_deepseek7b_deterministic_exit_suffix_v1.py",
                    "--config", str(args.config.resolve()),
                    "--output", str(manifest_path),
                ),
                output / "logs/prepare.log",
                environment,
            )
        run_manifest["stages"].append("sample_frozen")
        run_manifest["status"] = "determinism_gate"
        atomic_json(run_manifest, output / "RUN_MANIFEST.json")

        gate_roots = [output / "gate/run_a", output / "gate/run_b"]
        for index, gate_root in enumerate(gate_roots):
            run_logged(
                command(
                    args.python,
                    "scripts/collect_deepseek7b_deterministic_suffix_v1.py",
                    "--config", str(args.config.resolve()),
                    "--manifest", str(manifest_path),
                    "--output-root", str(gate_root),
                    "--gpu", str(args.gpu),
                    "--worker-id", f"gate_{index}",
                    "--limit", "1",
                    "--resume",
                ),
                output / f"logs/gate_{index}.log",
                environment,
            )
        left, right = branch_signature(only_artifact(gate_roots[0])), branch_signature(only_artifact(gate_roots[1]))
        gate = {
            "status": "complete",
            "all_exact": left == right,
            "comparison": "branch token ids, parsed predictions, correctness, and boxed detection",
            "left": left,
            "right": right,
        }
        atomic_json(gate, output / "DETERMINISM_GATE.json")
        if not gate["all_exact"]:
            raise AssertionError("suffix generation determinism gate failed")
        run_manifest["stages"].append("determinism_gate_exact")
        run_manifest["status"] = "collecting_suffixes"
        atomic_json(run_manifest, output / "RUN_MANIFEST.json")

        run_logged(
            command(
                args.python,
                "scripts/collect_deepseek7b_deterministic_suffix_v1.py",
                "--config", str(args.config.resolve()),
                "--manifest", str(manifest_path),
                "--output-root", str(output / "branches"),
                "--gpu", str(args.gpu),
                "--worker-id", "formal_gpu",
                "--resume",
            ),
            output / "logs/collect.log",
            environment,
        )
        run_manifest["stages"].append("suffix_collection")
        run_manifest["status"] = "analyzing"
        atomic_json(run_manifest, output / "RUN_MANIFEST.json")
        run_logged(
            command(
                args.python,
                "scripts/analyze_deepseek7b_deterministic_exit_suffix_v1.py",
                "--config", str(args.config.resolve()),
                "--experiment-root", str(output),
                "--output", str(output / "analysis"),
            ),
            output / "logs/analyze.log",
            environment,
        )
        run_manifest["stages"].append("analysis_and_audit")
        run_manifest["status"] = "complete"
        run_manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(run_manifest, output / "RUN_MANIFEST.json")
        atomic_json(
            {
                "status": "complete",
                "completed_at": run_manifest["completed_at"],
                "git_commit": provenance["git"]["commit"],
                "analysis": str(output / "analysis/RESULTS.json"),
                "audit": str(output / "analysis/AUDIT.json"),
            },
            output / "EXPERIMENT_COMPLETE.json",
        )
    except Exception as error:
        run_manifest["status"] = "failed"
        run_manifest["failed_at"] = datetime.now(timezone.utc).isoformat()
        run_manifest["error_type"] = type(error).__name__
        run_manifest["error"] = str(error)
        atomic_json(run_manifest, output / "RUN_MANIFEST.json")
        raise


if __name__ == "__main__":
    main()


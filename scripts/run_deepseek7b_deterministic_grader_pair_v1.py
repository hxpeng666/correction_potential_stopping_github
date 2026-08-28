#!/usr/bin/env python3
"""Run a strictly paired original-vs-forced-cap grader experiment.

Correctness is executed first as a negative control.  The pipeline aborts unless
the two independently trained correctness probes are bitwise identical, because
that target and all of its inputs are grader-invariant by construction.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reproducibility import code_provenance, deterministic_subprocess_environment


METHODS = (
    ("correctness", "correctness", "bce"),
    ("consistency", "consistency", "bce"),
    ("last_switch", "last_switch", "bce"),
    ("bce", "correction", "bce"),
    ("bce_traj", "correction", "bce_traj"),
)
ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.1)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str], log: Path, environment: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def train_command(
    *, condition_root: Path, data_root: Path, dataset: str, label: str,
    method: str, loss: str, config: Path, gpu: int,
) -> list[str]:
    training_dataset = "gsm8k" if dataset == "gsm8k" else "math"
    heldout_dataset = "gsm8k" if dataset == "gsm8k" else "math500"
    return [
        sys.executable,
        "scripts/train_deepseek7b_ablation_v1.py",
        "--dataset", dataset,
        "--config", str(config),
        "--raw-root", str(data_root / training_dataset),
        "--heldout-root", str(data_root / heldout_dataset),
        "--output", str(condition_root / "probes" / dataset / label),
        "--method", method,
        "--seed", "0",
        "--gpu", str(gpu),
        "--schedule", "sentence",
        "--actual-schedule-label", "paragraph",
        "--layer", "16",
        "--feature-kind", "full_no_delta",
        "--loss", loss,
        "--trajectory-aggregation", "normalized_softmin",
        "--trajectory-beta", "0.5",
        "--trajectory-weight", "1.0",
        "--resume",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original-data-root", type=Path, required=True)
    parser.add_argument("--forced-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    environment = deterministic_subprocess_environment(seed=0)
    code_identity = code_provenance(
        ROOT,
        (
            "scripts/run_deepseek7b_deterministic_grader_pair_v1.py",
            "scripts/train_deepseek7b_ablation_v1.py",
            "scripts/evaluate_deepseek7b_ood_v2.py",
            "scripts/recalibrate_deepseek7b_five_ablation_ltt_v1.py",
            "scripts/audit_deterministic_probe_pair_v1.py",
            "src/reproducibility.py",
        ),
    )
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    conditions = {
        "original_grader": args.original_data_root.resolve(),
        "forced_cap_grader": args.forced_data_root.resolve(),
    }
    state = {
        "status": "negative_control",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "code_identity": code_identity,
        "config": str(args.config.resolve()),
        "conditions": {key: str(value) for key, value in conditions.items()},
        "gpu": args.gpu,
        "execution": "single process per GPU; strict deterministic algorithms",
        "completed": [],
    }
    atomic_json(state, output / "RUN_MANIFEST.json")

    # Run correctness twice from grader-specific views and demand bitwise parity.
    for dataset in ("gsm8k", "math500"):
        for condition, data_root in conditions.items():
            condition_root = output / "conditions" / condition
            command = train_command(
                condition_root=condition_root,
                data_root=data_root,
                dataset=dataset,
                label="correctness",
                method="correctness",
                loss="bce",
                config=args.config.resolve(),
                gpu=args.gpu,
            )
            run(command, logs / f"train_{condition}_{dataset}_correctness.log", environment)
            state["completed"].append(f"{condition}/{dataset}/correctness")
            atomic_json(state, output / "RUN_MANIFEST.json")
        audit = output / f"NEGATIVE_CONTROL_{dataset.upper()}.json"
        run(
            [
                sys.executable,
                "scripts/audit_deterministic_probe_pair_v1.py",
                "--left", str(output / "conditions/original_grader/probes" / dataset / "correctness"),
                "--right", str(output / "conditions/forced_cap_grader/probes" / dataset / "correctness"),
                "--output", str(audit),
            ],
            logs / f"audit_negative_control_{dataset}.log",
            environment,
        )
        state["completed"].append(f"negative_control/{dataset}")
        atomic_json(state, output / "RUN_MANIFEST.json")

    state["status"] = "paired_training"
    atomic_json(state, output / "RUN_MANIFEST.json")
    for dataset in ("gsm8k", "math500"):
        for label, method, loss in METHODS[1:]:
            for condition, data_root in conditions.items():
                condition_root = output / "conditions" / condition
                run(
                    train_command(
                        condition_root=condition_root,
                        data_root=data_root,
                        dataset=dataset,
                        label=label,
                        method=method,
                        loss=loss,
                        config=args.config.resolve(),
                        gpu=args.gpu,
                    ),
                    logs / f"train_{condition}_{dataset}_{label}.log",
                    environment,
                )
                state["completed"].append(f"{condition}/{dataset}/{label}")
                atomic_json(state, output / "RUN_MANIFEST.json")

    state["status"] = "ood_evaluation"
    atomic_json(state, output / "RUN_MANIFEST.json")
    for condition, data_root in conditions.items():
        condition_root = output / "conditions" / condition
        for label, _, _ in METHODS:
            run(
                [
                    sys.executable,
                    "scripts/evaluate_deepseek7b_ood_v2.py",
                    "--dataset", "aime",
                    "--source-probe", str(condition_root / "probes/math500" / label),
                    "--heldout-root", str(data_root / "aime"),
                    "--output", str(condition_root / "probes/aime" / label),
                    "--gpu", str(args.gpu),
                    "--resume",
                ],
                logs / f"eval_{condition}_aime_{label}.log",
                environment,
            )
            state["completed"].append(f"{condition}/aime/{label}")
            atomic_json(state, output / "RUN_MANIFEST.json")

    state["status"] = "calibration"
    atomic_json(state, output / "RUN_MANIFEST.json")
    for condition, data_root in conditions.items():
        condition_root = output / "conditions" / condition
        for alpha in ALPHAS:
            tag = str(alpha).replace(".", "p")
            run(
                [
                    sys.executable,
                    "scripts/recalibrate_deepseek7b_five_ablation_ltt_v1.py",
                    "--probe-root", str(condition_root),
                    "--gsm-data-root", str(data_root / "gsm8k"),
                    "--math-data-root", str(data_root / "math"),
                    "--gsm-heldout-root", str(data_root / "gsm8k"),
                    "--math-heldout-root", str(data_root / "math500"),
                    "--aime-heldout-root", str(data_root / "aime"),
                    "--output", str(condition_root / f"ltt_alpha_{tag}"),
                    "--alpha", str(alpha),
                    "--delta", "0.05",
                    "--grid-size", "101",
                ],
                logs / f"ltt_{condition}_{tag}.log",
                environment,
            )
            state["completed"].append(f"{condition}/ltt/{alpha}")
            atomic_json(state, output / "RUN_MANIFEST.json")

    state["status"] = "complete"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(state, output / "RUN_MANIFEST.json")
    atomic_json(
        {
            "status": "complete",
            "completed_at": state["completed_at"],
            "git_commit": code_identity["git"]["commit"],
            "negative_controls": ["gsm8k", "math500"],
            "conditions": list(conditions),
            "alphas": list(ALPHAS),
        },
        output / "EXPERIMENT_COMPLETE.json",
    )


if __name__ == "__main__":
    main()

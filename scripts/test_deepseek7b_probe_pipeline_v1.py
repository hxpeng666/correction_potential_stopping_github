#!/usr/bin/env python3
"""Synthetic end-to-end contract test for the 3590-D probe pipeline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from deepseek7b_protocol_v1 import numeric_value, success


PROJECT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def artifact(
    problem_id: str, split: str, index: int, target: Path, dataset: str = "gsm8k"
) -> None:
    dense_success = index % 3 != 0
    dense_prediction = "1" if dense_success else "0"
    rows = []
    hidden = []
    generator = torch.Generator().manual_seed(1000 + index)
    for local, checkpoint in enumerate((32, 64, 96, 128)):
        current_success = bool((index + local) % 4 != 0)
        current_prediction = "1" if current_success else "0"
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "problem_id": problem_id,
                "checkpoint": checkpoint,
                "checkpoint_schedules": ["sentence"],
                "actual_checkpoint_schedule": "paragraph",
                "gold_answer": "1",
                "dense_prediction": dense_prediction,
                "dense_success": dense_success,
                "dense_tokens": 160,
                "dense_wall_ms": 160.0,
                "current_prediction": current_prediction,
                "current_success": current_success,
                "consistency": current_prediction == dense_prediction,
                "correction": (not current_success) and dense_success,
                "damage": current_success and (not dense_success),
                "branch_tokens": 5,
                "forced_answer_decoding": "greedy_argmax",
                "prefix_mean_entropy_tail8": 0.2 + 0.01 * local,
            }
        )
        hidden.append(torch.randn(3584, generator=generator, dtype=torch.float32).half())
    payload = {
        "status": "complete",
        "problem_id": problem_id,
        "dataset": dataset,
        "split": split,
        "dtype": "bfloat16",
        "protocol_fingerprint": "synthetic",
        "primary_replay_view_fingerprint": "synthetic:paragraph",
        "actual_checkpoint_schedule": "paragraph",
        "capture_layers": [16],
        "rows": rows,
        "hidden": torch.stack(hidden)[:, None, :],
        "record": {},
        "gold_answer": "1",
        "source_dense_artifact": str(target.resolve()),
        "forced_answer_decoding": {
            "strategy": "greedy_argmax",
            "do_sample": False,
            "max_new_tokens": 48,
        },
        "dense": {
            "prediction": dense_prediction,
            "success": dense_success,
            "reasoning_tokens": 160,
            "wall_ms": 160.0,
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deepseek7b_main_v1.yaml")
    args = parser.parse_args()
    # Regression guard: GSM8K/AIME use the numeric branch of success().  A
    # previously misplaced function body made every numeric answer incorrect.
    assert numeric_value("448,000") == 448000.0
    assert numeric_value(r"\frac{3}{4}") == 0.75
    assert success("gsm8k", "448000", "448,000") is True
    assert success("aime", "9", "9") is True
    assert success("gsm8k", "9", "10") is False
    (PROJECT / "results").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="deepseek7b_probe_contract_", dir=PROJECT / "results") as temporary:
        root = Path(temporary)
        raw = root / "raw"
        for split, count in (("probe_train", 20), ("calibration", 10), ("heldout", 10)):
            for index in range(count):
                problem_id = f"synthetic_{split}_{index:03d}"
                target = raw / split / f"sample_{problem_id}.pt"
                artifact(problem_id, split, index, target)
        ood = root / "ood"
        for index in range(10):
            problem_id = f"synthetic_aime_heldout_{index:03d}"
            artifact(
                problem_id,
                "heldout",
                index,
                ood / "heldout" / f"sample_{problem_id}.pt",
                dataset="aime",
            )
        commands = []
        for label, method, loss in (
            ("correctness", "correctness", "bce"),
            ("bce_traj", "correction", "bce_traj"),
        ):
            command = [
                PYTHON,
                "scripts/train_deepseek7b_ablation_v1.py",
                "--dataset",
                "gsm8k",
                "--config",
                args.config,
                "--raw-root",
                str(raw),
                "--output",
                str(root / label),
                "--method",
                method,
                "--seed",
                "0",
                "--gpu",
                "-1",
                "--schedule",
                "sentence",
                "--actual-schedule-label",
                "paragraph",
                "--layer",
                "16",
                "--feature-kind",
                "full_no_delta",
                "--loss",
                loss,
                "--trajectory-aggregation",
                "normalized_softmin",
                "--trajectory-beta",
                "0.5",
                "--trajectory-weight",
                "1.0",
                "--epochs",
                "2",
            ]
            subprocess.run(command, cwd=PROJECT, check=True)
            result = json.loads((root / label / "probe.json").read_text(encoding="utf-8"))
            assert result["status"] == "complete"
            assert result["run_spec"]["architecture"] == [3590, 384, 96, 1]
            assert set(result["frozen_policy_results"]["empirical_B"]) == {"0", "1", "2", "4", "10"}
            commands.append(command)
        ood_command = [
            PYTHON,
            "scripts/evaluate_deepseek7b_ood_v2.py",
            "--dataset",
            "aime",
            "--source-probe",
            str(root / "correctness"),
            "--heldout-root",
            str(ood),
            "--output",
            str(root / "aime_correctness"),
            "--gpu",
            "-1",
        ]
        subprocess.run(ood_command, cwd=PROJECT, check=True)
        ood_result = json.loads(
            (root / "aime_correctness" / "probe.json").read_text(encoding="utf-8")
        )
        assert ood_result["status"] == "complete"
        assert ood_result["run_spec"]["no_retraining"] is True
        assert ood_result["run_spec"]["no_recalibration"] is True
        assert all(
            value["heldout"]["problems"] == 10
            for value in ood_result["frozen_policy_results"]["empirical_B"].values()
        )
        print(json.dumps({"status": "complete", "tested": ["correctness", "bce_traj"]}, indent=2))


if __name__ == "__main__":
    main()

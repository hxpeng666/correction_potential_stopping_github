#!/usr/bin/env python3
"""Fail-closed completeness audit and final pipeline marker."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from src.utils import atomic_json


EXPECTED = {
    "gsm8k": {"probe_train": 6473, "calibration": 1000, "heldout": 1319, "variance": 128},
    "mmlu": {"probe_train": 5700, "calibration": 1140, "heldout": 14042, "variance": 114},
}
SEEDS = (20260803,)
METHODS = ("correctness", "consistency", "last_switch", "correction")
REQUIRED_TABLES = (
    "main_results.csv",
    "risk_frontier.csv",
    "target_ablation.csv",
    "feature_ablation.csv",
    "loss_ablation.csv",
    "checkpoint_ablation.csv",
    "layer_ablation.csv",
    "mmlu_subject_results.csv",
)
REQUIRED_FIGURES = (
    "gsm8k_risk_latency_frontier.png",
    "mmlu_risk_latency_frontier.png",
    "accuracy_latency_tradeoff.png",
    "token_walltime_comparison.png",
)
FATAL = re.compile(
    r"Traceback|CUDA out of memory|RuntimeError|NaN|missing sample|row/vector mismatch|phase failed",
    re.IGNORECASE,
)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def json_status(path: Path, accepted: set[str] = {"complete"}) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") in accepted
    except (OSError, json.JSONDecodeError):
        return False


def sample_count(path: Path) -> int:
    return len(list(path.glob("sample_*.pt")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/final_paper_v1"))
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    failures: list[str] = []
    evidence: dict[str, object] = {}

    audit = root / "ENVIRONMENT_AUDIT.json"
    require(json_status(audit, {"environment_audit_complete"}), "environment audit incomplete", failures)
    require((root / "splits/gsm8k_split.json").is_file(), "missing GSM8K split", failures)
    require((root / "splits/mmlu_split.json").is_file(), "missing MMLU split", failures)

    for dataset, counts in EXPECTED.items():
        dataset_root = root / dataset
        require((dataset_root / "collection.complete").is_file(), f"{dataset} collection marker missing", failures)
        for split in ("probe_train", "calibration", "heldout"):
            dense = dataset_root / f"raw/dense_seed_20260803/{split}"
            checkpoints = dataset_root / f"raw/checkpoints_seed_20260803/{split}"
            require(
                sample_count(dense) == counts[split],
                f"{dataset} dense {split}: {sample_count(dense)} != {counts[split]}",
                failures,
            )
            require(
                sample_count(checkpoints) == counts[split],
                f"{dataset} checkpoints {split}: {sample_count(checkpoints)} != {counts[split]}",
                failures,
            )
        for seed in SEEDS:
            for method in METHODS:
                run = dataset_root / f"seeds/stopper_seed_{seed}/target_{method}"
                require(json_status(run / "phase.complete"), f"missing {dataset}/{seed}/{method}", failures)
                if (run / "probe.pt").is_file():
                    probe = torch.load(run / "probe.pt", map_location="cpu", weights_only=False)
                    require(
                        probe["run_spec"]["scaler_fit_scope"] == "probe_train_fit_only",
                        f"invalid scaler scope {dataset}/{seed}/{method}",
                        failures,
                    )
                    if method == "correction":
                        require(
                            probe["run_spec"]["architecture"] == [5126, 384, 96, 1],
                            f"invalid architecture {dataset}/{seed}/{method}",
                            failures,
                        )
        for ablation in (
            "loss_bce",
            "feature_h_only",
            "feature_h_delta",
            "feature_full_no_entropy",
            "feature_full_no_position",
            "checkpoint_fixed",
            "checkpoint_hybrid",
            "layer_8",
            "layer_35",
        ):
            require(
                json_status(dataset_root / f"ablations/{ablation}/phase.complete"),
                f"missing {dataset} ablation {ablation}",
                failures,
            )
        require(json_status(dataset_root / "baselines/phase.complete"), f"{dataset} baselines missing", failures)
        require(
            json_status(dataset_root / "heldout/online_seed_20260803/phase.complete"),
            f"{dataset} online full missing",
            failures,
        )
        require(
            sample_count(dataset_root / "heldout/online_seed_20260803/raw") == counts["heldout"],
            f"{dataset} online full sample count mismatch",
            failures,
        )
        require(
            json_status(dataset_root / "heldout/online_variance_seed_20260803/phase.complete"),
            f"{dataset} online variance missing",
            failures,
        )
        require(
            sample_count(dataset_root / "heldout/online_variance_seed_20260803/raw") == counts["variance"],
            f"{dataset} online variance sample count mismatch",
            failures,
        )
        online = dataset_root / "heldout/online_seed_20260803/online_summary.json"
        if online.is_file():
            summary = json.loads(online.read_text(encoding="utf-8"))
            require(summary.get("warmup_examples", 0) >= 20, f"{dataset} warmup <20", failures)
            require(summary.get("paired_interleaved") is True, f"{dataset} online not paired/interleaved", failures)
            require(summary.get("batch_size") == 1, f"{dataset} online batch !=1", failures)
            require(
                set(summary.get("summary", {})) >= {"dense", "strict", "balanced", "aggressive"},
                f"{dataset} missing online workpoints",
                failures,
            )
        validation = dataset_root / "heldout/online_replay_validation.json"
        require(json_status(validation, {"PASS"}), f"{dataset} online/replay validation failed", failures)
        bootstrap = dataset_root / "bootstrap_10000.json"
        require(json_status(bootstrap), f"{dataset} bootstrap missing", failures)
        if bootstrap.is_file():
            require(
                json.loads(bootstrap.read_text()).get("replicates") == 10000,
                f"{dataset} bootstrap repetitions !=10000",
                failures,
            )

    for name in REQUIRED_TABLES:
        path = root / "tables" / name
        require(path.is_file() and path.stat().st_size > 0, f"missing table {name}", failures)
    subject_path = root / "tables/mmlu_subject_results.csv"
    if subject_path.is_file():
        subjects = pd.read_csv(subject_path)
        require(len(subjects) == 57, f"MMLU subject rows {len(subjects)} !=57", failures)
        require(int(subjects.n.sum()) == 14042, "MMLU subject n does not sum to 14042", failures)
    for name in REQUIRED_FIGURES:
        path = root / "figures" / name
        require(path.is_file() and path.stat().st_size > 1000, f"missing figure {name}", failures)
    report = root / "FINAL_PAPER_REPORT.md"
    reproducibility = root / "REPRODUCIBILITY.md"
    require(report.is_file(), "missing FINAL_PAPER_REPORT.md", failures)
    require(reproducibility.is_file(), "missing REPRODUCIBILITY.md", failures)
    if report.is_file():
        text = report.read_text(encoding="utf-8")
        for section in range(1, 16):
            require(f"## {section}." in text, f"report missing section {section}", failures)
        require("全部实验只使用 seed 20260803" in text, "single-seed protocol not disclosed", failures)
        require("actual online" in text.lower(), "actual online results not distinguished", failures)
        require("replay" in text.lower(), "replay results not distinguished", failures)

    formal_log_roots = [
        ROOT / "logs/final_paper_v1/gsm8k_collection",
        ROOT / "logs/final_paper_v1/mmlu_collection",
        ROOT / "logs/final_paper_v1/gsm8k_post",
        ROOT / "logs/final_paper_v1/mmlu_post",
    ]
    fatal_hits = []
    for directory in formal_log_roots:
        for path in directory.glob("*.log"):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                if FATAL.search(line):
                    fatal_hits.append(f"{path}:{line_number}:{line[:300]}")
    require(not fatal_hits, f"fatal patterns in formal logs: {fatal_hits[:10]}", failures)
    evidence["fatal_log_hits"] = fatal_hits
    evidence["dense_seed_count"] = 1
    evidence["stopper_seed_count"] = 1
    evidence["heldout_counts"] = {
        "gsm8k": EXPECTED["gsm8k"]["heldout"],
        "mmlu": EXPECTED["mmlu"]["heldout"],
    }

    payload = {
        "status": "complete" if not failures else "failed",
        "failures": failures,
        "evidence": evidence,
    }
    atomic_json(payload, root / "COMPLETENESS_AUDIT.json")
    if failures:
        print(json.dumps(payload, indent=2))
        raise SystemExit(1)
    atomic_json(
        {
            "status": "complete",
            "completion_gate_count": 10,
            "completeness_audit": "COMPLETENESS_AUDIT.json",
            "report": "FINAL_PAPER_REPORT.md",
            "dense_seeds": 1,
            "stopper_seeds": 1,
            "bootstrap_replicates": 10000,
        },
        root / "pipeline.complete",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Finalize preserved vLLM risk artifacts under a self-reproducibility policy."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_vllm_risk_pair_v1 import metrics
from run_qwen3_14b_vllm_full_v1 import gate_artifact
from run_qwen3_14b_vllm_risk_matrix_v1 import audit_exact, audit_risk, phase_speedup
from src.reproducibility import code_provenance, sha256_file


POLICY_NAME = "same_profile_same_and_cross_gpu_exact_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def artifact_metrics(root: Path, problems: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        metrics(torch.load(gate_artifact(root, problem), map_location="cpu", weights_only=False))
        for problem in problems
    ]
    sums = {
        name: sum(float(value[name]) for value in values)
        for name in (
            "dense_wall_ms",
            "hidden_wall_ms",
            "branch_wall_ms",
            "branch_cached_tokens",
            "branch_context_tokens",
        )
    }
    sums["branch_cache_fraction"] = (
        sums["branch_cached_tokens"] / sums["branch_context_tokens"]
        if sums["branch_context_tokens"]
        else 0.0
    )
    return {"problems": values, "sum": sums}


def require_completed_runs(
    manifest: dict[str, Any], profiles: list[str], problem_ids: list[str]
) -> None:
    expected = {
        *(
            f"{profile}/{phase}"
            for profile in profiles
            for phase in ("repeat0", "repeat1", "cross")
        ),
        "composition/paired_reverse",
        *(f"composition/solo_{problem_id}" for problem_id in problem_ids),
    }
    runs = manifest.get("runs", {})
    missing = sorted(expected - set(runs))
    failed = sorted(
        name for name in expected if runs.get(name, {}).get("returncode") != 0
    )
    if missing or failed:
        raise RuntimeError(f"incomplete preserved risk workers: missing={missing}, failed={failed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    profiles = [str(value) for value in config["vllm_risk_gate"]["candidate_profiles"]]
    problems = [dict(value) for value in config["determinism_gate"]["problems"]]
    problem_ids = [str(value["problem_id"]) for value in problems]
    output = args.output_root.resolve()
    source_manifest_path = output / "RISK_GATE_MANIFEST.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    require_completed_runs(source_manifest, profiles, problem_ids)

    identity = code_provenance(
        ROOT,
        (
            "configs/qwen3_14b_deterministic_ood13k_vllm_full_v1.yaml",
            "scripts/finalize_qwen3_14b_vllm_self_reproducibility_v1.py",
            "scripts/audit_deterministic_collection_pair_v1.py",
            "scripts/audit_vllm_risk_pair_v1.py",
            "scripts/run_qwen3_14b_vllm_risk_matrix_v1.py",
            "src/reproducibility.py",
        ),
    )
    baseline_root = output / "profiles" / "baseline_b1" / "same_gpu_repeat0"
    baseline_metrics = artifact_metrics(baseline_root, problems)
    results: dict[str, Any] = {}
    for profile in profiles:
        root = output / "profiles" / profile
        value: dict[str, Any] = {
            "same_gpu_exact": {},
            "cross_gpu_exact": {},
            "vs_baseline": {},
            "metrics": artifact_metrics(root / "same_gpu_repeat0", problems),
        }
        for problem in problems:
            problem_id = str(problem["problem_id"])
            reference = gate_artifact(root / "same_gpu_repeat0", problem)
            value["same_gpu_exact"][problem_id] = audit_exact(
                args.python,
                config,
                reference,
                gate_artifact(root / "same_gpu_repeat1", problem),
                output / "audits_self_reproducibility" / f"{profile}_{problem_id}_same.json",
                "same",
            )
            value["cross_gpu_exact"][problem_id] = audit_exact(
                args.python,
                config,
                reference,
                gate_artifact(root / "cross_gpu0", problem),
                output / "audits_self_reproducibility" / f"{profile}_{problem_id}_cross.json",
                "distinct",
            )
            value["vs_baseline"][problem_id] = audit_risk(
                args.python,
                config,
                gate_artifact(baseline_root, problem),
                reference,
                output / "audits_self_reproducibility" / f"{profile}_{problem_id}_vs_baseline.json",
            )
        value["speedup_vs_baseline"] = phase_speedup(baseline_metrics, value["metrics"])
        value["baseline_equivalent"] = all(
            item["all_scientific_exact"] for item in value["vs_baseline"].values()
        )
        value["self_reproducibility_accepted"] = bool(
            all(value["same_gpu_exact"].values())
            and all(value["cross_gpu_exact"].values())
        )
        results[profile] = value

    composition_profile = "full_apc_b2"
    composition_root = output / "composition" / composition_profile
    paired_root = output / "profiles" / composition_profile / "same_gpu_repeat0"
    composition: dict[str, Any] = {}
    for problem in problems:
        problem_id = str(problem["problem_id"])
        reference = gate_artifact(paired_root, problem)
        composition[f"{problem_id}/paired_reverse"] = audit_risk(
            args.python,
            config,
            reference,
            gate_artifact(composition_root / "paired_reverse", problem),
            output / "audits_self_reproducibility" / f"composition_{problem_id}_reverse.json",
        )
        composition[f"{problem_id}/solo"] = audit_risk(
            args.python,
            config,
            reference,
            gate_artifact(composition_root / f"solo_{problem_id}", problem),
            output / "audits_self_reproducibility" / f"composition_{problem_id}_solo.json",
        )

    accepted = [
        profile for profile in profiles if results[profile]["self_reproducibility_accepted"]
    ]
    ranked = sorted(
        accepted,
        key=lambda name: float(
            results[name]["speedup_vs_baseline"]["measured_phase_total"] or 0.0
        ),
        reverse=True,
    )
    recommendation = ranked[0] if ranked else None
    report = {
        "status": "complete",
        "all_model_runs_completed": True,
        "acceptance_policy": {
            "name": POLICY_NAME,
            "same_gpu_repeat_exact_required": True,
            "cross_gpu_exact_required": True,
            "vs_baseline_exact_required": False,
            "batch_composition_order_exact_required": False,
            "rationale": "User authorized self-reproducibility as the sole hard gate; baseline and composition divergences remain fully audited and disclosed.",
        },
        "source_failed_manifest": {
            "path": str(source_manifest_path),
            "sha256": sha256_file(source_manifest_path),
            "status": source_manifest.get("status"),
            "error_type": source_manifest.get("error_type"),
            "error": source_manifest.get("error"),
        },
        "finalizer_code_identity": identity,
        "results": results,
        "batch_composition": composition,
        "accepted_profiles": accepted,
        "recommended_profile": recommendation,
        "created_at": utc_now(),
    }
    atomic_json(report, output / "RISK_MATRIX.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

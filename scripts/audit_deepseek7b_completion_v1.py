#!/usr/bin/env python3
"""Requirement-by-requirement completion audit for the full experiment."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml

from deepseek7b_protocol_v1 import canonical_fingerprint
from train_deepseek7b_ablation_v1 import artifact_manifest


METHODS = ("correctness", "consistency", "last_switch", "bce", "bce_traj")
EXPECTED_TARGET = {
    "correctness": "correctness",
    "consistency": "consistency",
    "last_switch": "last_switch",
    "bce": "correction",
    "bce_traj": "correction",
}
EXPECTED_CALIBRATION = {"gsm8k": 500, "math500": 700, "aime": 700}
EXPECTED_ARCHITECTURE = [3590, 384, 96, 1]
EXPECTED_SPLITS = {
    "gsm8k/probe_train": 1000,
    "gsm8k/calibration": 500,
    "gsm8k/heldout": 1319,
    "math/probe_train": 1400,
    "math/calibration": 700,
    "math500/heldout": 500,
    "aime/heldout": 30,
}
EXPECTED_B = {"0", "1", "2", "4", "10"}


def same_json(left, right) -> bool:
    """Compare JSON-compatible protocol payloads without object-identity shortcuts."""
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def close_number(left, right, *, tolerance: float = 1e-10) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def bounded_number(value, lower: float, upper: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and lower <= number <= upper


def non_nan_number(value) -> bool:
    """Allow finite thresholds and +/-inf sentinels, but never NaN."""
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]


def normalized_questions(rows: list[dict]) -> set[str]:
    return {" ".join(str(row.get("question", "")).split()) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if Path(config["output_root"]).resolve() != output.resolve():
        raise ValueError("config/output-root mismatch")
    checks = []
    errors = []

    selective_extension = config.get("selective_dense_extension", {})
    selective_extension_enabled = bool(selective_extension.get("enabled", False))
    dense_budget_requirement = (
        "Selective Dense max_new_tokens=32768 with 13K non-capped reuse"
        if selective_extension_enabled
        else "Dense max_new_tokens=13000"
    )
    if selective_extension_enabled:
        dense_budget_passed = (
            config["generation"]["dense_max_new_tokens"] == 32768
            and selective_extension.get("source_dense_max_new_tokens") == 13000
            and selective_extension.get("target_dense_max_new_tokens") == 32768
            and selective_extension.get("eligibility")
            == "source_dense_reached_max_tokens_true"
            and selective_extension.get("preserve_noncapped_generation_exactly") is True
            and selective_extension.get("require_first_13000_token_identity") is True
        )
    else:
        dense_budget_passed = config["generation"]["dense_max_new_tokens"] == 13000

    protocol_invariants = {
        "frozen DeepSeek-R1-Distill-Qwen-7B identity": (
            config["model"]["id"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            and Path(config["model"]["local_path"]).resolve()
            == Path(
                "models/DeepSeek-R1-Distill-Qwen-7B"
            ).resolve()
            and config["model"]["frozen"] is True
            and config["model"]["dtype"] == "bfloat16"
            and config["model"]["hidden_size"] == 3584
            and config["model"]["num_hidden_layers"] == 28
        ),
        dense_budget_requirement: dense_budget_passed,
        "Dense decoding temperature/top_p/top_k": (
            config["generation"]["temperature"] == 0.6
            and config["generation"]["top_p"] == 0.95
            and config["generation"]["top_k"] == 20
            and config["generation"]["do_sample"] is True
        ),
        "forced-answer greedy max_new_tokens=48": (
            config["generation"]["force_answer_max_new_tokens"] == 48
            and config["generation"]["forced_answer_strategy"] == "greedy_argmax"
            and config["generation"]["forced_answer_do_sample"] is False
        ),
        "paragraph checkpoint without range filtering": (
            config["checkpoint"]["schedule"] == "paragraph"
            and config["checkpoint"]["range_filter"] == "none"
            and config["checkpoint"]["zero_checkpoint_policy"] == "dense_fallback"
        ),
        "layer 16 full_no_delta feature": (
            config["features"]["layer_zero_based"] == 16
            and config["features"]["primary_kind"] == "full_no_delta"
            and config["features"]["primary_width"] == 3590
            and config["probe"]["shared_architecture"] == EXPECTED_ARCHITECTURE
        ),
        "normalized trajectory main configuration": (
            config["probe"]["trajectory_softmin_beta"] == 0.5
            and config["probe"]["trajectory_weight"] == 1.0
        ),
        "empirical calibration grid and budgets": (
            config["calibration"]["quantile_grid_size"] == 101
            and config["calibration"]["empirical_lost_correct_budgets"]
            == [0, 1, 2, 4, 10]
        ),
    }
    for requirement, passed in protocol_invariants.items():
        checks.append({"requirement": requirement, "passed": passed})
        if not passed:
            errors.append(f"protocol invariant failed: {requirement}")

    prepared = Path(config["data"]["prepared_root"])
    prepared_rows = {
        "gsm8k/probe_train": read_jsonl(prepared / "gsm8k/probe_train.jsonl"),
        "gsm8k/calibration": read_jsonl(prepared / "gsm8k/calibration.jsonl"),
        "gsm8k/heldout": read_jsonl(prepared / "gsm8k/heldout.jsonl"),
        "math/probe_train": read_jsonl(prepared / "math/probe_train.jsonl"),
        "math/calibration": read_jsonl(prepared / "math/calibration.jsonl"),
        "math500/heldout": read_jsonl(prepared / "math500/heldout.jsonl"),
        "aime/heldout": read_jsonl(prepared / "aime/heldout.jsonl"),
    }
    prepared_counts_passed = all(
        len(prepared_rows[split]) == expected
        and len({row["problem_id"] for row in prepared_rows[split]}) == expected
        for split, expected in EXPECTED_SPLITS.items()
    )
    checks.append(
        {
            "requirement": "prepared problem counts and unique IDs",
            "passed": prepared_counts_passed,
            "counts": {split: len(rows) for split, rows in prepared_rows.items()},
            "unique_ids": {
                split: len({row["problem_id"] for row in rows})
                for split, rows in prepared_rows.items()
            },
        }
    )
    if not prepared_counts_passed:
        errors.append("prepared split count/unique-ID mismatch")

    for split, target in (("probe_train", 200), ("calibration", 100)):
        rows = prepared_rows[f"math/{split}"]
        counts = collections.Counter(str(row["category"]) for row in rows)
        passed = set(counts) == set(config["data"]["math"]["categories"]) and all(
            value == target for value in counts.values()
        )
        checks.append(
            {
                "requirement": f"MATH exact per-category {split}",
                "passed": passed,
                "counts": dict(sorted(counts.items())),
            }
        )
        if not passed:
            errors.append(f"MATH per-category {split} mismatch")
    fit_ids = {row["problem_id"] for row in prepared_rows["math/probe_train"]}
    calibration_ids = {row["problem_id"] for row in prepared_rows["math/calibration"]}
    disjoint = not (fit_ids & calibration_ids)
    checks.append({"requirement": "MATH fit/calibration problem disjointness", "passed": disjoint})
    if not disjoint:
        errors.append("MATH fit/calibration overlap")

    gsm_id_sets = [
        {row["problem_id"] for row in prepared_rows[f"gsm8k/{split}"]}
        for split in ("probe_train", "calibration", "heldout")
    ]
    gsm_disjoint = all(
        not (gsm_id_sets[left] & gsm_id_sets[right])
        for left in range(len(gsm_id_sets))
        for right in range(left + 1, len(gsm_id_sets))
    )
    checks.append(
        {"requirement": "GSM8K problem-level split disjointness", "passed": gsm_disjoint}
    )
    if not gsm_disjoint:
        errors.append("GSM8K problem split overlap")

    math_supervision_questions = normalized_questions(
        prepared_rows["math/probe_train"] + prepared_rows["math/calibration"]
    )
    ood_overlaps = {
        dataset: len(math_supervision_questions & normalized_questions(prepared_rows[dataset]))
        for dataset in ("math500/heldout", "aime/heldout")
    }
    ood_disjoint = all(count == 0 for count in ood_overlaps.values())
    checks.append(
        {
            "requirement": "MATH supervision versus OOD exact-question disjointness",
            "passed": ood_disjoint,
            "normalized_question_overlaps": ood_overlaps,
        }
    )
    if not ood_disjoint:
        errors.append("MATH supervision/OOD exact-question overlap")

    migration_path = output / "CACHE_MIGRATION.json"
    migration = (
        json.loads(migration_path.read_text(encoding="utf-8"))
        if migration_path.is_file()
        else {}
    )
    migration_passed = migration.get("status") == "complete"
    if selective_extension_enabled:
        manifest_path = Path(selective_extension["manifest"])
        extension_manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        expected_total = sum(EXPECTED_SPLITS.values())
        eligible_count = int(extension_manifest.get("eligible_count", -1))
        migration_passed = migration_passed and (
            extension_manifest.get("target_protocol_id") == config["protocol_id"]
            and extension_manifest.get("source_dense_max_new_tokens") == 13000
            and extension_manifest.get("target_dense_max_new_tokens") == 32768
            and eligible_count >= 0
            and int(migration.get("migrated_noncapped", 0))
            + int(migration.get("already_migrated_noncapped", 0))
            == expected_total - eligible_count
            and int(migration.get("unavailable_for_collection", -1)) == eligible_count
        )
    checks.append({"requirement": "audited cache migration", "passed": migration_passed})
    if not migration_passed:
        errors.append("missing/invalid CACHE_MIGRATION.json")

    repair_path = output / "NUMERIC_LABEL_REPAIR.json"
    repair = json.loads(repair_path.read_text(encoding="utf-8")) if repair_path.is_file() else {}
    repair_script_path = Path(__file__).with_name("repair_deepseek7b_numeric_labels_v2.py")
    repair_passed = (
        repair.get("status") == "complete"
        and repair.get("protocol_id") == config["protocol_id"]
        and repair.get("verified_after_repair") is True
        and repair.get("datasets", {}).get("gsm8k", {}).get("files") == 2819
        and repair.get("datasets", {}).get("aime", {}).get("files") == 30
        and repair_script_path.is_file()
        and repair.get("repair_script_sha256") == sha256(repair_script_path)
    )
    checks.append(
        {
            "requirement": "audited GSM8K/AIME numeric-label repair",
            "passed": repair_passed,
            "report": str(repair_path),
            "repair_script": str(repair_script_path),
            "reported_script_sha256": repair.get("repair_script_sha256"),
            "current_script_sha256": (
                sha256(repair_script_path) if repair_script_path.is_file() else None
            ),
        }
    )
    if not repair_passed:
        errors.append("missing/invalid NUMERIC_LABEL_REPAIR.json")

    freeze_path = output / "PROTOCOL_FREEZE.json"
    if not freeze_path.is_file():
        checks.append({"requirement": "frozen protocol identities", "passed": False})
        errors.append("missing PROTOCOL_FREEZE.json")
    else:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        drift = []
        freeze_files = freeze.get("files", [])
        if not isinstance(freeze_files, list):
            freeze_files = []
            drift.append("<malformed freeze files list>")
        frozen_paths = set()
        for item in freeze_files:
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                drift.append("<malformed freeze file identity>")
                continue
            path = Path(item["path"])
            frozen_paths.add(str(path.resolve()))
            if not path.is_file() or sha256(path) != item["sha256"]:
                drift.append(str(path))
        model_root = Path(config["model"]["local_path"])
        required_frozen_paths = {
            str(args.config.resolve()),
            str((prepared / "MANIFEST.json").resolve()),
            *(str(path.resolve()) for path in prepared.glob("*/*.jsonl")),
            str((model_root / "config.json").resolve()),
            str((model_root / "generation_config.json").resolve()),
            str((model_root / "model.safetensors.index.json").resolve()),
            str(Path(__file__).resolve()),
            str(Path(__file__).with_name("deepseek7b_protocol_v1.py").resolve()),
            str(Path(__file__).with_name("collect_deepseek7b_paragraph_v1.py").resolve()),
            str(Path(__file__).with_name("train_deepseek7b_ablation_v1.py").resolve()),
            str(Path(__file__).with_name("evaluate_deepseek7b_ood_v2.py").resolve()),
            str(Path(__file__).with_name("summarize_deepseek7b_results_v1.py").resolve()),
            str(Path(__file__).with_name("supervise_deepseek7b_experiment_v1.py").resolve()),
        }
        if selective_extension_enabled:
            required_frozen_paths.add(
                str(
                    Path(__file__)
                    .with_name("migrate_deepseek7b_selective_budget_v3.py")
                    .resolve()
                )
            )
        freeze_metadata_passed = (
            freeze.get("status") == "frozen"
            and freeze.get("protocol_id") == config["protocol_id"]
            and freeze.get("config_fingerprint") == canonical_fingerprint(config)
            and required_frozen_paths <= frozen_paths
        )
        freeze_passed = not drift and freeze_metadata_passed
        checks.append(
            {
                "requirement": "frozen protocol identities",
                "passed": freeze_passed,
                "metadata_verified": freeze_metadata_passed,
                "tracked_files": len(frozen_paths),
                "missing_required_files": sorted(required_frozen_paths - frozen_paths),
                "drift": drift,
            }
        )
        if not freeze_metadata_passed:
            errors.append("protocol freeze metadata/coverage mismatch")
        errors.extend(f"protocol drift: {path}" for path in drift)

    collection_path = output / "COLLECTION_AUDIT.json"
    if not collection_path.is_file():
        errors.append("missing COLLECTION_AUDIT.json")
    else:
        collection = json.loads(collection_path.read_text(encoding="utf-8"))
        passed = collection.get("status") == "complete"
        if selective_extension_enabled:
            extension_audit = collection.get("selective_dense_extension", {})
            passed = passed and (
                extension_audit.get("passed") is True
                and extension_audit.get("source_dense_max_new_tokens") == 13000
                and extension_audit.get("target_dense_max_new_tokens") == 32768
                and extension_audit.get("eligible")
                == extension_audit.get("generated_at_32768")
                == extension_audit.get("prefix_identity_verified")
            )
        checks.append({"requirement": "collection schema audit", "passed": passed})
        if not passed:
            errors.append("collection audit is not complete")

    for split, expected in EXPECTED_SPLITS.items():
        actual = len(list((output / "cache" / split).glob("sample_*.pt")))
        passed = actual == expected
        checks.append(
            {
                "requirement": f"cache count {split}",
                "passed": passed,
                "expected": expected,
                "actual": actual,
            }
        )
        if not passed:
            errors.append(f"cache count {split}: expected {expected}, got {actual}")

    cache = output / "cache"
    current_inputs = {
        "gsm8k": {
            "input": artifact_manifest(cache / "gsm8k"),
            "heldout_input": None,
        },
        "math500": {
            "input": artifact_manifest(cache / "math"),
            "heldout_input": artifact_manifest(cache / "math500"),
        },
        "aime": {
            "heldout_input": artifact_manifest(cache / "aime"),
        },
    }
    probe_count = 0
    expected_csv_rows = {}
    for dataset in ("gsm8k", "math500", "aime"):
        expected_test = EXPECTED_SPLITS[f"{dataset}/heldout"]
        for method in METHODS:
            root = output / "probes" / dataset / method
            marker = root / "phase.complete"
            report = root / "probe.json"
            required_artifacts = (
                ("probe.json", "probe.pt", "scores.pt", "policy_records.pt")
                if dataset != "aime"
                else ("probe.json", "scores.pt", "policy_records.pt")
            )
            artifacts_passed = marker.is_file() and all(
                (root / name).is_file() for name in required_artifacts
            )
            passed = artifacts_passed
            detail = {"requirement": f"probe {dataset}/{method}", "passed": passed}
            if passed:
                marker_value = json.loads(marker.read_text(encoding="utf-8"))
                payload = json.loads(report.read_text(encoding="utf-8"))
                frozen_results = payload.get("frozen_policy_results", {}).get("empirical_B", {})
                budgets = set(frozen_results)
                heldout_counts = {
                    str(budget): int(value["heldout"]["problems"])
                    for budget, value in frozen_results.items()
                    if isinstance(value, dict)
                    and isinstance(value.get("heldout"), dict)
                    and "problems" in value["heldout"]
                }
                calibration_counts = {
                    str(budget): int(value["calibration"]["problems"])
                    for budget, value in frozen_results.items()
                    if isinstance(value, dict)
                    and isinstance(value.get("calibration"), dict)
                    and "problems" in value["calibration"]
                }
                risk_constraints_passed = budgets == EXPECTED_B and all(
                    calibration_counts.get(budget) == EXPECTED_CALIBRATION[dataset]
                    and int(frozen_results[budget]["calibration"]["lost_correct_count"])
                    <= int(budget)
                    and non_nan_number(frozen_results[budget]["calibration"]["threshold"])
                    for budget in EXPECTED_B
                )
                metric_ranges_passed = budgets == EXPECTED_B and all(
                    bounded_number(value["heldout"].get("dense_accuracy"), 0.0, 1.0)
                    and bounded_number(value["heldout"].get("accuracy"), 0.0, 1.0)
                    and bounded_number(value["heldout"].get("token_reduction"), 0.0, 1.0)
                    and bounded_number(value["heldout"].get("coverage"), 0.0, 1.0)
                    and 0
                    <= int(value["heldout"].get("lost_correct_count", -1))
                    <= expected_test
                    and float(value["heldout"].get("mean_reasoning_and_answer_tokens", 0))
                    > 0
                    and float(value["heldout"].get("mean_dense_reasoning_tokens", 0)) > 0
                    for value in frozen_results.values()
                )
                expected_split_counts = (
                    {"probe_train": 1000, "calibration": 500, "heldout": 1319}
                    if dataset == "gsm8k"
                    else (
                        {"probe_train": 1400, "calibration": 700, "heldout": 500}
                        if dataset == "math500"
                        else {"heldout": 30}
                    )
                )
                split_counts = payload.get("split_counts", {})
                split_counts_passed = all(
                    split_counts.get(split, {}).get("problems") == expected
                    for split, expected in expected_split_counts.items()
                )
                run_spec = payload.get("run_spec", {})
                expected_loss = "bce_traj" if method == "bce_traj" else "bce"
                if dataset == "aime":
                    input_identity_passed = same_json(
                        payload.get("heldout_input"),
                        current_inputs["aime"]["heldout_input"],
                    )
                    fingerprint_passed = marker_value.get(
                        "invocation_fingerprint"
                    ) == canonical_fingerprint(payload.get("source_probe", {}))
                else:
                    input_identity_passed = same_json(
                        payload.get("input"), current_inputs[dataset]["input"]
                    ) and same_json(
                        payload.get("heldout_input"),
                        current_inputs[dataset]["heldout_input"],
                    )
                    fingerprint_passed = marker_value.get(
                        "run_spec_fingerprint"
                    ) == canonical_fingerprint(run_spec)
                passed = (
                    marker_value.get("status") == "complete"
                    and payload.get("status") == "complete"
                    and budgets == EXPECTED_B
                    and all(value == expected_test for value in heldout_counts.values())
                    and run_spec.get("protocol_id") == config["protocol_id"]
                    and run_spec.get("actual_schedule_label") == "paragraph"
                    and run_spec.get("layer") == 16
                    and run_spec.get("feature_kind") == "full_no_delta"
                    and run_spec.get("loss") == expected_loss
                    and run_spec.get("method") == EXPECTED_TARGET[method]
                    and run_spec.get("architecture") == EXPECTED_ARCHITECTURE
                    and split_counts_passed
                    and risk_constraints_passed
                    and metric_ranges_passed
                    and input_identity_passed
                    and fingerprint_passed
                )
                if method == "bce_traj":
                    passed = passed and (
                        run_spec.get("trajectory_aggregation") == "normalized_softmin"
                        and run_spec.get("trajectory_normalize_by_count") is True
                        and close_number(run_spec.get("trajectory_softmin_beta"), 0.5)
                        and close_number(run_spec.get("trajectory_weight"), 1.0)
                    )
                if dataset == "aime":
                    source_root = output / "probes" / "math500" / method
                    source_json_path = source_root / "probe.json"
                    source_pt_path = source_root / "probe.pt"
                    source_marker_path = source_root / "phase.complete"
                    source_payload = (
                        json.loads(source_json_path.read_text(encoding="utf-8"))
                        if source_json_path.is_file()
                        else None
                    )
                    source_identity = payload.get("source_probe", {})
                    source_results = (
                        source_payload.get("frozen_policy_results", {}).get("empirical_B", {})
                        if source_payload is not None
                        else {}
                    )
                    shared_probe_passed = (
                        marker_value.get("no_retraining") is True
                        and marker_value.get("no_recalibration") is True
                        and run_spec.get("evaluation_mode") == "frozen_shared_math_probe"
                        and run_spec.get("no_retraining") is True
                        and run_spec.get("no_recalibration") is True
                        and "/probes/math500/" in str(marker_value.get("source_probe"))
                        and source_json_path.is_file()
                        and source_pt_path.is_file()
                        and source_marker_path.is_file()
                        and source_payload is not None
                        and source_identity.get("source_probe_json_sha256") == sha256(source_json_path)
                        and source_identity.get("source_probe_pt_sha256") == sha256(source_pt_path)
                        and source_identity.get("source_marker_sha256") == sha256(source_marker_path)
                        and same_json(payload.get("calibration"), source_payload.get("calibration"))
                        and same_json(
                            payload.get("online_workpoints"),
                            source_payload.get("online_workpoints"),
                        )
                        and set(source_results) == EXPECTED_B
                    )
                    if shared_probe_passed:
                        for budget in EXPECTED_B:
                            shared_probe_passed = shared_probe_passed and same_json(
                                frozen_results[budget]["calibration"],
                                source_results[budget]["calibration"],
                            )
                    passed = passed and shared_probe_passed
                    detail["shared_math_probe_verified"] = shared_probe_passed
                detail.update(
                    {
                        "passed": passed,
                        "budgets": sorted(budgets),
                        "heldout_counts": heldout_counts,
                        "calibration_counts": calibration_counts,
                        "expected_heldout": expected_test,
                        "risk_constraints_verified": risk_constraints_passed,
                        "metric_ranges_verified": metric_ranges_passed,
                        "split_counts_verified": split_counts_passed,
                        "target_verified": run_spec.get("method") == EXPECTED_TARGET[method],
                        "required_artifacts": list(required_artifacts),
                        "artifacts_verified": artifacts_passed,
                        "input_identity_verified": input_identity_passed,
                        "marker_fingerprint_verified": fingerprint_passed,
                    }
                )
            checks.append(detail)
            if not passed:
                errors.append(f"probe incomplete/invalid: {dataset}/{method}")
            else:
                probe_count += 1
                for budget, value in frozen_results.items():
                    calibration = value["calibration"]
                    heldout = value["heldout"]
                    expected_csv_rows[(dataset, method, str(budget))] = {
                        "threshold": calibration["threshold"],
                        "calibration_lost_correct": calibration["lost_correct_count"],
                        "test_problems": heldout["problems"],
                        "dense_accuracy": heldout["dense_accuracy"],
                        "accuracy": heldout["accuracy"],
                        "accuracy_delta_pp": 100
                        * (heldout["accuracy"] - heldout["dense_accuracy"]),
                        "token_reduction": heldout["token_reduction"],
                        "lost_correct_count": heldout["lost_correct_count"],
                        "coverage": heldout["coverage"],
                        "mean_tokens": heldout["mean_reasoning_and_answer_tokens"],
                        "mean_dense_tokens": heldout["mean_dense_reasoning_tokens"],
                    }

    expected_identities = {
        (dataset, method, budget)
        for dataset in ("gsm8k", "math500", "aime")
        for method in METHODS
        for budget in EXPECTED_B
    }
    numeric_fields = (
        "threshold",
        "calibration_lost_correct",
        "test_problems",
        "dense_accuracy",
        "accuracy",
        "accuracy_delta_pp",
        "token_reduction",
        "lost_correct_count",
        "coverage",
        "mean_tokens",
        "mean_dense_tokens",
    )
    csv_path = output / "RESULTS_ALL_B.csv"
    if not csv_path.is_file():
        checks.append({"requirement": "75-row all-B result table", "passed": False})
        errors.append("missing RESULTS_ALL_B.csv")
    else:
        with csv_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        identities = {(row["dataset"], row["method"], row["budget_B"]) for row in rows}
        mismatches = []
        for row in rows:
            identity = (row["dataset"], row["method"], row["budget_B"])
            expected = expected_csv_rows.get(identity)
            if expected is None:
                mismatches.append({"identity": identity, "reason": "unexpected row"})
                continue
            bad_fields = [
                field
                for field in numeric_fields
                if not close_number(row.get(field), expected.get(field))
            ]
            if bad_fields:
                mismatches.append({"identity": identity, "fields": bad_fields})
        passed = (
            len(rows) == 75
            and identities == expected_identities
            and len(expected_csv_rows) == 75
            and not mismatches
        )
        checks.append(
            {
                "requirement": "75-row all-B result table",
                "passed": passed,
                "rows": len(rows),
                "unique_identities": len(identities),
                "source_rows": len(expected_csv_rows),
                "mismatches": mismatches,
            }
        )
        if not passed:
            errors.append("result table identity/count mismatch")

    json_path = output / "RESULTS_ALL_B.json"
    json_rows_passed = False
    json_mismatches = []
    if json_path.is_file():
        try:
            json_rows = json.loads(json_path.read_text(encoding="utf-8"))
            json_identities = {
                (row["dataset"], row["method"], str(row["budget_B"])) for row in json_rows
            }
            for row in json_rows:
                identity = (row["dataset"], row["method"], str(row["budget_B"]))
                expected = expected_csv_rows.get(identity)
                if expected is None:
                    json_mismatches.append({"identity": identity, "reason": "unexpected row"})
                    continue
                bad_fields = [
                    field
                    for field in numeric_fields
                    if not close_number(row.get(field), expected.get(field))
                ]
                if bad_fields:
                    json_mismatches.append({"identity": identity, "fields": bad_fields})
            json_rows_passed = (
                len(json_rows) == 75
                and json_identities == expected_identities
                and not json_mismatches
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            json_mismatches.append({"reason": type(exc).__name__, "message": str(exc)})
    checks.append(
        {
            "requirement": "75-row JSON result table",
            "passed": json_rows_passed,
            "mismatches": json_mismatches,
        }
    )
    if not json_rows_passed:
        errors.append("missing/invalid RESULTS_ALL_B.json")

    b2_path = output / "RESULTS_B2.md"
    b2_passed = False
    b2_rows = []
    if b2_path.is_file():
        b2_rows = [
            line
            for line in b2_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("| ")
            and not line.startswith("| Dataset ")
            and not line.startswith("|---")
        ]
        b2_passed = len(b2_rows) == 15 and all(
            any(f"| {dataset} | {method} |" in line for line in b2_rows)
            for dataset in ("gsm8k", "math500", "aime")
            for method in METHODS
        )
    checks.append(
        {
            "requirement": "15-row B=2 Markdown summary",
            "passed": b2_passed,
            "rows": len(b2_rows),
        }
    )
    if not b2_passed:
        errors.append("missing/invalid RESULTS_B2.md")

    complete_path = output / "EXPERIMENT_COMPLETE.json"
    marker_passed = False
    if complete_path.is_file():
        complete_marker = json.loads(complete_path.read_text(encoding="utf-8"))
        marker_passed = (
            complete_marker.get("status") == "complete"
            and complete_marker.get("protocol_id") == config["protocol_id"]
            and Path(complete_marker.get("collection_audit", "")).resolve()
            == (output / "COLLECTION_AUDIT.json").resolve()
            and Path(complete_marker.get("result_table", "")).resolve() == csv_path.resolve()
            and Path(complete_marker.get("completion_audit", "")).resolve()
            == (output / "COMPLETION_AUDIT.json").resolve()
        )
    checks.append({"requirement": "supervisor completion marker", "passed": marker_passed})
    if not marker_passed:
        errors.append("missing/invalid experiment completion marker")

    payload = {
        "status": "complete" if not errors else "failed",
        "requirements_checked": len(checks),
        "valid_probes": probe_count,
        "checks": checks,
        "errors": errors,
    }
    target = output / "COMPLETION_AUDIT.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

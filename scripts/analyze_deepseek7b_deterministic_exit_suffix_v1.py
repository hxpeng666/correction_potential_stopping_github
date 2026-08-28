#!/usr/bin/env python3
"""Analyze full held-out exit causes and paired suffix sensitivity."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from recalibrate_deepseek7b_method_exploration_ltt_v1 import align_scores, load_replay_data
from src.reproducibility import code_provenance, sha256_file, strict_reproducibility


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def transition(current: bool, dense: bool) -> str:
    return f"{'C' if current else 'W'}_to_{'C' if dense else 'W'}"


def exit_summary(data, scores: np.ndarray, threshold: float, sentinel: bool) -> dict[str, Any]:
    counts = Counter()
    token_savings = Counter()
    exit_fractions: dict[str, list[float]] = defaultdict(list)
    exited = 0
    total_dense = total_used = 0
    for start, end, dense_success, dense_tokens in data.groups:
        local = scores[start:end]
        mask = local <= threshold
        should_stop = bool(mask.any()) and not sentinel
        total_dense += dense_tokens
        if should_stop:
            local_index = int(np.argmax(mask))
            current = bool(data.row_current_success[start + local_index])
            checkpoint = int(data.row_checkpoints[start + local_index])
            name = transition(current, dense_success)
            counts[name] += 1
            token_savings[name] += dense_tokens - checkpoint
            exit_fractions[name].append(checkpoint / max(1, dense_tokens))
            total_used += checkpoint
            exited += 1
        else:
            total_used += dense_tokens
            counts["fallback_with_checkpoints"] += 1
    for _problem_id, _dense_success, dense_tokens in data.fallbacks:
        total_dense += dense_tokens
        total_used += dense_tokens
        counts["fallback_no_checkpoint"] += 1
    transitions = {}
    for name in ("C_to_C", "C_to_W", "W_to_C", "W_to_W"):
        transitions[name] = {
            "count": int(counts[name]),
            "share_of_exits": counts[name] / max(1, exited),
            "tokens_saved": int(token_savings[name]),
            "mean_exit_fraction": float(np.mean(exit_fractions[name])) if exit_fractions[name] else None,
        }
    return {
        "problems": data.problems,
        "exits": exited,
        "coverage": exited / data.problems,
        "fallback_with_checkpoints": int(counts["fallback_with_checkpoints"]),
        "fallback_no_checkpoint": int(counts["fallback_no_checkpoint"]),
        "token_reduction": 1.0 - total_used / max(1, total_dense),
        "transitions": transitions,
    }


def metric_counter() -> Counter:
    return Counter()


def update_variant(counter: Counter, row: dict[str, Any], dense_success: bool) -> None:
    current = bool(row["success"])
    counter["rows"] += 1
    counter["correct"] += int(current)
    counter["complete_boxed"] += int(bool(row["complete_boxed"]))
    counter["grader_parseable"] += int(bool(row["grader_parseable"]))
    counter["max_hit"] += int(bool(row["max_hit"]))
    counter["branch_tokens"] += int(row["branch_tokens"])
    counter[transition(current, dense_success)] += 1


def finish_variant(counter: Counter) -> dict[str, Any]:
    rows = max(1, int(counter["rows"]))
    result = {key: int(value) for key, value in counter.items()}
    result.update(
        {
            "accuracy": counter["correct"] / rows,
            "complete_boxed_rate": counter["complete_boxed"] / rows,
            "grader_parse_rate": counter["grader_parseable"] / rows,
            "max_hit_rate": counter["max_hit"] / rows,
            "mean_branch_tokens": counter["branch_tokens"] / rows,
            "W_to_C_rate": counter["W_to_C"] / rows,
        }
    )
    return result


def suffix_analysis(experiment: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = {
        (row["dataset"], row["split"], row["problem_id"]): row
        for row in manifest["entries"]
    }
    artifacts = sorted((experiment / "branches").glob("*/*/*.pt"))
    variants: dict[str, Counter] = defaultdict(metric_counter)
    by_group: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(metric_counter))
    paired: dict[str, Counter] = defaultdict(metric_counter)
    gaps: dict[str, Counter] = defaultdict(metric_counter)
    seen = set()
    commits = set()
    collection_fingerprints = set()
    for path in artifacts:
        artifact = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        key = (str(artifact["dataset"]), str(artifact["split"]), str(artifact["problem_id"]))
        if key not in expected or key in seen:
            raise AssertionError(f"unexpected or duplicate suffix artifact: {path}")
        seen.add(key)
        expected_checkpoints = list(map(int, expected[key]["checkpoints"]))
        if [int(row["checkpoint"]) for row in artifact["records"]] != expected_checkpoints:
            raise AssertionError(f"checkpoint pairing mismatch: {path}")
        commits.add(artifact["code_identity"]["git"]["commit"])
        collection_fingerprints.add(artifact["collection_fingerprint"])
        group = f"{key[0]}/{key[1]}"
        for record in artifact["records"]:
            dense_success = bool(record["dense_success"])
            baseline = record["reference"]
            update_variant(variants["current_box"], baseline, dense_success)
            update_variant(by_group[group]["current_box"], baseline, dense_success)
            candidates = record["variants"]
            successes = {"current_box": bool(baseline["success"])}
            for label, candidate in candidates.items():
                update_variant(variants[label], candidate, dense_success)
                update_variant(by_group[group][label], candidate, dense_success)
                successes[label] = bool(candidate["success"])
                local = paired[label]
                local["rows"] += 1
                local["answer_changed"] += int(candidate["prediction"] != baseline["prediction"])
                local["success_flip"] += int(bool(candidate["success"]) != bool(baseline["success"]))
                local["baseline_wrong_candidate_correct"] += int(not baseline["success"] and candidate["success"])
                local["baseline_correct_candidate_wrong"] += int(baseline["success"] and not candidate["success"])
                local["baseline_W_to_C_removed"] += int(
                    dense_success and not baseline["success"] and candidate["success"]
                )
                local["candidate_W_to_C_added"] += int(
                    dense_success and baseline["success"] and not candidate["success"]
                )
            local_gap = gaps[group]
            local_gap["rows"] += 1
            any_alternative_correct = any(value for label, value in successes.items() if label != "current_box")
            all_correct = all(successes.values())
            any_correct = any(successes.values())
            local_gap["robust_correct"] += int(all_correct)
            local_gap["robust_wrong"] += int(not any_correct)
            local_gap["suffix_sensitive"] += int(any_correct and not all_correct)
            local_gap["detection_extraction_gap"] += int(not successes["current_box"] and any_alternative_correct)
            local_gap["potential_fake_W_to_C"] += int(
                dense_success and not successes["current_box"] and any_alternative_correct
            )
    if seen != set(expected):
        missing = sorted(set(expected) - seen)[:10]
        raise AssertionError(f"missing suffix artifacts: {missing}")
    if len(commits) != 1 or len(collection_fingerprints) != 1:
        raise AssertionError("mixed collection provenance")

    def finish_gap(counter: Counter) -> dict[str, Any]:
        rows = max(1, int(counter["rows"]))
        return {
            **{key: int(value) for key, value in counter.items()},
            "suffix_sensitive_rate": counter["suffix_sensitive"] / rows,
            "detection_extraction_gap_rate": counter["detection_extraction_gap"] / rows,
            "potential_fake_W_to_C_rate": counter["potential_fake_W_to_C"] / rows,
        }

    combined_gap = Counter()
    for counter in gaps.values():
        combined_gap.update(counter)
    return {
        "trajectory_count": len(seen),
        "checkpoint_count": sum(counter["rows"] for counter in gaps.values()),
        "collection_git_commit": next(iter(commits)),
        "collection_fingerprint": next(iter(collection_fingerprints)),
        "variants": {label: finish_variant(counter) for label, counter in sorted(variants.items())},
        "by_group": {
            group: {label: finish_variant(counter) for label, counter in sorted(local.items())}
            for group, local in sorted(by_group.items())
        },
        "paired_against_current_box": {
            label: {key: int(value) for key, value in counter.items()}
            for label, counter in sorted(paired.items())
        },
        "gap_overall": finish_gap(combined_gap),
        "gap_by_group": {group: finish_gap(counter) for group, counter in sorted(gaps.items())},
    }


def markdown(exit_results: dict[str, Any], suffix: dict[str, Any]) -> str:
    lines = [
        "# Deterministic exit attribution and suffix robustness",
        "",
        "## Exit attribution",
        "",
        "| Dataset | alpha | exits | coverage | C→C | C→W | W→C | W→W | token reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, local in exit_results.items():
        for alpha, value in local.items():
            transitions = value["transitions"]
            lines.append(
                f"| {dataset} | {100 * float(alpha):g}% | {value['exits']} | {100 * value['coverage']:.2f}% | "
                f"{transitions['C_to_C']['count']} | {transitions['C_to_W']['count']} | "
                f"{transitions['W_to_C']['count']} | {transitions['W_to_W']['count']} | "
                f"{100 * value['token_reduction']:.2f}% |"
            )
    lines.extend(
        [
            "",
            "## Suffix-level paired robustness",
            "",
            "| Suffix | accuracy | complete boxed | grader parseable | W→C | max-hit | mean branch tokens |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, value in suffix["variants"].items():
        lines.append(
            f"| {label} | {100 * value['accuracy']:.2f}% | {100 * value['complete_boxed_rate']:.2f}% | "
            f"{100 * value['grader_parse_rate']:.2f}% | {value.get('W_to_C', 0)} | "
            f"{100 * value['max_hit_rate']:.2f}% | {value['mean_branch_tokens']:.2f} |"
        )
    gap = suffix["gap_overall"]
    lines.extend(
        [
            "",
            "## Detection–Extraction gap",
            "",
            f"- Suffix-sensitive checkpoints: {gap['suffix_sensitive']} / {gap['rows']} ({100 * gap['suffix_sensitive_rate']:.2f}%).",
            f"- Current suffix wrong but at least one alternative correct: {gap['detection_extraction_gap']} ({100 * gap['detection_extraction_gap_rate']:.2f}%).",
            f"- Potential fake W→C checkpoints: {gap['potential_fake_W_to_C']} ({100 * gap['potential_fake_W_to_C_rate']:.2f}%).",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    strict_reproducibility(seed=0, num_threads=1)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment = args.experiment_root.resolve()
    output = args.output.resolve()
    provenance = code_provenance(
        ROOT,
        (
            "configs/deepseek7b_deterministic_exit_suffix_v1.yaml",
            "scripts/analyze_deepseek7b_deterministic_exit_suffix_v1.py",
            "scripts/recalibrate_deepseek7b_method_exploration_ltt_v1.py",
            "src/reproducibility.py",
        ),
    )
    manifest = json.loads((experiment / "SAMPLE_MANIFEST.json").read_text(encoding="utf-8"))
    gate = json.loads((experiment / "DETERMINISM_GATE.json").read_text(encoding="utf-8"))
    if gate.get("status") != "complete" or gate.get("all_exact") is not True:
        raise AssertionError("suffix determinism gate did not pass")
    data_root = Path(config["source"]["data_root"])
    probe_root = Path(config["source"]["probe_root"])
    calibration = json.loads((probe_root / "ltt/CALIBRATION_DETAILS.json").read_text(encoding="utf-8"))
    roots = {
        "gsm8k": ("gsm8k", data_root / "gsm8k/heldout", "heldout"),
        "math500": ("math", data_root / "math500/heldout", "heldout"),
        "aime2024": ("math", data_root / "aime/heldout", "ood"),
    }
    score_cache: dict[str, Any] = {}
    exit_results: dict[str, Any] = {}
    for reported, (training_dataset, directory, score_split) in roots.items():
        data = load_replay_data(directory)
        if training_dataset not in score_cache:
            score_cache[training_dataset] = torch.load(
                probe_root / f"probes/primary/{training_dataset}/scores.pt",
                map_location="cpu",
                weights_only=False,
            )
        saved = score_cache[training_dataset]
        scores = align_scores(
            data,
            {"scores": {"aligned": saved["scores"][score_split]}, "keys": {"aligned": saved["keys"][score_split]}},
            "aligned",
        )
        exit_results[reported] = {}
        for alpha in config["exit_attribution"]["alphas"]:
            detail = calibration["primary"][training_dataset][str(alpha)]
            exit_results[reported][str(alpha)] = exit_summary(
                data,
                scores,
                float(detail["threshold"]),
                int(detail["selected_grid_index"]) == 0,
            )
    suffix = suffix_analysis(experiment, manifest)
    results = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config["protocol_id"],
        "exit_attribution": exit_results,
        "suffix_robustness": suffix,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(results, output / "RESULTS.json")
    (output / "RESULTS.md").write_text(markdown(exit_results, suffix), encoding="utf-8")
    audit = {
        "status": "complete",
        "created_at": results["created_at"],
        "code_identity": provenance,
        "determinism_gate": gate,
        "sample_manifest_sha256": sha256_file(experiment / "SAMPLE_MANIFEST.json"),
        "trajectory_count": suffix["trajectory_count"],
        "checkpoint_count": suffix["checkpoint_count"],
        "expected_trajectory_count": manifest["trajectory_count"],
        "expected_checkpoint_count": manifest["checkpoint_count"],
        "heldout_used_for_selection": False,
        "fixed_empirical_B_used": False,
        "aime_retrained_or_recalibrated": False,
    }
    if audit["trajectory_count"] != audit["expected_trajectory_count"]:
        raise AssertionError("trajectory count audit failed")
    if audit["checkpoint_count"] != audit["expected_checkpoint_count"]:
        raise AssertionError("checkpoint count audit failed")
    atomic_json(audit, output / "AUDIT.json")
    atomic_json(
        {"status": "complete", "created_at": results["created_at"], "audit": str(output / "AUDIT.json")},
        output / "EXPERIMENT_COMPLETE.json",
    )


if __name__ == "__main__":
    main()


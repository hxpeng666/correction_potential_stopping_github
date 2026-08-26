#!/usr/bin/env python3
"""Summarize corrected LTS/LYNX reproductions against the frozen current method."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "src").is_dir():
    ROOT = Path(__file__).resolve().parents[2]

DATASETS = ("gsm8k", "mmlu_pro")
BASELINES = {
    "learn_to_stop": {
        "label": "LTS",
        "paper_key": "0.999",
        "selection": "paper threshold p>=0.999",
    },
    "lynx": {
        "label": "LYNX",
        "paper_key": "0.03",
        "selection": "paper conformal delta=0.03",
    },
}
OURS = {
    "correction_bce": "Ours BCE",
    "correction_trajectory_normalized": "Ours BCE+normalized trajectory",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_ours(dataset: str) -> dict[str, dict[str, str]]:
    path = (
        ROOT
        / (
            "results/gsm8k_checkpoint_schedule_normalized_trajectory_v1/"
            "checkpoint_probe_matrix_five_targets.csv"
            if dataset == "gsm8k"
            else "results/mmlu_pro_paragraph_target_extension_v1/"
            "paragraph_all_targets_all_b.csv"
        )
    )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        row["target"]: row
        for row in rows
        if row["schedule"] == "paragraph"
        and row["budget_B"] == "2"
        and row["target"] in OURS
    }
    if set(selected) != set(OURS):
        raise RuntimeError(f"missing current-method rows in {path}: {sorted(selected)}")
    return selected


def baseline_row(
    result_root: Path,
    dataset: str,
    method: str,
    schedule: str,
    comparison_view: str,
) -> dict[str, Any]:
    meta = BASELINES[method]
    payload = read_json(result_root / dataset / method / schedule / "probe" / "results.json")
    point = payload["original_operating_points"][meta["paper_key"]]
    heldout = point["heldout"]
    calibration = point["calibration"]
    return {
        "comparison_view": comparison_view,
        "dataset": dataset,
        "method": meta["label"],
        "checkpoint_schedule": schedule,
        "selection": meta["selection"],
        "uses_common_B": False,
        "n": int(heldout["n"]),
        "dense_accuracy": float(heldout["dense_accuracy"]),
        "accuracy": float(heldout["accuracy"]),
        "accuracy_delta_pp": float(heldout["accuracy_delta_pp"]),
        "dense_mean_reasoning_tokens": float(heldout["dense_mean_reasoning_tokens"]),
        "mean_total_tokens_including_forced_answer": float(
            heldout["mean_total_tokens_including_forced_answer"]
        ),
        "total_token_reduction_pct": float(heldout["total_token_reduction_pct"]),
        "reasoning_only_token_reduction_pct": float(heldout["reasoning_token_reduction_pct"]),
        "lost_correct": int(heldout["lost_correct"]),
        "helped": int(heldout["helped"]),
        "stop_rate": float(heldout["stop_rate"]),
        "mean_checks": float(heldout["mean_checks"]),
        "calibration_lost_correct": int(calibration["lost_correct"]),
    }


def ours_row(
    dataset: str,
    target: str,
    source: dict[str, str],
    comparison_view: str,
) -> dict[str, Any]:
    n = 1319 if dataset == "gsm8k" else 1000
    dense_accuracy = float(source["heldout_dense_accuracy"])
    accuracy = float(source["heldout_accuracy"])
    lost = int(float(source["heldout_lost_correct_count"]))
    helped = round((accuracy - dense_accuracy) * n + lost)
    return {
        "comparison_view": comparison_view,
        "dataset": dataset,
        "method": OURS[target],
        "checkpoint_schedule": "paragraph",
        "selection": "calibration empirical lost-correct B=2",
        "uses_common_B": True,
        "n": n,
        "dense_accuracy": dense_accuracy,
        "accuracy": accuracy,
        "accuracy_delta_pp": 100.0 * (accuracy - dense_accuracy),
        "dense_mean_reasoning_tokens": float(source["heldout_mean_dense_reasoning_tokens"]),
        "mean_total_tokens_including_forced_answer": float(
            source["heldout_mean_reasoning_and_answer_tokens"]
        ),
        "total_token_reduction_pct": 100.0 * float(source["heldout_token_reduction"]),
        "reasoning_only_token_reduction_pct": None,
        "lost_correct": lost,
        "helped": helped,
        "stop_rate": float(source["heldout_coverage"]),
        "mean_checks": (
            float(source["mean_policy_checks_heldout"])
            if source.get("mean_policy_checks_heldout") not in (None, "")
            else None
        ),
        "calibration_lost_correct": int(float(source["calibration_lost_correct_count"])),
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| 数据集 | 方法 | checkpoint | operating point | Acc | Delta Acc(pp) | total-token reduction | lost | helped | stop rate | checks |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {schedule} | {selection} | {accuracy}% | {delta} | "
            "{reduction}% | {lost} | {helped} | {stop}% | {checks} |".format(
                dataset=row["dataset"],
                method=row["method"],
                schedule=row["checkpoint_schedule"],
                selection=row["selection"],
                accuracy=fmt(100.0 * row["accuracy"]),
                delta=fmt(row["accuracy_delta_pp"]),
                reduction=fmt(row["total_token_reduction_pct"]),
                lost=row["lost_correct"],
                helped=row["helped"],
                stop=fmt(100.0 * row["stop_rate"]),
                checks=fmt(row["mean_checks"]),
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result_root = ROOT / config["output_root"]

    views = {
        "paper_native_checkpoints": "Baselines use paper-native checkpoints; ours uses paragraph",
        "common_paragraph_checkpoints": "All methods use paragraph checkpoints",
    }
    rows: list[dict[str, Any]] = []
    dense_reference: dict[str, Any] = {}
    for dataset in DATASETS:
        current = read_ours(dataset)
        dataset_rows = []
        for view in views:
            schedule = "native" if view == "paper_native_checkpoints" else "paragraph"
            for method in BASELINES:
                row = baseline_row(result_root, dataset, method, schedule, view)
                rows.append(row)
                dataset_rows.append(row)
            for target, source in current.items():
                row = ours_row(dataset, target, source, view)
                rows.append(row)
                dataset_rows.append(row)
        dense_values = {round(row["dense_accuracy"], 12) for row in dataset_rows}
        dense_tokens = {round(row["dense_mean_reasoning_tokens"], 6) for row in dataset_rows}
        n_values = {row["n"] for row in dataset_rows}
        if len(dense_values) != 1 or len(dense_tokens) != 1 or len(n_values) != 1:
            raise RuntimeError(
                f"comparison scope mismatch for {dataset}: "
                f"dense_accuracy={dense_values}, dense_tokens={dense_tokens}, n={n_values}"
            )
        dense_reference[dataset] = {
            "n": next(iter(n_values)),
            "accuracy": next(iter(dense_values)),
            "mean_reasoning_tokens": next(iter(dense_tokens)),
        }

    summary = {
        "status": "complete",
        "scope": "LTS, LYNX, and current method only",
        "metric_definition": (
            "total-token reduction includes forced-answer branch tokens for every method; "
            "accuracy and all deltas are heldout/test metrics"
        ),
        "baseline_operating_points_do_not_use_B": True,
        "ours_operating_point": "calibration empirical lost-correct B=2",
        "views": views,
        "dense_reference": dense_reference,
        "strict_corrections": config.get("strict_corrections", {}),
        "rows": rows,
    }
    (result_root / "STRICT_SUBSET_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (result_root / "STRICT_SUBSET_SUMMARY.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Corrected strict summary: LTS, LYNX, and ours",
        "",
        "All accuracy and reduction numbers are heldout/test results. Total-token reduction includes the forced-answer branch. LTS and LYNX use their paper operating points and do not use B; ours uses the frozen calibration-only B=2 operating point.",
        "",
        "Dense references: "
        + "; ".join(
            f"{dataset}: Acc={100.0 * value['accuracy']:.2f}%, mean reasoning tokens={value['mean_reasoning_tokens']:.1f}, n={value['n']}"
            for dataset, value in dense_reference.items()
        ),
        "",
        "## Paper-native checkpoint comparison",
        "",
        table([row for row in rows if row["comparison_view"] == "paper_native_checkpoints"]),
        "",
        "## Common paragraph checkpoint comparison",
        "",
        table([row for row in rows if row["comparison_view"] == "common_paragraph_checkpoints"]),
        "",
        "## Strict corrections applied",
        "",
        "- LTS native labels use the last sentence-level greedy forced answer as the reverse-search terminal reference.",
        "- LTS paragraph labels use the paired full-reasoning greedy forced answer as the terminal reference.",
        "- LYNX adds the official 70%-of-thinking-span synthetic checkpoint only for no-cue train/calibration samples; no-cue heldout samples remain Dense fallbacks.",
    ]
    (result_root / "STRICT_SUBSET_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "summary": str(result_root / "STRICT_SUBSET_SUMMARY.md")}, indent=2))


if __name__ == "__main__":
    main()

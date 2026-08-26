#!/usr/bin/env python3
"""Analyze overthinking/over-reflection in the frozen Qwen3-4B MMLU-Pro cache.

The analysis is deliberately descriptive: it never trains a model or selects a
policy.  Correctness at a sentence checkpoint is defined by the already cached
forced-answer branch, and the Dense endpoint is appended as the final state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


DEFAULT_RUN = (
    "results/final_paper_mmlu_pro_independent_token_v2_train1000_cal500_test1000/"
    "run_float16_seed20260803"
)
DEFAULT_SPLIT = (
    "results/final_paper_mmlu_pro_independent_token_v2_train1000_cal500_test1000/"
    "splits/mmlu_pro_independent_split.json"
)
DEFAULT_OUTPUT = "results/final_paper_motivation_overthinking_v1"
BOOTSTRAP_SEED = 20260803


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def compress_states(states: Iterable[bool]) -> list[bool]:
    result: list[bool] = []
    for state in states:
        state = bool(state)
        if not result or result[-1] != state:
            result.append(state)
    return result


def contains_pattern(states: list[bool], pattern: list[bool]) -> bool:
    return any(states[i : i + len(pattern)] == pattern for i in range(len(states) - len(pattern) + 1))


def state_string(states: list[bool]) -> str:
    return "→".join("C" if state else "W" for state in states)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None, "p95": None}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
    }


def bootstrap_rate_ci(flags: np.ndarray, repeats: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = int(flags.size)
    # For a binary empirical distribution, non-parametric bootstrap counts are
    # Binomial(n, observed_rate); this avoids materializing a 10000 x n matrix.
    draws = rng.binomial(n, float(flags.mean()), size=repeats) / n
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def bootstrap_conditional_ci(
    numerator: np.ndarray, denominator: np.ndarray, repeats: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    n = int(numerator.size)
    values = []
    for _ in range(repeats):
        idx = rng.integers(0, n, size=n)
        den = int(denominator[idx].sum())
        if den:
            values.append(float(numerator[idx].sum() / den))
    return [percentile(values, 0.025), percentile(values, 0.975)]


def make_runs(rows: list[dict[str, Any]], dense_success: bool, dense_prediction: Any, dense_tokens: int) -> list[dict[str, Any]]:
    points = [
        {
            "position": int(row["checkpoint"]),
            "label": str(row["checkpoint"]),
            "success": bool(row["current_success"]),
            "prediction": row.get("current_prediction"),
            "kind": "sentence_checkpoint",
        }
        for row in rows
    ]
    points.append(
        {
            "position": int(dense_tokens),
            "label": "Dense",
            "success": bool(dense_success),
            "prediction": dense_prediction,
            "kind": "dense_endpoint",
        }
    )
    runs: list[dict[str, Any]] = []
    for point in points:
        if not runs or runs[-1]["success"] != point["success"]:
            runs.append(
                {
                    "state": "C" if point["success"] else "W",
                    "success": point["success"],
                    "start": point["label"],
                    "end": point["label"],
                    "start_position": point["position"],
                    "end_position": point["position"],
                    "count": 1,
                    "predictions": [point["prediction"]],
                    "ends_at_dense": point["kind"] == "dense_endpoint",
                }
            )
        else:
            run = runs[-1]
            run["end"] = point["label"]
            run["end_position"] = point["position"]
            run["count"] += 1
            if point["prediction"] not in run["predictions"]:
                run["predictions"].append(point["prediction"])
            run["ends_at_dense"] = point["kind"] == "dense_endpoint"
    return runs


def format_runs(runs: list[dict[str, Any]]) -> str:
    parts = []
    for run in runs:
        span = str(run["start"]) if run["start"] == run["end"] else f'{run["start"]}–{run["end"]}'
        answers = "/".join("missing" if value is None else str(value) for value in run["predictions"])
        parts.append(f'{run["state"]}@{span}({answers})')
    return " → ".join(parts)


def tail_snippet(text: str, limit: int = 420) -> str:
    compact = " ".join(text.strip().split())
    return compact if len(compact) <= limit else "…" + compact[-limit:]


def select_representatives(cases: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    used: set[str] = set()

    def choose(label: str, pool: list[dict[str, Any]], key: Any) -> None:
        for case in sorted(pool, key=key, reverse=True):
            if case["problem_id"] not in used:
                selected.append((label, case))
                used.add(case["problem_id"])
                return

    noncap = [case for case in cases if not case["reached_max_tokens"]]
    choose(
        "非触顶的长程 W→C→W",
        [case for case in noncap if case["compressed_path"] == "W→C→W"],
        lambda case: (case["tokens_after_last_correct"], case["problem_id"]),
    )
    choose(
        "非触顶的多次反思振荡",
        [case for case in noncap if case["switch_count"] >= 5 and case["c_to_w_opportunity"]],
        lambda case: (case["switch_count"], case["tokens_after_last_correct"], case["problem_id"]),
    )
    choose(
        "非触顶的最大多余推理",
        [case for case in noncap if case["c_to_w_opportunity"]],
        lambda case: (case["tokens_after_last_correct"], case["switch_count"], case["problem_id"]),
    )
    choose(
        "4096 触顶的清晰 W→C→W",
        [case for case in cases if case["reached_max_tokens"] and case["compressed_path"] == "W→C→W"],
        lambda case: (case["tokens_after_last_correct"], case["problem_id"]),
    )
    return selected


def svg_bar_chart(metrics: list[dict[str, Any]], path: Path) -> None:
    width, height = 1120, 540
    left, right, top, bottom = 105, 45, 72, 112
    plot_w, plot_h = width - left - right, height - top - bottom
    max_pct = 50.0
    bar_w = 125
    gap = (plot_w - bar_w * len(metrics)) / max(1, len(metrics) - 1)
    colors = ["#38598b", "#4f86c6", "#ff8c42", "#d1495b", "#8f5aa8"]
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#1f2937}.axis{stroke:#6b7280;stroke-width:1}.grid{stroke:#d1d5db;stroke-width:1;stroke-dasharray:4 4}.bar{rx:5}</style>',
        '<text x="560" y="34" text-anchor="middle" font-size="24" font-weight="700">Qwen3-4B 在 MMLU-Pro-1k 上的答案状态反转</text>',
        '<text x="560" y="58" text-anchor="middle" font-size="14" fill="#4b5563">sentence-step forced-answer trajectory + Dense endpoint；问题级比例</text>',
    ]
    for pct in range(0, 51, 10):
        y = top + plot_h * (1 - pct / max_pct)
        items.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        items.append(f'<text x="{left-14}" y="{y+5:.1f}" text-anchor="end" font-size="14">{pct}%</text>')
    items.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    items.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}"/>')
    for i, metric in enumerate(metrics):
        x = left + i * (bar_w + gap)
        pct = float(metric["rate"] * 100)
        h = plot_h * pct / max_pct
        y = top + plot_h - h
        items.append(f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{colors[i % len(colors)]}"/>')
        items.append(f'<text x="{x+bar_w/2:.1f}" y="{y-10:.1f}" text-anchor="middle" font-size="18" font-weight="700">{pct:.1f}%</text>')
        label_lines = metric["short_label"].split("|")
        for j, line in enumerate(label_lines):
            items.append(f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+30+j*19:.1f}" text-anchor="middle" font-size="14">{line}</text>')
    items.append(f'<text x="18" y="{top+plot_h/2:.1f}" text-anchor="middle" font-size="15" transform="rotate(-90 18 {top+plot_h/2:.1f})">held-out 问题比例</text>')
    items.append('</svg>')
    atomic_text(path, "\n".join(items) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-dir", default=DEFAULT_RUN)
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    run_dir = (root / args.run_dir).resolve()
    split_path = (root / args.split_manifest).resolve()
    output = (root / args.output_dir).resolve()
    cache_dir = run_dir / "cache" / "merged" / "heldout"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    heldout_ids = list(split["role_ids"]["heldout"])
    assert len(heldout_ids) == 1000, f"Expected 1000 held-out IDs, got {len(heldout_ids)}"
    assert len(set(heldout_ids)) == len(heldout_ids), "Duplicate held-out IDs"

    cases: list[dict[str, Any]] = []
    raw_by_id: dict[str, dict[str, Any]] = {}
    protocol_fingerprints: set[str] = set()
    dtypes: set[str] = set()
    backends: set[str] = set()
    model_revisions: set[str] = set()
    missing_files: list[str] = []
    duplicate_checkpoints: list[str] = []
    invalid_schedules: list[str] = []
    transition_edges: Counter[tuple[bool, bool]] = Counter()
    category_total: Counter[str] = Counter()
    category_metrics: dict[str, Counter[str]] = defaultdict(Counter)

    for problem_id in heldout_ids:
        path = cache_dir / f"sample_{problem_id}.pt"
        if not path.exists():
            missing_files.append(problem_id)
            continue
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        raw_by_id[problem_id] = artifact
        assert artifact["problem_id"] == problem_id
        assert artifact["status"] == "complete"
        assert artifact["record"]["policy_role"] == "heldout"
        protocol_fingerprints.add(str(artifact["protocol_fingerprint"]))
        dtypes.add(str(artifact["dtype"]))
        backends.add(str(artifact["attention_backend"]))
        model_revisions.add(str(artifact.get("model_audit", {}).get("fingerprint", artifact.get("model_audit", {}).get("path"))))
        rows = sorted(
            [row for row in artifact["rows"] if row.get("is_sentence_checkpoint")],
            key=lambda row: int(row["checkpoint"]),
        )
        positions = [int(row["checkpoint"]) for row in rows]
        if len(positions) != len(set(positions)):
            duplicate_checkpoints.append(problem_id)
        if positions != artifact["schedules"]["sentence"]:
            invalid_schedules.append(problem_id)
        current_states = [bool(row["current_success"]) for row in rows]
        dense_success = bool(artifact["dense"]["success"])
        augmented = current_states + [dense_success]
        compressed = compress_states(augmented)
        checkpoint_compressed = compress_states(current_states)
        for pair in zip(augmented, augmented[1:]):
            transition_edges[pair] += 1

        correct_rows = [row for row in rows if bool(row["current_success"])]
        max_consecutive_correct = 0
        current_consecutive_correct = 0
        for state in current_states:
            current_consecutive_correct = current_consecutive_correct + 1 if state else 0
            max_consecutive_correct = max(max_consecutive_correct, current_consecutive_correct)
        first_correct = int(correct_rows[0]["checkpoint"]) if correct_rows else None
        last_correct = int(correct_rows[-1]["checkpoint"]) if correct_rows else None
        dense_content_tokens = int(artifact["dense_content_tokens"])
        c_to_w_opportunity = bool(correct_rows and not dense_success)
        local_c_to_w = any(a and not b for a, b in zip(augmented, augmented[1:]))
        local_w_to_c = any((not a) and b for a, b in zip(augmented, augmented[1:]))
        wcw = contains_pattern(compressed, [False, True, False])
        terminal_wcw = len(compressed) >= 3 and compressed[-3:] == [False, True, False]
        checkpoint_wcw = contains_pattern(checkpoint_compressed, [False, True, False])
        category = str(artifact["record"]["category"])
        category_total[category] += 1
        category_metrics[category]["c_to_w_opportunity"] += int(c_to_w_opportunity)
        category_metrics[category]["local_c_to_w"] += int(local_c_to_w)
        category_metrics[category]["w_c_w"] += int(wcw)
        category_metrics[category]["terminal_w_c_w"] += int(terminal_wcw)
        runs = make_runs(
            rows,
            dense_success=dense_success,
            dense_prediction=artifact["dense"].get("prediction"),
            dense_tokens=dense_content_tokens,
        )
        case = {
            "problem_id": problem_id,
            "category": category,
            "question": artifact["record"]["question"],
            "choices": artifact["record"]["choices"],
            "gold_answer": artifact["gold_answer"],
            "dense_prediction": artifact["dense"].get("prediction"),
            "dense_success": dense_success,
            "dense_content_tokens": dense_content_tokens,
            "reached_max_tokens": bool(artifact["dense"]["reached_max_tokens"]),
            "sentence_checkpoint_count": len(rows),
            "has_correct_checkpoint": bool(correct_rows),
            "first_correct_checkpoint": first_correct,
            "last_correct_checkpoint": last_correct,
            "correct_checkpoint_count": len(correct_rows),
            "max_consecutive_correct_checkpoints": max_consecutive_correct,
            "tokens_after_first_correct": dense_content_tokens - first_correct if first_correct is not None else None,
            "tokens_after_last_correct": dense_content_tokens - last_correct if last_correct is not None else None,
            "c_to_w_opportunity": c_to_w_opportunity,
            "local_c_to_w": local_c_to_w,
            "local_w_to_c": local_w_to_c,
            "w_c_w": wcw,
            "checkpoint_only_w_c_w": checkpoint_wcw,
            "terminal_w_c_w": terminal_wcw,
            "compressed_path": state_string(compressed),
            "switch_count": max(0, len(compressed) - 1),
            "state_runs": runs,
            "state_path_detailed": format_runs(runs),
        }
        cases.append(case)

    if missing_files:
        raise RuntimeError(f"Missing {len(missing_files)} held-out artifacts")
    assert len(cases) == 1000
    assert len(protocol_fingerprints) == 1
    assert len(dtypes) == 1
    assert len(backends) == 1
    assert not duplicate_checkpoints
    assert not invalid_schedules

    flags = {
        "has_any_state_switch": np.asarray([case["switch_count"] > 0 for case in cases], dtype=np.int8),
        "local_c_to_w": np.asarray([case["local_c_to_w"] for case in cases], dtype=np.int8),
        "w_c_w": np.asarray([case["w_c_w"] for case in cases], dtype=np.int8),
        "c_to_w_opportunity": np.asarray([case["c_to_w_opportunity"] for case in cases], dtype=np.int8),
        "terminal_w_c_w": np.asarray([case["terminal_w_c_w"] for case in cases], dtype=np.int8),
    }
    label_map = {
        "has_any_state_switch": "至少发生一次 C/W 状态切换",
        "local_c_to_w": "至少发生一次局部 C→W",
        "w_c_w": "压缩轨迹包含 W→C→W",
        "c_to_w_opportunity": "曾在 checkpoint 正确但 Dense 最终错误",
        "terminal_w_c_w": "压缩轨迹最终以 W→C→W 结束",
    }
    short_map = {
        "has_any_state_switch": "任意状态|切换",
        "local_c_to_w": "局部|C→W",
        "w_c_w": "含|W→C→W",
        "c_to_w_opportunity": "中间正确|Dense 错误",
        "terminal_w_c_w": "末段|W→C→W",
    }
    metrics = []
    for index, (name, values) in enumerate(flags.items()):
        count = int(values.sum())
        metrics.append(
            {
                "metric": name,
                "label": label_map[name],
                "short_label": short_map[name],
                "count": count,
                "denominator": len(cases),
                "rate": count / len(cases),
                "bootstrap_95_ci": bootstrap_rate_ci(values, args.bootstrap_repeats, BOOTSTRAP_SEED + index),
            }
        )

    dense_wrong = np.asarray([not case["dense_success"] for case in cases], dtype=np.int8)
    opportunity = flags["c_to_w_opportunity"]
    noncap_cases = [case for case in cases if not case["reached_max_tokens"]]
    opportunity_cases = [case for case in cases if case["c_to_w_opportunity"]]
    noncap_opportunity = [case for case in noncap_cases if case["c_to_w_opportunity"]]
    checkpoint_to_dense_states: Counter[str] = Counter()
    for problem_id in heldout_ids:
        artifact = raw_by_id[problem_id]
        dense_state = "C" if artifact["dense"]["success"] else "W"
        for row in artifact["rows"]:
            if row.get("is_sentence_checkpoint"):
                current_state = "C" if row["current_success"] else "W"
                checkpoint_to_dense_states[f"{current_state}_to_{dense_state}"] += 1
    summary = {
        "analysis_id": "qwen3_4b_mmlu_pro_1k_overthinking_motivation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "model": "Qwen3-4B",
            "dataset": "TIGER-Lab/MMLU-Pro",
            "split": "fixed held-out subset of official test",
            "heldout_n": len(cases),
            "global_seed": int(next(iter(raw_by_id.values()))["seed"]),
            "dtype": sorted(dtypes),
            "attention_backend": sorted(backends),
            "protocol_fingerprint": sorted(protocol_fingerprints),
            "split_fingerprint": split["fingerprint"],
            "checkpoint_definition": "pure sentence-step, 64–768 reasoning tokens, minimum gap 8",
            "state_definition": "forced-answer correctness at each sentence checkpoint; Dense correctness appended as endpoint",
        },
        "headline_metrics": metrics,
        "dense": {
            "correct": int(sum(case["dense_success"] for case in cases)),
            "wrong": int(dense_wrong.sum()),
            "accuracy": float(np.mean([case["dense_success"] for case in cases])),
            "reached_4096": int(sum(case["reached_max_tokens"] for case in cases)),
            "reached_4096_rate": float(np.mean([case["reached_max_tokens"] for case in cases])),
        },
        "recoverability_oracle_descriptive_only": {
            "dense_wrong_with_earlier_correct_checkpoint": int(opportunity.sum()),
            "share_of_all_questions": float(opportunity.mean()),
            "share_of_dense_errors": float(opportunity.sum() / dense_wrong.sum()),
            "share_of_dense_errors_bootstrap_95_ci": bootstrap_conditional_ci(
                opportunity, dense_wrong, args.bootstrap_repeats, BOOTSTRAP_SEED + 100
            ),
            "dense_or_earliest_correct_oracle_accuracy": float((sum(case["dense_success"] for case in cases) + opportunity.sum()) / len(cases)),
            "oracle_gain_over_dense_pp": float(100 * opportunity.mean()),
            "warning": "Oracle upper bound only; it uses future correctness and is not deployable.",
        },
        "transition_edges": {
            f'{"C" if a else "W"}_to_{"C" if b else "W"}': int(count)
            for (a, b), count in sorted(transition_edges.items())
        },
        "checkpoint_to_dense_four_states": {
            "definition": "Each sentence checkpoint is compared with the Dense endpoint of the same problem.",
            "total_checkpoint_rows": int(sum(checkpoint_to_dense_states.values())),
            "counts": dict(sorted(checkpoint_to_dense_states.items())),
            "rates": {
                key: value / sum(checkpoint_to_dense_states.values())
                for key, value in sorted(checkpoint_to_dense_states.items())
            },
        },
        "correct_window_robustness": {
            "opportunity_problem_count": len(opportunity_cases),
            "max_consecutive_correct_checkpoints": describe(
                [case["max_consecutive_correct_checkpoints"] for case in opportunity_cases]
            ),
            "problems_with_at_least_k_consecutive_correct_checkpoints": {
                str(k): sum(case["max_consecutive_correct_checkpoints"] >= k for case in opportunity_cases)
                for k in [1, 2, 3, 5, 10]
            },
        },
        "opportunity_token_statistics": {
            "all_139": {
                "first_correct_checkpoint": describe([case["first_correct_checkpoint"] for case in opportunity_cases]),
                "last_correct_checkpoint": describe([case["last_correct_checkpoint"] for case in opportunity_cases]),
                "correct_checkpoint_count": describe([case["correct_checkpoint_count"] for case in opportunity_cases]),
                "max_consecutive_correct_checkpoints": describe(
                    [case["max_consecutive_correct_checkpoints"] for case in opportunity_cases]
                ),
                "tokens_after_first_correct": describe([case["tokens_after_first_correct"] for case in opportunity_cases]),
                "tokens_after_last_correct": describe([case["tokens_after_last_correct"] for case in opportunity_cases]),
            },
            "excluding_4096_cap": {
                "n": len(noncap_opportunity),
                "tokens_after_first_correct": describe([case["tokens_after_first_correct"] for case in noncap_opportunity]),
                "tokens_after_last_correct": describe([case["tokens_after_last_correct"] for case in noncap_opportunity]),
            },
        },
        "non_4096_sensitivity": {
            "denominator": len(noncap_cases),
            "c_to_w_opportunity_count": sum(case["c_to_w_opportunity"] for case in noncap_cases),
            "c_to_w_opportunity_rate": float(np.mean([case["c_to_w_opportunity"] for case in noncap_cases])),
            "local_c_to_w_count": sum(case["local_c_to_w"] for case in noncap_cases),
            "local_c_to_w_rate": float(np.mean([case["local_c_to_w"] for case in noncap_cases])),
            "w_c_w_count": sum(case["w_c_w"] for case in noncap_cases),
            "w_c_w_rate": float(np.mean([case["w_c_w"] for case in noncap_cases])),
            "terminal_w_c_w_count": sum(case["terminal_w_c_w"] for case in noncap_cases),
            "terminal_w_c_w_rate": float(np.mean([case["terminal_w_c_w"] for case in noncap_cases])),
        },
        "interpretation_guardrails": [
            "Checkpoint states come from cached stochastic forced-answer branches and are operational outcomes, not directly observed latent beliefs.",
            "W→C→W is computed after compressing consecutive equal correctness states and includes the Dense endpoint.",
            "All statistics are descriptive held-out diagnostics; no policy or threshold was selected here.",
            "MMLU-Pro-1k is a fixed held-out subset of official test, not the complete MMLU-Pro benchmark.",
        ],
    }

    category_rows = []
    for category in sorted(category_total):
        n = category_total[category]
        category_rows.append(
            {
                "category": category,
                "n": n,
                "c_to_w_opportunity_count": category_metrics[category]["c_to_w_opportunity"],
                "c_to_w_opportunity_rate": category_metrics[category]["c_to_w_opportunity"] / n,
                "local_c_to_w_count": category_metrics[category]["local_c_to_w"],
                "local_c_to_w_rate": category_metrics[category]["local_c_to_w"] / n,
                "w_c_w_count": category_metrics[category]["w_c_w"],
                "w_c_w_rate": category_metrics[category]["w_c_w"] / n,
                "terminal_w_c_w_count": category_metrics[category]["terminal_w_c_w"],
                "terminal_w_c_w_rate": category_metrics[category]["terminal_w_c_w"] / n,
            }
        )

    selected = select_representatives(cases)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            root / "models" / "Qwen3-4B", local_files_only=True
        )
    except Exception:
        tokenizer = None
    representative_rows = []
    for label, case in selected:
        artifact = raw_by_id[case["problem_id"]]
        row_by_checkpoint = {
            int(row["checkpoint"]): row
            for row in artifact["rows"]
            if row.get("is_sentence_checkpoint")
        }
        transition_details = []
        previous = None
        for run in case["state_runs"]:
            if previous is not None:
                checkpoint = run["start_position"]
                if run["start"] == "Dense":
                    snippet = tail_snippet(artifact["dense"]["text"])
                elif tokenizer is not None and checkpoint in row_by_checkpoint:
                    prefix = tokenizer.decode(
                        row_by_checkpoint[checkpoint]["prefix_token_ids"], skip_special_tokens=False
                    )
                    snippet = tail_snippet(prefix)
                else:
                    snippet = ""
                transition_details.append(
                    {
                        "transition": f'{previous["state"]}→{run["state"]}',
                        "at": run["start"],
                        "prediction": run["predictions"][0],
                        "reasoning_prefix_tail": snippet,
                    }
                )
            previous = run
        representative_rows.append(
            {
                "selection_reason": label,
                **{key: case[key] for key in [
                    "problem_id", "category", "question", "choices", "gold_answer",
                    "dense_prediction", "dense_success", "dense_content_tokens",
                    "reached_max_tokens", "first_correct_checkpoint", "last_correct_checkpoint",
                    "tokens_after_last_correct", "compressed_path", "state_path_detailed",
                    "switch_count",
                ]},
                "transition_details": transition_details,
                "dense_reasoning_tail": tail_snippet(artifact["dense"]["text"], 650),
            }
        )

    pattern_counts = Counter(case["compressed_path"] for case in cases)
    pattern_rows = [
        {"compressed_path": path, "count": count, "rate": count / len(cases)}
        for path, count in pattern_counts.most_common()
    ]
    flat_case_rows = []
    for case in cases:
        flat_case_rows.append(
            {key: case[key] for key in [
                "problem_id", "category", "gold_answer", "dense_prediction", "dense_success",
                "dense_content_tokens", "reached_max_tokens", "sentence_checkpoint_count",
                "has_correct_checkpoint", "first_correct_checkpoint", "last_correct_checkpoint",
                "correct_checkpoint_count", "tokens_after_first_correct", "tokens_after_last_correct",
                "max_consecutive_correct_checkpoints",
                "c_to_w_opportunity", "local_c_to_w", "local_w_to_c", "w_c_w",
                "checkpoint_only_w_c_w", "terminal_w_c_w", "compressed_path", "switch_count",
                "state_path_detailed",
            ]}
        )

    audit = {
        "status": "pass",
        "heldout_manifest_count": len(heldout_ids),
        "loaded_count": len(cases),
        "unique_problem_ids": len({case["problem_id"] for case in cases}),
        "missing_files": missing_files,
        "duplicate_checkpoint_samples": duplicate_checkpoints,
        "schedule_mismatch_samples": invalid_schedules,
        "unique_protocol_fingerprints": sorted(protocol_fingerprints),
        "unique_dtypes": sorted(dtypes),
        "unique_attention_backends": sorted(backends),
        "heldout_only": all(raw_by_id[problem_id]["record"]["policy_role"] == "heldout" for problem_id in heldout_ids),
        "selection_or_training_performed": False,
        "new_llm_generation_performed": False,
    }

    atomic_json(output / "summary.json", summary)
    atomic_json(output / "representative_cases.json", representative_rows)
    atomic_json(output / "INTEGRITY_AUDIT.json", audit)
    atomic_csv(output / "tables" / "problem_level_cases.csv", flat_case_rows, list(flat_case_rows[0]))
    atomic_csv(output / "tables" / "category_breakdown.csv", category_rows, list(category_rows[0]))
    atomic_csv(output / "tables" / "compressed_pattern_counts.csv", pattern_rows, list(pattern_rows[0]))
    svg_bar_chart(metrics, output / "figures" / "overthinking_problem_rates.svg")

    metric_by_name = {item["metric"]: item for item in metrics}
    top_categories = sorted(category_rows, key=lambda row: (row["c_to_w_opportunity_rate"], row["n"]), reverse=True)[:5]
    report = f"""# Motivation 数据审计：Qwen3-4B 在 MMLU-Pro-1k 上的 overthinking / over-reflection

## 结论先行

在冻结的 MMLU-Pro-1k held-out 子集上，Qwen3-4B 的 Dense accuracy 为 **{summary['dense']['accuracy']*100:.1f}%**。然而，**{int(opportunity.sum())}/1000（{opportunity.mean()*100:.1f}%）** 的题目至少在一个 sentence checkpoint 上能够正确作答，却在继续推理后以错误的 Dense 答案结束。换言之，在 Dense 的 {int(dense_wrong.sum())} 个最终错误中，**{opportunity.sum()/dense_wrong.sum()*100:.1f}%** 曾经出现过可被捕获的中间正确状态。

这不是只有 4096-token 触顶才会出现的伪象：排除全部 {summary['dense']['reached_4096']} 个触顶样本后，剩余 {len(noncap_cases)} 题中仍有 **{len(noncap_opportunity)}/{len(noncap_cases)}（{len(noncap_opportunity)/len(noncap_cases)*100:.1f}%）** 呈现“checkpoint 正确、Dense 最终错误”。

## 统计定义

- 数据：TIGER-Lab/MMLU-Pro official test 的固定、分层 held-out 1,000 题；这不是完整 MMLU-Pro test。
- 模型：Qwen3-4B，FP16，SDPA，单 seed 20260803，5-shot，thinking enabled。
- checkpoint：pure sentence-step，64–768 reasoning tokens，最小间隔 8 tokens。
- 每个 checkpoint 的 C/W：从该 reasoning prefix 追加统一 suffix 后，缓存 forced-answer 是否正确。
- 轨迹末尾追加 Dense endpoint 的 C/W；连续相同状态先压缩，再识别 W→C→W。
- `C→W opportunity`：至少一个 checkpoint 为 C，但 Dense endpoint 为 W。这对应“继续思考破坏了已经可用的正确答案”的问题级机会，而不是某个策略实际停止的计数。

## 核心数字

| 现象 | 问题数 / 1000 | 比例 | problem-bootstrap 95% CI |
|---|---:|---:|---:|
"""
    for name in ["has_any_state_switch", "local_c_to_w", "w_c_w", "c_to_w_opportunity", "terminal_w_c_w"]:
        item = metric_by_name[name]
        report += f"| {item['label']} | {item['count']} | {item['rate']*100:.1f}% | [{item['bootstrap_95_ci'][0]*100:.1f}%, {item['bootstrap_95_ci'][1]*100:.1f}%] |\n"
    report += f"""

另外，在全部相邻 sentence checkpoint 加 Dense endpoint 的状态边上，共观察到 **{transition_edges[(True, False)]} 次局部 C→W** 与 **{transition_edges[(False, True)]} 次局部 W→C**。这些是“边”的数量，不能与上表的问题数相加。

按原方法的四类定义，将每个 sentence checkpoint 与同题 Dense endpoint 比较，在 {sum(checkpoint_to_dense_states.values())} 个 checkpoint 行中有：**C→W {checkpoint_to_dense_states['C_to_W']}（{checkpoint_to_dense_states['C_to_W']/sum(checkpoint_to_dense_states.values())*100:.1f}%）**、W→C {checkpoint_to_dense_states['W_to_C']}、W→W {checkpoint_to_dense_states['W_to_W']}、C→C {checkpoint_to_dense_states['C_to_C']}。这里的 C→W 行集中在 {len(opportunity_cases)} 道问题上，因此论文主文应优先报告问题级 13.9%，checkpoint 行数作为机制补充。

### 正确窗口之后还继续了多久

对 {len(opportunity_cases)} 个 `checkpoint C / Dense W` 问题：

- 首次正确 checkpoint 的中位位置为 {summary['opportunity_token_statistics']['all_139']['first_correct_checkpoint']['median']:.0f} tokens；
- 最后一次正确 checkpoint 后，模型仍继续生成中位 **{summary['opportunity_token_statistics']['all_139']['tokens_after_last_correct']['median']:.0f} tokens**；
- 排除 4096-token 触顶后，这一中位数仍为 **{summary['opportunity_token_statistics']['excluding_4096_cap']['tokens_after_last_correct']['median']:.0f} tokens**；
- 每题正确 checkpoint 数的中位数为 {summary['opportunity_token_statistics']['all_139']['correct_checkpoint_count']['median']:.0f}，说明总体信号不只来自单个孤立 checkpoint。
- {summary['correct_window_robustness']['problems_with_at_least_k_consecutive_correct_checkpoints']['2']}/{len(opportunity_cases)} 个机会样本至少连续两个 checkpoint 正确，{summary['correct_window_robustness']['problems_with_at_least_k_consecutive_correct_checkpoints']['5']}/{len(opportunity_cases)} 个至少连续五个正确；最长连续正确窗口的中位数为 {summary['correct_window_robustness']['max_consecutive_correct_checkpoints']['median']:.0f} 个 checkpoint。这一结果降低了“全部由单个 forced-answer 随机命中造成”的解释力度，但不能完全消除采样噪声。

描述性 earliest-correct oracle 若能事后知道正确性，可将 accuracy 从 {summary['dense']['accuracy']*100:.1f}% 提高到 **{summary['recoverability_oracle_descriptive_only']['dense_or_earliest_correct_oracle_accuracy']*100:.1f}%**。该数字只是 opportunity upper bound，使用了未来 correctness，**不是可部署方法的性能**。

## 类别分解

`checkpoint C / Dense W` 比例最高的五类如下。类别样本量不同，因此主要用于说明现象跨领域存在，而不是进行类别排名推断。

| 类别 | n | C→W opportunity | 比例 |
|---|---:|---:|---:|
"""
    for row in top_categories:
        report += f"| {row['category']} | {row['n']} | {row['c_to_w_opportunity_count']} | {row['c_to_w_opportunity_rate']*100:.1f}% |\n"
    report += "\n## 代表性真实样例\n\n"
    for index, case in enumerate(representative_rows, 1):
        choices = "; ".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(case["choices"]))
        report += f"""### 样例 {index}：{case['selection_reason']}

- ID / 类别：`{case['problem_id']}` / {case['category']}
- 问题：{case['question']}
- 选项：{choices}
- Gold / Dense：**{case['gold_answer']} / {case['dense_prediction']}（{'C' if case['dense_success'] else 'W'}）**
- Dense 长度：{case['dense_content_tokens']} tokens；4096 触顶：{'是' if case['reached_max_tokens'] else '否'}
- 状态路径：`{case['state_path_detailed']}`
- 最后正确 checkpoint：{case['last_correct_checkpoint']}；其后仍生成 {case['tokens_after_last_correct']} tokens

"""
        for transition in case["transition_details"]:
            report += f"- {transition['transition']} @ {transition['at']}，分支答案 {transition['prediction']}；prefix 尾部：{transition['reasoning_prefix_tail']}\n"
        report += f"- Dense reasoning 尾部：{case['dense_reasoning_tail']}\n\n"
    report += """## 论文写作时应如何表述

推荐主句：

> On a fixed 1,000-example MMLU-Pro held-out subset, 13.9% of problems were answerable correctly at least once during sentence-step reasoning but ended with an incorrect Dense answer; these cases accounted for 34.6% of all Dense errors. Moreover, 21.1% of trajectories exhibited a W→C→W correctness motif, providing direct evidence that additional reflection can both repair and subsequently damage an answer.

中文可写为：

> 在固定的 MMLU-Pro-1k held-out 子集上，13.9% 的问题曾在 sentence-step reasoning 的至少一个检查点上正确作答，却在 Dense 终点退化为错误；它们占全部 Dense 错误的 34.6%。同时，21.1% 的轨迹出现 W→C→W 正确性反转，直接表明额外反思既可能修正答案，也可能再次破坏已经获得的正确结果。

## 边界与限制

1. forced-answer branch 使用冻结协议下的采样生成，因此 C/W 是“立即停止并作答”的操作性结果，不等价于直接观测模型的潜在信念；局部振荡可能同时包含 reasoning 演化和短答案采样方差。
2. 这是 held-out 描述性诊断，不用于训练、阈值或策略选择。
3. MMLU-Pro-1k 是 official test 的固定分层子集，不应写成完整 MMLU-Pro benchmark。
4. 4096 触顶会放大多余 token 数，因此报告同时给出排除触顶样本的敏感性结果。
5. earliest-correct oracle 偷看未来 correctness，只能说明可恢复空间，不能作为实际 baseline。
"""
    atomic_text(output / "MOTIVATION_OVERTHINKING_REPORT_ZH.md", report)

    ledger = f"""# 实验账本：Motivation overthinking 审计

## Charter

```yaml
objective: 统计 Qwen3-4B 在 MMLU-Pro-1k held-out 上的 C→W 与 W→C→W 真实轨迹现象
hypothesis: Dense 推理会在非忽略比例的问题上破坏中间已经正确的答案
method_version: descriptive_cache_audit_v1
baseline: Dense endpoint
primary_metric:
  name: checkpoint_correct_dense_wrong_rate
  direction: descriptive
repetitions:
  seeds: [20260803]
  aggregation: problem-level
  uncertainty_method: 10000-repeat problem bootstrap
train_validation_test_policy: 仅读取 frozen held-out IDs；不训练、不选策略
success_condition: 完整读取1000题且定义、分母、指纹和案例均可复核
stop_conditions: [缓存缺失, 指纹混合, checkpoint错位]
```

## Run record

```yaml
run_id: qwen3_4b_mmlu_pro_1k_overthinking_motivation_v1
purpose: diagnostic
status: completed
command: python scripts/analyze_motivation_overthinking_mmlupro_v1.py
data_split_id: {split['fingerprint']}
seed: 20260803
artifact_dir: {output}
results:
  checkpoint_correct_dense_wrong: {int(opportunity.sum())}/1000
  any_W_C_W: {int(flags['w_c_w'].sum())}/1000
  local_C_W_problem: {int(flags['local_c_to_w'].sum())}/1000
verdict: 支持 overthinking/over-reflection motivation；需保留 forced-answer 随机性与 MMLU-Pro-1k 范围限制
```
"""
    atomic_text(output / "EXPERIMENT_LEDGER.md", ledger)
    atomic_text(output / "pipeline.complete", "complete\n")
    print(json.dumps({"status": "complete", "output": str(output), "headline": summary["headline_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    random.seed(BOOTSTRAP_SEED)
    np.random.seed(BOOTSTRAP_SEED)
    main()

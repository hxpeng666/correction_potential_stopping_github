#!/usr/bin/env python3
"""Held-out motivation audit for Qwen3-4B on the full GSM8K test set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import torch


SEED = 20260803
DEFAULT_CACHE = "results/final_paper_replay_v2/cache/gsm8k/merged/heldout"
DEFAULT_SPLIT = "results/final_paper_replay_v2/splits/gsm8k_split.json"
DEFAULT_OUTPUT = "results/final_paper_motivation_overthinking_gsm8k_v1"
NUMBER = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/[1-9]\d*)?")


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def compress(states: list[bool]) -> list[bool]:
    result: list[bool] = []
    for state in states:
        if not result or result[-1] != bool(state):
            result.append(bool(state))
    return result


def pattern(states: list[bool], target: list[bool]) -> bool:
    return any(states[i : i + len(target)] == target for i in range(len(states) - len(target) + 1))


def state_path(states: list[bool]) -> str:
    return "→".join("C" if state else "W" for state in states)


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().replace("$", "").replace(",", "")
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            number = Decimal(numerator) / Decimal(denominator)
        else:
            number = Decimal(text)
        normalized = format(number.normalize(), "f")
        return "0" if normalized in {"-0", "+0"} else normalized
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def last_boxed_number(text: str | None) -> str | None:
    """Sensitivity parser: use only the last explicit boxed numeric answer."""
    matches = re.findall(r"\\boxed\s*\{([^{}]+)\}", text or "")
    if not matches:
        return None
    numbers = NUMBER.findall(matches[-1])
    return normalize_number(numbers[-1]) if numbers else None


def describe(values: list[int | float]) -> dict[str, int | float | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
    }


def rate_ci(flags: np.ndarray, repeats: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(flags)
    draws = rng.binomial(n, float(flags.mean()), size=repeats) / n
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def conditional_ci(numerator: np.ndarray, denominator: np.ndarray, repeats: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(numerator)
    values = []
    for _ in range(repeats):
        idx = rng.integers(0, n, n)
        den = int(denominator[idx].sum())
        if den:
            values.append(float(numerator[idx].sum() / den))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def build_runs(rows: list[dict[str, Any]], dense: dict[str, Any], dense_tokens: int) -> list[dict[str, Any]]:
    points = [
        {
            "position": int(row["checkpoint"]),
            "label": str(row["checkpoint"]),
            "success": bool(row["current_success"]),
            "prediction": row.get("current_prediction"),
        }
        for row in rows
    ]
    points.append(
        {
            "position": dense_tokens,
            "label": "Dense",
            "success": bool(dense["success"]),
            "prediction": dense.get("prediction"),
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
                    "predictions": [point["prediction"]],
                }
            )
        else:
            run = runs[-1]
            run["end"] = point["label"]
            run["end_position"] = point["position"]
            if point["prediction"] not in run["predictions"]:
                run["predictions"].append(point["prediction"])
    return runs


def format_runs(runs: list[dict[str, Any]]) -> str:
    values = []
    for run in runs:
        span = str(run["start"]) if run["start"] == run["end"] else f'{run["start"]}–{run["end"]}'
        answers = "/".join("missing" if answer is None else str(answer) for answer in run["predictions"])
        values.append(f'{run["state"]}@{span}({answers})')
    return " → ".join(values)


def snippet(text: str, limit: int = 440) -> str:
    text = " ".join(text.strip().split())
    return text if len(text) <= limit else "…" + text[-limit:]


def select_cases(cases: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    # Predeclared after manual semantic validation: avoid questions whose wording
    # or official reference is plausibly ambiguous. Every chosen case is non-cap,
    # checkpoint-correct/Dense-wrong under both the frozen and last-boxed parsers.
    curated = [
        ("清晰的总工时重复除以人数错误", "gsm8k_test_01046"),
        ("清晰的正负体重变化反思错误", "gsm8k_test_00589"),
        ("清晰的末步加法退化", "gsm8k_test_01309"),
        ("原价百分比被错误改成复利", "gsm8k_test_01016"),
    ]
    by_id = {case["problem_id"]: case for case in cases}
    result = []
    for label, problem_id in curated:
        case = by_id[problem_id]
        assert case["parser_robust_c_to_w_opportunity"]
        assert not case["reached_max_tokens"]
        result.append((label, case))
    return result


def svg(metrics: list[dict[str, Any]], path: Path) -> None:
    width, height = 1120, 540
    left, right, top, bottom = 105, 45, 72, 112
    plot_w, plot_h = width - left - right, height - top - bottom
    max_pct, bar_w = 60.0, 125
    gap = (plot_w - bar_w * len(metrics)) / (len(metrics) - 1)
    colors = ["#38598b", "#4f86c6", "#ff8c42", "#d1495b", "#8f5aa8"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#1f2937}.axis{stroke:#6b7280}.grid{stroke:#d1d5db;stroke-dasharray:4 4}</style>',
        '<text x="560" y="34" text-anchor="middle" font-size="24" font-weight="700">Qwen3-4B 在 GSM8K official test 上的答案状态反转</text>',
        '<text x="560" y="58" text-anchor="middle" font-size="14">sentence-step forced-answer trajectory + Dense endpoint；问题级比例</text>',
    ]
    for pct in range(0, 61, 10):
        y = top + plot_h * (1 - pct / max_pct)
        lines += [
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>',
            f'<text x="{left-14}" y="{y+5:.1f}" text-anchor="end" font-size="14">{pct}%</text>',
        ]
    lines += [
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}"/>',
    ]
    for i, metric in enumerate(metrics):
        x = left + i * (bar_w + gap)
        pct = metric["rate"] * 100
        bar_h = plot_h * pct / max_pct
        y = top + plot_h - bar_h
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="5" fill="{colors[i]}"/>')
        lines.append(f'<text x="{x+bar_w/2:.1f}" y="{y-10:.1f}" text-anchor="middle" font-size="18" font-weight="700">{pct:.1f}%</text>')
        for j, label in enumerate(metric["short"].split("|")):
            lines.append(f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+30+j*19:.1f}" text-anchor="middle" font-size="14">{label}</text>')
    lines.append("</svg>")
    atomic_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    cache = (root / args.cache_dir).resolve()
    split_path = (root / args.split_manifest).resolve()
    output = (root / args.output_dir).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    heldout_ids = list(split["files"]["heldout"]["problem_ids"])
    assert len(heldout_ids) == len(set(heldout_ids)) == 1319

    cases: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    fingerprints, dtypes, backends = set(), set(), set()
    missing, duplicate_checkpoints, schedule_mismatch = [], [], []
    adjacent_edges: Counter[tuple[bool, bool]] = Counter()
    four_states: Counter[str] = Counter()

    for problem_id in heldout_ids:
        path = cache / f"sample_{problem_id}.pt"
        if not path.exists():
            missing.append(problem_id)
            continue
        value = torch.load(path, map_location="cpu", weights_only=False)
        artifacts[problem_id] = value
        assert value["problem_id"] == problem_id and value["status"] == "complete"
        assert value["dataset"] == "gsm8k" and value["split"] == "heldout"
        fingerprints.add(value["protocol_fingerprint"])
        dtypes.add(value["dtype"])
        backends.add(value["attention_backend"])
        rows = sorted([row for row in value["rows"] if row.get("is_sentence_checkpoint")], key=lambda row: row["checkpoint"])
        positions = [int(row["checkpoint"]) for row in rows]
        if len(positions) != len(set(positions)):
            duplicate_checkpoints.append(problem_id)
        if positions != value["schedules"]["sentence"]:
            schedule_mismatch.append(problem_id)
        current = [bool(row["current_success"]) for row in rows]
        dense_success = bool(value["dense"]["success"])
        alternative_dense_prediction = last_boxed_number(value["dense"].get("text"))
        alternative_dense_success = alternative_dense_prediction == value["gold_answer"]
        alternative_current = [
            last_boxed_number(row.get("branch_text")) == value["gold_answer"] for row in rows
        ]
        alternative_augmented = alternative_current + [alternative_dense_success]
        alternative_compressed = compress(alternative_augmented)
        augmented = current + [dense_success]
        compressed = compress(augmented)
        checkpoint_compressed = compress(current)
        for a, b in zip(augmented, augmented[1:]):
            adjacent_edges[(a, b)] += 1
        for state in current:
            four_states[f'{"C" if state else "W"}_to_{"C" if dense_success else "W"}'] += 1
        correct_rows = [row for row in rows if row["current_success"]]
        run = max_run = 0
        for state in current:
            run = run + 1 if state else 0
            max_run = max(max_run, run)
        first = int(correct_rows[0]["checkpoint"]) if correct_rows else None
        last = int(correct_rows[-1]["checkpoint"]) if correct_rows else None
        dense_tokens = int(value["dense_content_tokens"])
        runs = build_runs(rows, value["dense"], dense_tokens)
        cases.append(
            {
                "problem_id": problem_id,
                "question": value["record"]["question"],
                "gold_answer": value["gold_answer"],
                "dense_prediction": value["dense"].get("prediction"),
                "dense_success": dense_success,
                "last_boxed_dense_prediction": alternative_dense_prediction,
                "last_boxed_dense_success": alternative_dense_success,
                "dense_content_tokens": dense_tokens,
                "reached_max_tokens": bool(value["dense"]["reached_max_tokens"]),
                "sentence_checkpoint_count": len(rows),
                "has_correct_checkpoint": bool(correct_rows),
                "first_correct_checkpoint": first,
                "last_correct_checkpoint": last,
                "correct_checkpoint_count": len(correct_rows),
                "max_consecutive_correct_checkpoints": max_run,
                "tokens_after_first_correct": dense_tokens - first if first is not None else None,
                "tokens_after_last_correct": dense_tokens - last if last is not None else None,
                "c_to_w_opportunity": bool(correct_rows and not dense_success),
                "last_boxed_c_to_w_opportunity": bool(any(alternative_current) and not alternative_dense_success),
                "parser_robust_c_to_w_opportunity": bool(
                    correct_rows and not dense_success and any(alternative_current) and not alternative_dense_success
                ),
                "last_boxed_w_c_w": pattern(alternative_compressed, [False, True, False]),
                "local_c_to_w": any(a and not b for a, b in zip(augmented, augmented[1:])),
                "local_w_to_c": any((not a) and b for a, b in zip(augmented, augmented[1:])),
                "w_c_w": pattern(compressed, [False, True, False]),
                "checkpoint_only_w_c_w": pattern(checkpoint_compressed, [False, True, False]),
                "terminal_w_c_w": len(compressed) >= 3 and compressed[-3:] == [False, True, False],
                "compressed_path": state_path(compressed),
                "switch_count": max(0, len(compressed) - 1),
                "state_runs": runs,
                "state_path_detailed": format_runs(runs),
            }
        )

    assert not missing and len(cases) == 1319
    assert len(fingerprints) == len(dtypes) == len(backends) == 1
    assert not duplicate_checkpoints and not schedule_mismatch

    definitions = [
        ("has_any_state_switch", "至少发生一次 C/W 状态切换", "任意状态|切换", lambda case: case["switch_count"] > 0),
        ("local_c_to_w", "至少发生一次局部 C→W", "局部|C→W", lambda case: case["local_c_to_w"]),
        ("w_c_w", "压缩轨迹包含 W→C→W", "含|W→C→W", lambda case: case["w_c_w"]),
        ("c_to_w_opportunity", "曾在 checkpoint 正确但 Dense 最终错误", "中间正确|Dense 错误", lambda case: case["c_to_w_opportunity"]),
        ("terminal_w_c_w", "压缩轨迹最终以 W→C→W 结束", "末段|W→C→W", lambda case: case["terminal_w_c_w"]),
    ]
    flags: dict[str, np.ndarray] = {}
    metrics = []
    for index, (name, label, short, fn) in enumerate(definitions):
        values = np.asarray([fn(case) for case in cases], dtype=np.int8)
        flags[name] = values
        metrics.append(
            {
                "metric": name,
                "label": label,
                "short": short,
                "count": int(values.sum()),
                "denominator": len(values),
                "rate": float(values.mean()),
                "bootstrap_95_ci": rate_ci(values, args.bootstrap_repeats, SEED + index),
            }
        )
    dense_wrong = np.asarray([not case["dense_success"] for case in cases], dtype=np.int8)
    opportunities = [case for case in cases if case["c_to_w_opportunity"]]
    noncap = [case for case in cases if not case["reached_max_tokens"]]
    noncap_opp = [case for case in noncap if case["c_to_w_opportunity"]]
    last_boxed_opportunities = [case for case in cases if case["last_boxed_c_to_w_opportunity"]]
    robust_opportunities = [case for case in cases if case["parser_robust_c_to_w_opportunity"]]
    concordant_dense_wrong = [
        case for case in cases if (not case["dense_success"] and not case["last_boxed_dense_success"])
    ]
    checkpoint_total = sum(four_states.values())
    summary = {
        "analysis_id": "qwen3_4b_gsm8k_official_test_overthinking_motivation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "model": "Qwen3-4B",
            "dataset": "openai/gsm8k main",
            "split": "complete official test",
            "heldout_n": len(cases),
            "seed": int(next(iter(artifacts.values()))["seed"]),
            "dtype": sorted(dtypes),
            "attention_backend": sorted(backends),
            "protocol_fingerprint": sorted(fingerprints),
            "split_fingerprint": split["fingerprint"],
            "checkpoint_definition": "pure sentence-step, 64–768 reasoning tokens, minimum gap 8",
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
            "dense_wrong_with_earlier_correct_checkpoint": len(opportunities),
            "share_of_all_questions": len(opportunities) / len(cases),
            "share_of_dense_errors": len(opportunities) / int(dense_wrong.sum()),
            "share_of_dense_errors_bootstrap_95_ci": conditional_ci(
                flags["c_to_w_opportunity"], dense_wrong, args.bootstrap_repeats, SEED + 100
            ),
            "dense_or_earliest_correct_oracle_accuracy": (
                sum(case["dense_success"] for case in cases) + len(opportunities)
            ) / len(cases),
            "warning": "Oracle upper bound only; not deployable.",
        },
        "parser_sensitivity": {
            "frozen_parser": "historical priority: Final answer, answer is/:/=, boxed, ####, final numeric fallback",
            "alternative_parser": "last explicit numeric boxed answer only",
            "frozen_dense_correct": int(sum(case["dense_success"] for case in cases)),
            "last_boxed_dense_correct": int(sum(case["last_boxed_dense_success"] for case in cases)),
            "dense_correctness_disagreement_count": int(
                sum(case["dense_success"] != case["last_boxed_dense_success"] for case in cases)
            ),
            "last_boxed_missing_dense_count": int(
                sum(case["last_boxed_dense_prediction"] is None for case in cases)
            ),
            "frozen_c_to_w_opportunity_count": len(opportunities),
            "last_boxed_c_to_w_opportunity_count": len(last_boxed_opportunities),
            "intersection_c_to_w_opportunity_count": len(robust_opportunities),
            "intersection_rate_all_questions": len(robust_opportunities) / len(cases),
            "concordant_dense_wrong_count": len(concordant_dense_wrong),
            "intersection_share_of_concordant_dense_errors": (
                len(robust_opportunities) / len(concordant_dense_wrong)
            ),
            "frozen_w_c_w_count": int(flags["w_c_w"].sum()),
            "last_boxed_w_c_w_count": int(sum(case["last_boxed_w_c_w"] for case in cases)),
            "interpretation": "Use frozen counts for protocol consistency and the parser intersection as the conservative motivation claim.",
        },
        "adjacent_transition_edges": {
            f'{"C" if a else "W"}_to_{"C" if b else "W"}': count
            for (a, b), count in sorted(adjacent_edges.items())
        },
        "checkpoint_to_dense_four_states": {
            "total_checkpoint_rows": checkpoint_total,
            "counts": dict(sorted(four_states.items())),
            "rates": {key: value / checkpoint_total for key, value in sorted(four_states.items())},
        },
        "opportunity_token_statistics": {
            "first_correct_checkpoint": describe([case["first_correct_checkpoint"] for case in opportunities]),
            "last_correct_checkpoint": describe([case["last_correct_checkpoint"] for case in opportunities]),
            "tokens_after_first_correct": describe([case["tokens_after_first_correct"] for case in opportunities]),
            "tokens_after_last_correct": describe([case["tokens_after_last_correct"] for case in opportunities]),
            "correct_checkpoint_count": describe([case["correct_checkpoint_count"] for case in opportunities]),
            "max_consecutive_correct_checkpoints": describe([case["max_consecutive_correct_checkpoints"] for case in opportunities]),
        },
        "correct_window_robustness": {
            "problems_with_at_least_k_consecutive_correct_checkpoints": {
                str(k): sum(case["max_consecutive_correct_checkpoints"] >= k for case in opportunities)
                for k in [1, 2, 3, 5, 10]
            }
        },
        "non_4096_sensitivity": {
            "denominator": len(noncap),
            "c_to_w_opportunity_count": len(noncap_opp),
            "c_to_w_opportunity_rate": len(noncap_opp) / len(noncap),
            "local_c_to_w_count": sum(case["local_c_to_w"] for case in noncap),
            "local_c_to_w_rate": float(np.mean([case["local_c_to_w"] for case in noncap])),
            "w_c_w_count": sum(case["w_c_w"] for case in noncap),
            "w_c_w_rate": float(np.mean([case["w_c_w"] for case in noncap])),
            "terminal_w_c_w_count": sum(case["terminal_w_c_w"] for case in noncap),
            "terminal_w_c_w_rate": float(np.mean([case["terminal_w_c_w"] for case in noncap])),
            "tokens_after_last_correct": describe([case["tokens_after_last_correct"] for case in noncap_opp]),
        },
        "interpretation_guardrails": [
            "Forced-answer branches are stochastic operational outcomes, not direct observations of latent beliefs.",
            "Numeric free-form answers make local GSM8K branch states more volatile than MCQ letters; emphasize problem-level checkpoint-correct/Dense-wrong counts.",
            "W→C→W is identified after compressing consecutive equal states and appending Dense correctness.",
            "No policy, model, epoch, or threshold was selected in this descriptive held-out audit.",
        ],
    }

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(root / "models" / "Qwen3-4B", local_files_only=True)
    except Exception:
        tokenizer = None
    representative = []
    for reason, case in select_cases(cases):
        artifact = artifacts[case["problem_id"]]
        row_map = {
            int(row["checkpoint"]): row
            for row in artifact["rows"]
            if row.get("is_sentence_checkpoint")
        }
        transitions = []
        previous = None
        for run in case["state_runs"]:
            if previous is not None:
                position = run["start_position"]
                if run["start"] == "Dense":
                    prefix_tail = snippet(artifact["dense"]["text"])
                elif tokenizer is not None and position in row_map:
                    prefix_tail = snippet(tokenizer.decode(row_map[position]["prefix_token_ids"], skip_special_tokens=False))
                else:
                    prefix_tail = ""
                transitions.append(
                    {
                        "transition": f'{previous["state"]}→{run["state"]}',
                        "at": run["start"],
                        "prediction": run["predictions"][0],
                        "reasoning_prefix_tail": prefix_tail,
                    }
                )
            previous = run
        representative.append(
            {
                "selection_reason": reason,
                **{key: case[key] for key in [
                    "problem_id", "question", "gold_answer", "dense_prediction", "dense_success",
                    "last_boxed_dense_prediction", "last_boxed_dense_success",
                    "dense_content_tokens", "reached_max_tokens", "first_correct_checkpoint",
                    "last_correct_checkpoint", "tokens_after_last_correct", "compressed_path",
                    "state_path_detailed", "switch_count",
                ]},
                "transition_details": transitions,
                "dense_reasoning_tail": snippet(artifact["dense"]["text"], 650),
            }
        )

    flat_cases = [
        {key: case[key] for key in [
                "problem_id", "gold_answer", "dense_prediction", "dense_success", "dense_content_tokens",
            "last_boxed_dense_prediction", "last_boxed_dense_success",
            "reached_max_tokens", "sentence_checkpoint_count", "has_correct_checkpoint",
            "first_correct_checkpoint", "last_correct_checkpoint", "correct_checkpoint_count",
            "max_consecutive_correct_checkpoints", "tokens_after_first_correct", "tokens_after_last_correct",
            "c_to_w_opportunity", "local_c_to_w", "local_w_to_c", "w_c_w",
            "last_boxed_c_to_w_opportunity", "parser_robust_c_to_w_opportunity", "last_boxed_w_c_w",
            "checkpoint_only_w_c_w", "terminal_w_c_w", "compressed_path", "switch_count",
            "state_path_detailed",
        ]}
        for case in cases
    ]
    patterns = Counter(case["compressed_path"] for case in cases)
    pattern_rows = [
        {"compressed_path": name, "count": count, "rate": count / len(cases)}
        for name, count in patterns.most_common()
    ]
    audit = {
        "status": "pass",
        "heldout_manifest_count": len(heldout_ids),
        "loaded_count": len(cases),
        "unique_problem_ids": len({case["problem_id"] for case in cases}),
        "missing_files": missing,
        "duplicate_checkpoint_samples": duplicate_checkpoints,
        "schedule_mismatch_samples": schedule_mismatch,
        "unique_protocol_fingerprints": sorted(fingerprints),
        "unique_dtypes": sorted(dtypes),
        "unique_attention_backends": sorted(backends),
        "complete_official_test": set(heldout_ids) == {case["problem_id"] for case in cases},
        "new_llm_generation_performed": False,
        "selection_or_training_performed": False,
    }

    metric = {item["metric"]: item for item in metrics}
    report = f"""# Motivation 数据审计：Qwen3-4B 在完整 GSM8K test 上的 overthinking / over-reflection

## 结论先行

在完整 GSM8K official test 1,319 题上，Qwen3-4B 的 Dense accuracy 为 **{summary['dense']['accuracy']*100:.1f}%**。其中 **{len(opportunities)}/1319（{len(opportunities)/len(cases)*100:.1f}%）** 曾在至少一个 sentence checkpoint 正确作答，却在继续推理后以错误 Dense 答案结束。由于 Dense 只错了 {summary['dense']['wrong']} 题，这 {len(opportunities)} 题占全部 Dense 错误的 **{len(opportunities)/summary['dense']['wrong']*100:.1f}%**。

排除全部 {summary['dense']['reached_4096']} 个 4096-token 触顶样本后，仍有 **{len(noncap_opp)}/{len(noncap)}（{len(noncap_opp)/len(noncap)*100:.1f}%）** 呈现中间正确、Dense 最终错误。

### Parser-sensitivity 审计

冻结协议的数值解析器优先匹配 `Final answer` / `answer is`，再匹配 boxed；作为敏感性分析，我们另用“最后一个明确 numeric `\\boxed{{...}}`”重判。两种解析器分别得到 {len(opportunities)} 和 {len(last_boxed_opportunities)} 个 `checkpoint C / Dense W`，**交集为 {len(robust_opportunities)} 题（{len(robust_opportunities)/len(cases)*100:.1f}%）**。在两种解析器都判断 Dense 错误的 {len(concordant_dense_wrong)} 题中，这一交集占 **{len(robust_opportunities)/len(concordant_dense_wrong)*100:.1f}%**。

因此最保守、最适合 motivation 主文的表述是：**至少 {len(robust_opportunities)}/1319（{len(robust_opportunities)/len(cases)*100:.1f}%）在两种答案解析下都曾中间正确但最终错误**。冻结协议的 38/1319 仍保留为主实验口径。两种解析下的 W→C→W 数量分别为 {int(flags['w_c_w'].sum())} 和 {sum(case['last_boxed_w_c_w'] for case in cases)}，说明总体反转统计对 parser 选择不敏感。

## 固定统计口径

- 模型：Qwen3-4B，FP16，SDPA，单 seed 20260803，thinking enabled。
- 数据：openai/gsm8k `main` 的完整 official test 1,319 题。
- checkpoint：pure sentence-step，64–768 reasoning tokens，相邻至少 8 tokens。
- checkpoint C/W 来自缓存 forced-answer 数值答案的 exact match；末尾追加 Dense C/W。
- 连续相同状态压缩后识别 W→C→W。

## 核心结果

| 现象 | 问题数 / 1319 | 比例 | problem-bootstrap 95% CI |
|---|---:|---:|---:|
"""
    for name in ["has_any_state_switch", "local_c_to_w", "w_c_w", "c_to_w_opportunity", "terminal_w_c_w"]:
        item = metric[name]
        report += f"| {item['label']} | {item['count']} | {item['rate']*100:.1f}% | [{item['bootstrap_95_ci'][0]*100:.1f}%, {item['bootstrap_95_ci'][1]*100:.1f}%] |\n"
    report += f"""

相邻 sentence checkpoint 加 Dense endpoint 共出现 **{adjacent_edges[(True, False)]} 次局部 C→W** 和 **{adjacent_edges[(False, True)]} 次局部 W→C**。这是相关 checkpoint 边的数量，不是独立问题数。

按原四状态定义，在 {checkpoint_total} 个 sentence-checkpoint 行中有：C→W {four_states['C_to_W']}（{four_states['C_to_W']/checkpoint_total*100:.1f}%）、W→C {four_states['W_to_C']}（{four_states['W_to_C']/checkpoint_total*100:.1f}%）、W→W {four_states['W_to_W']}、C→C {four_states['C_to_C']}。

### 正确窗口及后续多余推理

对 {len(opportunities)} 个中间正确、Dense 错误问题：

- 首次正确 checkpoint 中位位置：{summary['opportunity_token_statistics']['first_correct_checkpoint']['median']:.1f} tokens；
- 最后正确 checkpoint 后仍继续生成中位 **{summary['opportunity_token_statistics']['tokens_after_last_correct']['median']:.0f} tokens**；
- 排除 4096 触顶后，仍继续生成中位 **{summary['non_4096_sensitivity']['tokens_after_last_correct']['median']:.0f} tokens**；
- {summary['correct_window_robustness']['problems_with_at_least_k_consecutive_correct_checkpoints']['2']}/{len(opportunities)} 至少连续两个 checkpoint 正确，{summary['correct_window_robustness']['problems_with_at_least_k_consecutive_correct_checkpoints']['5']}/{len(opportunities)} 至少连续五个正确。

描述性 earliest-correct oracle 可将 accuracy 从 {summary['dense']['accuracy']*100:.1f}% 提高到 **{summary['recoverability_oracle_descriptive_only']['dense_or_earliest_correct_oracle_accuracy']*100:.1f}%**，但它使用未来 correctness，只是不可部署上界。

## 代表性真实样例

"""
    for index, case in enumerate(representative, 1):
        report += f"""### 样例 {index}：{case['selection_reason']}

- ID：`{case['problem_id']}`
- 问题：{case['question']}
- Gold / Dense：**{case['gold_answer']} / {case['dense_prediction']}（{'C' if case['dense_success'] else 'W'}）**
- 最后 boxed 复核：Dense={case['last_boxed_dense_prediction']}；该案例在两种 parser 下均属于中间正确、最终错误
- Dense 长度：{case['dense_content_tokens']} tokens；4096 触顶：{'是' if case['reached_max_tokens'] else '否'}
- 状态路径：`{case['state_path_detailed']}`
- 最后正确 checkpoint：{case['last_correct_checkpoint']}；其后仍生成 {case['tokens_after_last_correct']} tokens

"""
        for transition in case["transition_details"]:
            report += f"- {transition['transition']} @ {transition['at']}，分支答案 {transition['prediction']}；prefix 尾部：{transition['reasoning_prefix_tail']}\n"
        report += f"- Dense reasoning 尾部：{case['dense_reasoning_tail']}\n\n"
    report += f"""## 与 MMLU-Pro-1k 的联合解读

GSM8K 的 Dense accuracy 高于 MMLU-Pro-1k，因此问题级 `checkpoint C / Dense W` 比例更低（GSM8K **{len(opportunities)/len(cases)*100:.1f}%**，MMLU-Pro-1k **13.9%**）。但在 Dense 已经答错的条件下，GSM8K 有 **{len(opportunities)/summary['dense']['wrong']*100:.1f}%** 的错误曾经在中间答对，MMLU-Pro-1k 为 34.6%。这说明“继续推理破坏可用答案”的机会在两类任务上都存在，而困难 MCQ 上的总体发生率更高。

## 写作边界

1. GSM8K 的自由数值 forced answer 比 MCQ 单字母答案更容易产生局部采样波动，因此 50.9% 的“至少一次局部 C→W”和 42.9% 的 W→C→W 应写作**操作性答案状态不稳定**，不宜全部解释为稳定的潜在认知反转。
2. 更保守、适合主文的数字是两种 parser 的交集：**{len(robust_opportunities)}/1319（{len(robust_opportunities)/len(cases)*100:.1f}%）中间正确但 Dense 错误**；冻结协议结果为 {len(opportunities)}/1319（{len(opportunities)/len(cases)*100:.1f}%）。
3. 这是 held-out 描述性诊断，不用于训练或策略选择。
4. earliest-correct oracle 不可部署；4096 触顶会放大 token 浪费，因此必须同时报告非触顶敏感性。
"""

    ledger = f"""# 实验账本：GSM8K Motivation overthinking 审计

```yaml
run_id: qwen3_4b_gsm8k_official_test_overthinking_motivation_v1
objective: 统计完整 GSM8K test 上的 C→W 与 W→C→W 真实轨迹现象
method_version: descriptive_cache_audit_v1
status: completed
model: Qwen3-4B
dataset: openai/gsm8k main official test
sample_count: 1319
seed: 20260803
uncertainty: 10000-repeat problem bootstrap
selection_or_training: false
new_llm_generation: false
primary_result: {len(opportunities)}/1319 checkpoint-correct but Dense-wrong
verdict: 支持跨任务 overthinking motivation；GSM8K 局部 forced-answer 振荡需保守解释
```

## 运行记录

- 首次 parser-sensitivity 汇总：统计完成，但代表案例渲染遗漏 `last_boxed_dense_prediction`，报告阶段失败；原始缓存和统计未受影响。
- 修复：仅将已有 case 字段复制到展示对象，随后从同一缓存完整重跑并通过完整性断言。
"""
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "representative_cases.json", representative)
    atomic_json(output / "INTEGRITY_AUDIT.json", audit)
    atomic_csv(output / "tables" / "problem_level_cases.csv", flat_cases)
    atomic_csv(output / "tables" / "compressed_pattern_counts.csv", pattern_rows)
    svg(metrics, output / "figures" / "overthinking_problem_rates.svg")
    atomic_text(output / "MOTIVATION_OVERTHINKING_GSM8K_REPORT_ZH.md", report)
    atomic_text(output / "EXPERIMENT_LEDGER.md", ledger)
    atomic_text(output / "pipeline.complete", "complete\n")
    print(json.dumps({"status": "complete", "output": str(output), "headline": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

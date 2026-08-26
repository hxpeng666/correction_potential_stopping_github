#!/usr/bin/env python3
"""为 replay-v3 注入累计 checkpoint 检查成本后复用原 probe 训练入口。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

import src.final_paper_probe as probe


def simulate_policy_v3(
    frame: pd.DataFrame,
    scores,
    direction: str,
    threshold: float,
    *,
    include_records: bool = False,
    fallback_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(frame) != len(scores):
        raise ValueError("frame/score mismatch")
    scored = frame.copy()
    scored["score"] = scores
    records: list[dict[str, Any]] = []
    for problem_id, group in scored.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        first = ordered.iloc[0]
        eligible = ordered[ordered.score >= threshold] if direction == "high" else ordered[ordered.score <= threshold]
        fallback = eligible.empty
        dense_reference = float(first.get("dense_reference_wall_ms", first.dense_wall_ms))
        if fallback:
            current_success = bool(first.dense_success)
            method_tokens = int(first.dense_tokens)
            replay_wall_ms = float(first.get("adaptive_fallback_wall_ms", first.dense_wall_ms))
            checkpoint = None
            transition = "fallback"
            prediction = first.dense_prediction
        else:
            chosen = eligible.iloc[0]
            current_success = bool(chosen.current_success)
            method_tokens = min(int(chosen.dense_tokens), int(chosen.checkpoint) + int(chosen.branch_tokens))
            replay_wall_ms = float(chosen.dense_prefill_cuda_ms + chosen.prefix_decode_cuda_ms + chosen.branch_wall_ms)
            checkpoint = int(chosen.checkpoint)
            transition = probe.transition_name(current_success, bool(chosen.dense_success))
            prediction = chosen.current_prediction
        records.append({
            "problem_id": str(problem_id),
            "subject": first.get("subject"),
            "category": first.get("category"),
            "fallback": bool(fallback),
            "checkpoint": checkpoint,
            "transition": transition,
            "method_prediction": prediction,
            "dense_prediction": first.dense_prediction,
            "gold_answer": first.gold_answer,
            "method_success": bool(current_success),
            "dense_success": bool(first.dense_success),
            "method_tokens": method_tokens,
            "dense_tokens": int(first.dense_tokens),
            "replay_wall_ms": replay_wall_ms,
            "dense_wall_ms": dense_reference,
        })
    seen = {row["problem_id"] for row in records}
    for base in fallback_records or []:
        if base["problem_id"] in seen:
            raise ValueError(f"duplicate fallback problem {base['problem_id']}")
        records.append({
            "problem_id": base["problem_id"],
            "subject": base.get("subject"),
            "category": base.get("category"),
            "fallback": True,
            "checkpoint": None,
            "transition": "fallback",
            "method_prediction": base["dense_prediction"],
            "dense_prediction": base["dense_prediction"],
            "gold_answer": base["gold_answer"],
            "method_success": bool(base["dense_success"]),
            "dense_success": bool(base["dense_success"]),
            "method_tokens": int(base["dense_tokens"]),
            "dense_tokens": int(base["dense_tokens"]),
            "replay_wall_ms": float(base["dense_wall_ms"]),
            "dense_wall_ms": float(base["dense_wall_ms"]),
        })
        seen.add(base["problem_id"])
    summary = probe.summarize_policy_records(records)
    summary["threshold"] = float(threshold)
    if include_records:
        summary["records"] = records
    return summary


# calibrate_policies 在 src 模块中按运行时全局名称解析 simulate_policy。
probe.simulate_policy = simulate_policy_v3

import train_final_paper_probe as entry

entry.simulate_policy = simulate_policy_v3

if __name__ == "__main__":
    entry.main()

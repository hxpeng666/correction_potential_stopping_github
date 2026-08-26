#!/usr/bin/env python3
"""从同一 calibration 阈值曲线生成独立命名的 formal-certified 补充表。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_paper_probe import binomial_upper_simultaneous
from src.legacy_empirical_probe_v4 import (
    load_checkpoint_split,
    method_direction,
    simulate_policy,
)
from src.utils import atomic_json


PROBES = {
    "correctness": "Correctness (controlled)",
    "consistency": "Consistency (controlled)",
    "last_switch": "Last-switch (controlled)",
    "correction_bce": "Correction BCE only",
    "correction_trajectory": "Correction + trajectory",
}


def selected_point(curve: list[dict[str, Any]], alpha: float, grid_size: int) -> dict[str, Any]:
    rows = []
    sentinel = None
    for item in curve:
        row = dict(item)
        disabled = bool(row.get("is_no_stop_sentinel", False))
        if disabled:
            sentinel = row
            row["simultaneous_upper_95"] = 0.0
            continue
        row["simultaneous_upper_95"] = binomial_upper_simultaneous(
            int(row["lost_correct_count"]),
            int(row["problems"]),
            confidence=0.95,
            grid_size=grid_size,
        )
        rows.append(row)
    feasible = [row for row in rows if row["simultaneous_upper_95"] <= alpha]
    if feasible:
        chosen = min(
            feasible,
            key=lambda row: (
                row["mean_replay_wall_ms"],
                -row["token_reduction"],
                -row["coverage"],
                row["threshold"],
            ),
        )
        chosen = dict(chosen)
        chosen["dense_fallback"] = False
    else:
        if sentinel is None:
            raise RuntimeError("calibration curve 缺少 Dense sentinel")
        chosen = dict(sentinel)
        chosen["simultaneous_upper_95"] = 0.0
        chosen["dense_fallback"] = True
    chosen["alpha"] = alpha
    return chosen


def verify_score_alignment(score_payload: dict[str, Any], frame: pd.DataFrame, split: str) -> np.ndarray:
    ids = frame.problem_id.astype(str).tolist()
    checkpoints = frame.checkpoint.astype(int).tolist()
    if score_payload["problem_ids"][split] != ids:
        raise ValueError(f"{split} score/problem ID 错位")
    if score_payload["checkpoints"][split] != checkpoints:
        raise ValueError(f"{split} score/checkpoint 错位")
    scores = score_payload["scores"][split].numpy().astype(np.float64)
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError(f"{split} score 非有限或长度错误")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.01, 0.02, 0.05))
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"拒绝覆盖 formal-certified 表：{args.output}")
    output_rows = []
    audit_rows = []
    for dataset in ("gsm8k", "mmlu"):
        heldout_frame, _, _, heldout_fallbacks = load_checkpoint_split(
            args.replay_root / dataset / "heldout", "sentence"
        )
        for key, display in PROBES.items():
            directory = args.run_root / dataset / "probes" / key
            probe = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
            scores_payload = torch.load(
                directory / "scores.pt", map_location="cpu", weights_only=False
            )
            scores = verify_score_alignment(scores_payload, heldout_frame, "heldout")
            curve = probe["calibration"]["curve"]
            grid_size = int(probe["calibration"]["grid_declared_size"])
            for alpha in args.alphas:
                frozen = selected_point(curve, float(alpha), grid_size)
                result = simulate_policy(
                    heldout_frame,
                    scores,
                    method_direction("correction" if key.startswith("correction") else key),
                    float(frozen["threshold"]),
                    fallback_records=heldout_fallbacks,
                    force_dense=bool(frozen["dense_fallback"]),
                )
                counts = result.pop("counts")
                output_rows.append(
                    {
                        "dataset": dataset,
                        "method": display,
                        "workpoint": f"formal-certified-{100 * alpha:g}%",
                        "alpha": alpha,
                        "threshold": frozen["threshold"],
                        "dense_fallback": frozen["dense_fallback"],
                        "calibration_lost_correct": frozen["lost_correct_count"],
                        "calibration_simultaneous_upper_95": frozen["simultaneous_upper_95"],
                        **result,
                        **counts,
                    }
                )
                audit_rows.append(
                    {
                        "dataset": dataset,
                        "method": key,
                        "alpha": alpha,
                        "heldout_used_for_selection": False,
                        "calibration_grid_size": grid_size,
                        "dense_fallback": frozen["dense_fallback"],
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(args.output, index=False)
    atomic_json(
        {
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "name_guard": "formal-certified only; Strict/Balanced/Aggressive reserved for empirical B=1/2/4",
            "heldout_used_for_selection": False,
            "rows": audit_rows,
        },
        args.output.with_suffix(".audit.json"),
    )
    print(json.dumps({"status": "complete", "rows": len(output_rows)}, indent=2))


if __name__ == "__main__":
    main()

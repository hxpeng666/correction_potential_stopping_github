#!/usr/bin/env python3
"""审计动态策略动作路径不访问在线不可观测的未来字段。"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.dynamic_optimal_stopping_deployable_v2 import (
    decide_current_action,
    simulate_deployable_dynamic_policy,
)
from src.legacy_empirical_probe_v4 import load_checkpoint_split
from src.utils import atomic_json


def chosen_checkpoints_from_stream(
    frame, stop: np.ndarray, risk: np.ndarray, q_continue: np.ndarray, mu: float,
) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for problem_id, group in frame.groupby("problem_id", sort=False):
        chosen = None
        for position in group.sort_values("checkpoint").index.to_numpy(dtype=np.int64):
            q_stop = float(stop[position] - mu * risk[position])
            if decide_current_action(q_stop, float(q_continue[position])):
                chosen = int(frame.loc[position, "checkpoint"])
                break
        result[str(problem_id)] = chosen
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu", "mmlu_pro"), required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    probe = json.loads((args.probe_root / "probe.json").read_text(encoding="utf-8"))
    predictions = torch.load(
        args.probe_root / "predictions.pt", map_location="cpu", weights_only=False
    )
    frame, _, _, fallbacks = load_checkpoint_split(args.raw_root / "calibration", "sentence")
    expected_ids = predictions["problem_ids"]["calibration"]
    expected_checkpoints = predictions["checkpoints"]["calibration"]
    if expected_ids != frame.problem_id.astype(str).tolist():
        raise ValueError("calibration problem ID错位")
    if expected_checkpoints != frame.checkpoint.astype(int).tolist():
        raise ValueError("calibration checkpoint错位")
    stop = predictions["stop_probability"]["calibration"].numpy().astype(np.float64)
    risk = predictions["risk_probability"]["calibration"].numpy().astype(np.float64)
    q_bank = predictions["q_continue_values"]["calibration"].numpy().astype(np.float64)

    primitive_source = inspect.getsource(decide_current_action)
    tree = ast.parse(primitive_source)
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    forbidden = {
        "frame", "dense_tokens", "next_checkpoint", "transition_token_costs",
        "future_entropy", "future_correctness", "next_state",
    }
    forbidden_present = sorted(identifiers & forbidden)
    mismatches = []
    future_mutation_mismatches = []
    candidate_grid = probe["run_spec"]["candidate_grid"]
    for candidate in candidate_grid:
        index = int(candidate["candidate_index"])
        mu = float(candidate["mu"])
        streaming = chosen_checkpoints_from_stream(
            frame, stop, risk, q_bank[:, index], mu
        )
        replay = simulate_deployable_dynamic_policy(
            frame, stop, risk, q_bank[:, index], mu_value=mu,
            fallback_records=fallbacks, include_records=True,
        )
        replay_choice = {
            str(row["problem_id"]): (
                None if row["fallback"] else int(row["checkpoint"])
            )
            for row in replay["records"]
        }
        for problem_id, value in streaming.items():
            if replay_choice[problem_id] != value:
                mismatches.append({
                    "candidate": index, "problem_id": problem_id,
                    "streaming": value, "offline": replay_choice[problem_id],
                })

        # 保持行顺序，任意改变未来间隔和 Dense endpoint；当前动作索引必须不变。
        mutated = frame.copy()
        for _, group in mutated.groupby("problem_id", sort=False):
            positions = group.sort_values("checkpoint").index.to_numpy(dtype=np.int64)
            mutated.loc[positions, "checkpoint"] = np.arange(1, len(positions) + 1) * 1000
            mutated.loc[positions, "dense_tokens"] = 99999
        mutated_replay = simulate_deployable_dynamic_policy(
            mutated, stop, risk, q_bank[:, index], mu_value=mu,
            fallback_records=fallbacks, include_records=True,
        )
        mutated_choice = {
            str(row["problem_id"]): (
                None if row["fallback"] else int(row["checkpoint"] // 1000 - 1)
            )
            for row in mutated_replay["records"]
        }
        # checkpoint数值被改过，因此比较 first-hit 的行位置而不是数值。
        for problem_id, group in frame.groupby("problem_id", sort=False):
            positions = group.sort_values("checkpoint").index.to_numpy(dtype=np.int64)
            original_stop = None
            for local_index, position in enumerate(positions):
                value = decide_current_action(
                    float(stop[position] - mu * risk[position]), float(q_bank[position, index])
                )
                if value and original_stop is None:
                    original_stop = local_index
            mutated_stop = mutated_choice[str(problem_id)]
            if original_stop != mutated_stop:
                future_mutation_mismatches.append({
                    "candidate": index, "problem_id": str(problem_id),
                    "original_index": original_stop, "mutated_index": mutated_stop,
                })

    decision_text = probe["run_spec"].get("decision", "")
    q_semantics = probe["run_spec"].get("q_continue_semantics", "")
    passed = (
        not forbidden_present
        and not mismatches
        and not future_mutation_mismatches
        and "Q_continue_hat(z_t)" in decision_text
        and "delta_t_next" in q_semantics
    )
    payload = {
        "status": "complete" if passed else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "candidates_checked": len(candidate_grid),
        "calibration_rows_checked": len(frame),
        "calibration_problems_checked": int(frame.problem_id.nunique()) + len(fallbacks),
        "decision_primitive_signature": str(inspect.signature(decide_current_action)),
        "decision_primitive_identifiers": sorted(identifiers),
        "forbidden_identifiers_present": forbidden_present,
        "streaming_offline_mismatch_count": len(mismatches),
        "future_mutation_mismatch_count": len(future_mutation_mismatches),
        "mismatch_examples": mismatches[:10],
        "future_mutation_examples": future_mutation_mismatches[:10],
        "future_fields_used_by_action": False if passed else None,
        "training_target_may_use_future_increment": True,
        "heldout_used": False,
    }
    atomic_json(payload, args.output / f"{args.dataset}_online_parity_audit.json")
    if not passed:
        raise SystemExit(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

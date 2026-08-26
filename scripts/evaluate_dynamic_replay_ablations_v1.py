#!/usr/bin/env python3
"""在完整动态模型的冻结预测上执行无需重训的策略级消融。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.dynamic_optimal_stopping_v1 import (
    clopper_pearson_upper, simulate_dynamic_policy, summarize_token_records,
)
from src.final_paper_inference import atomic_torch_save
from src.legacy_empirical_probe_v4 import load_checkpoint_split, transition_name
from src.utils import atomic_json, load_yaml


def strip_records(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = dict(payload)
    records = summary.pop("records")
    return summary, records


def select_policy(
    curve: list[dict[str, Any]],
    dense: dict[str, Any],
    *,
    family: str,
    bound: float | int,
    epsilon: float,
) -> dict[str, Any]:
    if family == "formal_alpha":
        feasible = [
            row for row in curve
            if float(row["lost_correct_ucb_simultaneous95"]) <= float(bound)
            and float(row["accuracy"]) >= float(dense["dense_accuracy"]) - epsilon
        ]
    elif family == "empirical_B":
        feasible = [
            row for row in curve
            if int(row["lost_correct_count"]) <= int(bound)
            and float(row["accuracy"]) >= float(dense["dense_accuracy"]) - epsilon
        ]
    else:
        raise ValueError(f"未知选择族：{family}")
    if feasible:
        selected = dict(min(feasible, key=lambda row: (
            float(row["mean_reasoning_tokens"]),
            -float(row["accuracy"]),
            int(row["lost_correct_count"]),
            int(row["candidate_index"]),
        )))
        selected["dense_fallback"] = False
    else:
        selected = dict(dense)
        selected.update({"selected_candidate": "dense", "candidate_index": None, "dense_fallback": True})
    selected["selection_family"] = family
    selected["selection_bound"] = bound
    selected["accuracy_epsilon"] = epsilon
    return selected


def oracle_earliest_correct(frame, fallback_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """使用heldout标签的不可部署上界；仅作描述性诊断。"""
    records: list[dict[str, Any]] = []
    for problem_id, group in frame.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        first = ordered.iloc[0]
        correct = ordered[ordered.current_success.astype(bool)]
        if correct.empty:
            records.append({
                "problem_id": str(problem_id), "subject": first.get("subject"),
                "category": first.get("category"), "fallback": True, "checkpoint": None,
                "transition": "fallback", "method_prediction": first.dense_prediction,
                "dense_prediction": first.dense_prediction, "gold_answer": first.gold_answer,
                "method_success": bool(first.dense_success), "dense_success": bool(first.dense_success),
                "method_tokens": int(first.dense_tokens), "dense_tokens": int(first.dense_tokens),
            })
        else:
            chosen = correct.iloc[0]
            current_success = bool(chosen.current_success)
            records.append({
                "problem_id": str(problem_id), "subject": first.get("subject"),
                "category": first.get("category"), "fallback": False,
                "checkpoint": int(chosen.checkpoint),
                "transition": transition_name(current_success, bool(chosen.dense_success)),
                "method_prediction": chosen.current_prediction,
                "dense_prediction": chosen.dense_prediction, "gold_answer": chosen.gold_answer,
                "method_success": current_success, "dense_success": bool(chosen.dense_success),
                "method_tokens": min(int(chosen.checkpoint), int(chosen.dense_tokens)),
                "dense_tokens": int(chosen.dense_tokens),
            })
    seen = {row["problem_id"] for row in records}
    for source in fallback_records:
        problem_id = str(source["problem_id"])
        if problem_id in seen:
            raise ValueError(f"oracle fallback重复：{problem_id}")
        records.append({
            "problem_id": problem_id, "subject": source.get("subject"),
            "category": source.get("category"), "fallback": True, "checkpoint": None,
            "transition": "fallback", "method_prediction": source["dense_prediction"],
            "dense_prediction": source["dense_prediction"], "gold_answer": source["gold_answer"],
            "method_success": bool(source["dense_success"]), "dense_success": bool(source["dense_success"]),
            "method_tokens": int(source["dense_tokens"]), "dense_tokens": int(source["dense_tokens"]),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--full-probe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    marker = args.output / "phase.complete"
    if args.resume and marker.is_file():
        print(json.dumps({"status": "skipped_complete", "output": str(args.output)}))
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"拒绝覆盖非空目录：{args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    config = load_yaml(args.config)
    dynamic = config["dynamic_policy"]
    probe = json.loads((args.full_probe_root / "probe.json").read_text(encoding="utf-8"))
    predictions = torch.load(args.full_probe_root / "predictions.pt", map_location="cpu", weights_only=False)
    frames: dict[str, Any] = {}
    fallbacks: dict[str, list[dict[str, Any]]] = {}
    for split in ("calibration", "heldout"):
        frame, _, _, fallback = load_checkpoint_split(args.raw_root / split, "sentence")
        frames[split] = frame
        fallbacks[split] = fallback
        expected_ids = [str(value) for value in predictions["problem_ids"][split]]
        expected_checkpoints = [int(value) for value in predictions["checkpoints"][split]]
        if expected_ids != frame.problem_id.astype(str).tolist():
            raise ValueError(f"{split} problem ID/row与冻结预测不对齐")
        if expected_checkpoints != frame.checkpoint.astype(int).tolist():
            raise ValueError(f"{split} checkpoint与冻结预测不对齐")

    candidates = probe["run_spec"]["candidate_grid"]
    curve_all = probe["calibration"]["curve"]
    dense_calibration = probe["calibration"]["dense_sentinel"]
    epsilon_main = float(dynamic["accuracy_epsilon"])
    definitions = {
        "no_risk_penalty_mu0": {
            "description": "固定mu=0，移除局部lost-correct风险惩罚",
            "indices": {int(row["candidate_index"]) for row in candidates if float(row["mu"]) == 0.0},
            "epsilon": epsilon_main,
            "stop_mode": "frozen", "risk_mode": "frozen", "value_mode": "frozen",
        },
        "no_compute_cost_lambda0": {
            "description": "固定lambda=0，移除继续动作的token成本",
            "indices": {int(row["candidate_index"]) for row in candidates if float(row["lambda"]) == 0.0},
            "epsilon": epsilon_main,
            "stop_mode": "frozen", "risk_mode": "frozen", "value_mode": "frozen",
        },
        "no_calibration_accuracy_constraint": {
            "description": "保留风险约束，但移除calibration总体准确率下限",
            "indices": {int(row["candidate_index"]) for row in candidates},
            "epsilon": 1.0,
            "stop_mode": "frozen", "risk_mode": "frozen", "value_mode": "frozen",
        },
        "no_stop_correctness_pS0": {
            "description": "决策时令p_stop_correct=0，移除当前停止正确率项",
            "indices": {int(row["candidate_index"]) for row in candidates},
            "epsilon": epsilon_main,
            "stop_mode": "zero", "risk_mode": "frozen", "value_mode": "frozen",
        },
        "no_continuation_value_M0": {
            "description": "决策时令continuation value M=0，移除未来状态价值",
            "indices": {int(row["candidate_index"]) for row in candidates},
            "epsilon": epsilon_main,
            "stop_mode": "frozen", "risk_mode": "frozen", "value_mode": "zero",
        },
    }
    stop_probability = {
        split: predictions["stop_probability"][split].numpy().astype(np.float64)
        for split in ("calibration", "heldout")
    }
    risk_probability = {
        split: predictions["risk_probability"][split].numpy().astype(np.float64)
        for split in ("calibration", "heldout")
    }
    continuation = {
        split: predictions["continuation_values"][split].numpy().astype(np.float64)
        for split in ("calibration", "heldout")
    }
    results: dict[str, Any] = {}
    saved_records: dict[str, Any] = {}
    for name, definition in definitions.items():
        transformed_stop = {
            split: (np.zeros_like(stop_probability[split]) if definition["stop_mode"] == "zero" else stop_probability[split])
            for split in ("calibration", "heldout")
        }
        transformed_risk = {
            split: (np.zeros_like(risk_probability[split]) if definition["risk_mode"] == "zero" else risk_probability[split])
            for split in ("calibration", "heldout")
        }
        local_curve = []
        for candidate in candidates:
            index = int(candidate["candidate_index"])
            if index not in definition["indices"]:
                continue
            candidate_value = (
                np.zeros(len(frames["calibration"]), dtype=np.float64)
                if definition["value_mode"] == "zero"
                else continuation["calibration"][:, index]
            )
            row = simulate_dynamic_policy(
                frames["calibration"], transformed_stop["calibration"], transformed_risk["calibration"],
                candidate_value, lambda_value=float(candidate["lambda"]), mu_value=float(candidate["mu"]),
                cost_unit_tokens=float(dynamic["cost_unit_tokens"]), fallback_records=fallbacks["calibration"],
            )
            row.update(candidate)
            row["lost_correct_ucb_simultaneous95"] = clopper_pearson_upper(
                int(row["lost_correct_count"]), int(row["problems"]),
                float(dynamic["formal_delta"]) / len(candidates),
            )
            local_curve.append(row)
        if not local_curve:
            raise ValueError(f"{name}候选为空")
        results[name] = {"definition": definition["description"], "candidate_count": len(local_curve), "selected": {}}
        saved_records[name] = {}
        for family, bounds in (
            ("formal_alpha", dynamic["formal_alpha"]),
            ("empirical_B", dynamic["empirical_B"]),
        ):
            results[name]["selected"][family] = {}
            saved_records[name][family] = {}
            for raw_bound in bounds:
                key = str(int(raw_bound)) if family == "empirical_B" else str(float(raw_bound))
                selected = select_policy(
                    local_curve, dense_calibration, family=family, bound=raw_bound,
                    epsilon=float(definition["epsilon"]),
                )
                if selected.get("dense_fallback"):
                    evaluated = simulate_dynamic_policy(
                        frames["heldout"], transformed_stop["heldout"], transformed_risk["heldout"],
                        np.zeros(len(frames["heldout"]), dtype=np.float64), lambda_value=0.0,
                        mu_value=0.0, cost_unit_tokens=float(dynamic["cost_unit_tokens"]),
                        fallback_records=fallbacks["heldout"], include_records=True, force_dense=True,
                    )
                else:
                    index = int(selected["candidate_index"])
                    candidate = candidates[index]
                    heldout_value = (
                        np.zeros(len(frames["heldout"]), dtype=np.float64)
                        if definition["value_mode"] == "zero"
                        else continuation["heldout"][:, index]
                    )
                    evaluated = simulate_dynamic_policy(
                        frames["heldout"], transformed_stop["heldout"], transformed_risk["heldout"],
                        heldout_value, lambda_value=float(candidate["lambda"]),
                        mu_value=float(candidate["mu"]), cost_unit_tokens=float(dynamic["cost_unit_tokens"]),
                        fallback_records=fallbacks["heldout"], include_records=True,
                    )
                summary, records = strip_records(evaluated)
                results[name]["selected"][family][key] = {"calibration": selected, "heldout": summary}
                saved_records[name][family][key] = records

    oracle_records = oracle_earliest_correct(frames["heldout"], fallbacks["heldout"])
    oracle_summary = summarize_token_records(oracle_records)
    results["oracle_earliest_correct"] = {
        "definition": "读取heldout checkpoint正确性标签的不可部署描述性上界，不参与策略选择",
        "heldout": oracle_summary,
    }
    saved_records["oracle_earliest_correct"] = oracle_records
    payload = {
        "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset, "source_full_probe": str(args.full_probe_root),
        "source_run_spec_fingerprint": probe["run_spec_fingerprint"],
        "cost": "reasoning_tokens_only", "short_answer_cost": 0,
        "formal_ucb_note": "沿用完整48候选的Bonferroni同时上界，不因消融子集缩小而放松",
        "heldout_used_for_selection": False, "results": results,
    }
    atomic_json(payload, args.output / "replay_ablations.json")
    atomic_torch_save({"status": "complete", "records": saved_records}, args.output / "policy_records.pt")
    atomic_json({
        "status": "complete", "dataset": args.dataset,
        "source_run_spec_fingerprint": probe["run_spec_fingerprint"],
        "artifacts": ["replay_ablations.json", "policy_records.pt"],
    }, marker)
    print(json.dumps({"status": "complete", "dataset": args.dataset, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

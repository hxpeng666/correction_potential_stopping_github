#!/usr/bin/env python3
"""OS-Pruner确定性0.5 first-hit敏感性分析；只复用冻结stop probability。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.dynamic_optimal_stopping_deployable_v2 import simulate_deployable_dynamic_policy
from src.final_paper_inference import atomic_torch_save
from src.legacy_empirical_probe_v4 import load_checkpoint_split
from src.utils import atomic_json, load_yaml


def choose(curve, dense, budget: int, epsilon: float):
    feasible = [
        row for row in curve
        if int(row["lost_correct_count"]) <= budget
        and float(row["accuracy"]) >= float(dense["dense_accuracy"]) - epsilon
    ]
    if not feasible:
        result = dict(dense)
        result.update({"candidate_index": None, "dense_fallback": True})
    else:
        result = dict(min(feasible, key=lambda row: (
            float(row["mean_reasoning_tokens"]), -float(row["token_reduction"]),
            -float(row["coverage"]), int(row["candidate_index"]),
        )))
        result["dense_fallback"] = False
    result["budget_B"] = budget
    return result


def strip_records(payload: dict[str, Any]):
    summary = dict(payload)
    return summary, summary.pop("records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-filter", choices=("matched", "all"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"拒绝覆盖非空目录：{args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    config = load_yaml(args.config)
    probe = json.loads((args.source_root / "probe.json").read_text(encoding="utf-8"))
    pred = torch.load(args.source_root / "predictions.pt", map_location="cpu", weights_only=False)
    candidates = probe["run_spec"]["candidate_grid"]
    if args.candidate_filter == "matched":
        candidates = [row for row in candidates if float(row["mu"]) == 0.0]
    frames, fallbacks = {}, {}
    probabilities = {}
    for split in ("calibration", "heldout"):
        frame, _, _, fallback = load_checkpoint_split(args.raw_root / split, "sentence")
        if frame.problem_id.astype(str).tolist() != [str(v) for v in pred["problem_ids"][split]]:
            raise ValueError(f"{split} ID不对齐")
        frames[split], fallbacks[split] = frame, fallback
        probability_key = "stop_probability" if "stop_probability" in pred else "stop_probabilities"
        probabilities[split] = pred[probability_key][split].numpy().astype(np.float64)

    dense = simulate_deployable_dynamic_policy(
        frames["calibration"], np.zeros(len(frames["calibration"])),
        np.zeros(len(frames["calibration"])), np.zeros(len(frames["calibration"])),
        mu_value=0.0, fallback_records=fallbacks["calibration"], force_dense=True,
    )
    dense.update({"candidate_index": None, "lambda": None, "mu": None})
    curve = []
    for candidate in candidates:
        index = int(candidate["candidate_index"])
        row = simulate_deployable_dynamic_policy(
            frames["calibration"], probabilities["calibration"][:, index],
            np.zeros(len(frames["calibration"])), np.full(len(frames["calibration"]), 0.5),
            mu_value=0.0, fallback_records=fallbacks["calibration"],
        )
        row.update(candidate); curve.append(row)
    epsilon = float(config["dynamic_policy"]["accuracy_epsilon"])
    selected, records_by_B, frozen = {}, {}, {}
    for raw_budget in config["dynamic_policy"]["empirical_B"]:
        budget = int(raw_budget); key = str(budget)
        selected[key] = choose(curve, dense, budget, epsilon)
        selection = selected[key]
        if selection.get("dense_fallback"):
            evaluated = simulate_deployable_dynamic_policy(
                frames["heldout"], np.zeros(len(frames["heldout"])),
                np.zeros(len(frames["heldout"])), np.zeros(len(frames["heldout"])),
                mu_value=0.0, fallback_records=fallbacks["heldout"],
                include_records=True, force_dense=True,
            )
        else:
            index = int(selection["candidate_index"])
            evaluated = simulate_deployable_dynamic_policy(
                frames["heldout"], probabilities["heldout"][:, index],
                np.zeros(len(frames["heldout"])), np.full(len(frames["heldout"]), 0.5),
                mu_value=0.0, fallback_records=fallbacks["heldout"], include_records=True,
            )
        summary, records = strip_records(evaluated)
        frozen[key] = {"calibration": selection, "heldout": summary}
        records_by_B[key] = records
    payload = {
        "status": "complete", "dataset": args.dataset, "label": args.label,
        "source": str(args.source_root), "candidate_filter": args.candidate_filter,
        "deployment_rule": "deterministic first-hit: stop_probability >= 0.5",
        "calibration": {"curve": curve, "selected": selected},
        "frozen_policy_results": frozen, "heldout_used_for_selection": False,
    }
    atomic_json(payload, args.output / "evaluation.json")
    atomic_torch_save({"status": "complete", "records": {"empirical_B": records_by_B}}, args.output / "policy_records.pt")
    atomic_json({"status": "complete", "label": args.label}, args.output / "phase.complete")
    print(json.dumps({"status": "complete", "dataset": args.dataset, "label": args.label}, ensure_ascii=False))


if __name__ == "__main__":
    main()

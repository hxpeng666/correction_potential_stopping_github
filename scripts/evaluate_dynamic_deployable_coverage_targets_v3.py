#!/usr/bin/env python3
"""在 calibration 上冻结可部署动态策略的 coverage-targeted 工作点。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.dynamic_optimal_stopping_deployable_v2 import simulate_deployable_dynamic_policy
from src.final_paper_inference import atomic_torch_save
from src.legacy_empirical_probe_v4 import load_checkpoint_split
from src.utils import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu", "mmlu_pro"), required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    marker = args.output / "phase.complete"
    if args.resume and marker.is_file():
        print(json.dumps({"status": "skipped_complete", "output": str(args.output)}))
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"拒绝覆盖非空输出：{args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    probe = json.loads((args.probe_root / "probe.json").read_text(encoding="utf-8"))
    predictions = torch.load(
        args.probe_root / "predictions.pt", map_location="cpu", weights_only=False
    )
    if probe["run_spec"].get("online_observability") != (
        "current z_t only; no next checkpoint/dense remainder"
    ):
        raise ValueError("输入probe不是当前可部署Q_continue协议")
    frame, _, _, fallbacks = load_checkpoint_split(args.raw_root / "heldout", "sentence")
    if predictions["problem_ids"]["heldout"] != frame.problem_id.astype(str).tolist():
        raise ValueError("heldout problem ID/row与预测不对齐")
    if [int(value) for value in predictions["checkpoints"]["heldout"]] != (
        frame.checkpoint.astype(int).tolist()
    ):
        raise ValueError("heldout checkpoint与预测不对齐")

    stop = predictions["stop_probability"]["heldout"].numpy().astype(np.float64)
    risk = predictions["risk_probability"]["heldout"].numpy().astype(np.float64)
    q_values = predictions["q_continue_values"]["heldout"].numpy().astype(np.float64)
    curve = probe["calibration"]["curve"]
    candidates = probe["run_spec"]["candidate_grid"]
    results = {}
    records_output = {}
    for target in (30, 40, 50, 60, 70, 80, 90):
        selected = dict(min(curve, key=lambda row: (
            abs(float(row["coverage"]) - target / 100.0),
            float(row["mean_reasoning_tokens"]),
            int(row["lost_correct_count"]),
            int(row["candidate_index"]),
        )))
        index = int(selected["candidate_index"])
        candidate = candidates[index]
        evaluated = simulate_deployable_dynamic_policy(
            frame,
            stop,
            risk,
            q_values[:, index],
            mu_value=float(candidate["mu"]),
            fallback_records=fallbacks,
            include_records=True,
        )
        records = evaluated.pop("records")
        if evaluated.get("future_fields_used_by_action") is not False:
            raise AssertionError("coverage replay错误访问未来字段")
        results[str(target)] = {
            "target_percent": target,
            "calibration": selected,
            "heldout": evaluated,
        }
        records_output[str(target)] = records

    atomic_json({
        "status": "complete",
        "dataset": args.dataset,
        "source_run_spec_fingerprint": probe["run_spec_fingerprint"],
        "selection": "closest calibration coverage; tie: fewer tokens, fewer W_to_C, candidate index",
        "heldout_used_for_selection": False,
        "future_fields_used_by_action": False,
        "results": results,
    }, args.output / "coverage_targeted.json")
    atomic_torch_save(
        {"status": "complete", "records": records_output},
        args.output / "policy_records.pt",
    )
    atomic_json({
        "status": "complete",
        "artifacts": ["coverage_targeted.json", "policy_records.pt"],
        "source_run_spec_fingerprint": probe["run_spec_fingerprint"],
    }, marker)
    print(json.dumps({"status": "complete", "dataset": args.dataset, "output": str(args.output)}))


if __name__ == "__main__":
    main()

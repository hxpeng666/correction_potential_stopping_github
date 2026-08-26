#!/usr/bin/env python3
"""诊断Full与M=0差异，并审计value预测误差与M=0候选退化。"""
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

from src.dynamic_optimal_stopping_deployable_v2 import recursive_q_continue_targets
from src.legacy_empirical_probe_v4 import load_checkpoint_split
from src.utils import atomic_json


def index_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["problem_id"]): row for row in records}
    if len(result) != len(records):
        raise ValueError("policy records存在重复problem ID")
    return result


def pair_summary(full_records, m0_records) -> dict[str, Any]:
    full = index_records(full_records)
    m0 = index_records(m0_records)
    if set(full) != set(m0):
        raise ValueError("Full/M0 problem ID不一致")
    counts = {
        "same_action": 0, "both_stop_same_checkpoint": 0,
        "full_only_stop": 0, "m0_only_stop": 0,
        "both_stop_full_earlier": 0, "both_stop_m0_earlier": 0,
        "full_correct_m0_wrong": 0, "full_wrong_m0_correct": 0,
    }
    transition_by_region: dict[str, dict[str, int]] = {}
    token_delta = []
    differing_rows = []
    for problem_id in sorted(full):
        a, b = full[problem_id], m0[problem_id]
        a_stop = not bool(a["fallback"])
        b_stop = not bool(b["fallback"])
        if a_stop == b_stop and a.get("checkpoint") == b.get("checkpoint"):
            counts["same_action"] += 1
            if a_stop:
                counts["both_stop_same_checkpoint"] += 1
            region = "same_action"
        elif a_stop and not b_stop:
            counts["full_only_stop"] += 1
            region = "full_only_stop"
        elif b_stop and not a_stop:
            counts["m0_only_stop"] += 1
            region = "m0_only_stop"
        elif int(a["checkpoint"]) < int(b["checkpoint"]):
            counts["both_stop_full_earlier"] += 1
            region = "both_stop_full_earlier"
        else:
            counts["both_stop_m0_earlier"] += 1
            region = "both_stop_m0_earlier"
        if bool(a["method_success"]) and not bool(b["method_success"]):
            counts["full_correct_m0_wrong"] += 1
        if bool(b["method_success"]) and not bool(a["method_success"]):
            counts["full_wrong_m0_correct"] += 1
        transition_by_region.setdefault(region, {})
        key = f"full={a['transition']}|m0={b['transition']}"
        transition_by_region[region][key] = transition_by_region[region].get(key, 0) + 1
        delta = int(b["method_tokens"]) - int(a["method_tokens"])
        token_delta.append(delta)
        if region != "same_action":
            differing_rows.append({
                "problem_id": problem_id, "region": region,
                "dense_success": bool(a["dense_success"]),
                "full_success": bool(a["method_success"]),
                "m0_success": bool(b["method_success"]),
                "full_transition": a["transition"], "m0_transition": b["transition"],
                "full_checkpoint": a.get("checkpoint"), "m0_checkpoint": b.get("checkpoint"),
                "full_tokens": int(a["method_tokens"]), "m0_tokens": int(b["method_tokens"]),
                "m0_minus_full_tokens": delta,
            })
    return {
        "problems": len(full), "action_relationship": counts,
        "transition_by_region": transition_by_region,
        "mean_m0_minus_full_tokens": float(np.mean(token_delta)),
        "median_m0_minus_full_tokens": float(np.median(token_delta)),
        "differing_rows": differing_rows,
    }


def value_error_summary(frame, target, prediction, candidate_index: int) -> dict[str, Any]:
    residual = target[:, candidate_index].astype(np.float64) - prediction[:, candidate_index]
    checkpoints = frame.checkpoint.to_numpy(dtype=np.int64)
    bins = [(64, 192), (193, 384), (385, 576), (577, 768)]
    by_position = []
    for low, high in bins:
        mask = (checkpoints >= low) & (checkpoints <= high)
        if not mask.any():
            continue
        by_position.append({
            "checkpoint_range": f"{low}-{high}", "rows": int(mask.sum()),
            "mean_target_minus_prediction": float(residual[mask].mean()),
            "mae": float(np.abs(residual[mask]).mean()),
            "rmse": float(np.sqrt(np.square(residual[mask]).mean())),
        })
    return {
        "rows": int(len(residual)),
        "mean_target_minus_prediction": float(residual.mean()),
        "median_target_minus_prediction": float(np.median(residual)),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "by_checkpoint_position": by_position,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--full-probe-root", type=Path, required=True)
    parser.add_argument("--replay-ablation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    probe = json.loads((args.full_probe_root / "probe.json").read_text(encoding="utf-8"))
    predictions = torch.load(args.full_probe_root / "predictions.pt", map_location="cpu", weights_only=False)
    full_records = torch.load(args.full_probe_root / "policy_records.pt", map_location="cpu", weights_only=False)["records"]
    replay_json = json.loads((args.replay_ablation_root / "replay_ablations.json").read_text(encoding="utf-8"))
    m0_records = torch.load(
        args.replay_ablation_root / "policy_records.pt", map_location="cpu", weights_only=False
    )["records"]["no_continuation_value_M0"]["empirical_B"]
    candidates = probe["run_spec"]["candidate_grid"]
    lambdas = np.asarray([float(row["lambda"]) for row in candidates], dtype=np.float64)
    mus = np.asarray([float(row["mu"]) for row in candidates], dtype=np.float64)

    frames = {}
    fallbacks = {}
    stop = {}
    risk = {}
    qpred = {}
    targets = {}
    for split in ("probe_train", "calibration", "heldout"):
        frame, _, _, fallback = load_checkpoint_split(args.raw_root / split, "sentence")
        if frame.problem_id.astype(str).tolist() != [str(v) for v in predictions["problem_ids"][split]]:
            raise ValueError(f"{split} problem ID不对齐")
        frames[split], fallbacks[split] = frame, fallback
        stop[split] = predictions["stop_probability"][split].numpy().astype(np.float64)
        risk[split] = predictions["risk_probability"][split].numpy().astype(np.float64)
        qpred[split] = predictions["q_continue_values"][split].numpy().astype(np.float64)
        targets[split], _ = recursive_q_continue_targets(
            frame, stop[split], risk[split], lambdas, mus, cost_unit_tokens=4096.0
        )

    validation_ids = set(str(value) for value in probe["validation_problem_ids"])
    validation_mask = frames["probe_train"].problem_id.astype(str).isin(validation_ids).to_numpy()
    selected_full = probe["calibration"]["selected"]["empirical_B"]
    selected_m0 = replay_json["results"]["no_continuation_value_M0"]["selected"]["empirical_B"]

    # M=0对lambda完全不敏感：计算实际q_stop action signature确认候选退化数量。
    signatures = {}
    for candidate in candidates:
        index = int(candidate["candidate_index"])
        action = stop["calibration"] - float(candidate["mu"]) * risk["calibration"] >= 0.0
        signatures.setdefault(np.packbits(action).tobytes().hex(), []).append(index)

    by_budget = {}
    value_diagnostics = {}
    for budget in (0, 1, 2, 4, 10):
        key = str(budget)
        by_budget[key] = pair_summary(
            full_records["empirical_B"][key], m0_records[key]
        )
        full_selection = selected_full[key]
        if not full_selection.get("dense_fallback"):
            candidate_index = int(full_selection["candidate_index"])
            value_diagnostics[key] = {
                "candidate": candidates[candidate_index],
                "internal_validation": value_error_summary(
                    frames["probe_train"].loc[validation_mask].reset_index(drop=True),
                    targets["probe_train"][validation_mask],
                    qpred["probe_train"][validation_mask], candidate_index,
                ),
                "calibration_descriptive_only": value_error_summary(
                    frames["calibration"], targets["calibration"], qpred["calibration"], candidate_index
                ),
                "heldout_descriptive_only": value_error_summary(
                    frames["heldout"], targets["heldout"], qpred["heldout"], candidate_index
                ),
            }

    payload = {
        "status": "complete", "dataset": args.dataset,
        "heldout_used_for_selection": False,
        "heldout_future_labels_used_for": "post-hoc descriptive value-error audit only",
        "m0_candidate_degeneracy": {
            "nominal_candidates": len(candidates),
            "unique_calibration_action_signatures": len(signatures),
            "expected_unique_due_to_mu_only": len(set(mus.tolist())),
            "signature_candidate_indices": list(signatures.values()),
            "explanation": "M=0时lambda只存在于被清零的value head中，因此48候选退化为最多6个mu策略。",
        },
        "selected_calibration": {"full": selected_full, "m0": selected_m0},
        "full_vs_m0_by_B": by_budget,
        "selected_full_value_error": value_diagnostics,
    }
    atomic_json(payload, args.output / f"{args.dataset}_m0_mechanism_audit.json")

    b4 = by_budget["4"]
    action = b4["action_relationship"]
    lines = [
        f"# {args.dataset}：Full 与 M=0 机制审计", "",
        "该报告只做冻结结果的事后机制分析；held-out没有参与候选或工作点选择。", "",
        "## B=4逐样本差异", "",
        f"- 完全相同动作：{action['same_action']}题。",
        f"- 只有Full早停：{action['full_only_stop']}题；只有M=0早停：{action['m0_only_stop']}题。",
        f"- 两者都早停但Full更早：{action['both_stop_full_earlier']}题；M=0更早：{action['both_stop_m0_earlier']}题。",
        f"- Full正确而M=0错误：{action['full_correct_m0_wrong']}题；反向：{action['full_wrong_m0_correct']}题。",
        f"- M=0相对Full每题token差均值：{b4['mean_m0_minus_full_tokens']:.2f}。", "",
        "## 关键诊断", "",
        f"名义48个M=0候选实际只有{len(signatures)}种动作签名，因为清零value后lambda失效；因此它不是完整的48策略动态前沿。",
        "完整逐样本分区、transition组合及value残差见同目录JSON。",
    ]
    (args.output / f"{args.dataset}_M0_MECHANISM_AUDIT_ZH.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "complete", "dataset": args.dataset,
        "unique_m0_policies": len(signatures), "B4": action,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

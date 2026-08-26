"""四状态概率 probe 与无需校准阈值的固定效用差停止规则。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.legacy_empirical_probe_v4 import summarize_policy_records, transition_name


STATE_NAMES = ("W_to_C", "C_to_W", "W_to_W", "C_to_C")
STATE_TO_INDEX = {name: index for index, name in enumerate(STATE_NAMES)}


class FourStateUtilityProbe(nn.Module):
    """与原标量 probe 共享骨干，只将末层改为四分类输出。"""

    def __init__(self, width: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(384, 96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 4),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def four_state_targets(frame: pd.DataFrame) -> np.ndarray:
    """按 [W→C, C→W, W→W, C→C] 顺序构造互斥完备标签。"""
    current = frame.current_success.to_numpy(dtype=bool)
    dense = frame.dense_success.to_numpy(dtype=bool)
    result = np.empty(len(frame), dtype=np.int64)
    result[(~current) & dense] = STATE_TO_INDEX["W_to_C"]
    result[current & (~dense)] = STATE_TO_INDEX["C_to_W"]
    result[(~current) & (~dense)] = STATE_TO_INDEX["W_to_W"]
    result[current & dense] = STATE_TO_INDEX["C_to_C"]
    return result


def multiclass_point_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """未加权四分类交叉熵；softmax 输出保留概率语义。"""
    return F.cross_entropy(logits, targets)


def legacy_weighted_multiclass_trajectory_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    remaining: torch.Tensor,
    offsets: np.ndarray,
    *,
    beta: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把旧加权 point loss 与 weakest-point protection 原样迁移到四分类 margin。"""
    point_terms = F.cross_entropy(logits, targets, reduction="none")
    danger = targets == STATE_TO_INDEX["W_to_C"]
    weights = torch.where(
        danger,
        torch.full_like(point_terms, 1.5),
        1.0 + remaining,
    )
    point = (point_terms * weights).mean()
    trajectory_terms: list[torch.Tensor] = []
    # softmax 后 p_WC-p_CW 的符号与 logit_WC-logit_CW 完全相同。
    continuation_margin = logits[:, STATE_TO_INDEX["W_to_C"]] - logits[:, STATE_TO_INDEX["C_to_W"]]
    for start, end in zip(offsets[:-1], offsets[1:]):
        local_danger = danger[start:end]
        if local_danger.any():
            values = continuation_margin[start:end][local_danger]
            soft_minimum = -beta * torch.logsumexp(-values / beta, dim=0)
            trajectory_terms.append(F.softplus(-soft_minimum))
    trajectory = (
        torch.stack(trajectory_terms).mean()
        if trajectory_terms
        else logits.sum() * 0.0
    )
    return point + trajectory, point, trajectory


@torch.no_grad()
def predict_probabilities(
    model: FourStateUtilityProbe,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(features), batch_size):
        values = torch.from_numpy(features[start:start + batch_size]).to(device)
        chunks.append(torch.softmax(model(values), dim=-1).float().cpu().numpy())
    result = np.concatenate(chunks).astype(np.float64, copy=False)
    if result.shape != (len(features), 4):
        raise ValueError(f"四分类概率形状错误：{result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("四分类概率包含 NaN/Inf")
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("四分类概率和不为1")
    return result


def probability_diagnostics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    predicted = probabilities.argmax(axis=1)
    one_hot = np.eye(4, dtype=np.float64)[targets]
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    per_class = {}
    for index, name in enumerate(STATE_NAMES):
        mask = targets == index
        per_class[name] = {
            "count": int(mask.sum()),
            "rate": float(mask.mean()),
            "recall": float((predicted[mask] == index).mean()) if mask.any() else None,
            "mean_predicted_probability": float(probabilities[:, index].mean()),
        }
    return {
        "rows": int(len(targets)),
        "cross_entropy": float(-np.log(clipped[np.arange(len(targets)), targets]).mean()),
        "accuracy": float((predicted == targets).mean()),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "probability_sum_max_abs_error": float(np.abs(probabilities.sum(axis=1) - 1.0).max()),
        "class_order": list(STATE_NAMES),
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
    }


def _fallback_policy_record(base: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem_id": str(base["problem_id"]),
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
        "replay_wall_ms": float(base.get("adaptive_fallback_wall_ms", base["dense_wall_ms"])),
        "dense_wall_ms": float(base["dense_wall_ms"]),
        "decision_utility": None,
        "p_WC": None,
        "p_CW": None,
        "p_WW": None,
        "p_CC": None,
    }


def simulate_zero_utility_policy(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    include_records: bool = False,
    fallback_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """首个 P(W→C)-P(C→W)<0 的 sentence checkpoint 停止；等于0继续。"""
    if len(frame) != len(probabilities):
        raise ValueError("frame/probability 行数不一致")
    if probabilities.shape != (len(frame), 4):
        raise ValueError(f"概率形状错误：{probabilities.shape}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("概率和不为1")
    scored = frame.copy()
    for index, name in enumerate(("p_WC", "p_CW", "p_WW", "p_CC")):
        scored[name] = probabilities[:, index]
    scored["decision_utility"] = scored.p_WC - scored.p_CW
    records: list[dict[str, Any]] = []
    for problem_id, group in scored.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        first = ordered.iloc[0]
        eligible = ordered[ordered.decision_utility < 0.0]
        fallback = eligible.empty
        if fallback:
            record = _fallback_policy_record({
                "problem_id": str(problem_id),
                "subject": first.get("subject"),
                "category": first.get("category"),
                "dense_prediction": first.dense_prediction,
                "gold_answer": first.gold_answer,
                "dense_success": first.dense_success,
                "dense_tokens": first.dense_tokens,
                "dense_wall_ms": first.dense_wall_ms,
                "adaptive_fallback_wall_ms": first.get("adaptive_fallback_wall_ms", first.dense_wall_ms),
            })
        else:
            chosen = eligible.iloc[0]
            current_success = bool(chosen.current_success)
            record = {
                "problem_id": str(problem_id),
                "subject": first.get("subject"),
                "category": first.get("category"),
                "fallback": False,
                "checkpoint": int(chosen.checkpoint),
                "transition": transition_name(current_success, bool(chosen.dense_success)),
                "method_prediction": chosen.current_prediction,
                "dense_prediction": chosen.dense_prediction,
                "gold_answer": chosen.gold_answer,
                "method_success": current_success,
                "dense_success": bool(chosen.dense_success),
                "method_tokens": min(
                    int(chosen.dense_tokens),
                    int(chosen.checkpoint) + int(chosen.branch_tokens),
                ),
                "dense_tokens": int(chosen.dense_tokens),
                "replay_wall_ms": float(chosen.get(
                    "replay_stop_wall_ms",
                    chosen.dense_prefill_cuda_ms + chosen.prefix_decode_cuda_ms + chosen.branch_wall_ms,
                )),
                "dense_wall_ms": float(chosen.dense_wall_ms),
                "decision_utility": float(chosen.decision_utility),
                "p_WC": float(chosen.p_WC),
                "p_CW": float(chosen.p_CW),
                "p_WW": float(chosen.p_WW),
                "p_CC": float(chosen.p_CC),
            }
        records.append(record)
    seen = {row["problem_id"] for row in records}
    for base in fallback_records or []:
        if str(base["problem_id"]) in seen:
            raise ValueError(f"重复 fallback 样本：{base['problem_id']}")
        records.append(_fallback_policy_record(base))
    summary = summarize_policy_records(records)
    summary.update({
        "decision_rule": "continue iff p_WC_minus_p_CW >= 0; stop otherwise",
        "threshold_calibration_used": False,
        "decision_boundary": 0.0,
    })
    if include_records:
        summary["records"] = records
    return summary

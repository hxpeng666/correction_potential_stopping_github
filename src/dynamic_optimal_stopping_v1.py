"""风险约束动态推理最优停止：多头模型、Bellman目标与token回放。"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.legacy_empirical_probe_v4 import correction_loss, transition_name


DYNAMIC_FEATURE_KINDS = (
    "full",
    "h_only",
    "delta_only",
    "scalars_only",
    "h_delta",
    "h_delta_plus_t",
    "h_delta_plus_log_t",
    "h_delta_plus_delta_t",
    "h_delta_plus_entropy",
    "h_delta_plus_delta_norm",
    "h_delta_plus_cosine",
    "full_no_t",
    "full_no_log_t",
    "full_no_delta_t",
    "full_no_entropy",
    "full_no_delta_norm",
    "full_no_cosine",
    "full_no_position",
    "full_no_geometry",
    "full_no_hidden",
    "full_no_delta",
    "main_no_t",
    "main_no_log_t",
    "main_no_delta_t",
    "main_no_entropy",
    "main_no_delta_norm",
    "main_no_cosine",
    "main_no_position",
    "main_no_geometry",
)


def _hidden_deltas(frame: pd.DataFrame, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.zeros_like(current, dtype=np.float32)
    delta_t = np.zeros((len(frame), 1), dtype=np.float32)
    for _, group in frame.groupby("problem_id", sort=False):
        positions = group.sort_values("checkpoint").index.to_numpy(dtype=np.int64)
        checkpoints = frame.loc[positions, "checkpoint"].to_numpy(dtype=np.float32)
        if len(positions) > 1:
            delta[positions[1:]] = current[positions[1:]] - current[positions[:-1]]
        delta_t[positions, 0] = np.diff(np.concatenate(([0.0], checkpoints)))
    return delta, delta_t


def build_dynamic_features(
    frame: pd.DataFrame,
    hidden: np.ndarray,
    capture_layers: list[int],
    *,
    layer: int = 20,
    feature_kind: str = "full",
) -> np.ndarray:
    """构造动态方法的分组/逐标量特征消融，不改变缓存或标签。"""
    if feature_kind not in DYNAMIC_FEATURE_KINDS:
        raise ValueError(f"未知动态特征：{feature_kind}")
    if layer not in capture_layers:
        raise ValueError(f"layer {layer}不在缓存层{capture_layers}中")
    current = hidden[:, capture_layers.index(layer), :].astype(np.float32, copy=False)
    delta, delta_t = _hidden_deltas(frame, current)
    checkpoint = frame.checkpoint.to_numpy(dtype=np.float32)[:, None]
    components = {
        "h": current,
        "delta": delta,
        "t": checkpoint,
        "log_t": np.log1p(checkpoint).astype(np.float32),
        "delta_t": delta_t,
        "entropy": frame.prefix_mean_entropy_tail8.to_numpy(dtype=np.float32)[:, None],
        "delta_norm": np.linalg.norm(delta, axis=1, keepdims=True).astype(np.float32),
    }
    current_norm = np.linalg.norm(current, axis=1, keepdims=True).astype(np.float32)
    components["cosine"] = (
        np.sum(current * delta, axis=1, keepdims=True)
        / np.maximum(current_norm * components["delta_norm"], 1e-6)
    ).astype(np.float32)
    scalars = ("t", "log_t", "delta_t", "entropy", "delta_norm", "cosine")
    if feature_kind == "full":
        names = ("h", "delta", *scalars)
    elif feature_kind == "h_only":
        names = ("h",)
    elif feature_kind == "delta_only":
        names = ("delta",)
    elif feature_kind == "scalars_only":
        names = scalars
    elif feature_kind == "h_delta":
        names = ("h", "delta")
    elif feature_kind.startswith("h_delta_plus_"):
        suffix = feature_kind.removeprefix("h_delta_plus_")
        names = ("h", "delta", suffix)
    elif feature_kind.startswith("full_no_"):
        removed = feature_kind.removeprefix("full_no_")
        removal = {
            "position": {"t", "log_t", "delta_t"},
            "geometry": {"delta_norm", "cosine"},
            "hidden": {"h"},
            "delta": {"delta"},
        }.get(removed, {removed})
        names = tuple(name for name in ("h", "delta", *scalars) if name not in removal)
    elif feature_kind.startswith("main_no_"):
        # 当前论文主输入是 full_no_delta=[h, 六个标量]；这些变体只从该
        # 2566维母体移除指定标量组，不能与旧5126维full的逐项删除混淆。
        removed = feature_kind.removeprefix("main_no_")
        removal = {
            "position": {"t", "log_t", "delta_t"},
            "geometry": {"delta_norm", "cosine"},
        }.get(removed, {removed})
        names = tuple(name for name in ("h", *scalars) if name not in removal)
    else:
        raise AssertionError(feature_kind)
    if not names or any(name not in components for name in names):
        raise ValueError(f"非法特征组合{feature_kind}: {names}")
    values = np.concatenate([components[name] for name in names], axis=1).astype(np.float32, copy=False)
    expected_width = sum(components[name].shape[1] for name in names)
    if values.shape != (len(frame), expected_width):
        raise ValueError(f"特征维度错误：{values.shape} != {(len(frame), expected_width)}")
    if not np.isfinite(values).all():
        raise ValueError(f"{feature_kind}包含NaN/Inf")
    return values


class DynamicLocalHeads(nn.Module):
    """共享5126→384→96表示，以及stop-correctness和lost-correct风险头。"""

    def __init__(self, width: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(width, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(384, 96),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.stop_correctness_head = nn.Linear(96, 1)
        self.lost_correct_risk_head = nn.Linear(96, 1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        representation = self.trunk(values)
        return (
            representation,
            self.stop_correctness_head(representation)[:, 0],
            self.lost_correct_risk_head(representation)[:, 0],
        )


class ContinuationValueBank(nn.Module):
    """每个预声明(lambda,mu)候选对应一个从共享表示到M_k的标量head。"""

    def __init__(self, candidates: int):
        super().__init__()
        self.heads = nn.Linear(96, candidates)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return self.heads(representation)


def supervised_local_loss(
    stop_logits: torch.Tensor,
    risk_logits: torch.Tensor,
    stop_targets: torch.Tensor,
    risk_targets: torch.Tensor,
    remaining: torch.Tensor,
    offsets: np.ndarray,
    *,
    beta: float,
    gamma_risk: float,
    gamma_trajectory: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_targets)
    risk_total, risk_point, risk_trajectory = correction_loss(
        risk_logits,
        risk_targets,
        remaining,
        offsets,
        beta=beta,
        trajectory=True,
    )
    risk_combined = risk_point + gamma_trajectory * risk_trajectory
    total = stop_loss + gamma_risk * risk_combined
    return total, {
        "stop_bce": stop_loss,
        "risk_point": risk_point,
        "risk_trajectory": risk_trajectory,
        "risk_combined": risk_combined,
        "risk_original_total": risk_total,
    }


def group_positions_offsets(frame: pd.DataFrame, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = [
        group.sort_values("checkpoint").index.to_numpy(dtype=np.int64)
        for _, group in frame.loc[mask].groupby("problem_id", sort=False)
    ]
    if not groups:
        raise ValueError("轨迹组为空")
    positions = np.concatenate(groups)
    offsets = np.cumsum([0] + [len(group) for group in groups])
    return positions, offsets


@torch.no_grad()
def predict_local_heads(
    model: DynamicLocalHeads,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    representations = []
    stop = []
    risk = []
    for start in range(0, len(features), batch_size):
        values = torch.from_numpy(features[start:start + batch_size]).to(device)
        local_representation, stop_logits, risk_logits = model(values)
        representations.append(local_representation.float().cpu().numpy())
        stop.append(torch.sigmoid(stop_logits).float().cpu().numpy())
        risk.append(torch.sigmoid(risk_logits).float().cpu().numpy())
    representation = np.concatenate(representations).astype(np.float32, copy=False)
    stop_probability = np.concatenate(stop).astype(np.float64, copy=False)
    risk_probability = np.concatenate(risk).astype(np.float64, copy=False)
    for name, values in (("representation", representation), ("stop", stop_probability), ("risk", risk_probability)):
        if not np.isfinite(values).all():
            raise ValueError(f"{name}包含NaN/Inf")
    return representation, stop_probability, risk_probability


@torch.no_grad()
def predict_continuation_values(
    model: ContinuationValueBank,
    representations: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(representations), batch_size):
        values = torch.from_numpy(representations[start:start + batch_size]).to(device)
        output.append(model(values).float().cpu().numpy())
    result = np.concatenate(output).astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("continuation value包含NaN/Inf")
    return result


def transition_token_costs(
    frame: pd.DataFrame,
    cost_unit_tokens: float,
    cost_mode: str = "incremental",
) -> np.ndarray:
    """计算继续动作的增量成本，或endpoint消融中的全部剩余成本。"""
    if cost_mode not in {"incremental", "remaining_to_dense"}:
        raise ValueError(f"未知cost_mode：{cost_mode}")
    result = np.zeros(len(frame), dtype=np.float64)
    for _, group in frame.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        positions = ordered.index.to_numpy(dtype=np.int64)
        checkpoints = ordered.checkpoint.to_numpy(dtype=np.float64)
        dense_tokens = float(ordered.iloc[0].dense_tokens)
        if cost_mode == "incremental":
            next_positions = np.concatenate((checkpoints[1:], [dense_tokens]))
            increments = np.maximum(next_positions - checkpoints, 0.0)
        else:
            increments = np.maximum(dense_tokens - checkpoints, 0.0)
        result[positions] = increments / float(cost_unit_tokens)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("非法transition token cost")
    return result


def backward_value_targets(
    frame: pd.DataFrame,
    stop_probability: np.ndarray,
    risk_probability: np.ndarray,
    lambdas: np.ndarray,
    mus: np.ndarray,
    *,
    cost_unit_tokens: float,
) -> tuple[np.ndarray, np.ndarray]:
    """沿每条已缓存Dense轨迹从后向前计算M_k=V(z_{k+1})监督。"""
    policies = len(lambdas)
    if len(mus) != policies:
        raise ValueError("lambda/mu候选数不一致")
    if len(frame) != len(stop_probability) or len(frame) != len(risk_probability):
        raise ValueError("frame/local prediction行数不一致")
    transition_cost = transition_token_costs(frame, cost_unit_tokens)
    targets = np.zeros((len(frame), policies), dtype=np.float32)
    state_values = np.zeros((len(frame), policies), dtype=np.float32)
    for _, group in frame.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        positions = ordered.index.to_numpy(dtype=np.int64)
        terminal_value = np.full(policies, float(ordered.iloc[0].dense_success), dtype=np.float64)
        next_value = terminal_value
        for position in positions[::-1]:
            targets[position] = next_value.astype(np.float32)
            q_stop = stop_probability[position] - mus * risk_probability[position]
            q_continue = -lambdas * transition_cost[position] + next_value
            current_value = np.maximum(q_stop, q_continue)
            state_values[position] = current_value.astype(np.float32)
            next_value = current_value
    if not np.isfinite(targets).all() or not np.isfinite(state_values).all():
        raise ValueError("Bellman target包含NaN/Inf")
    return targets, state_values


def one_step_value_targets(
    frame: pd.DataFrame,
    stop_probability: np.ndarray,
    risk_probability: np.ndarray,
    mus: np.ndarray,
) -> np.ndarray:
    """只估计下一checkpoint立即停止价值，不执行递归max。"""
    targets = np.zeros((len(frame), len(mus)), dtype=np.float32)
    for _, group in frame.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        positions = ordered.index.to_numpy(dtype=np.int64)
        terminal = np.full(len(mus), float(ordered.iloc[0].dense_success), dtype=np.float64)
        for local_index, position in enumerate(positions):
            if local_index + 1 < len(positions):
                next_position = positions[local_index + 1]
                target = stop_probability[next_position] - mus * risk_probability[next_position]
            else:
                target = terminal
            targets[position] = target.astype(np.float32)
    if not np.isfinite(targets).all():
        raise ValueError("one-step target包含NaN/Inf")
    return targets


def dense_endpoint_value_targets(frame: pd.DataFrame, candidates: int) -> np.ndarray:
    """原式endpoint消融：继续价值只预测Dense最终正确性。"""
    target = frame.dense_success.to_numpy(dtype=np.float32)[:, None]
    return np.repeat(target, candidates, axis=1)


def _fallback_record(base: dict[str, Any]) -> dict[str, Any]:
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
        "q_stop": None,
        "q_continue": None,
        "p_stop_correct": None,
        "p_lost_correct": None,
        "continuation_value": None,
    }


def summarize_token_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("records为空")
    success = np.asarray([row["method_success"] for row in records], dtype=np.float64)
    dense = np.asarray([row["dense_success"] for row in records], dtype=np.float64)
    used = np.asarray([row["method_tokens"] for row in records], dtype=np.float64)
    dense_tokens = np.asarray([row["dense_tokens"] for row in records], dtype=np.float64)
    stopped = np.asarray([not row["fallback"] for row in records], dtype=bool)
    counts = {name: 0 for name in ("W_to_C", "C_to_W", "W_to_W", "C_to_C")}
    for row in records:
        if row["transition"] in counts:
            counts[row["transition"]] += 1
    return {
        "problems": len(records),
        "accuracy": float(success.mean()),
        "dense_accuracy": float(dense.mean()),
        "delta_dense_pp": float(100.0 * (success.mean() - dense.mean())),
        "accuracy_drop_pp": float(100.0 * (dense.mean() - success.mean())),
        "coverage": float(stopped.mean()),
        "fallback": int((~stopped).sum()),
        "mean_reasoning_tokens": float(used.mean()),
        "mean_dense_reasoning_tokens": float(dense_tokens.mean()),
        "token_reduction": float(1.0 - used.mean() / dense_tokens.mean()),
        "lost_correct_count": int(counts["W_to_C"]),
        "lost_correct_rate": float(counts["W_to_C"] / len(records)),
        "gained_correct_count": int(counts["C_to_W"]),
        "counts": counts,
        "short_answer_cost_ignored": True,
    }


def simulate_dynamic_policy(
    frame: pd.DataFrame,
    stop_probability: np.ndarray,
    risk_probability: np.ndarray,
    continuation_values: np.ndarray,
    *,
    lambda_value: float,
    mu_value: float,
    cost_unit_tokens: float,
    fallback_records: list[dict[str, Any]] | None = None,
    include_records: bool = False,
    force_dense: bool = False,
    cost_mode: str = "incremental",
) -> dict[str, Any]:
    """逐题执行Q_stop>=Q_continue的first-hit动态策略。"""
    if not (len(frame) == len(stop_probability) == len(risk_probability) == len(continuation_values)):
        raise ValueError("动态策略输入行数不一致")
    transition_cost = transition_token_costs(frame, cost_unit_tokens, cost_mode=cost_mode)
    q_stop = stop_probability - float(mu_value) * risk_probability
    q_continue = -float(lambda_value) * transition_cost + continuation_values
    records: list[dict[str, Any]] = []
    for problem_id, group in frame.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        positions = ordered.index.to_numpy(dtype=np.int64)
        first = ordered.iloc[0]
        chosen_position = None
        if not force_dense:
            for position in positions:
                if q_stop[position] >= q_continue[position]:
                    chosen_position = int(position)
                    break
        if chosen_position is None:
            records.append(_fallback_record({
                "problem_id": str(problem_id),
                "subject": first.get("subject"),
                "category": first.get("category"),
                "dense_prediction": first.dense_prediction,
                "gold_answer": first.gold_answer,
                "dense_success": first.dense_success,
                "dense_tokens": first.dense_tokens,
            }))
            continue
        chosen = frame.loc[chosen_position]
        current_success = bool(chosen.current_success)
        records.append({
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
            # 用户指定短答案cost暂时忽略，因此只计reasoning checkpoint位置。
            "method_tokens": min(int(chosen.dense_tokens), int(chosen.checkpoint)),
            "dense_tokens": int(chosen.dense_tokens),
            "q_stop": float(q_stop[chosen_position]),
            "q_continue": float(q_continue[chosen_position]),
            "p_stop_correct": float(stop_probability[chosen_position]),
            "p_lost_correct": float(risk_probability[chosen_position]),
            "continuation_value": float(continuation_values[chosen_position]),
        })
    seen = {row["problem_id"] for row in records}
    for base in fallback_records or []:
        if str(base["problem_id"]) in seen:
            raise ValueError(f"重复fallback问题：{base['problem_id']}")
        records.append(_fallback_record(base))
    summary = summarize_token_records(records)
    summary.update({
        "lambda": float(lambda_value),
        "mu": float(mu_value),
        "cost_unit_tokens": float(cost_unit_tokens),
        "decision_rule": "stop iff Q_stop >= Q_continue",
        "force_dense": bool(force_dense),
        "cost_mode": cost_mode,
    })
    if include_records:
        summary["records"] = records
    return summary


def binomial_cdf(k: int, n: int, probability: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    terms = [
        math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
        + j * log_p + (n - j) * log_q
        for j in range(k + 1)
    ]
    maximum = max(terms)
    return float(math.exp(maximum) * sum(math.exp(value - maximum) for value in terms))


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """单侧(1-delta) Clopper-Pearson上界，无SciPy依赖。"""
    if not 0 <= k <= n or not 0.0 < delta < 1.0:
        raise ValueError("非法binomial参数")
    if k == n:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if binomial_cdf(k, n, middle) > delta:
            low = middle
        else:
            high = middle
    return float(high)

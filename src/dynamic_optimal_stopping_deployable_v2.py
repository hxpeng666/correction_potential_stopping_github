"""可部署的动态最优停止与受控 OS-Pruner 基线。

本模块与 ``dynamic_optimal_stopping_v1`` 的关键区别是：策略在当前
sentence checkpoint 作决策时，不能读取下一 checkpoint 位置或 Dense
剩余长度。未来 token 成本只进入离线 Q-target；网络输出本身就是完整的
continue action value。
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.dynamic_optimal_stopping_v1 import (
    DynamicLocalHeads,
    build_dynamic_features,
    clopper_pearson_upper,
    group_positions_offsets,
    predict_local_heads,
    summarize_token_records,
    supervised_local_loss,
    transition_token_costs,
)
from src.legacy_empirical_probe_v4 import transition_name


class ContinuationQValueBank(nn.Module):
    """从当前可观测表示直接预测包含未来增量成本的 Q_continue。"""

    def __init__(self, candidates: int):
        super().__init__()
        self.heads = nn.Linear(96, candidates)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return self.heads(representation)


class OSPrunerPolicyBank(nn.Module):
    """严格受控 OS-Pruner：同一 2566→384→96 主干与多个 stop-policy 头。"""

    def __init__(self, width: int, candidates: int):
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
        self.stop_heads = nn.Linear(96, candidates)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.stop_heads(self.trunk(values))


def recursive_q_continue_targets(
    frame: pd.DataFrame,
    stop_probability: np.ndarray,
    risk_probability: np.ndarray,
    lambdas: np.ndarray,
    mus: np.ndarray,
    *,
    cost_unit_tokens: float,
) -> tuple[np.ndarray, np.ndarray]:
    """构造可部署的有限时域 Bellman Q_continue 监督。

    真实 ``delta_t_next`` 只在这里作为监督标签使用。输出 target 已经包含
    ``-lambda * delta_t_next``，部署决策不得再次读取或扣除未来成本。
    """
    policies = len(lambdas)
    if len(mus) != policies:
        raise ValueError("lambda/mu候选数不一致")
    if len(frame) != len(stop_probability) or len(frame) != len(risk_probability):
        raise ValueError("frame/local prediction行数不一致")
    transition_cost = transition_token_costs(frame, cost_unit_tokens, cost_mode="incremental")
    q_targets = np.zeros((len(frame), policies), dtype=np.float32)
    state_values = np.zeros((len(frame), policies), dtype=np.float32)
    for _, group in frame.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        positions = ordered.index.to_numpy(dtype=np.int64)
        next_value = np.full(policies, float(ordered.iloc[0].dense_success), dtype=np.float64)
        for position in positions[::-1]:
            q_continue = -lambdas * transition_cost[position] + next_value
            q_targets[position] = q_continue.astype(np.float32)
            q_stop = stop_probability[position] - mus * risk_probability[position]
            current_value = np.maximum(q_stop, q_continue)
            state_values[position] = current_value.astype(np.float32)
            next_value = current_value
    if not np.isfinite(q_targets).all() or not np.isfinite(state_values).all():
        raise ValueError("Bellman Q target包含NaN/Inf")
    return q_targets, state_values


def one_step_q_continue_targets(
    frame: pd.DataFrame,
    stop_probability: np.ndarray,
    risk_probability: np.ndarray,
    lambdas: np.ndarray,
    mus: np.ndarray,
    *,
    cost_unit_tokens: float,
) -> np.ndarray:
    """one-step 消融：预测走到下一状态并立即停止的完整 action value。"""
    if len(lambdas) != len(mus):
        raise ValueError("lambda/mu候选数不一致")
    transition_cost = transition_token_costs(frame, cost_unit_tokens, cost_mode="incremental")
    targets = np.zeros((len(frame), len(mus)), dtype=np.float32)
    for _, group in frame.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        positions = ordered.index.to_numpy(dtype=np.int64)
        terminal = np.full(len(mus), float(ordered.iloc[0].dense_success), dtype=np.float64)
        for local_index, position in enumerate(positions):
            if local_index + 1 < len(positions):
                next_position = positions[local_index + 1]
                next_stop_value = (
                    stop_probability[next_position]
                    - mus * risk_probability[next_position]
                )
            else:
                next_stop_value = terminal
            targets[position] = (
                -lambdas * transition_cost[position] + next_stop_value
            ).astype(np.float32)
    if not np.isfinite(targets).all():
        raise ValueError("one-step Q target包含NaN/Inf")
    return targets


def dense_endpoint_q_continue_targets(
    frame: pd.DataFrame,
    lambdas: np.ndarray,
    *,
    cost_unit_tokens: float,
) -> np.ndarray:
    """Dense-endpoint 消融：预测继续到 Dense 的完整剩余价值。"""
    remaining_cost = transition_token_costs(
        frame, cost_unit_tokens, cost_mode="remaining_to_dense"
    )
    dense = frame.dense_success.to_numpy(dtype=np.float64)[:, None]
    targets = dense - remaining_cost[:, None] * lambdas[None, :]
    if not np.isfinite(targets).all():
        raise ValueError("Dense-endpoint Q target包含NaN/Inf")
    return targets.astype(np.float32)


@torch.no_grad()
def predict_q_values(
    model: ContinuationQValueBank,
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
        raise ValueError("Q_continue预测包含NaN/Inf")
    return result


def decide_current_action(q_stop: float, q_continue: float) -> bool:
    """仅基于当前可见的两个标量作决策；True 表示 stop。"""
    if not np.isfinite(q_stop) or not np.isfinite(q_continue):
        raise ValueError("当前动作值包含NaN/Inf")
    return bool(q_stop >= q_continue)


def _fallback_record(base: Any) -> dict[str, Any]:
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
        "continuation_q_value": None,
    }


def simulate_deployable_dynamic_policy(
    frame: pd.DataFrame,
    stop_probability: np.ndarray,
    risk_probability: np.ndarray,
    q_continue_values: np.ndarray,
    *,
    mu_value: float,
    fallback_records: list[dict[str, Any]] | None = None,
    include_records: bool = False,
    force_dense: bool = False,
) -> dict[str, Any]:
    """可部署 first-hit replay；动作计算禁止读取任何下一状态字段。"""
    if not (
        len(frame) == len(stop_probability) == len(risk_probability) == len(q_continue_values)
    ):
        raise ValueError("动态策略输入行数不一致")
    q_stop = stop_probability - float(mu_value) * risk_probability
    records: list[dict[str, Any]] = []
    for problem_id, group in frame.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        positions = ordered.index.to_numpy(dtype=np.int64)
        first = ordered.iloc[0]
        chosen_position = None
        if not force_dense:
            for position in positions:
                if decide_current_action(q_stop[position], q_continue_values[position]):
                    chosen_position = int(position)
                    break
        if chosen_position is None:
            records.append(_fallback_record(first))
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
            "method_tokens": min(int(chosen.dense_tokens), int(chosen.checkpoint)),
            "dense_tokens": int(chosen.dense_tokens),
            "q_stop": float(q_stop[chosen_position]),
            "q_continue": float(q_continue_values[chosen_position]),
            "p_stop_correct": float(stop_probability[chosen_position]),
            "p_lost_correct": float(risk_probability[chosen_position]),
            "continuation_q_value": float(q_continue_values[chosen_position]),
        })
    seen = {row["problem_id"] for row in records}
    for base in fallback_records or []:
        if str(base["problem_id"]) in seen:
            raise ValueError(f"重复fallback问题：{base['problem_id']}")
        records.append(_fallback_record(base))
    summary = summarize_token_records(records)
    summary.update({
        "mu": float(mu_value),
        "decision_rule": "stop iff p_stop-mu*p_lost_correct >= Q_continue_hat(z_t)",
        "force_dense": bool(force_dense),
        "future_fields_used_by_action": False,
        "continuation_value_includes_future_cost": True,
    })
    if include_records:
        summary["records"] = records
    return summary


def os_pruner_expected_utility(
    logits: torch.Tensor,
    frame: pd.DataFrame,
    positions: np.ndarray,
    offsets: np.ndarray,
    lambdas: torch.Tensor,
    mus: torch.Tensor,
    *,
    cost_unit_tokens: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算 differentiable first-hit utility 及期望 lost-correct。"""
    if logits.ndim != 2 or logits.shape[1] != len(lambdas) or len(lambdas) != len(mus):
        raise ValueError("OS-Pruner候选维度不一致")
    local = frame.loc[positions]
    stop_prob = torch.sigmoid(logits)
    utilities = []
    risks = []
    expected_tokens = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        trajectory_prob = stop_prob[start:end]
        survival = torch.ones_like(trajectory_prob[0])
        stop_masses = []
        for row_probability in trajectory_prob:
            stop_masses.append(survival * row_probability)
            survival = survival * (1.0 - row_probability)
        stop_mass = torch.stack(stop_masses, dim=0)
        rows = local.iloc[int(start):int(end)]
        correctness = torch.as_tensor(
            rows.current_success.to_numpy(dtype=np.float32),
            device=logits.device,
        )[:, None]
        checkpoints = torch.as_tensor(
            rows.checkpoint.to_numpy(dtype=np.float32) / float(cost_unit_tokens),
            device=logits.device,
        )[:, None]
        dense_success = float(rows.iloc[0].dense_success)
        dense_tokens = float(rows.iloc[0].dense_tokens) / float(cost_unit_tokens)
        lost = dense_success * (1.0 - correctness)
        stop_utility = correctness - lambdas[None, :] * checkpoints - mus[None, :] * lost
        fallback_utility = dense_success - lambdas * dense_tokens
        utilities.append((stop_mass * stop_utility).sum(dim=0) + survival * fallback_utility)
        risks.append((stop_mass * lost).sum(dim=0))
        expected_tokens.append(
            (stop_mass * checkpoints).sum(dim=0) + survival * dense_tokens
        )
    utility = torch.stack(utilities).mean(dim=0)
    risk = torch.stack(risks).mean(dim=0)
    token_cost = torch.stack(expected_tokens).mean(dim=0)
    return -utility.mean(), {
        "mean_candidate_utility": utility.mean(),
        "mean_expected_lost_correct": risk.mean(),
        "mean_expected_normalized_tokens": token_cost.mean(),
    }


@torch.no_grad()
def predict_os_stop_probabilities(
    model: OSPrunerPolicyBank,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(features), batch_size):
        values = torch.from_numpy(features[start:start + batch_size]).to(device)
        output.append(torch.sigmoid(model(values)).float().cpu().numpy())
    result = np.concatenate(output).astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("OS-Pruner停止概率包含NaN/Inf")
    return result


def deterministic_action_uniform(
    action_seed: int,
    dataset: str,
    split: str,
    problem_id: str,
    checkpoint: int,
) -> float:
    """固定 common-random-number，使随机 policy rollout 可复现且与任务顺序无关。"""
    payload = f"{action_seed}|{dataset}|{split}|{problem_id}|{checkpoint}".encode("utf-8")
    raw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (raw + 0.5) / float(2**64)


def simulate_os_pruner_policy(
    frame: pd.DataFrame,
    stop_probability: np.ndarray,
    *,
    dataset: str,
    split: str,
    action_seed: int,
    fallback_records: list[dict[str, Any]] | None = None,
    include_records: bool = False,
    force_dense: bool = False,
) -> dict[str, Any]:
    """按原 OS-Pruner Bernoulli first-hit 语义执行固定随机数 rollout。"""
    if len(frame) != len(stop_probability):
        raise ValueError("OS-Pruner输入行数不一致")
    records: list[dict[str, Any]] = []
    for problem_id, group in frame.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        positions = ordered.index.to_numpy(dtype=np.int64)
        first = ordered.iloc[0]
        chosen_position = None
        if not force_dense:
            for position in positions:
                checkpoint = int(frame.loc[position, "checkpoint"])
                threshold = deterministic_action_uniform(
                    action_seed, dataset, split, str(problem_id), checkpoint
                )
                if float(stop_probability[position]) >= threshold:
                    chosen_position = int(position)
                    break
        if chosen_position is None:
            records.append(_fallback_record(first))
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
            "method_tokens": min(int(chosen.dense_tokens), int(chosen.checkpoint)),
            "dense_tokens": int(chosen.dense_tokens),
            "stop_probability": float(stop_probability[chosen_position]),
            "action_uniform": float(deterministic_action_uniform(
                action_seed, dataset, split, str(problem_id), int(chosen.checkpoint)
            )),
        })
    seen = {row["problem_id"] for row in records}
    for base in fallback_records or []:
        if str(base["problem_id"]) in seen:
            raise ValueError(f"重复fallback问题：{base['problem_id']}")
        records.append(_fallback_record(base))
    summary = summarize_token_records(records)
    summary.update({
        "policy_rollout": "fixed-seed Bernoulli first-hit",
        "action_seed": int(action_seed),
        "future_fields_used_by_action": False,
    })
    if include_records:
        summary["records"] = records
    return summary


def summarize_expected_os_policy(
    frame: pd.DataFrame,
    stop_probability: np.ndarray,
    fallback_records: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """解析计算随机 first-hit policy 的期望指标，供固定 rollout 稳定性审计。"""
    if len(frame) != len(stop_probability):
        raise ValueError("OS-Pruner期望指标输入行数不一致")
    accuracy_sum = 0.0
    lost_sum = 0.0
    token_sum = 0.0
    dense_token_sum = 0.0
    coverage_sum = 0.0
    problems = 0
    for _, group in frame.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        positions = ordered.index.to_numpy(dtype=np.int64)
        probability = np.clip(stop_probability[positions], 0.0, 1.0)
        survival = 1.0
        stop_mass = []
        for value in probability:
            stop_mass.append(survival * float(value))
            survival *= 1.0 - float(value)
        masses = np.asarray(stop_mass, dtype=np.float64)
        correctness = ordered.current_success.to_numpy(dtype=np.float64)
        checkpoints = ordered.checkpoint.to_numpy(dtype=np.float64)
        dense_success = float(ordered.iloc[0].dense_success)
        dense_tokens = float(ordered.iloc[0].dense_tokens)
        accuracy_sum += float(np.dot(masses, correctness) + survival * dense_success)
        lost_sum += float(np.dot(masses, dense_success * (1.0 - correctness)))
        token_sum += float(np.dot(masses, checkpoints) + survival * dense_tokens)
        dense_token_sum += dense_tokens
        coverage_sum += 1.0 - survival
        problems += 1
    for base in fallback_records or []:
        accuracy_sum += float(base["dense_success"])
        token_sum += float(base["dense_tokens"])
        dense_token_sum += float(base["dense_tokens"])
        problems += 1
    if problems == 0:
        raise ValueError("OS-Pruner期望指标没有问题")
    return {
        "expected_accuracy": accuracy_sum / problems,
        "expected_lost_correct_count": lost_sum,
        "expected_lost_correct_rate": lost_sum / problems,
        "expected_coverage": coverage_sum / problems,
        "expected_mean_reasoning_tokens": token_sum / problems,
        "expected_token_reduction": 1.0 - token_sum / dense_token_sum,
    }


def candidate_feasibility_counts(
    curve: list[dict[str, Any]],
    dense_accuracy: float,
    epsilon: float,
    budgets: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    """区分不会停止、准确率不合格和风险不合格三类机制。"""
    accuracy_rows = [row for row in curve if row["accuracy"] >= dense_accuracy - epsilon]
    return {
        "total_candidates": len(curve),
        "nonzero_stopping_candidates": sum(row["coverage"] > 0 for row in curve),
        "accuracy_feasible_candidates": len(accuracy_rows),
        "accuracy_and_budget_feasible": {
            str(int(budget)): sum(
                row["lost_correct_count"] <= int(budget) for row in accuracy_rows
            )
            for budget in budgets
        },
    }


__all__ = [
    "ContinuationQValueBank",
    "DynamicLocalHeads",
    "OSPrunerPolicyBank",
    "build_dynamic_features",
    "candidate_feasibility_counts",
    "clopper_pearson_upper",
    "decide_current_action",
    "dense_endpoint_q_continue_targets",
    "group_positions_offsets",
    "one_step_q_continue_targets",
    "os_pruner_expected_utility",
    "predict_local_heads",
    "predict_os_stop_probabilities",
    "predict_q_values",
    "recursive_q_continue_targets",
    "simulate_deployable_dynamic_policy",
    "simulate_os_pruner_policy",
    "summarize_expected_os_policy",
    "summarize_token_records",
    "supervised_local_loss",
]

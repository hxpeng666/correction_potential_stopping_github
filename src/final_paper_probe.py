"""Final-paper probe features, losses, trajectory replay, and risk calibration."""
from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta as beta_distribution
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("correctness", "consistency", "last_switch", "correction")
FEATURE_KINDS = (
    "h_only",
    "h_delta",
    "full",
    "full_no_entropy",
    "full_no_position",
)
SCHEDULES = ("fixed", "sentence", "hybrid")


class FinalPaperProbe(nn.Module):
    """The preregistered scalar 5126->384->96->1 MLP (or ablated input width)."""

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
            nn.Linear(96, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)[:, 0]


def _fallback_record(artifact: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(artifact["source_dense_artifact"])
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    dense = source["dense"]
    return {
        "problem_id": str(source["problem_id"]),
        "subject": source["record"].get("subject"),
        "category": source["record"].get("category"),
        "gold_answer": source["gold_answer"],
        "dense_prediction": dense["prediction"],
        "dense_success": bool(dense["success"]),
        "dense_tokens": int(dense["reasoning_tokens"]),
        "dense_wall_ms": float(dense["wall_ms"]),
    }


def _artifact_rows(
    path: Path,
    schedule: str,
) -> tuple[list[dict[str, Any]], torch.Tensor, list[int], dict[str, Any] | None]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("status") != "complete":
        raise ValueError(f"incomplete checkpoint artifact: {path}")
    rows = artifact.get("rows", [])
    hidden = artifact.get("hidden")
    if not torch.is_tensor(hidden) or len(rows) != int(hidden.shape[0]):
        raise ValueError(f"row/vector mismatch in {path}")
    selected = [
        index
        for index, row in enumerate(rows)
        if schedule in row.get("checkpoint_schedules", [])
    ]
    fallback = _fallback_record(artifact) if not selected else None
    return (
        [rows[index] for index in selected],
        hidden[selected].float(),
        [int(value) for value in artifact["capture_layers"]],
        fallback,
    )


def load_checkpoint_split(
    directory: Path,
    schedule: str,
) -> tuple[pd.DataFrame, np.ndarray, list[int], list[dict[str, Any]]]:
    """Load scorable checkpoints plus explicit Dense-only fallback problems."""
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule: {schedule}")
    paths = sorted(directory.glob("sample_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no checkpoint artifacts in {directory}")
    all_rows: list[dict[str, Any]] = []
    all_hidden: list[np.ndarray] = []
    fallbacks: list[dict[str, Any]] = []
    capture_layers: list[int] | None = None
    for path in paths:
        rows, hidden, layers, fallback = _artifact_rows(path, schedule)
        if capture_layers is None:
            capture_layers = layers
        elif capture_layers != layers:
            raise ValueError(f"capture-layer mismatch in {path}: {layers} != {capture_layers}")
        all_rows.extend(rows)
        all_hidden.extend(hidden.numpy())
        if fallback is not None:
            fallbacks.append(fallback)
    if not all_rows:
        raise ValueError(f"schedule {schedule} has no legal checkpoints in {directory}")
    frame = pd.DataFrame(all_rows)
    vectors = np.stack(all_hidden).astype(np.float32, copy=False)
    frame["_vector_index"] = np.arange(len(frame))
    frame = frame.sort_values(["problem_id", "checkpoint"], kind="stable").reset_index(drop=True)
    vectors = vectors[frame.pop("_vector_index").to_numpy(dtype=np.int64)]
    if frame.duplicated(["problem_id", "checkpoint"]).any():
        raise ValueError(f"duplicate problem/checkpoint rows in {directory}")
    if vectors.shape[0] != len(frame) or vectors.ndim != 3:
        raise ValueError(f"invalid hidden tensor shape {vectors.shape} in {directory}")
    if not np.isfinite(vectors).all():
        raise ValueError(f"NaN/Inf hidden state in {directory}")
    if set(frame.problem_id.astype(str)) & {row["problem_id"] for row in fallbacks}:
        raise ValueError(f"scorable/fallback problem overlap in {directory}")
    if int(frame.problem_id.nunique()) + len(fallbacks) != len(paths):
        raise ValueError(f"problem accounting mismatch in {directory}")
    assert capture_layers is not None
    return add_targets(frame), vectors, capture_layers, fallbacks


def _last_switch_flags(current: list[str], dense: str) -> list[bool]:
    sequence = current + [dense]
    final_switch = max(
        (
            index
            for index in range(len(sequence) - 1)
            if sequence[index] != sequence[index + 1]
        ),
        default=-1,
    )
    return [index > final_switch for index in range(len(current))]


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the four controlled targets without treating two missing answers as consistent."""
    result = frame.copy()
    result["target_last_switch"] = False
    for _, group in result.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        current = [
            str(value) if pd.notna(value) else "<MISSING>"
            for value in ordered.current_prediction
        ]
        dense_value = ordered.iloc[0].dense_prediction
        dense = str(dense_value) if pd.notna(dense_value) else "<MISSING>"
        result.loc[ordered.index, "target_last_switch"] = _last_switch_flags(current, dense)
    current_present = result.current_prediction.notna()
    dense_present = result.dense_prediction.notna()
    result["target_correctness"] = result.current_success.astype(bool)
    result["target_consistency"] = (
        current_present
        & dense_present
        & result.current_prediction.astype(str).eq(result.dense_prediction.astype(str))
    )
    result["target_correction"] = (
        (~result.current_success.astype(bool)) & result.dense_success.astype(bool)
    )
    return result


def target_values(frame: pd.DataFrame, method: str) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(method)
    return frame[f"target_{method}"].to_numpy(dtype=np.float32)


def method_direction(method: str) -> str:
    if method not in METHODS:
        raise ValueError(method)
    return "low" if method == "correction" else "high"


def _deltas(frame: pd.DataFrame, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.zeros_like(current, dtype=np.float32)
    delta_t = np.zeros((len(frame), 1), dtype=np.float32)
    for _, group in frame.groupby("problem_id", sort=False):
        positions = group.sort_values("checkpoint").index.to_numpy(dtype=np.int64)
        checkpoints = frame.loc[positions, "checkpoint"].to_numpy(dtype=np.float32)
        if len(positions) > 1:
            delta[positions[1:]] = current[positions[1:]] - current[positions[:-1]]
        delta_t[positions, 0] = np.diff(np.concatenate(([0.0], checkpoints)))
    return delta, delta_t


def build_features(
    frame: pd.DataFrame,
    hidden: np.ndarray,
    capture_layers: list[int],
    *,
    layer: int = 20,
    feature_kind: str = "full",
) -> np.ndarray:
    """Build layer-specific hidden-dynamics features; full has exactly 5126 columns."""
    if feature_kind not in FEATURE_KINDS:
        raise ValueError(feature_kind)
    if layer not in capture_layers:
        raise ValueError(f"layer {layer} absent from {capture_layers}")
    current = hidden[:, capture_layers.index(layer), :].astype(np.float32, copy=False)
    delta, delta_t = _deltas(frame, current)
    checkpoint = frame.checkpoint.to_numpy(dtype=np.float32)[:, None]
    log_checkpoint = np.log1p(checkpoint).astype(np.float32)
    entropy = frame.prefix_mean_entropy_tail8.to_numpy(dtype=np.float32)[:, None]
    delta_norm = np.linalg.norm(delta, axis=1, keepdims=True).astype(np.float32)
    current_norm = np.linalg.norm(current, axis=1, keepdims=True).astype(np.float32)
    cosine = (
        np.sum(current * delta, axis=1, keepdims=True)
        / np.maximum(current_norm * delta_norm, 1e-6)
    ).astype(np.float32)
    if feature_kind == "h_only":
        parts = [current]
    elif feature_kind == "h_delta":
        parts = [current, delta]
    elif feature_kind == "full":
        parts = [
            current,
            delta,
            checkpoint,
            log_checkpoint,
            delta_t,
            entropy,
            delta_norm,
            cosine,
        ]
    elif feature_kind == "full_no_entropy":
        parts = [
            current,
            delta,
            checkpoint,
            log_checkpoint,
            delta_t,
            delta_norm,
            cosine,
        ]
    else:
        parts = [current, delta, entropy, delta_norm, cosine]
    values = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    expected = {
        "h_only": 2560,
        "h_delta": 5120,
        "full": 5126,
        "full_no_entropy": 5125,
        "full_no_position": 5123,
    }[feature_kind]
    if values.shape != (len(frame), expected):
        raise ValueError(f"feature shape {values.shape}, expected {(len(frame), expected)}")
    if not np.isfinite(values).all():
        raise ValueError("NaN/Inf in constructed features")
    return values


def fit_validation_masks(
    frame: pd.DataFrame,
    dataset: str,
    *,
    fraction: float = 0.8,
    seed: int = 20260803,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed problem-level split; MMLU is stratified by subject."""
    fit_ids: set[str] = set()
    grouping: Iterable[tuple[Any, pd.DataFrame]]
    if dataset == "mmlu":
        grouping = frame.groupby("subject", sort=True)
    else:
        grouping = [("all", frame)]
    for stratum, group in grouping:
        ids = sorted(group.problem_id.astype(str).unique())
        ordered = sorted(
            ids,
            key=lambda value: hashlib.sha256(
                f"{seed}:{stratum}:{value}".encode("utf-8")
            ).hexdigest(),
        )
        cut = max(1, min(len(ordered) - 1, int(math.floor(fraction * len(ordered)))))
        fit_ids.update(ordered[:cut])
    fit = frame.problem_id.astype(str).isin(fit_ids).to_numpy()
    validation = ~fit
    # A deliberately tiny multi-subject smoke split can contain one problem per
    # subject. Preserve subject stratification for formal MMLU, but fall back
    # to the same fixed problem-level hash when every stratum is a singleton.
    if not validation.any():
        ids = sorted(frame.problem_id.astype(str).unique())
        ordered = sorted(
            ids,
            key=lambda value: hashlib.sha256(
                f"{seed}:global_fallback:{value}".encode("utf-8")
            ).hexdigest(),
        )
        cut = max(1, min(len(ordered) - 1, int(math.floor(fraction * len(ordered)))))
        fit_ids = set(ordered[:cut])
        fit = frame.problem_id.astype(str).isin(fit_ids).to_numpy()
        validation = ~fit
    if not fit.any() or not validation.any():
        raise ValueError("fit/validation problem split is empty")
    return fit, validation


def problem_batches(
    frame: pd.DataFrame,
    mask: np.ndarray,
    batch_problems: int,
    rng: random.Random,
):
    groups = [
        group.index.to_numpy(dtype=np.int64)
        for _, group in frame.loc[mask].groupby("problem_id", sort=False)
    ]
    rng.shuffle(groups)
    for left in range(0, len(groups), batch_problems):
        selected = groups[left:left + batch_problems]
        positions = np.concatenate(selected)
        offsets = np.cumsum([0] + [len(value) for value in selected])
        yield positions, offsets


def correction_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    remaining: torch.Tensor,
    offsets: np.ndarray,
    *,
    beta: float = 0.5,
    trajectory: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L_point plus optional beta-soft-min protection over each W->C trajectory."""
    point_terms = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = torch.where(
        target > 0.5,
        torch.full_like(point_terms, 1.5),
        1.0 + remaining,
    )
    point = (point_terms * weights).mean()
    trajectory_terms: list[torch.Tensor] = []
    if trajectory:
        for start, end in zip(offsets[:-1], offsets[1:]):
            dangerous = target[start:end] > 0.5
            if dangerous.any():
                values = logits[start:end][dangerous]
                soft_minimum = -beta * torch.logsumexp(-values / beta, dim=0)
                trajectory_terms.append(F.softplus(-soft_minimum))
    trajectory_value = (
        torch.stack(trajectory_terms).mean()
        if trajectory_terms
        else logits.sum() * 0.0
    )
    return point + trajectory_value, point, trajectory_value


def binary_point_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=positive_weight
    )


@torch.no_grad()
def predict_scores(
    model: FinalPaperProbe,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(features), batch_size):
        values = torch.from_numpy(features[start:start + batch_size]).to(device)
        output.append(torch.sigmoid(model(values)).float().cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def safe_ap_auc(truth: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    truth = truth.astype(np.int64)
    if len(np.unique(truth)) < 2:
        return 0.0, 0.5
    return (
        float(average_precision_score(truth, scores)),
        float(roc_auc_score(truth, scores)),
    )


def transition_name(current_success: bool, dense_success: bool) -> str:
    return ("C" if current_success else "W") + "_to_" + (
        "C" if dense_success else "W"
    )


def _summarize_policy_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if not n:
        raise ValueError("cannot summarize empty policy records")
    stopped = np.asarray([not row["fallback"] for row in records], dtype=bool)
    method_success = np.asarray([row["method_success"] for row in records], dtype=float)
    dense_success = np.asarray([row["dense_success"] for row in records], dtype=float)
    method_tokens = np.asarray([row["method_tokens"] for row in records], dtype=float)
    dense_tokens = np.asarray([row["dense_tokens"] for row in records], dtype=float)
    method_wall = np.asarray([row["replay_wall_ms"] for row in records], dtype=float)
    dense_wall = np.asarray([row["dense_wall_ms"] for row in records], dtype=float)
    counts = {name: 0 for name in ("W_to_C", "C_to_W", "W_to_W", "C_to_C")}
    for row in records:
        if row["transition"] in counts:
            counts[row["transition"]] += 1
    lost = counts["W_to_C"]
    return {
        "problems": n,
        "counts": counts,
        "fallback": int((~stopped).sum()),
        "fallback_rate": float((~stopped).mean()),
        "coverage": float(stopped.mean()),
        "accuracy": float(method_success.mean()),
        "dense_accuracy": float(dense_success.mean()),
        "accuracy_drop_pp": float(100.0 * (dense_success.mean() - method_success.mean())),
        "lost_correct_count": int(lost),
        "lost_correct_rate": float(lost / n),
        "mean_reasoning_and_answer_tokens": float(method_tokens.mean()),
        "mean_dense_reasoning_tokens": float(dense_tokens.mean()),
        "token_reduction": float(1.0 - method_tokens.mean() / dense_tokens.mean()),
        "mean_replay_wall_ms": float(method_wall.mean()),
        "mean_dense_wall_ms": float(dense_wall.mean()),
        "replay_wall_reduction": float(1.0 - method_wall.mean() / dense_wall.mean()),
        "p95_replay_wall_ms": float(np.percentile(method_wall, 95)),
        "p95_dense_wall_ms": float(np.percentile(dense_wall, 95)),
        "p95_replay_wall_reduction": float(
            1.0 - np.percentile(method_wall, 95) / np.percentile(dense_wall, 95)
        ),
    }


def simulate_policy(
    frame: pd.DataFrame,
    scores: np.ndarray,
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
        eligible = (
            ordered[ordered.score >= threshold]
            if direction == "high"
            else ordered[ordered.score <= threshold]
        )
        fallback = eligible.empty
        if fallback:
            current_success = bool(first.dense_success)
            method_tokens = int(first.dense_tokens)
            replay_wall_ms = float(first.dense_wall_ms)
            checkpoint = None
            transition = "fallback"
            prediction = first.dense_prediction
        else:
            chosen = eligible.iloc[0]
            current_success = bool(chosen.current_success)
            method_tokens = min(
                int(chosen.dense_tokens),
                int(chosen.checkpoint) + int(chosen.branch_tokens),
            )
            replay_wall_ms = float(
                chosen.dense_prefill_cuda_ms
                + chosen.prefix_decode_cuda_ms
                + chosen.branch_wall_ms
            )
            checkpoint = int(chosen.checkpoint)
            transition = transition_name(current_success, bool(chosen.dense_success))
            prediction = chosen.current_prediction
        records.append(
            {
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
                "dense_wall_ms": float(first.dense_wall_ms),
            }
        )
    seen = {row["problem_id"] for row in records}
    for base in fallback_records or []:
        if base["problem_id"] in seen:
            raise ValueError(f"duplicate fallback problem {base['problem_id']}")
        records.append(
            {
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
            }
        )
        seen.add(base["problem_id"])
    summary = _summarize_policy_records(records)
    summary["threshold"] = float(threshold)
    if include_records:
        summary["records"] = records
    return summary


def threshold_grid(
    scores: np.ndarray,
    direction: str,
    grid_size: int = 101,
) -> list[tuple[float, bool]]:
    """Return a finite calibration grid including exactly one disabled policy."""
    if grid_size < 2:
        raise ValueError("threshold grid must contain at least two values")
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, grid_size - 1))
    values = [float(value) for value in np.unique(quantiles)]
    disabled = (
        float(np.nextafter(np.max(scores), np.inf))
        if direction == "high"
        else float(np.nextafter(np.min(scores), -np.inf))
    )
    return [(disabled, True)] + [(value, False) for value in values]


def binomial_upper_simultaneous(
    events: int,
    trials: int,
    *,
    confidence: float,
    grid_size: int,
) -> float:
    """One-sided Clopper-Pearson bound with finite-grid Bonferroni correction."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if events < 0 or events > trials:
        raise ValueError("invalid event count")
    if events == trials:
        return 1.0
    tail = (1.0 - confidence) / float(grid_size)
    return float(beta_distribution.ppf(1.0 - tail, events + 1, trials - events))


def calibration_curve(
    frame: pd.DataFrame,
    scores: np.ndarray,
    direction: str,
    *,
    grid_size: int = 101,
    confidence: float = 0.95,
    fallback_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grid = threshold_grid(scores, direction, grid_size)
    curve = []
    for index, (threshold, disabled) in enumerate(grid):
        row = simulate_policy(
            frame,
            scores,
            direction,
            threshold,
            fallback_records=fallback_records,
        )
        row["grid_index"] = index
        row["disabled"] = bool(disabled)
        row["simultaneous_upper_95"] = (
            0.0
            if disabled
            else binomial_upper_simultaneous(
                row["lost_correct_count"],
                row["problems"],
                confidence=confidence,
                grid_size=grid_size,
            )
        )
        curve.append(row)
    return curve


def _fastest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (
            row["mean_replay_wall_ms"],
            row["lost_correct_count"],
            -row["coverage"],
            row["threshold"],
        ),
    )


def select_empirical_budget(
    curve: list[dict[str, Any]],
    percent: float,
    calibration_problems: int,
) -> dict[str, Any]:
    allowed = int(math.floor((percent / 100.0) * calibration_problems))
    feasible = [row for row in curve if row["lost_correct_count"] <= allowed]
    selected = dict(_fastest(feasible))
    selected["budget_percent"] = float(percent)
    selected["allowed_lost_correct"] = allowed
    return selected


def select_formal_bound(
    curve: list[dict[str, Any]],
    alpha: float,
) -> dict[str, Any]:
    feasible = [
        row
        for row in curve
        if (not row["disabled"]) and row["simultaneous_upper_95"] <= alpha
    ]
    if feasible:
        selected = dict(_fastest(feasible))
        selected["dense_fallback"] = False
    else:
        selected = dict(next(row for row in curve if row["disabled"]))
        selected["dense_fallback"] = True
    selected["alpha"] = float(alpha)
    return selected


def select_coverage(
    curve: list[dict[str, Any]],
    target: float,
) -> dict[str, Any]:
    selected = min(
        curve,
        key=lambda row: (
            abs(row["coverage"] - target),
            row["mean_replay_wall_ms"],
            row["lost_correct_count"],
        ),
    )
    result = dict(selected)
    result["coverage_target"] = float(target)
    return result


def calibrate_policies(
    frame: pd.DataFrame,
    scores: np.ndarray,
    direction: str,
    *,
    grid_size: int,
    confidence: float,
    empirical_percent: list[float],
    formal_alpha: list[float],
    coverage_targets: list[float],
    fallback_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    curve = calibration_curve(
        frame,
        scores,
        direction,
        grid_size=grid_size,
        confidence=confidence,
        fallback_records=fallback_records,
    )
    n = int(frame.problem_id.nunique()) + len(fallback_records or [])
    return {
        "grid_declared_size": int(grid_size),
        "grid_realized_size": len(curve),
        "curve": curve,
        "empirical": {
            str(value): select_empirical_budget(curve, value, n)
            for value in empirical_percent
        },
        "formal": {
            str(value): select_formal_bound(curve, value)
            for value in formal_alpha
        },
        "coverage": {
            str(int(round(100 * value))): select_coverage(curve, value)
            for value in coverage_targets
        },
    }



def simulate_fixed_budget(
    frame: pd.DataFrame,
    budget: int,
    *,
    include_records: bool = False,
) -> dict[str, Any]:
    """Replay a fixed-token budget using the same cached forced-answer branches."""
    records: list[dict[str, Any]] = []
    for problem_id, group in frame.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        first = ordered.iloc[0]
        exact = ordered[ordered.checkpoint == int(budget)]
        fallback = exact.empty
        if fallback:
            current_success = bool(first.dense_success)
            method_tokens = int(first.dense_tokens)
            replay_wall_ms = float(first.dense_wall_ms)
            checkpoint = None
            transition = "fallback"
            prediction = first.dense_prediction
        else:
            chosen = exact.iloc[0]
            current_success = bool(chosen.current_success)
            method_tokens = min(
                int(chosen.dense_tokens),
                int(chosen.checkpoint) + int(chosen.branch_tokens),
            )
            replay_wall_ms = float(
                chosen.dense_prefill_cuda_ms
                + chosen.prefix_decode_cuda_ms
                + chosen.branch_wall_ms
            )
            checkpoint = int(chosen.checkpoint)
            transition = transition_name(current_success, bool(chosen.dense_success))
            prediction = chosen.current_prediction
        records.append(
            {
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
                "dense_wall_ms": float(first.dense_wall_ms),
            }
        )
    summary = _summarize_policy_records(records)
    summary["fixed_budget"] = int(budget)
    if include_records:
        summary["records"] = records
    return summary


def summarize_policy_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Public summary helper for fixed/direct baseline and bootstrap scripts."""
    return _summarize_policy_records(records)



def build_online_feature(
    current: np.ndarray,
    previous: np.ndarray | None,
    checkpoint: int,
    previous_checkpoint: int,
    entropy_tail8: float,
    feature_kind: str = "full",
) -> np.ndarray:
    """Construct one deployable feature vector with the exact offline ordering."""
    current = np.asarray(current, dtype=np.float32).reshape(1, -1)
    if current.shape[1] != 2560:
        raise ValueError(f"online hidden width must be 2560, got {current.shape[1]}")
    prior = (
        np.zeros_like(current)
        if previous is None
        else np.asarray(previous, dtype=np.float32).reshape(1, -1)
    )
    delta = current if previous is not None else np.zeros_like(current)
    if previous is not None:
        delta = current - prior
    delta_norm = np.linalg.norm(delta, axis=1, keepdims=True).astype(np.float32)
    current_norm = np.linalg.norm(current, axis=1, keepdims=True).astype(np.float32)
    cosine = (
        np.sum(current * delta, axis=1, keepdims=True)
        / np.maximum(current_norm * delta_norm, 1e-6)
    ).astype(np.float32)
    position = np.asarray([[float(checkpoint)]], dtype=np.float32)
    log_position = np.log1p(position).astype(np.float32)
    delta_t = np.asarray(
        [[float(checkpoint - previous_checkpoint)]], dtype=np.float32
    )
    entropy = np.asarray([[float(entropy_tail8)]], dtype=np.float32)
    if feature_kind == "h_only":
        parts = [current]
    elif feature_kind == "h_delta":
        parts = [current, delta]
    elif feature_kind == "full":
        parts = [
            current,
            delta,
            position,
            log_position,
            delta_t,
            entropy,
            delta_norm,
            cosine,
        ]
    elif feature_kind == "full_no_entropy":
        parts = [
            current,
            delta,
            position,
            log_position,
            delta_t,
            delta_norm,
            cosine,
        ]
    elif feature_kind == "full_no_position":
        parts = [current, delta, entropy, delta_norm, cosine]
    else:
        raise ValueError(feature_kind)
    result = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("NaN/Inf online feature")
    return result

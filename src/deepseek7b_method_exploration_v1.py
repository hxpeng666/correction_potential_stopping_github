"""Core components for the DeepSeek-7B 13K method-exploration study.

The module deliberately separates three experimental axes:

* representation/readout construction;
* checkpoint-level risk loss;
* trajectory-level first-hit protection.

It contains no dataset splitting or threshold selection code.  That separation is
intentional: costs belong to calibration, while this module learns a risk score.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


READOUT_COLUMNS = (
    "top1_probability",
    "top1_top2_margin",
    "top20_entropy_renormalized",
    "numeric_probability_mass_top20",
    "top20_probability_mass",
    "argmax_same_previous",
    "argmax_run_length",
)

REPRESENTATION_KINDS = (
    "last",
    "last4_mean",
    "paragraph_mean",
    "prefix_mean",
)

FEATURE_KINDS = (
    "hidden_only",
    "hidden_scalars",
    "scalars_only",
    "one_step_only",
    "scalars_one_step",
    "hidden_scalars_one_step",
    "pca_hidden",
    "pca_hidden_scalars",
    "pca_hidden_scalars_one_step",
)

POINT_LOSSES = (
    "legacy_weighted",
    "checkpoint_proper",
    "problem_balanced_proper",
)

TRAJECTORY_AGGREGATIONS = (
    "none",
    "hard_min",
    "normalized_softmin",
    "bottomk_mean",
    "lower_tail_cvar",
)

TRAJECTORY_SCOPES = ("all_dangerous", "reachability_earliest_safe")


class RiskProbe(nn.Module):
    """Configurable scalar probe whose capacity follows the input dimension."""

    def __init__(self, width: int, architecture: str = "standard") -> None:
        super().__init__()
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        self.width = int(width)
        self.architecture = architecture
        if architecture == "linear":
            self.network = nn.Linear(width, 1)
            self.layer_widths = [width, 1]
        elif architecture == "compact":
            hidden = min(128, max(32, width // 2))
            bottleneck = min(32, max(16, hidden // 4))
            self.network = nn.Sequential(
                nn.Linear(width, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(hidden, bottleneck),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(bottleneck, 1),
            )
            self.layer_widths = [width, hidden, bottleneck, 1]
        elif architecture == "small":
            hidden = min(192, max(64, width))
            bottleneck = min(64, max(24, hidden // 3))
            self.network = nn.Sequential(
                nn.Linear(width, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden, bottleneck),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(bottleneck, 1),
            )
            self.layer_widths = [width, hidden, bottleneck, 1]
        elif architecture == "standard":
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
            self.layer_widths = [width, 384, 96, 1]
        else:
            raise ValueError(f"unknown architecture: {architecture}")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)[:, 0]

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _dynamics_scalars(frame: pd.DataFrame, current: np.ndarray) -> np.ndarray:
    """Return t, log(1+t), delta-t, entropy, delta norm, and cosine."""
    if current.ndim != 2 or len(frame) != len(current):
        raise ValueError("frame/hidden shape mismatch")
    delta = np.zeros_like(current, dtype=np.float32)
    delta_t = np.zeros((len(frame), 1), dtype=np.float32)
    for _problem_id, group in frame.groupby("problem_id", sort=False):
        positions = group.sort_values("checkpoint").index.to_numpy(dtype=np.int64)
        checkpoints = frame.loc[positions, "checkpoint"].to_numpy(dtype=np.float32)
        if len(positions) > 1:
            delta[positions[1:]] = current[positions[1:]] - current[positions[:-1]]
        delta_t[positions, 0] = np.diff(np.concatenate(([0.0], checkpoints)))
    checkpoint = frame.checkpoint.to_numpy(dtype=np.float32)[:, None]
    log_checkpoint = np.log1p(checkpoint).astype(np.float32)
    entropy = frame.prefix_mean_entropy_tail8.to_numpy(dtype=np.float32)[:, None]
    delta_norm = np.linalg.norm(delta, axis=1, keepdims=True).astype(np.float32)
    current_norm = np.linalg.norm(current, axis=1, keepdims=True).astype(np.float32)
    cosine = (
        np.sum(current * delta, axis=1, keepdims=True)
        / np.maximum(current_norm * delta_norm, 1e-6)
    ).astype(np.float32)
    result = np.concatenate(
        [checkpoint, log_checkpoint, delta_t, entropy, delta_norm, cosine], axis=1
    ).astype(np.float32, copy=False)
    if result.shape != (len(frame), 6) or not np.isfinite(result).all():
        raise ValueError("invalid dynamics scalars")
    return result


def load_auxiliary_split(
    directory: Path,
    frame: pd.DataFrame,
    representation_kind: str,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Load pooled hidden/readout features and align them by problem/checkpoint."""
    if representation_kind not in REPRESENTATION_KINDS:
        raise ValueError(representation_kind)
    paths = sorted(directory.glob("sample_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no auxiliary feature artifacts under {directory}")
    hidden_rows: list[np.ndarray] = []
    readout_rows: list[np.ndarray] = []
    keys: list[tuple[str, int]] = []
    fingerprints: set[str] = set()
    suffix_token_counts: set[int] = set()
    for path in paths:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("status") != "complete":
            raise ValueError(f"incomplete auxiliary artifact: {path}")
        problem_id = str(value["problem_id"])
        checkpoints = [int(item) for item in value["checkpoints"]]
        representations = value["representations"]
        hidden = representations[representation_kind]
        readout = value["one_step_readout"]
        if tuple(value["readout_columns"]) != READOUT_COLUMNS:
            raise ValueError(f"readout schema mismatch: {path}")
        if not torch.is_tensor(hidden) or not torch.is_tensor(readout):
            raise TypeError(f"non-tensor auxiliary fields: {path}")
        # Schema-v1 collectors originally materialized an empty Python list as
        # a rank-1 tensor for the rare Dense-fallback trajectory with zero legal
        # checkpoints.  It contributes no feature row, so normalize that legacy
        # representation without inventing data.  New collectors always write
        # the canonical [0, len(READOUT_COLUMNS)] tensor directly.
        if not checkpoints and readout.ndim == 1 and readout.numel() == 0:
            readout = readout.reshape(0, len(READOUT_COLUMNS))
        if hidden.ndim != 2 or readout.ndim != 2:
            raise ValueError(f"invalid auxiliary ranks: {path}")
        if readout.shape[1] != len(READOUT_COLUMNS):
            raise ValueError(f"invalid auxiliary readout width: {path}")
        if len(checkpoints) != len(hidden) or len(checkpoints) != len(readout):
            raise ValueError(f"auxiliary row mismatch: {path}")
        keys.extend((problem_id, checkpoint) for checkpoint in checkpoints)
        hidden_rows.append(hidden.float().numpy())
        readout_rows.append(readout.float().numpy())
        fingerprints.add(str(value.get("source_protocol_fingerprint")))
        suffix_token_counts.add(int(value["one_step_protocol"]["suffix_tokens"]))
    hidden_array = np.concatenate(hidden_rows, axis=0).astype(np.float32, copy=False)
    readout_array = np.concatenate(readout_rows, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(hidden_array).all() or not np.isfinite(readout_array).all():
        raise ValueError("NaN/Inf in auxiliary features")
    lookup = {key: index for index, key in enumerate(keys)}
    if len(lookup) != len(keys):
        raise ValueError(f"duplicate auxiliary keys under {directory}")
    desired = [
        (str(problem_id), int(checkpoint))
        for problem_id, checkpoint in zip(frame.problem_id, frame.checkpoint)
    ]
    missing = [key for key in desired if key not in lookup]
    if missing:
        raise ValueError(f"missing auxiliary keys under {directory}: {missing[:5]}")
    order = np.asarray([lookup[key] for key in desired], dtype=np.int64)
    return (
        hidden_array[order],
        readout_array[order],
        {
            "files": len(paths),
            "rows": len(order),
            "source_protocol_fingerprints": sorted(fingerprints),
            "suffix_token_counts": sorted(suffix_token_counts),
        },
    )


@dataclass
class FeatureTransform:
    feature_kind: str
    pca_components: np.ndarray | None
    pca_mean: np.ndarray | None
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    input_width: int
    pca_fit_rows: int

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_kind": self.feature_kind,
            "pca_components": self.pca_components,
            "pca_mean": self.pca_mean,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "input_width": self.input_width,
            "pca_fit_rows": self.pca_fit_rows,
        }

    @classmethod
    def from_state_dict(cls, value: dict[str, Any]) -> "FeatureTransform":
        return cls(**value)

    def _project_hidden(self, hidden: np.ndarray) -> np.ndarray:
        if self.pca_components is None:
            return hidden.astype(np.float32, copy=False)
        assert self.pca_mean is not None
        return ((hidden - self.pca_mean) @ self.pca_components.T).astype(
            np.float32, copy=False
        )

    def transform(
        self,
        frame: pd.DataFrame,
        hidden: np.ndarray,
        readout: np.ndarray | None,
    ) -> np.ndarray:
        scalars = _dynamics_scalars(frame, hidden)
        projected = self._project_hidden(hidden)
        raw = concatenate_feature_groups(
            self.feature_kind, projected, scalars, readout
        )
        if raw.shape[1] != self.input_width:
            raise ValueError(
                f"transformed width {raw.shape[1]} != fitted width {self.input_width}"
            )
        values = ((raw - self.scaler_mean) / self.scaler_scale).astype(
            np.float32, copy=False
        )
        if not np.isfinite(values).all():
            raise ValueError("NaN/Inf after feature transform")
        return values


def concatenate_feature_groups(
    feature_kind: str,
    hidden: np.ndarray,
    scalars: np.ndarray,
    readout: np.ndarray | None,
) -> np.ndarray:
    if feature_kind not in FEATURE_KINDS:
        raise ValueError(feature_kind)
    uses_readout = feature_kind in {
        "one_step_only",
        "scalars_one_step",
        "hidden_scalars_one_step",
        "pca_hidden_scalars_one_step",
    }
    if uses_readout and readout is None:
        raise ValueError(f"{feature_kind} requires one-step readout")
    if feature_kind in {"hidden_only", "pca_hidden"}:
        parts = [hidden]
    elif feature_kind in {"hidden_scalars", "pca_hidden_scalars"}:
        parts = [hidden, scalars]
    elif feature_kind == "scalars_only":
        parts = [scalars]
    elif feature_kind == "one_step_only":
        assert readout is not None
        parts = [readout]
    elif feature_kind == "scalars_one_step":
        assert readout is not None
        parts = [scalars, readout]
    else:
        assert readout is not None
        parts = [hidden, scalars, readout]
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def fit_feature_transform(
    frame: pd.DataFrame,
    hidden: np.ndarray,
    readout: np.ndarray | None,
    fit_mask: np.ndarray,
    *,
    feature_kind: str,
    pca_dim: int | None,
    pca_fit_max_rows: int = 20000,
    seed: int = 0,
) -> tuple[FeatureTransform, np.ndarray, dict[str, Any]]:
    if feature_kind.startswith("pca_") and (pca_dim is None or pca_dim <= 0):
        raise ValueError("PCA feature kind requires a positive pca_dim")
    if not feature_kind.startswith("pca_") and pca_dim is not None:
        raise ValueError("pca_dim is only valid for PCA feature kinds")
    fit_indices = np.flatnonzero(fit_mask)
    if not len(fit_indices):
        raise ValueError("empty feature-fit mask")
    pca_components = None
    pca_mean = None
    pca_fit_rows = 0
    projected = hidden.astype(np.float32, copy=False)
    explained = None
    if pca_dim is not None:
        if pca_dim > min(hidden.shape[1], len(fit_indices)):
            raise ValueError(f"PCA dimension {pca_dim} is infeasible")
        rng = np.random.default_rng(seed)
        if len(fit_indices) > pca_fit_max_rows:
            pca_indices = np.sort(
                rng.choice(fit_indices, size=pca_fit_max_rows, replace=False)
            )
        else:
            pca_indices = fit_indices
        pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed)
        pca.fit(hidden[pca_indices])
        pca_components = pca.components_.astype(np.float32)
        pca_mean = pca.mean_.astype(np.float32)
        projected = ((hidden - pca_mean) @ pca_components.T).astype(
            np.float32, copy=False
        )
        pca_fit_rows = len(pca_indices)
        explained = float(pca.explained_variance_ratio_.sum())
    scalars = _dynamics_scalars(frame, hidden)
    raw = concatenate_feature_groups(feature_kind, projected, scalars, readout)
    scaler = StandardScaler(copy=False)
    scaler.fit(raw[fit_mask])
    scale = scaler.scale_.astype(np.float32)
    scale[scale < 1e-8] = 1.0
    transform = FeatureTransform(
        feature_kind=feature_kind,
        pca_components=pca_components,
        pca_mean=pca_mean,
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scale,
        input_width=int(raw.shape[1]),
        pca_fit_rows=int(pca_fit_rows),
    )
    values = transform.transform(frame, hidden, readout)
    report = {
        "feature_kind": feature_kind,
        "input_width": int(values.shape[1]),
        "hidden_width": int(hidden.shape[1]),
        "readout_width": int(readout.shape[1]) if readout is not None else 0,
        "pca_dim": pca_dim,
        "pca_fit_rows": pca_fit_rows,
        "pca_explained_variance_ratio": explained,
        "scaler_fit_rows": int(fit_mask.sum()),
    }
    return transform, values, report


def problem_groups(frame: pd.DataFrame, mask: np.ndarray) -> list[np.ndarray]:
    return [
        group.index.to_numpy(dtype=np.int64)
        for _problem_id, group in frame.loc[mask].groupby("problem_id", sort=False)
    ]


def problem_batches(
    frame: pd.DataFrame,
    mask: np.ndarray,
    target: np.ndarray,
    batch_problems: int,
    rng: random.Random,
    *,
    stratified: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    groups = problem_groups(frame, mask)
    if stratified:
        positive = [group for group in groups if bool((target[group] > 0.5).any())]
        negative = [group for group in groups if not bool((target[group] > 0.5).any())]
        rng.shuffle(positive)
        rng.shuffle(negative)
        groups = []
        while positive or negative:
            if positive:
                groups.append(positive.pop())
            if negative:
                groups.append(negative.pop())
    else:
        rng.shuffle(groups)
    for left in range(0, len(groups), batch_problems):
        selected = groups[left : left + batch_problems]
        positions = np.concatenate(selected)
        offsets = np.cumsum([0] + [len(group) for group in selected])
        yield positions, offsets


def point_risk_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    remaining: torch.Tensor,
    offsets: np.ndarray,
    mode: str,
) -> torch.Tensor:
    if mode not in POINT_LOSSES:
        raise ValueError(mode)
    terms = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if mode == "legacy_weighted":
        weights = torch.where(
            target > 0.5,
            torch.full_like(terms, 1.5),
            1.0 + remaining,
        )
        return (terms * weights).mean()
    if mode == "checkpoint_proper":
        return terms.mean()
    per_problem = [
        terms[start:end].mean() for start, end in zip(offsets[:-1], offsets[1:])
    ]
    return torch.stack(per_problem).mean()


def lower_tail_location(
    values: torch.Tensor,
    aggregation: str,
    *,
    beta: float,
    rho: float,
) -> torch.Tensor:
    """Aggregate low dangerous logits; larger output means a safer protected prefix."""
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("lower-tail aggregation requires a non-empty vector")
    if aggregation not in TRAJECTORY_AGGREGATIONS or aggregation == "none":
        raise ValueError(aggregation)
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0,1], got {rho}")
    if aggregation == "hard_min":
        return values.min()
    if aggregation == "normalized_softmin":
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        return -beta * (
            torch.logsumexp(-values / beta, dim=0) - math.log(values.numel())
        )
    ordered = torch.sort(values).values
    mass = rho * values.numel()
    if aggregation == "bottomk_mean":
        count = max(1, int(math.ceil(mass)))
        return ordered[:count].mean()
    # Exact empirical lower-tail CVaR with fractional mass at the boundary.
    full = int(math.floor(mass + 1e-12))
    fraction = mass - full
    if full >= values.numel():
        return ordered.mean()
    numerator = ordered[:full].sum() if full else ordered.sum() * 0.0
    if fraction > 1e-12:
        numerator = numerator + fraction * ordered[full]
    elif full == 0:
        numerator = ordered[0] * mass
    return numerator / mass


@dataclass
class LossBreakdown:
    total: torch.Tensor
    point: torch.Tensor
    protect: torch.Tensor
    separation: torch.Tensor
    protected_trajectories: int
    separated_trajectories: int


def correction_objective(
    logits: torch.Tensor,
    target: torch.Tensor,
    current_success: torch.Tensor,
    dense_success: torch.Tensor,
    remaining: torch.Tensor,
    offsets: np.ndarray,
    *,
    point_mode: str,
    trajectory_scope: str,
    aggregation: str,
    beta: float,
    rho: float,
    lambda_protect: float,
    lambda_separation: float,
    gamma: float,
) -> LossBreakdown:
    if trajectory_scope not in TRAJECTORY_SCOPES:
        raise ValueError(trajectory_scope)
    if lambda_protect < 0 or lambda_separation < 0:
        raise ValueError("trajectory coefficients must be non-negative")
    point = point_risk_loss(logits, target, remaining, offsets, point_mode)
    protect_terms: list[torch.Tensor] = []
    separation_terms: list[torch.Tensor] = []
    if aggregation != "none" and (lambda_protect > 0 or lambda_separation > 0):
        for start, end in zip(offsets[:-1], offsets[1:]):
            local_logits = logits[start:end]
            local_target = target[start:end] > 0.5
            local_current = current_success[start:end] > 0.5
            local_dense = dense_success[start:end] > 0.5
            if not bool(local_dense[0]):
                continue
            safe_index: int | None = None
            if trajectory_scope == "all_dangerous":
                dangerous = local_target
            else:
                safe_positions = torch.nonzero(local_current, as_tuple=False).flatten()
                if safe_positions.numel():
                    safe_index = int(safe_positions[0].item())
                    prefix = torch.arange(
                        len(local_logits), device=logits.device
                    ) < safe_index
                    dangerous = prefix & (~local_current)
                else:
                    # Dense is correct but no intermediate checkpoint is safe:
                    # protect all observed pre-end wrong checkpoints.
                    dangerous = ~local_current
            if not bool(dangerous.any()):
                continue
            location = lower_tail_location(
                local_logits[dangerous], aggregation, beta=beta, rho=rho
            )
            protect_terms.append(F.softplus(-location))
            if (
                trajectory_scope == "reachability_earliest_safe"
                and safe_index is not None
                and lambda_separation > 0
            ):
                safe_logit = local_logits[safe_index]
                separation_terms.append(
                    F.softplus(gamma - (location - safe_logit))
                )
    zero = logits.sum() * 0.0
    protect = torch.stack(protect_terms).mean() if protect_terms else zero
    separation = (
        torch.stack(separation_terms).mean() if separation_terms else zero
    )
    total = point + lambda_protect * protect + lambda_separation * separation
    return LossBreakdown(
        total=total,
        point=point,
        protect=protect,
        separation=separation,
        protected_trajectories=len(protect_terms),
        separated_trajectories=len(separation_terms),
    )


@torch.no_grad()
def predict_scores(
    model: RiskProbe,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    values = []
    for start in range(0, len(features), batch_size):
        tensor = torch.from_numpy(features[start : start + batch_size]).to(device)
        values.append(torch.sigmoid(model(tensor)).float().cpu().numpy())
    return np.concatenate(values).astype(np.float64)

"""Core labels, cached-data loading, hidden-delta features, and probe losses."""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def _after_last_switch(values: list[str]) -> list[bool]:
    switch = max(
        (index for index in range(len(values) - 1) if values[index] != values[index + 1]),
        default=-1,
    )
    return [index > switch for index in range(len(values) - 1)]


def add_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["last_switch_raw"] = False
    for _, group in frame.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        current = ordered.current_prediction.fillna("<MISSING>").astype(str).tolist()
        dense = (
            str(ordered.iloc[0].dense_prediction)
            if pd.notna(ordered.iloc[0].dense_prediction)
            else "<MISSING>"
        )
        frame.loc[ordered.index, "last_switch_raw"] = _after_last_switch(current + [dense])
    return frame


def load_split(
    directory: Path, dense_reference_directory: Path | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    rows, hidden = [], []
    files = sorted(directory.glob("sample_*.pt"))
    if not files:
        raise FileNotFoundError(f"no sample checkpoints in {directory}")
    for path in files:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if len(artifact["rows"]) != len(artifact["hidden"]):
            raise ValueError(f"row/vector mismatch in {path}")
        rows.extend(artifact["rows"])
        hidden.extend(value.flatten().numpy() for value in artifact["hidden"].float())
    frame = pd.DataFrame(rows).sort_values(["problem_id", "checkpoint"]).reset_index(drop=True)
    vectors = np.stack(hidden).astype(np.float32)
    if len(frame) != len(vectors):
        raise ValueError("global row/vector mismatch")
    if dense_reference_directory is not None:
        references = {}
        for path in sorted(dense_reference_directory.glob("sample_*.pt")):
            value = torch.load(path, map_location="cpu", weights_only=False)
            references[int(value["problem_id"])] = value
        missing = sorted(set(frame.problem_id.astype(int)) - set(references))
        if missing:
            raise FileNotFoundError(
                f"missing dense references for {missing[:10]} ({len(missing)} total)"
            )
        for column in (
            "dense_prediction",
            "dense_success",
            "dense_tokens",
            "dense_wall_ms",
            "dense_prefill_cuda_ms",
        ):
            frame[column] = frame.problem_id.map(lambda pid: references[int(pid)][column])
        frame["correction"] = (
            (~frame.current_success.astype(bool)) & frame.dense_success.astype(bool)
        )
        frame["damage"] = (
            frame.current_success.astype(bool) & (~frame.dense_success.astype(bool))
        )
    return frame, vectors


def transition(row: pd.Series) -> str:
    return ("C" if bool(row.current_success) else "W") + "_to_" + (
        "C" if bool(row.dense_success) else "W"
    )


def _previous_delta(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    for _, group in frame.groupby("problem_id", sort=False):
        positions = group.sort_values("checkpoint").index.to_numpy()
        if len(positions) > 1:
            result[positions[1:]] = values[positions[1:]] - values[positions[:-1]]
    return result


def hidden_delta_feature(frame: pd.DataFrame, hidden: np.ndarray) -> np.ndarray:
    """Build the deployable 5126-D layer-20 hidden-dynamics feature."""
    current = hidden[:, 1, :]
    delta = _previous_delta(frame, current)
    delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
    current_norm = np.linalg.norm(current, axis=1, keepdims=True)
    cosine_delta = np.sum(current * delta, axis=1, keepdims=True) / np.maximum(
        current_norm * delta_norm, 1e-6
    )
    metadata = np.zeros((len(frame), 4), dtype=np.float32)
    for _, group in frame.groupby("problem_id", sort=False):
        previous_checkpoint = 0.0
        for position in group.sort_values("checkpoint").index:
            checkpoint = float(frame.loc[position, "checkpoint"])
            metadata[position] = np.asarray(
                [
                    checkpoint,
                    math.log1p(checkpoint),
                    checkpoint - previous_checkpoint,
                    float(frame.loc[position, "prefix_mean_entropy_tail8"]),
                ],
                dtype=np.float32,
            )
            previous_checkpoint = checkpoint
    return np.concatenate(
        [current, delta, metadata, delta_norm, cosine_delta], axis=1
    ).astype(np.float32, copy=False)


class Probe(nn.Module):
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
        return self.network(values)


def category_labels(frame: pd.DataFrame) -> np.ndarray:
    current = frame.current_success.astype(bool).to_numpy()
    dense = frame.dense_success.astype(bool).to_numpy()
    result = np.full(len(frame), 3, dtype=np.int64)
    result[(~current) & dense] = 0
    result[current & (~dense)] = 1
    result[(~current) & (~dense)] = 2
    return result


def split_problem_ids(
    frame: pd.DataFrame, fraction: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    ids = frame.problem_id.drop_duplicates().to_numpy()
    shuffled = np.random.default_rng(0).permutation(ids)
    cut = int(fraction * len(shuffled))
    fit_ids = set(shuffled[:cut].tolist())
    fit = frame.problem_id.isin(fit_ids).to_numpy()
    return fit, ~fit


def problem_batches(
    frame: pd.DataFrame,
    mask: np.ndarray,
    batch_problems: int,
    rng: random.Random,
):
    groups = [
        group.index.to_numpy()
        for _, group in frame.loc[mask].groupby("problem_id", sort=False)
    ]
    rng.shuffle(groups)
    for start in range(0, len(groups), batch_problems):
        selected = groups[start:start + batch_problems]
        positions = np.concatenate(selected)
        offsets = np.cumsum([0] + [len(value) for value in selected])
        yield positions, offsets


def _softmin(values: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    return -temperature * torch.logsumexp(-values / temperature, dim=0)


def correction_loss(
    logits: torch.Tensor,
    categories: torch.Tensor,
    remaining: torch.Tensor,
    offsets: np.ndarray,
) -> torch.Tensor:
    danger = (categories == 0).float()
    point = F.binary_cross_entropy_with_logits(logits[:, 0], danger, reduction="none")
    point = point * torch.where(danger > 0, torch.full_like(point, 1.5), 1.0 + remaining)
    loss = point.mean()
    trajectory_terms = []
    for left, right in zip(offsets[:-1], offsets[1:]):
        local = danger[left:right].bool()
        if local.any():
            weakest = _softmin(logits[left:right, 0][local], temperature=0.5)
            trajectory_terms.append(F.softplus(-weakest))
    if trajectory_terms:
        loss = loss + torch.stack(trajectory_terms).mean()
    return loss

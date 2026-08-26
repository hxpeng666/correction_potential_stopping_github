#!/usr/bin/env python3
"""Train paragraph readout and low-rank dynamics probes on frozen Qwen3-4B traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from src.final_paper_inference import atomic_torch_save
from src.dynamic_optimal_stopping_v1 import (
    DYNAMIC_FEATURE_KINDS,
    build_dynamic_features,
)
from src.legacy_empirical_probe_normalized_v1 import (
    FEATURE_KINDS,
    FinalPaperProbe,
    METHODS,
    SCHEDULES,
    binary_point_loss,
    calibrate_policies,
    correction_loss,
    fit_validation_masks,
    fit_validation_problem_ids,
    load_checkpoint_split,
    method_direction,
    problem_batches,
    safe_ap_auc,
    select_empirical_budget,
    simulate_policy,
    target_values,
    threshold_grid,
)
from src.final_paper_protocol import canonical_fingerprint
from src.utils import atomic_json, load_yaml, seed_everything


REPRESENTATIONS = (
    "boundary",
    "preboundary_nonblank",
    "last4_noncontrol_mean",
    "last8_noncontrol_mean",
    "sentence_mean",
    "paragraph_mean",
    "last8_noncontrol_ln_mean",
    "paragraph_ln_mean",
)
LOWRANK_FEATURE_KINDS = ("lowrank_u", "lowrank_u_delta", "lowrank_full")
LOWRANK_SCALAR_NAMES = (
    "t_over_tmax",
    "log_t_over_log_tmax",
    "delta_t_over_tmax",
    "entropy",
    "delta_entropy",
    "sampling_pmax",
    "sampling_probability_gap",
)


def artifact_manifest(root: Path) -> dict[str, Any]:
    packed_manifest = root / "PACKED_MANIFEST.json"
    if packed_manifest.is_file():
        payload = json.loads(packed_manifest.read_text(encoding="utf-8"))
        return {
            "root": str(root.resolve()),
            "schema": payload["schema"],
            "manifest_fingerprint": payload["manifest_fingerprint"],
            "representation_names": payload["representation_names"],
            "splits": payload["splits"],
        }
    paths = sorted(root.glob("*/sample_*.pt"))
    digest = hashlib.sha256()
    view_fingerprints: set[str] = set()
    for path in paths:
        relative = path.relative_to(root)
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        fingerprint = str(artifact.get("primary_replay_view_fingerprint"))
        view_fingerprints.add(fingerprint)
        digest.update(
            (
                f"{relative}:{path.stat().st_size}:{artifact.get('problem_id')}:"
                f"{artifact.get('dataset')}:{artifact.get('split')}:{artifact.get('dtype')}:"
                f"{artifact.get('protocol_fingerprint')}:{fingerprint}:"
                f"{artifact.get('token_pooling_fingerprint')}\n"
            ).encode("utf-8")
        )
    return {
        "root": str(root.resolve()),
        "files": len(paths),
        "artifact_identity_fingerprint": digest.hexdigest(),
        "primary_replay_view_fingerprints": sorted(view_fingerprints),
    }


def strip_records(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = dict(payload)
    records = summary.pop("records")
    return summary, records


def global_seed(config: dict[str, Any]) -> int:
    value = config["seed"]
    return int(value["global"] if isinstance(value, dict) else value)


def calibration_value(config: dict[str, Any], legacy: str, primary: str):
    calibration = config["calibration"]
    if legacy in calibration:
        return calibration[legacy]
    return calibration[primary]


def representation_names(root: Path) -> list[str]:
    packed_manifest = root / "PACKED_MANIFEST.json"
    if packed_manifest.is_file():
        payload = json.loads(packed_manifest.read_text(encoding="utf-8"))
        names = [str(value) for value in payload["representation_names"]]
        if names != list(REPRESENTATIONS):
            raise ValueError(f"unexpected packed representation schema: {names}")
        return names
    path = next(iter(sorted((root / "probe_train").glob("sample_*.pt"))), None)
    if path is None:
        raise FileNotFoundError(root / "probe_train")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    names = [str(value) for value in artifact.get("representation_names", [])]
    if names != list(REPRESENTATIONS):
        raise ValueError(f"unexpected representation schema in {path}: {names}")
    return names


def load_representation_split(
    root: Path,
    split: str,
    schedule: str,
    representation: str,
) -> tuple[Any, np.ndarray, list[int], list[dict[str, Any]], np.ndarray | None]:
    packed = root / representation / f"{split}.pt"
    if packed.is_file():
        payload = torch.load(packed, map_location="cpu", weights_only=False)
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete packed split: {packed}")
        if payload.get("representation") != representation:
            raise ValueError(f"packed representation mismatch: {packed}")
        frame = payload["frame"].copy()
        hidden = payload["hidden"].float().numpy()
        prompt = payload["prompt_hidden_ln"].float().numpy()
        if len(frame) != len(hidden) or len(frame) != len(prompt):
            raise ValueError(f"packed row mismatch: {packed}")
        return frame, hidden, [int(value) for value in payload["capture_layers"]], list(
            payload["fallbacks"]
        ), prompt
    frame, enriched_hidden, layers, fallbacks = load_checkpoint_split(
        root / split, schedule
    )
    names = representation_names(root)
    hidden = select_representation(enriched_hidden, names.index(representation))
    return frame, hidden, layers, fallbacks, None


def select_representation(hidden: np.ndarray, index: int) -> np.ndarray:
    if hidden.ndim != 3 or hidden.shape[1] != len(REPRESENTATIONS):
        raise ValueError(f"invalid enriched hidden shape: {hidden.shape}")
    return hidden[:, index:index + 1, :].astype(np.float32, copy=False)


def prompt_hidden_for_frame(directory: Path, frame: pd.DataFrame) -> np.ndarray:
    by_problem: dict[str, np.ndarray] = {}
    for path in sorted(directory.glob("sample_*.pt")):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        problem_id = str(artifact["problem_id"])
        value = artifact.get("prompt_hidden_ln")
        if not torch.is_tensor(value) or tuple(value.shape) != (2560,):
            raise ValueError(f"missing prompt_hidden_ln in {path}")
        by_problem[problem_id] = value.float().numpy()
    missing = sorted(set(frame.problem_id.astype(str)) - set(by_problem))
    if missing:
        raise ValueError(f"missing prompt state for {len(missing)} problems")
    return np.stack([by_problem[str(value)] for value in frame.problem_id]).astype(
        np.float32, copy=False
    )


def previous_vectors(
    frame: pd.DataFrame,
    current: np.ndarray,
    prompt: np.ndarray,
) -> np.ndarray:
    previous = prompt.copy()
    for _, group in frame.groupby("problem_id", sort=False):
        positions = group.sort_values("checkpoint").index.to_numpy(dtype=np.int64)
        if len(positions) > 1:
            previous[positions[1:]] = current[positions[:-1]]
    return previous


def lowrank_raw_features(
    frame: pd.DataFrame,
    hidden: np.ndarray,
    prompt: np.ndarray,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    current = hidden[:, 0, :].astype(np.float32, copy=False)
    previous = previous_vectors(frame, current, prompt)
    checkpoint = frame.checkpoint.to_numpy(dtype=np.float32)
    tmax = max(float(tmax), 1.0)
    delta_t = np.zeros(len(frame), dtype=np.float32)
    delta_entropy = np.zeros(len(frame), dtype=np.float32)
    entropy = frame.prefix_mean_entropy_tail8.to_numpy(dtype=np.float32)
    for _, group in frame.groupby("problem_id", sort=False):
        positions = group.sort_values("checkpoint").index.to_numpy(dtype=np.int64)
        ordered_t = checkpoint[positions]
        delta_t[positions] = np.diff(np.concatenate(([0.0], ordered_t)))
        if len(positions) > 1:
            delta_entropy[positions[1:]] = entropy[positions[1:]] - entropy[positions[:-1]]
    required = ("sampling_pmax", "sampling_probability_gap")
    if any(name not in frame.columns for name in required):
        raise ValueError(f"low-rank cache missing columns: {required}")
    scalar = np.stack(
        [
            checkpoint / tmax,
            np.log1p(checkpoint) / np.log1p(tmax),
            delta_t / tmax,
            entropy,
            delta_entropy,
            frame.sampling_pmax.to_numpy(dtype=np.float32),
            frame.sampling_probability_gap.to_numpy(dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    raw = np.concatenate([current, previous, prompt, scalar, delta_t[:, None]], axis=1).astype(
        np.float32, copy=False
    )
    if not np.isfinite(raw).all():
        raise ValueError("low-rank raw features contain NaN/Inf")
    return raw, delta_t


class LowRankProbe(nn.Module):
    """Shared learned projection retaining low-dimensional state dynamics."""

    def __init__(self, rank: int, feature_kind: str):
        super().__init__()
        if feature_kind not in LOWRANK_FEATURE_KINDS:
            raise ValueError(feature_kind)
        self.rank = int(rank)
        self.feature_kind = feature_kind
        self.projection = nn.Linear(2560, self.rank, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        widths = {
            "lowrank_u": self.rank + len(LOWRANK_SCALAR_NAMES),
            "lowrank_u_delta": 2 * self.rank + len(LOWRANK_SCALAR_NAMES) + 2,
            "lowrank_full": 3 * self.rank + len(LOWRANK_SCALAR_NAMES) + 2,
        }
        self.input_width = widths[feature_kind]
        self.network = nn.Sequential(
            nn.Linear(self.input_width, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(384, 96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        current, previous, origin = torch.split(values[:, : 3 * 2560], 2560, dim=1)
        scalars = values[:, 3 * 2560 : 3 * 2560 + len(LOWRANK_SCALAR_NAMES)]
        raw_delta_t = values[:, 3 * 2560 + len(LOWRANK_SCALAR_NAMES)].clamp_min(1.0)
        u = self.projection(current)
        if self.feature_kind == "lowrank_u":
            probe_input = torch.cat([u, scalars], dim=1)
        else:
            u_previous = self.projection(previous)
            delta = u - u_previous
            delta_norm = torch.linalg.vector_norm(delta, dim=1, keepdim=True) / torch.sqrt(
                raw_delta_t[:, None]
            )
            cosine = F.cosine_similarity(u, u_previous, dim=1)[:, None]
            geometry = torch.cat([delta_norm, cosine], dim=1)
            parts = [u, delta]
            if self.feature_kind == "lowrank_full":
                parts.append(u - self.projection(origin))
            probe_input = torch.cat([*parts, scalars, geometry], dim=1)
        return self.network(probe_input)[:, 0]


@torch.no_grad()
def predict_model(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        values = torch.from_numpy(features[start : start + batch_size]).float().to(device)
        output.append(torch.sigmoid(model(values)).float().cpu().numpy())
    result = np.concatenate(output).astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("predicted scores contain NaN/Inf")
    return result


def evaluate_frozen(
    frame,
    scores: np.ndarray,
    direction: str,
    calibrated: dict[str, Any],
    fallback_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluated: dict[str, Any] = {}
    records: dict[str, Any] = {}
    for family in ("empirical_B", "coverage"):
        evaluated[family] = {}
        records[family] = {}
        for key, frozen in calibrated[family].items():
            heldout, local_records = strip_records(
                simulate_policy(
                    frame,
                    scores,
                    direction,
                    float(frozen["threshold"]),
                    include_records=True,
                    fallback_records=fallback_records,
                    force_dense=bool(frozen.get("is_no_stop_sentinel", False)),
                )
            )
            evaluated[family][key] = {
                "calibration": frozen,
                "heldout": heldout,
            }
            records[family][key] = local_records
    return evaluated, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--schedule", choices=SCHEDULES, default="sentence")
    parser.add_argument("--actual-schedule-label")
    parser.add_argument("--layer", type=int, choices=(8, 20, 35), default=20)
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="boundary")
    parser.add_argument(
        "--feature-kind",
        choices=(*DYNAMIC_FEATURE_KINDS, *LOWRANK_FEATURE_KINDS),
        default="full_no_delta",
    )
    parser.add_argument("--rank", type=int, choices=(64, 128), default=64)
    parser.add_argument("--loss", choices=("bce", "bce_traj"), default="bce_traj")
    parser.add_argument(
        "--trajectory-aggregation",
        choices=("unnormalized_softmin", "normalized_softmin"),
        default="normalized_softmin",
    )
    parser.add_argument("--trajectory-beta", type=float)
    parser.add_argument("--trajectory-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.seed != 0:
        raise ValueError("legacy-v4 probe initialization/training seed must be 0")

    if args.method != "correction" and args.loss != "bce":
        raise ValueError("controlled target baselines use checkpoint BCE only")
    lowrank = args.feature_kind in LOWRANK_FEATURE_KINDS
    if lowrank and args.representation not in (
        "last8_noncontrol_ln_mean",
        "paragraph_ln_mean",
    ):
        raise ValueError("low-rank projection requires a per-token-LN mean representation")
    config = load_yaml(args.config)
    trajectory_beta = float(
        args.trajectory_beta
        if args.trajectory_beta is not None
        else config["probe"]["trajectory_softmin_beta"]
    )
    if trajectory_beta <= 0:
        raise ValueError("--trajectory-beta must be positive")
    if args.trajectory_weight < 0:
        raise ValueError("--trajectory-weight must be non-negative")
    if config.get("primary") is True and config.get("runnable") is not True:
        raise RuntimeError("论文主配置尚未通过 dtype 配对审计，禁止启动 probe 训练")
    destination = args.output if args.output.is_absolute() else ROOT / args.output
    invocation_spec = {
        "protocol_id": config["protocol_id"],
        "dataset": args.dataset,
        "method": args.method,
        "probe_seed": args.seed,
        "schedule": args.schedule,
        "actual_schedule_label": args.actual_schedule_label or args.schedule,
        "layer": args.layer,
        "representation": args.representation,
        "feature_kind": args.feature_kind,
        "lowrank_rank": args.rank if args.feature_kind in LOWRANK_FEATURE_KINDS else None,
        "loss": args.loss,
        "trajectory_aggregation": args.trajectory_aggregation,
        "trajectory_beta": trajectory_beta,
        "trajectory_weight": float(args.trajectory_weight),
        "raw_input": artifact_manifest(args.raw_root),
        "probe_config": config["probe"],
        "calibration_config": config["calibration"],
    }
    invocation_fingerprint = canonical_fingerprint(invocation_spec)
    complete_path = destination / "phase.complete"
    if args.resume and complete_path.is_file():
        marker = json.loads(complete_path.read_text(encoding="utf-8"))
        if (
            marker.get("status") == "complete"
            and marker.get("invocation_fingerprint") == invocation_fingerprint
            and (destination / "probe.pt").is_file()
        ):
            print(json.dumps({"status": "skipped_complete", "output": str(destination)}))
            return
        raise RuntimeError(f"拒绝 resume 不同指纹的 probe 输出：{destination}")
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"拒绝覆盖既有 probe 输出；请使用新的目录或同指纹 --resume：{destination}")
    destination.mkdir(parents=True, exist_ok=True)
    seed_everything(0)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    torch.set_num_threads(min(16, torch.get_num_threads()))

    representation_names(args.raw_root)
    frames: dict[str, Any] = {}
    raw_features: dict[str, np.ndarray] = {}
    features: dict[str, np.ndarray] = {}
    fallbacks: dict[str, list[dict[str, Any]]] = {}
    capture_layers = [args.layer]
    for split in ("probe_train", "calibration", "heldout"):
        frame, hidden, loaded_layers, split_fallbacks, packed_prompt = load_representation_split(
            args.raw_root,
            split,
            args.schedule,
            args.representation,
        )
        if loaded_layers != capture_layers:
            raise ValueError(f"capture layers differ for {split}: {loaded_layers}")
        frames[split] = frame
        fallbacks[split] = split_fallbacks
        if lowrank:
            prompt = (
                packed_prompt
                if packed_prompt is not None
                else prompt_hidden_for_frame(args.raw_root / split, frame)
            )
            raw, _ = lowrank_raw_features(
                frame,
                hidden,
                prompt,
                float(config["generation"]["dense_max_new_tokens"]),
            )
            raw_features[split] = raw
            del prompt
        else:
            raw_features[split] = build_dynamic_features(
                frame,
                hidden,
                capture_layers,
                layer=args.layer,
                feature_kind=args.feature_kind,
            )
        del hidden

    train_frame = frames["probe_train"]
    train_fallbacks = fallbacks["probe_train"]
    fallback_train_ids = [row["problem_id"] for row in train_fallbacks]
    fit_problem_ids, validation_problem_ids = fit_validation_problem_ids(
        train_frame,
        args.dataset,
        seed=0,
        additional_problem_ids=fallback_train_ids,
    )
    fit_mask, validation_mask = fit_validation_masks(
        train_frame,
        args.dataset,
        seed=0,
        additional_problem_ids=fallback_train_ids,
    )
    validation_fallbacks = [
        row for row in train_fallbacks
        if row["problem_id"] in validation_problem_ids
    ]
    scaler = StandardScaler(copy=True)
    if lowrank:
        scalar_start = 3 * 2560
        scalar_stop = scalar_start + len(LOWRANK_SCALAR_NAMES)
        scaler.fit(raw_features["probe_train"][fit_mask, scalar_start:scalar_stop])
        for split, raw in raw_features.items():
            values = raw.copy()
            values[:, scalar_start:scalar_stop] = scaler.transform(
                values[:, scalar_start:scalar_stop]
            )
            features[split] = values.astype(np.float32, copy=False)
    else:
        scaler.fit(raw_features["probe_train"][fit_mask])
        for split, raw in raw_features.items():
            features[split] = scaler.transform(raw).astype(np.float32, copy=False)
    del raw_features

    # 本轮统一忽略短答案成本，并以reasoning token作为唯一成本。
    for split, frame in frames.items():
        frame["branch_tokens"] = 0
        frame["replay_stop_wall_ms"] = frame.checkpoint.astype(float)
        frame["dense_wall_ms"] = frame.dense_tokens.astype(float)
        frame["adaptive_fallback_wall_ms"] = frame.dense_tokens.astype(float)
        for fallback in fallbacks[split]:
            fallback["dense_wall_ms"] = float(fallback["dense_tokens"])
            fallback["adaptive_fallback_wall_ms"] = float(fallback["dense_tokens"])

    labels = {
        split: target_values(frame, args.method)
        for split, frame in frames.items()
    }
    train_labels = labels["probe_train"]
    positives = float(train_labels[fit_mask].sum())
    negatives = float(fit_mask.sum()) - positives
    positive_weight = torch.tensor(
        negatives / positives if positives > 0 else 1.0,
        dtype=torch.float32,
        device=device,
    )
    remaining = np.clip(
        (
            train_frame.dense_tokens.to_numpy(dtype=np.float32)
            - train_frame.checkpoint.to_numpy(dtype=np.float32)
        )
        / np.maximum(train_frame.dense_tokens.to_numpy(dtype=np.float32), 1.0),
        0.0,
        1.0,
    ).astype(np.float32)

    stored_width = int(features["probe_train"].shape[1])
    if lowrank:
        if stored_width != 3 * 2560 + len(LOWRANK_SCALAR_NAMES) + 1:
            raise ValueError(f"invalid low-rank stored width: {stored_width}")
        model = LowRankProbe(args.rank, args.feature_kind).to(device)
        width = model.input_width
    else:
        width = stored_width
        if args.feature_kind == "full_no_delta" and width != 2566:
            raise ValueError(f"full_no_delta input width must be 2566, got {width}")
        model = FinalPaperProbe(width).to(device)
    probe_config = config["probe"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(probe_config["learning_rate"]),
        weight_decay=float(probe_config["weight_decay"]),
    )
    maximum_epochs = int(args.epochs or probe_config["max_epochs"])
    patience_limit = int(probe_config["patience"])
    rng = random.Random(0)
    history: list[dict[str, Any]] = []
    best: tuple[tuple[float, ...], int, dict[str, torch.Tensor]] | None = None
    patience = 0
    direction = method_direction(args.method)

    for epoch in range(maximum_epochs):
        model.train()
        losses: list[float] = []
        point_losses: list[float] = []
        trajectory_losses: list[float] = []
        for positions, offsets in problem_batches(
            train_frame,
            fit_mask,
            int(probe_config["trajectory_batch_size"]),
            rng,
        ):
            values = torch.from_numpy(features["probe_train"][positions]).float().to(device)
            target = torch.from_numpy(train_labels[positions]).to(device)
            logits = model(values)
            if args.method == "correction":
                loss, point, trajectory = correction_loss(
                    logits,
                    target,
                    torch.from_numpy(remaining[positions]).to(device),
                    offsets,
                    beta=trajectory_beta,
                    trajectory=args.loss == "bce_traj",
                    normalize_by_count=(
                        args.trajectory_aggregation == "normalized_softmin"
                    ),
                    trajectory_weight=float(args.trajectory_weight),
                )
            else:
                loss = binary_point_loss(logits, target, positive_weight)
                point = loss
                trajectory = loss.detach() * 0.0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(probe_config["gradient_clip"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            point_losses.append(float(point.detach().cpu()))
            trajectory_losses.append(float(trajectory.detach().cpu()))

        validation_scores = predict_model(
            model,
            features["probe_train"][validation_mask],
            device,
        )
        validation_truth = train_labels[validation_mask]
        validation_ap, validation_auc = safe_ap_auc(
            validation_truth, validation_scores
        )
        validation_frame = train_frame.loc[validation_mask].reset_index(drop=True)
        if args.method == "correction":
            internal_grid = [
                (
                    float(value),
                    no_stop,
                )
                for value, no_stop in threshold_grid(
                    validation_scores,
                    direction,
                    grid_size=int(
                        calibration_value(
                            config, "quantile_grid_size", "threshold_quantiles"
                        )
                    ),
                )
            ]
            internal_curve = []
            for threshold, no_stop in internal_grid:
                row = simulate_policy(
                    validation_frame,
                    validation_scores,
                    direction,
                    threshold,
                    force_dense=no_stop,
                    fallback_records=validation_fallbacks,
                )
                row["is_no_stop_sentinel"] = no_stop
                internal_curve.append(row)
            strict = select_empirical_budget(internal_curve, 0)
            key = (
                strict["replay_wall_reduction"],
                validation_ap,
                validation_auc,
            )
        else:
            strict = None
            key = (
                validation_ap,
                validation_auc,
                -float(np.mean(losses)),
            )
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "point_loss": float(np.mean(point_losses)),
            "trajectory_loss": float(np.mean(trajectory_losses)),
            "validation_ap": validation_ap,
            "validation_auc": validation_auc,
            "validation_strict_replay": strict,
        }
        history.append(record)
        print(json.dumps({"method": args.method, **record}), flush=True)
        if best is None or key > best[0]:
            best = (
                key,
                epoch,
                {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
            )
            patience = 0
        else:
            patience += 1
        if patience >= patience_limit:
            break

    if best is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best[2])
    scores = {split: predict_model(model, values, device) for split, values in features.items()}
    calibration_config = config["calibration"]
    calibrated = calibrate_policies(
        frames["calibration"],
        scores["calibration"],
        direction,
        grid_size=int(
            calibration_value(config, "quantile_grid_size", "threshold_quantiles")
        ),
        empirical_budgets=sorted(set(
            int(value) for value in (
                calibration_value(
                    config,
                    "empirical_lost_correct_budgets",
                    "empirical_lost_correct_B",
                )
                + calibration_config.get("rate_matched_empirical_budgets", [])
            )
        )),
        fallback_records=fallbacks["calibration"],
        coverage_targets=[
            float(value) for value in calibration_config["coverage_targets"]
        ],
    )
    # 与动态方法相同：B约束之外还要求calibration准确率不低于Dense 1pp。
    curve = calibrated["curve"]
    dense_row = next(row for row in curve if row.get("is_no_stop_sentinel"))
    epsilon = float(config["dynamic_policy"]["accuracy_epsilon"])
    for raw_budget in calibration_value(
        config, "empirical_lost_correct_budgets", "empirical_lost_correct_B"
    ):
        budget = int(raw_budget)
        feasible = [
            row for row in curve
            if int(row["lost_correct_count"]) <= budget
            and float(row["accuracy"]) >= float(dense_row["dense_accuracy"]) - epsilon
        ]
        if not feasible:
            chosen = dict(dense_row)
        else:
            chosen = dict(min(feasible, key=lambda row: (
                float(row["mean_reasoning_and_answer_tokens"]),
                -float(row["token_reduction"]),
                -float(row["coverage"]),
                float(row["threshold"]),
            )))
        chosen["budget_B"] = budget
        chosen["accuracy_epsilon"] = epsilon
        calibrated["empirical_B"][str(budget)] = chosen
    evaluated, policy_records = evaluate_frozen(
        frames["heldout"],
        scores["heldout"],
        direction,
        calibrated,
        fallbacks["heldout"],
    )
    heldout_ap, heldout_auc = safe_ap_auc(
        labels["heldout"], scores["heldout"]
    )
    fit_ids = sorted(fit_problem_ids)
    validation_ids = sorted(validation_problem_ids)
    online_workpoints = {}
    workpoints = calibration_value(
        config, "historical_workpoints", "named_workpoints"
    )
    for name, budget in workpoints.items():
        family = "empirical_B"
        key = str(int(budget))
        online_workpoints[name] = {
            "family": family,
            "key": key,
            "calibration": evaluated[family][key]["calibration"],
            "heldout_metrics_loaded_by_online_runner": False,
        }

    run_spec = {
        "dataset": args.dataset,
        "method": args.method,
        "stop_direction": direction,
        "seed": 0,
        "dense_seed": global_seed(config),
        "protocol_id": config["protocol_id"],
        "schedule": args.schedule,
        "actual_schedule_label": args.actual_schedule_label or args.schedule,
        "layer": args.layer,
        "representation": args.representation,
        "feature_kind": args.feature_kind,
        "lowrank_rank": args.rank if lowrank else None,
        "loss": args.loss,
        "architecture": (
            {
                "shared_projection": [2560, args.rank],
                "projected_probe_input_width": width,
                "probe_trunk": [width, 384, 96, 1],
            }
            if lowrank
            else [width, 384, 96, 1]
        ),
        "stored_feature_width": stored_width,
        "trainable_parameters": int(sum(value.numel() for value in model.parameters())),
        "optimizer": "AdamW",
        "learning_rate": float(probe_config["learning_rate"]),
        "weight_decay": float(probe_config["weight_decay"]),
        "trajectory_batch_size": int(probe_config["trajectory_batch_size"]),
        "gradient_clip": float(probe_config["gradient_clip"]),
        "maximum_epochs": maximum_epochs,
        "patience": patience_limit,
        "trajectory_softmin_beta": trajectory_beta,
        "trajectory_aggregation": args.trajectory_aggregation,
        "trajectory_normalize_by_count": (
            args.trajectory_aggregation == "normalized_softmin"
        ),
        "trajectory_weight": float(args.trajectory_weight),
        "fit_split_seed": 0,
        "fit_fraction": 0.8,
        "scaler_fit_scope": (
            "probe_train_fit_only_lowrank_scalars_only"
            if lowrank
            else "probe_train_fit_only_all_features"
        ),
        "lowrank_scalar_names": list(LOWRANK_SCALAR_NAMES) if lowrank else None,
        "lowrank_origin": "prompt_last_token_layernorm" if lowrank else None,
        "lowrank_previous_first": "prompt_last_token_layernorm" if lowrank else None,
        "cost_protocol": "reasoning_tokens_only; short_answer_cost=0",
        "calibration_accuracy_epsilon": float(config["dynamic_policy"]["accuracy_epsilon"]),
    }
    payload = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_spec": run_spec,
        "run_spec_fingerprint": canonical_fingerprint(run_spec),
        "input": artifact_manifest(args.raw_root),
        "split_counts": {
            split: {
                "problems": int(frame.problem_id.nunique()) + len(fallbacks[split]),
                "scorable_problems": int(frame.problem_id.nunique()),
                "fallback_only_problems": len(fallbacks[split]),
                "checkpoints": len(frame),
                "positive_labels": int(labels[split].sum()),
                "missing_forced_answers": int(frame.current_prediction.isna().sum()),
            }
            for split, frame in frames.items()
        },
        "fit_problem_ids": fit_ids,
        "validation_problem_ids": validation_ids,
        "best_epoch": int(best[1]),
        "history": history,
        "heldout_label_ap_descriptive": heldout_ap,
        "heldout_label_auc_descriptive": heldout_auc,
        "calibration": calibrated,
        "frozen_policy_results": evaluated,
        "online_workpoints": online_workpoints,
    }
    atomic_torch_save(
        {
            "status": "complete",
            "state_dict": best[2],
            "run_spec": run_spec,
            "capture_layers": capture_layers,
            "input_width": width,
            "stored_feature_width": stored_width,
            "scaler_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
            "scaler_scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
            "scaler_feature_scope": (
                list(LOWRANK_SCALAR_NAMES) if lowrank else "all_input_features"
            ),
            "online_workpoints": online_workpoints,
        },
        destination / "probe.pt",
    )
    atomic_torch_save(
        {
            "status": "complete",
            "scores": {
                split: torch.from_numpy(value.astype(np.float32))
                for split, value in scores.items()
            },
            "problem_ids": {
                split: frame.problem_id.astype(str).tolist()
                for split, frame in frames.items()
            },
            "checkpoints": {
                split: frame.checkpoint.astype(int).tolist()
                for split, frame in frames.items()
            },
        },
        destination / "scores.pt",
    )
    atomic_torch_save(
        {
            "status": "complete",
            "records": policy_records,
        },
        destination / "policy_records.pt",
    )
    atomic_json(payload, destination / "probe.json")
    atomic_json(
        {
            "status": "complete",
            "invocation_fingerprint": invocation_fingerprint,
            "run_spec_fingerprint": payload["run_spec_fingerprint"],
            "best_epoch": int(best[1]),
            "artifacts": ["probe.json", "probe.pt", "scores.pt", "policy_records.pt"],
        },
        complete_path,
    )
    print(json.dumps({
        "status": "complete",
        "output": str(destination),
        "best_epoch": int(best[1]),
        "heldout_label_ap_descriptive": heldout_ap,
        "heldout_label_auc_descriptive": heldout_auc,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

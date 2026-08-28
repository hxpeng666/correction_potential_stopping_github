#!/usr/bin/env python3
"""Train and evaluate one controlled DeepSeek-7B correction-risk variant.

Hyperparameters are selected only on the problem-level 20% internal split of
``probe_train``.  The external calibration set is used only after a candidate is
frozen, and held-out/OOD sets are never used for model or threshold selection.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from deepseek7b_protocol_v1 import canonical_fingerprint
from src.deepseek7b_method_exploration_v1 import (
    FEATURE_KINDS,
    POINT_LOSSES,
    REPRESENTATION_KINDS,
    TRAJECTORY_AGGREGATIONS,
    TRAJECTORY_SCOPES,
    FeatureTransform,
    RiskProbe,
    correction_objective,
    fit_feature_transform,
    load_auxiliary_split,
    predict_scores,
    problem_batches,
)
from src.final_paper_inference import atomic_torch_save
from src.legacy_empirical_probe_normalized_v1 import (
    fit_validation_masks,
    fit_validation_problem_ids,
    load_checkpoint_split,
    threshold_grid,
)
from src.reproducibility import (
    code_provenance,
    environment_provenance,
    sha256_array,
    sha256_json,
    sha256_state_dict,
    strict_reproducibility,
)
from src.utils import atomic_json, load_yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_manifest(root: Path) -> dict[str, Any]:
    paths = sorted(root.glob("sample_*.pt"))
    digest = hashlib.sha256()
    capped = 0
    checkpoints = 0
    for path in paths:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        capped += int(bool(artifact["dense"].get("reached_max_tokens")))
        checkpoints += len(artifact["rows"])
        digest.update(
            (
                f"{path.name}:{path.stat().st_size}:{artifact['problem_id']}:"
                f"{artifact.get('protocol_fingerprint')}:"
                f"{artifact.get('primary_replay_view_fingerprint')}\n"
            ).encode()
        )
    return {
        "root": str(root.resolve()),
        "files": len(paths),
        "checkpoints": checkpoints,
        "capped_dense_trajectories": capped,
        "identity_fingerprint": digest.hexdigest(),
    }


def auxiliary_audit_identity(root: Path | None, split: str) -> dict[str, Any] | None:
    if root is None:
        return None
    path = root / split / "AUDIT.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing auxiliary audit: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete auxiliary audit: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "artifacts": int(payload["artifacts"]),
        "output_checkpoints": int(payload["output_checkpoints"]),
        "protocol_fingerprint": str(payload["protocol_fingerprint"]),
    }


def safe_ap_auc(truth: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(np.unique(truth.astype(np.int64))) < 2:
        return 0.0, 0.5
    return (
        float(average_precision_score(truth, scores)),
        float(roc_auc_score(truth, scores)),
    )


def prepare_cost_columns(
    frame: pd.DataFrame, fallbacks: list[dict[str, Any]]
) -> None:
    frame["branch_tokens"] = 0
    frame["replay_stop_wall_ms"] = frame.checkpoint.astype(float)
    frame["dense_wall_ms"] = frame.dense_tokens.astype(float)
    frame["adaptive_fallback_wall_ms"] = frame.dense_tokens.astype(float)
    for fallback in fallbacks:
        fallback["dense_wall_ms"] = float(fallback["dense_tokens"])
        fallback["adaptive_fallback_wall_ms"] = float(fallback["dense_tokens"])


def replay_policy(
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    *,
    force_dense: bool,
    readout_suffix_tokens: int,
    fallbacks: list[dict[str, Any]],
    include_records: bool = False,
) -> dict[str, Any]:
    """Problem-level first-hit replay with explicit one-step feature cost."""
    if len(frame) != len(scores):
        raise ValueError("frame/score length mismatch")
    records: list[dict[str, Any]] = []
    counts = {"W_to_C": 0, "C_to_W": 0, "W_to_W": 0, "C_to_C": 0}
    total_dense = 0.0
    total_reasoning = 0.0
    total_deployed = 0.0
    total_readout = 0.0
    correct = lost = helped = stopped = 0
    score_series = pd.Series(scores, index=frame.index)
    for problem_id, group in frame.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        local_scores = score_series.loc[ordered.index].to_numpy(dtype=np.float64)
        chosen_position = None
        if not force_dense:
            candidates = np.flatnonzero(local_scores <= threshold)
            if len(candidates):
                chosen_position = int(candidates[0])
        dense_success = bool(ordered.iloc[0].dense_success)
        dense_tokens = int(ordered.iloc[0].dense_tokens)
        visited = len(ordered) if chosen_position is None else chosen_position + 1
        readout_tokens = visited * readout_suffix_tokens
        if chosen_position is None:
            current_success = dense_success
            reasoning_tokens = dense_tokens
            checkpoint = dense_tokens
            stopped_now = False
        else:
            row = ordered.iloc[chosen_position]
            current_success = bool(row.current_success)
            reasoning_tokens = int(row.checkpoint)
            checkpoint = int(row.checkpoint)
            stopped_now = True
            stopped += 1
        transition = ("C" if current_success else "W") + "_to_" + (
            "C" if dense_success else "W"
        )
        counts[transition] += 1
        correct += int(current_success)
        lost += int(dense_success and not current_success)
        helped += int((not dense_success) and current_success)
        total_dense += dense_tokens
        total_reasoning += reasoning_tokens
        total_readout += readout_tokens
        total_deployed += reasoning_tokens + readout_tokens
        if include_records:
            records.append(
                {
                    "problem_id": str(problem_id),
                    "dense_success": dense_success,
                    "stopped": stopped_now,
                    "stop_checkpoint": checkpoint,
                    "current_success": current_success,
                    "transition": transition,
                    "visited_checkpoints": visited,
                    "one_step_suffix_tokens": readout_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "deployed_token_equivalent": reasoning_tokens + readout_tokens,
                    "dense_tokens": dense_tokens,
                }
            )
    for fallback in fallbacks:
        dense_success = bool(fallback["dense_success"])
        dense_tokens = int(fallback["dense_tokens"])
        transition = "C_to_C" if dense_success else "W_to_W"
        counts[transition] += 1
        correct += int(dense_success)
        total_dense += dense_tokens
        total_reasoning += dense_tokens
        total_deployed += dense_tokens
        if include_records:
            records.append(
                {
                    "problem_id": str(fallback["problem_id"]),
                    "dense_success": dense_success,
                    "stopped": False,
                    "stop_checkpoint": dense_tokens,
                    "current_success": dense_success,
                    "transition": transition,
                    "visited_checkpoints": 0,
                    "one_step_suffix_tokens": 0,
                    "reasoning_tokens": dense_tokens,
                    "deployed_token_equivalent": dense_tokens,
                    "dense_tokens": dense_tokens,
                    "fallback_only": True,
                }
            )
    problems = int(frame.problem_id.nunique()) + len(fallbacks)
    if problems <= 0 or total_dense <= 0:
        raise ValueError("empty replay")
    result = {
        "problems": problems,
        "threshold": float(threshold),
        "is_no_stop_sentinel": bool(force_dense),
        "counts": counts,
        "accuracy": correct / problems,
        "dense_accuracy": (counts["W_to_C"] + counts["C_to_C"]) / problems,
        "accuracy_delta_pp": 100.0
        * (
            correct / problems
            - (counts["W_to_C"] + counts["C_to_C"]) / problems
        ),
        "lost_correct_count": lost,
        "lost_correct_rate": lost / problems,
        "helped_count": helped,
        "coverage": stopped / problems,
        "mean_reasoning_tokens": total_reasoning / problems,
        "mean_dense_tokens": total_dense / problems,
        "token_reduction": 1.0 - total_reasoning / total_dense,
        "mean_one_step_suffix_tokens": total_readout / problems,
        "mean_deployed_token_equivalent": total_deployed / problems,
        "deployed_token_reduction": 1.0 - total_deployed / total_dense,
        "readout_suffix_tokens_per_checkpoint": readout_suffix_tokens,
    }
    if include_records:
        result["records"] = records
    return result


def policy_curve(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    grid_size: int,
    readout_suffix_tokens: int,
    fallbacks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    curve = []
    for threshold, force_dense in threshold_grid(scores, "low", grid_size=grid_size):
        curve.append(
            replay_policy(
                frame,
                scores,
                float(threshold),
                force_dense=bool(force_dense),
                readout_suffix_tokens=readout_suffix_tokens,
                fallbacks=fallbacks,
            )
        )
    return curve


def choose_empirical_budget(
    curve: list[dict[str, Any]], budget: int, accuracy_epsilon: float
) -> dict[str, Any]:
    dense = next(row for row in curve if row["is_no_stop_sentinel"])
    feasible = [
        row
        for row in curve
        if int(row["lost_correct_count"]) <= budget
        and float(row["accuracy"])
        >= float(dense["dense_accuracy"]) - accuracy_epsilon
    ]
    chosen = dict(
        min(
            feasible or [dense],
            key=lambda row: (
                float(row["mean_deployed_token_equivalent"]),
                -float(row["deployed_token_reduction"]),
                -float(row["token_reduction"]),
                -float(row["coverage"]),
                float(row["threshold"]),
            ),
        )
    )
    chosen["budget_B"] = int(budget)
    chosen["accuracy_epsilon"] = float(accuracy_epsilon)
    chosen["selection_objective"] = "maximize calibration deployed-token reduction"
    return chosen


def load_representation(
    frame: pd.DataFrame,
    cached_hidden: np.ndarray,
    capture_layers: list[int],
    *,
    layer: int,
    representation_kind: str,
    feature_kind: str,
    readout_kind: str,
    auxiliary_root: Path | None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    if layer not in capture_layers:
        raise ValueError(f"layer {layer} absent from {capture_layers}")
    last = cached_hidden[:, capture_layers.index(layer), :].astype(
        np.float32, copy=False
    )
    needs_readout = "one_step" in feature_kind
    needs_auxiliary = representation_kind != "last" or needs_readout
    if not needs_auxiliary:
        return last, None, {"mode": "frozen_cached_last_hidden"}
    if auxiliary_root is None:
        raise ValueError("selected representation/readout requires --auxiliary root")
    auxiliary_hidden, readout, audit = load_auxiliary_split(
        auxiliary_root, frame, representation_kind
    )
    if representation_kind == "last":
        if not np.array_equal(auxiliary_hidden.astype(np.float16), last.astype(np.float16)):
            raise AssertionError("auxiliary/cached last-hidden mismatch")
        hidden = last
    else:
        hidden = auxiliary_hidden
    if needs_readout:
        if readout_kind == "full":
            selected_readout = readout
        elif readout_kind == "distribution":
            selected_readout = readout[:, :5]
        elif readout_kind == "stability":
            selected_readout = readout[:, 5:]
        else:
            raise ValueError(readout_kind)
        audit = {**audit, "readout_kind": readout_kind, "readout_width": int(selected_readout.shape[1])}
    else:
        selected_readout = None
    return hidden, selected_readout, audit


def load_split(
    root: Path,
    split: str,
    *,
    schedule: str,
    layer: int,
    representation_kind: str,
    feature_kind: str,
    readout_kind: str,
    auxiliary_root: Path | None,
) -> dict[str, Any]:
    frame, cached_hidden, layers, fallbacks = load_checkpoint_split(
        root / split, schedule
    )
    hidden, readout, auxiliary_audit = load_representation(
        frame,
        cached_hidden,
        layers,
        layer=layer,
        representation_kind=representation_kind,
        feature_kind=feature_kind,
        readout_kind=readout_kind,
        auxiliary_root=(auxiliary_root / split if auxiliary_root is not None else None),
    )
    del cached_hidden
    prepare_cost_columns(frame, fallbacks)
    return {
        "frame": frame,
        "hidden": hidden,
        "readout": readout,
        "capture_layers": layers,
        "fallbacks": fallbacks,
        "auxiliary_audit": auxiliary_audit,
    }


def transform_split(
    loaded: dict[str, Any], transform: FeatureTransform
) -> np.ndarray:
    return transform.transform(
        loaded["frame"], loaded["hidden"], loaded["readout"]
    )


def acquire_output_lock(output: Path):
    """Serialize all writers for one experiment output directory.

    The lock lives beside, rather than inside, ``output`` so it cannot make an
    otherwise fresh output look incompatible.  It is held for the process
    lifetime; a second runner waits, then observes ``phase.complete`` and exits
    through the normal fingerprint-checked resume path.
    """
    lock_root = output.parent / ".locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    handle = (lock_root / f"{output.name}.lock").open("a", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path)
    parser.add_argument("--ood-root", type=Path)
    parser.add_argument("--aux-raw-root", type=Path)
    parser.add_argument("--aux-heldout-root", type=Path)
    parser.add_argument("--aux-ood-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--schedule", default="sentence")
    parser.add_argument("--representation-kind", choices=REPRESENTATION_KINDS, default="last")
    parser.add_argument("--feature-kind", choices=FEATURE_KINDS, default="hidden_scalars")
    parser.add_argument(
        "--readout-kind",
        choices=("full", "distribution", "stability"),
        default="full",
    )
    parser.add_argument("--pca-dim", type=int)
    parser.add_argument("--pca-fit-max-rows", type=int, default=20000)
    parser.add_argument(
        "--probe-architecture",
        choices=("linear", "compact", "small", "standard"),
        default="standard",
    )
    parser.add_argument("--point-loss", choices=POINT_LOSSES, default="legacy_weighted")
    parser.add_argument("--stratified-problem-batches", action="store_true")
    parser.add_argument("--trajectory-scope", choices=TRAJECTORY_SCOPES, default="all_dangerous")
    parser.add_argument(
        "--trajectory-aggregation",
        choices=TRAJECTORY_AGGREGATIONS,
        default="normalized_softmin",
    )
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--lambda-protect", type=float, default=1.0)
    parser.add_argument("--lambda-separation", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-problems", type=int)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_lock = acquire_output_lock(args.output)

    config = load_yaml(args.config)
    reproducibility = strict_reproducibility(seed=0, num_threads=1)
    code_identity = code_provenance(
        ROOT,
        (
            "scripts/train_deepseek7b_method_exploration_v1.py",
            "scripts/deepseek7b_protocol_v1.py",
            "src/deepseek7b_method_exploration_v1.py",
            "src/reproducibility.py",
            "src/legacy_empirical_probe_normalized_v1.py",
            "src/final_paper_inference.py",
        ),
    )
    if args.feature_kind.startswith("pca_") != (args.pca_dim is not None):
        raise ValueError("PCA feature kinds and --pca-dim must be used together")
    if args.trajectory_aggregation == "none" and (
        args.lambda_protect != 0 or args.lambda_separation != 0
    ):
        raise ValueError("trajectory aggregation none requires zero trajectory weights")
    if args.lambda_separation > 0 and args.trajectory_scope != "reachability_earliest_safe":
        raise ValueError("separation loss requires reachability_earliest_safe")
    if args.screen_only and (args.heldout_root or args.ood_root):
        raise ValueError("screen-only mode cannot read heldout/OOD roots")

    invocation = {
        "protocol_id": "deepseek7b_method_exploration13k_v1",
        "dataset": args.dataset,
        "source_config_sha256": sha256(args.config),
        "raw_manifest": artifact_manifest(args.raw_root / "probe_train"),
        "representation_kind": args.representation_kind,
        "feature_kind": args.feature_kind,
        "pca_dim": args.pca_dim,
        "pca_fit_max_rows": args.pca_fit_max_rows,
        "probe_architecture": args.probe_architecture,
        "point_loss": args.point_loss,
        "stratified_problem_batches": args.stratified_problem_batches,
        "trajectory_scope": args.trajectory_scope,
        "trajectory_aggregation": args.trajectory_aggregation,
        "beta": args.beta,
        "rho": args.rho,
        "lambda_protect": args.lambda_protect,
        "lambda_separation": args.lambda_separation,
        "gamma": args.gamma,
        "screen_only": args.screen_only,
        "reproducibility_protocol": reproducibility,
        "code_identity": code_identity,
    }
    # Preserve the v1 fingerprint for variants that use only the original cache;
    # auxiliary identity becomes part of the fingerprint exactly when it matters.
    if "one_step" in args.feature_kind:
        invocation["readout_kind"] = args.readout_kind
    if args.aux_raw_root is not None:
        invocation["aux_raw_probe_train"] = auxiliary_audit_identity(
            args.aux_raw_root, "probe_train"
        )
        if not args.screen_only:
            invocation["aux_raw_calibration"] = auxiliary_audit_identity(
                args.aux_raw_root, "calibration"
            )
    if not args.screen_only and args.aux_heldout_root is not None:
        invocation["aux_heldout"] = auxiliary_audit_identity(
            args.aux_heldout_root, "heldout"
        )
    if not args.screen_only and args.aux_ood_root is not None:
        invocation["aux_ood"] = auxiliary_audit_identity(
            args.aux_ood_root, "heldout"
        )
    invocation_fingerprint = canonical_fingerprint(invocation)
    marker = args.output / "phase.complete"
    if args.resume and marker.is_file():
        saved = json.loads(marker.read_text())
        if saved.get("invocation_fingerprint") == invocation_fingerprint:
            print(json.dumps({"status": "skipped_complete", "output": str(args.output)}))
            return
        raise RuntimeError(f"incompatible resume: {args.output}")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        # The probe is tiny relative to the frozen LLM.  CPU screening allows
        # the GPU feature collectors to remain saturated without serializing
        # otherwise independent ablation axes.
        device = torch.device("cpu")
    runtime_identity = environment_provenance(device)
    train = load_split(
        args.raw_root,
        "probe_train",
        schedule=args.schedule,
        layer=args.layer,
        representation_kind=args.representation_kind,
        feature_kind=args.feature_kind,
        readout_kind=args.readout_kind,
        auxiliary_root=args.aux_raw_root,
    )
    frame = train["frame"]
    fallback_ids = [row["problem_id"] for row in train["fallbacks"]]
    fit_ids, validation_ids = fit_validation_problem_ids(
        frame, args.dataset, seed=0, additional_problem_ids=fallback_ids
    )
    fit_mask, validation_mask = fit_validation_masks(
        frame, args.dataset, seed=0, additional_problem_ids=fallback_ids
    )
    validation_fallbacks = [
        row for row in train["fallbacks"] if row["problem_id"] in validation_ids
    ]
    transform, train_features, feature_report = fit_feature_transform(
        frame,
        train["hidden"],
        train["readout"],
        fit_mask,
        feature_kind=args.feature_kind,
        pca_dim=args.pca_dim,
        pca_fit_max_rows=args.pca_fit_max_rows,
        seed=0,
    )
    target = ((~frame.current_success.astype(bool)) & frame.dense_success.astype(bool)).to_numpy(np.float32)
    current_success = frame.current_success.to_numpy(np.float32)
    dense_success = frame.dense_success.to_numpy(np.float32)
    remaining = np.clip(
        (frame.dense_tokens.to_numpy(np.float32) - frame.checkpoint.to_numpy(np.float32))
        / np.maximum(frame.dense_tokens.to_numpy(np.float32), 1.0),
        0.0,
        1.0,
    ).astype(np.float32)
    readout_suffix_tokens = 6 if "one_step" in args.feature_kind else 0
    input_identity = {
        "row_keys_sha256": sha256_json(
            list(
                zip(
                    frame.problem_id.astype(str).tolist(),
                    frame.checkpoint.astype(int).tolist(),
                )
            )
        ),
        "features_sha256": sha256_array(train_features),
        "target_sha256": sha256_array(target),
        "fit_mask_sha256": sha256_array(fit_mask),
        "validation_mask_sha256": sha256_array(validation_mask),
    }
    model = RiskProbe(transform.input_width, args.probe_architecture).to(device)
    initial_state_sha256 = sha256_state_dict(model.state_dict())
    probe_config = config["probe"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(probe_config["learning_rate"]),
        weight_decay=float(probe_config["weight_decay"]),
        foreach=False,
        fused=False,
    )
    epochs = int(args.epochs or probe_config["max_epochs"])
    patience_limit = int(args.patience or probe_config["patience"])
    batch_problems = int(args.batch_problems or probe_config["trajectory_batch_size"])
    rng = random.Random(0)
    best = None
    patience = 0
    history = []
    for epoch in range(epochs):
        model.train()
        totals = []
        points = []
        protects = []
        separations = []
        protected_count = separated_count = 0
        for positions, offsets in problem_batches(
            frame,
            fit_mask,
            target,
            batch_problems,
            rng,
            stratified=args.stratified_problem_batches,
        ):
            logits = model(torch.from_numpy(train_features[positions]).to(device))
            breakdown = correction_objective(
                logits,
                torch.from_numpy(target[positions]).to(device),
                torch.from_numpy(current_success[positions]).to(device),
                torch.from_numpy(dense_success[positions]).to(device),
                torch.from_numpy(remaining[positions]).to(device),
                offsets,
                point_mode=args.point_loss,
                trajectory_scope=args.trajectory_scope,
                aggregation=args.trajectory_aggregation,
                beta=args.beta,
                rho=args.rho,
                lambda_protect=args.lambda_protect,
                lambda_separation=args.lambda_separation,
                gamma=args.gamma,
            )
            optimizer.zero_grad(set_to_none=True)
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(probe_config["gradient_clip"]), foreach=False
            )
            optimizer.step()
            totals.append(float(breakdown.total.detach().cpu()))
            points.append(float(breakdown.point.detach().cpu()))
            protects.append(float(breakdown.protect.detach().cpu()))
            separations.append(float(breakdown.separation.detach().cpu()))
            protected_count += breakdown.protected_trajectories
            separated_count += breakdown.separated_trajectories

        validation_scores = predict_scores(
            model, train_features[validation_mask], device
        )
        validation_truth = target[validation_mask]
        ap, auc = safe_ap_auc(validation_truth, validation_scores)
        validation_frame = frame.loc[validation_mask].reset_index(drop=True)
        curve = policy_curve(
            validation_frame,
            validation_scores,
            grid_size=101,
            readout_suffix_tokens=readout_suffix_tokens,
            fallbacks=validation_fallbacks,
        )
        strict = choose_empirical_budget(curve, 0, 0.01)
        record = {
            "epoch": epoch,
            "loss": float(np.mean(totals)),
            "point_loss": float(np.mean(points)),
            "protect_loss": float(np.mean(protects)),
            "separation_loss": float(np.mean(separations)),
            "protected_trajectory_batches": protected_count,
            "separated_trajectory_batches": separated_count,
            "validation_ap": ap,
            "validation_auc": auc,
            "validation_B0": strict,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        key = (
            float(strict["deployed_token_reduction"]),
            ap,
            auc,
            -record["loss"],
        )
        if best is None or key > best[0]:
            best = (
                key,
                epoch,
                {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            )
            patience = 0
        else:
            patience += 1
        if patience >= patience_limit:
            break
    if best is None:
        raise RuntimeError("training produced no model")
    model.load_state_dict(best[2])
    train_scores = predict_scores(model, train_features, device)
    final_state_sha256 = sha256_state_dict(best[2])
    internal_scores = train_scores[validation_mask]
    internal_curve = policy_curve(
        frame.loc[validation_mask].reset_index(drop=True),
        internal_scores,
        grid_size=101,
        readout_suffix_tokens=readout_suffix_tokens,
        fallbacks=validation_fallbacks,
    )
    internal_results = {
        str(budget): choose_empirical_budget(internal_curve, budget, 0.01)
        for budget in (0, 1, 2, 4, 10)
    }
    payload: dict[str, Any] = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "invocation": invocation,
        "invocation_fingerprint": invocation_fingerprint,
        "feature_report": feature_report,
        "probe": {
            "architecture": model.architecture,
            "layer_widths": model.layer_widths,
            "parameters": model.parameter_count,
        },
        "split": {
            "fit_problem_ids": sorted(fit_ids),
            "validation_problem_ids": sorted(validation_ids),
            "fit_checkpoints": int(fit_mask.sum()),
            "validation_checkpoints": int(validation_mask.sum()),
            "fit_positive_checkpoints": int(target[fit_mask].sum()),
            "validation_positive_checkpoints": int(target[validation_mask].sum()),
        },
        "auxiliary_audit": {"probe_train": train["auxiliary_audit"]},
        "best_epoch": int(best[1]),
        "history": history,
        "internal_validation": {
            "label_ap": safe_ap_auc(target[validation_mask], internal_scores)[0],
            "label_auc": safe_ap_auc(target[validation_mask], internal_scores)[1],
            "empirical_B": internal_results,
            "curve": internal_curve,
        },
        "screen_only": bool(args.screen_only),
        "reproducibility": {
            "settings": reproducibility,
            "code": code_identity,
            "environment": runtime_identity,
            "input": input_identity,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": final_state_sha256,
            "probe_train_scores_sha256": sha256_array(train_scores),
        },
    }

    saved_scores = {
        "probe_train": torch.from_numpy(train_scores.astype(np.float32))
    }
    saved_keys = {
        "probe_train": {
            "problem_ids": frame.problem_id.astype(str).tolist(),
            "checkpoints": frame.checkpoint.astype(int).tolist(),
        }
    }
    if not args.screen_only:
        if args.heldout_root is None:
            raise ValueError("full evaluation requires --heldout-root")
        calibration = load_split(
            args.raw_root,
            "calibration",
            schedule=args.schedule,
            layer=args.layer,
            representation_kind=args.representation_kind,
            feature_kind=args.feature_kind,
            readout_kind=args.readout_kind,
            auxiliary_root=args.aux_raw_root,
        )
        heldout = load_split(
            args.heldout_root,
            "heldout",
            schedule=args.schedule,
            layer=args.layer,
            representation_kind=args.representation_kind,
            feature_kind=args.feature_kind,
            readout_kind=args.readout_kind,
            auxiliary_root=args.aux_heldout_root,
        )
        calibration_features = transform_split(calibration, transform)
        heldout_features = transform_split(heldout, transform)
        calibration_scores = predict_scores(model, calibration_features, device)
        heldout_scores = predict_scores(model, heldout_features, device)
        calibration_curve = policy_curve(
            calibration["frame"],
            calibration_scores,
            grid_size=int(config["calibration"]["quantile_grid_size"]),
            readout_suffix_tokens=readout_suffix_tokens,
            fallbacks=calibration["fallbacks"],
        )
        budgets = [
            int(value)
            for value in config["calibration"]["empirical_lost_correct_budgets"]
        ]
        selected = {
            str(budget): choose_empirical_budget(
                calibration_curve,
                budget,
                float(config["dynamic_policy"]["accuracy_epsilon"]),
            )
            for budget in budgets
        }
        frozen_results = {}
        frozen_records = {}
        for budget, choice in selected.items():
            result = replay_policy(
                heldout["frame"],
                heldout_scores,
                float(choice["threshold"]),
                force_dense=bool(choice["is_no_stop_sentinel"]),
                readout_suffix_tokens=readout_suffix_tokens,
                fallbacks=heldout["fallbacks"],
                include_records=True,
            )
            frozen_records[budget] = result.pop("records")
            frozen_results[budget] = {
                "calibration": choice,
                "heldout": result,
            }
        payload["calibration"] = {
            "selection_objective": "token-only including one-step suffix prefill when used",
            "curve": calibration_curve,
            "empirical_B": selected,
        }
        payload["heldout"] = {
            "label_ap": safe_ap_auc(
                ((~heldout["frame"].current_success.astype(bool)) & heldout["frame"].dense_success.astype(bool)).to_numpy(np.float32),
                heldout_scores,
            )[0],
            "label_auc": safe_ap_auc(
                ((~heldout["frame"].current_success.astype(bool)) & heldout["frame"].dense_success.astype(bool)).to_numpy(np.float32),
                heldout_scores,
            )[1],
            "empirical_B": frozen_results,
        }
        payload["auxiliary_audit"].update(
            {
                "calibration": calibration["auxiliary_audit"],
                "heldout": heldout["auxiliary_audit"],
            }
        )
        saved_scores.update(
            {
                "calibration": torch.from_numpy(calibration_scores.astype(np.float32)),
                "heldout": torch.from_numpy(heldout_scores.astype(np.float32)),
            }
        )
        saved_keys.update(
            {
                "calibration": {
                    "problem_ids": calibration["frame"].problem_id.astype(str).tolist(),
                    "checkpoints": calibration["frame"].checkpoint.astype(int).tolist(),
                },
                "heldout": {
                    "problem_ids": heldout["frame"].problem_id.astype(str).tolist(),
                    "checkpoints": heldout["frame"].checkpoint.astype(int).tolist(),
                },
            }
        )
        ood_records = {}
        if args.ood_root is not None:
            ood = load_split(
                args.ood_root,
                "heldout",
                schedule=args.schedule,
                layer=args.layer,
                representation_kind=args.representation_kind,
                feature_kind=args.feature_kind,
                readout_kind=args.readout_kind,
                auxiliary_root=args.aux_ood_root,
            )
            ood_features = transform_split(ood, transform)
            ood_scores = predict_scores(model, ood_features, device)
            ood_results = {}
            for budget, choice in selected.items():
                result = replay_policy(
                    ood["frame"],
                    ood_scores,
                    float(choice["threshold"]),
                    force_dense=bool(choice["is_no_stop_sentinel"]),
                    readout_suffix_tokens=readout_suffix_tokens,
                    fallbacks=ood["fallbacks"],
                    include_records=True,
                )
                ood_records[budget] = result.pop("records")
                ood_results[budget] = result
            payload["ood"] = {
                "mode": "frozen MATH probe and calibration threshold; no retraining/recalibration",
                "empirical_B": ood_results,
            }
            payload["auxiliary_audit"]["ood"] = ood["auxiliary_audit"]
            saved_scores["ood"] = torch.from_numpy(ood_scores.astype(np.float32))
            saved_keys["ood"] = {
                "problem_ids": ood["frame"].problem_id.astype(str).tolist(),
                "checkpoints": ood["frame"].checkpoint.astype(int).tolist(),
            }
        atomic_torch_save(
            {
                "status": "complete",
                "records": {"heldout": frozen_records, "ood": ood_records},
            },
            args.output / "policy_records.pt",
        )

    atomic_torch_save(
        {
            "status": "complete",
            "state_dict": best[2],
            "reproducibility": payload["reproducibility"],
            "feature_transform": transform.state_dict(),
            "probe_architecture": args.probe_architecture,
            "probe_layer_widths": model.layer_widths,
            "representation_kind": args.representation_kind,
            "layer": args.layer,
            "readout_suffix_tokens": readout_suffix_tokens,
            "invocation_fingerprint": invocation_fingerprint,
        },
        args.output / "probe.pt",
    )
    atomic_torch_save(
        {"status": "complete", "scores": saved_scores, "keys": saved_keys},
        args.output / "scores.pt",
    )
    atomic_json(payload, args.output / "probe.json")
    atomic_json(
        {
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "invocation_fingerprint": invocation_fingerprint,
            "best_epoch": int(best[1]),
            "artifacts": ["probe.json", "probe.pt", "scores.pt"],
        },
        marker,
    )
    output_lock.close()
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "best_epoch": int(best[1]),
                "feature_report": feature_report,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

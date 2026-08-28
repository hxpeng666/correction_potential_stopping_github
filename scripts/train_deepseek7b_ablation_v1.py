#!/usr/bin/env python3
"""Train the five controlled DeepSeek-7B probes with normalized trajectory loss."""
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
import torch
from sklearn.preprocessing import StandardScaler

from src.final_paper_inference import atomic_torch_save
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
    predict_scores,
    problem_batches,
    safe_ap_auc,
    select_empirical_budget,
    simulate_policy,
    target_values,
    threshold_grid,
)
from src.final_paper_protocol import canonical_fingerprint
from src.reproducibility import (
    code_provenance,
    enforce_runtime_lock,
    environment_provenance,
    sha256_array,
    sha256_json,
    sha256_state_dict,
    strict_reproducibility,
)
from src.utils import atomic_json, load_yaml
from deepseek7b_protocol_v1 import success as answer_equivalent


def build_dynamic_features(frame, hidden, capture_layers, *, layer, feature_kind):
    """Build the deployed h + six scalar features for any decoder width."""
    if feature_kind != "full_no_delta":
        raise ValueError(feature_kind)
    if layer not in capture_layers:
        raise ValueError(f"layer {layer} absent from {capture_layers}")
    current = hidden[:, capture_layers.index(layer), :].astype(np.float32, copy=False)
    delta = np.zeros_like(current, dtype=np.float32)
    delta_t = np.zeros((len(frame), 1), dtype=np.float32)
    for _, group in frame.groupby("problem_id", sort=False):
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
    values = np.concatenate(
        [current, checkpoint, log_checkpoint, delta_t, entropy, delta_norm, cosine], axis=1
    ).astype(np.float32, copy=False)
    expected = current.shape[1] + 6
    if values.shape != (len(frame), expected) or not np.isfinite(values).all():
        raise ValueError(f"invalid feature shape/values: {values.shape}, expected {(len(frame), expected)}")
    return values


def apply_semantic_answer_targets(frame):
    """Use dataset-aware answer equivalence for consistency and last-switch.

    The legacy GSM8K path compared normalized strings.  That is insufficient for
    MATH, where equivalent LaTeX forms are common.  Missing-vs-missing remains a
    stable state for last-switch, but is never a positive consistency label.
    """
    result = frame.copy()

    def equivalent(dataset, left, right, *, missing_equal=False):
        left_missing = left is None or (isinstance(left, float) and np.isnan(left))
        right_missing = right is None or (isinstance(right, float) and np.isnan(right))
        if left_missing or right_missing:
            return bool(missing_equal and left_missing and right_missing)
        return answer_equivalent(str(dataset), str(left), str(right))

    result["target_consistency"] = [
        equivalent(dataset, current, dense)
        for dataset, current, dense in zip(
            result.dataset, result.current_prediction, result.dense_prediction
        )
    ]
    result["target_last_switch"] = False
    for _problem_id, group in result.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        dataset = str(ordered.iloc[0].dataset)
        sequence = ordered.current_prediction.tolist() + [ordered.iloc[0].dense_prediction]
        final_switch = max(
            (
                index
                for index in range(len(sequence) - 1)
                if not equivalent(
                    dataset, sequence[index], sequence[index + 1], missing_equal=True
                )
            ),
            default=-1,
        )
        result.loc[ordered.index, "target_last_switch"] = [
            index > final_switch for index in range(len(ordered))
        ]
    return result


def artifact_manifest(root: Path) -> dict[str, Any]:
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
                f"{artifact.get('protocol_fingerprint')}:{fingerprint}\n"
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
    parser.add_argument(
        "--dataset", choices=("gsm8k", "math", "math500", "aime"), required=True
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--schedule", choices=SCHEDULES, default="sentence")
    parser.add_argument("--actual-schedule-label")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--feature-kind", choices=("full_no_delta",), default="full_no_delta")
    parser.add_argument("--loss", choices=("bce", "bce_traj"), default="bce_traj")
    parser.add_argument(
        "--trajectory-aggregation",
        choices=("unnormalized_softmin", "normalized_softmin"),
        default="unnormalized_softmin",
    )
    parser.add_argument("--trajectory-beta", type=float)
    parser.add_argument("--trajectory-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-unlocked-legacy",
        action="store_true",
        help=(
            "Explicitly run a historical config without a committed runtime lock. "
            "Such output is non-formal and must not be mixed with locked results."
        ),
    )
    args = parser.parse_args()

    if args.seed != 0:
        raise ValueError("legacy-v4 probe initialization/training seed must be 0")

    if args.method != "correction" and args.loss != "bce":
        raise ValueError("controlled target baselines use checkpoint BCE only")
    config = load_yaml(args.config)
    reproducibility = strict_reproducibility(seed=0, num_threads=1)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    runtime_identity = environment_provenance(device)
    runtime_lock_value = config.get("reproducibility", {}).get("runtime_lock")
    if runtime_lock_value is None and not args.allow_unlocked_legacy:
        raise RuntimeError(
            "formal probe training requires reproducibility.runtime_lock; use a "
            "committed locked config, or pass --allow-unlocked-legacy only to "
            "inspect historical protocols"
        )
    runtime_lock_audit = None
    if runtime_lock_value is not None:
        runtime_lock_path = Path(runtime_lock_value)
        if not runtime_lock_path.is_absolute():
            runtime_lock_path = ROOT / runtime_lock_path
        runtime_lock_audit = enforce_runtime_lock(runtime_lock_path, runtime_identity)
    code_identity = code_provenance(
        ROOT,
        (
            "scripts/train_deepseek7b_ablation_v1.py",
            "scripts/deepseek7b_protocol_v1.py",
            "src/reproducibility.py",
            "src/legacy_empirical_probe_normalized_v1.py",
            "src/final_paper_inference.py",
            "src/final_paper_protocol.py",
        ),
    )
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
        "feature_kind": args.feature_kind,
        "loss": args.loss,
        "trajectory_aggregation": args.trajectory_aggregation,
        "trajectory_beta": trajectory_beta,
        "trajectory_weight": float(args.trajectory_weight),
        "raw_input": artifact_manifest(args.raw_root),
        "heldout_input": artifact_manifest(args.heldout_root) if args.heldout_root else None,
        "probe_config": config["probe"],
        "calibration_config": config["calibration"],
        "reproducibility_protocol": reproducibility,
        "runtime_lock": runtime_lock_audit,
        "code_identity": code_identity,
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

    frames: dict[str, Any] = {}
    features: dict[str, np.ndarray] = {}
    fallbacks: dict[str, list[dict[str, Any]]] = {}
    train_frame, train_hidden, capture_layers, train_fallbacks = load_checkpoint_split(
        args.raw_root / "probe_train", args.schedule
    )
    if args.method in {"consistency", "last_switch"}:
        train_frame = apply_semantic_answer_targets(train_frame)
    frames["probe_train"] = train_frame
    fallbacks["probe_train"] = train_fallbacks
    raw_train = build_dynamic_features(
        train_frame,
        train_hidden,
        capture_layers,
        layer=args.layer,
        feature_kind=args.feature_kind,
    )
    del train_hidden
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
    scaler = StandardScaler(copy=False)
    scaler.fit(raw_train[fit_mask])
    features["probe_train"] = scaler.transform(raw_train).astype(np.float32, copy=False)
    del raw_train

    for split in ("calibration", "heldout"):
        split_root = args.heldout_root if split == "heldout" and args.heldout_root else args.raw_root
        frame, hidden, layers, split_fallbacks = load_checkpoint_split(
            split_root / split, args.schedule
        )
        if args.method in {"consistency", "last_switch"}:
            frame = apply_semantic_answer_targets(frame)
        if layers != capture_layers:
            raise ValueError(f"capture layers differ for {split}: {layers}")
        raw = build_dynamic_features(
            frame,
            hidden,
            layers,
            layer=args.layer,
            feature_kind=args.feature_kind,
        )
        del hidden
        frames[split] = frame
        fallbacks[split] = split_fallbacks
        features[split] = scaler.transform(raw).astype(np.float32, copy=False)
        del raw

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
    input_identity = {
        split: {
            "row_keys_sha256": sha256_json(
                list(
                    zip(
                        frame.problem_id.astype(str).tolist(),
                        frame.checkpoint.astype(int).tolist(),
                    )
                )
            ),
            "features_sha256": sha256_array(features[split]),
            "labels_sha256": sha256_array(labels[split]),
        }
        for split, frame in frames.items()
    }
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

    width = int(features["probe_train"].shape[1])
    if args.feature_kind == "full_no_delta" and width != 3590:
        raise ValueError(f"DeepSeek full_no_delta input width must be 3590, got {width}")
    model = FinalPaperProbe(width).to(device)
    initial_state_sha256 = sha256_state_dict(model.state_dict())
    probe_config = config["probe"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(probe_config["learning_rate"]),
        weight_decay=float(probe_config["weight_decay"]),
        foreach=False,
        fused=False,
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
            values = torch.from_numpy(features["probe_train"][positions]).to(device)
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
                model.parameters(), float(probe_config["gradient_clip"]), foreach=False
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            point_losses.append(float(point.detach().cpu()))
            trajectory_losses.append(float(trajectory.detach().cpu()))

        validation_scores = predict_scores(
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
    scores = {
        split: predict_scores(model, values, device)
        for split, values in features.items()
    }
    final_state_sha256 = sha256_state_dict(best[2])
    score_sha256 = {split: sha256_array(value) for split, value in scores.items()}
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
        "feature_kind": args.feature_kind,
        "loss": args.loss,
        "architecture": [width, 384, 96, 1],
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
        "scaler_fit_scope": "probe_train_fit_only",
        "cost_protocol": "reasoning_tokens_only; short_answer_cost=0",
        "calibration_accuracy_epsilon": float(config["dynamic_policy"]["accuracy_epsilon"]),
        "determinism_protocol_id": reproducibility["protocol_id"],
        "git_commit": code_identity["git"]["commit"],
    }
    payload = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_spec": run_spec,
        "run_spec_fingerprint": canonical_fingerprint(run_spec),
        "reproducibility": {
            "settings": reproducibility,
            "code": code_identity,
            "environment": runtime_identity,
            "runtime_lock": runtime_lock_audit,
            "input": input_identity,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": final_state_sha256,
            "score_sha256": score_sha256,
        },
        "input": artifact_manifest(args.raw_root),
        "heldout_input": artifact_manifest(args.heldout_root) if args.heldout_root else None,
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
            "scaler_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
            "scaler_scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
            "online_workpoints": online_workpoints,
            "reproducibility": {
                "code": code_identity,
                "settings": reproducibility,
                "environment": runtime_identity,
                "runtime_lock": runtime_lock_audit,
                "input": input_identity,
                "initial_state_sha256": initial_state_sha256,
                "final_state_sha256": final_state_sha256,
                "score_sha256": score_sha256,
            },
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

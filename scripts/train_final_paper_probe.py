#!/usr/bin/env python3
"""Train and calibrate one final-paper stopper target/ablation configuration."""
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
from src.final_paper_probe import (
    FEATURE_KINDS,
    FinalPaperProbe,
    METHODS,
    SCHEDULES,
    binary_point_loss,
    build_features,
    calibrate_policies,
    correction_loss,
    fit_validation_masks,
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
from src.utils import atomic_json, load_yaml, seed_everything


def artifact_manifest(root: Path) -> dict[str, Any]:
    paths = sorted(root.glob("*/sample_*.pt"))
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root)
        digest.update(f"{relative}:{path.stat().st_size}\n".encode("utf-8"))
    return {
        "root": str(root.resolve()),
        "files": len(paths),
        "name_size_fingerprint": digest.hexdigest(),
    }


def strip_records(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = dict(payload)
    records = summary.pop("records")
    return summary, records


def evaluate_frozen(
    frame,
    scores: np.ndarray,
    direction: str,
    calibrated: dict[str, Any],
    fallback_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluated: dict[str, Any] = {}
    records: dict[str, Any] = {}
    for family in ("empirical", "formal", "coverage"):
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
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--schedule", choices=SCHEDULES, default="sentence")
    parser.add_argument("--layer", type=int, choices=(8, 20, 35), default=20)
    parser.add_argument("--feature-kind", choices=FEATURE_KINDS, default="full")
    parser.add_argument("--loss", choices=("bce", "bce_traj"), default="bce_traj")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.method != "correction" and args.loss != "bce":
        raise ValueError("controlled target baselines use checkpoint BCE only")
    config = load_yaml(args.config)
    destination = args.output if args.output.is_absolute() else ROOT / args.output
    complete_path = destination / "phase.complete"
    if args.resume and complete_path.is_file():
        marker = json.loads(complete_path.read_text(encoding="utf-8"))
        if marker.get("status") == "complete" and (destination / "probe.pt").is_file():
            print(json.dumps({"status": "skipped_complete", "output": str(destination)}))
            return
    destination.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    torch.set_num_threads(min(16, torch.get_num_threads()))

    frames: dict[str, Any] = {}
    features: dict[str, np.ndarray] = {}
    fallbacks: dict[str, list[dict[str, Any]]] = {}
    train_frame, train_hidden, capture_layers, train_fallbacks = load_checkpoint_split(
        args.raw_root / "probe_train", args.schedule
    )
    frames["probe_train"] = train_frame
    fallbacks["probe_train"] = train_fallbacks
    raw_train = build_features(
        train_frame,
        train_hidden,
        capture_layers,
        layer=args.layer,
        feature_kind=args.feature_kind,
    )
    del train_hidden
    fit_mask, validation_mask = fit_validation_masks(
        train_frame, args.dataset, seed=20260803
    )
    scaler = StandardScaler(copy=False)
    scaler.fit(raw_train[fit_mask])
    features["probe_train"] = scaler.transform(raw_train).astype(np.float32, copy=False)
    del raw_train

    for split in ("calibration", "heldout"):
        frame, hidden, layers, split_fallbacks = load_checkpoint_split(
            args.raw_root / split, args.schedule
        )
        if layers != capture_layers:
            raise ValueError(f"capture layers differ for {split}: {layers}")
        raw = build_features(
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

    width = int(features["probe_train"].shape[1])
    if args.feature_kind == "full" and width != 5126:
        raise ValueError(f"full input width must be 5126, got {width}")
    model = FinalPaperProbe(width).to(device)
    probe_config = config["probe"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(probe_config["learning_rate"]),
        weight_decay=float(probe_config["weight_decay"]),
    )
    maximum_epochs = int(args.epochs or probe_config["max_epochs"])
    patience_limit = int(probe_config["patience"])
    rng = random.Random(args.seed)
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
                    beta=float(probe_config["trajectory_softmin_beta"]),
                    trajectory=args.loss == "bce_traj",
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
                    disabled,
                )
                for value, disabled in threshold_grid(
                    validation_scores, direction, grid_size=21
                )
            ]
            internal_curve = []
            for threshold, disabled in internal_grid:
                row = simulate_policy(
                    validation_frame, validation_scores, direction, threshold
                )
                row["disabled"] = disabled
                internal_curve.append(row)
            strict = select_empirical_budget(
                internal_curve,
                0.0,
                int(validation_frame.problem_id.nunique()),
            )
            key = (
                strict["replay_wall_reduction"],
                validation_ap,
                validation_auc,
                -float(np.mean(losses)),
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
    calibration_config = config["calibration"]
    calibrated = calibrate_policies(
        frames["calibration"],
        scores["calibration"],
        direction,
        grid_size=int(calibration_config["quantile_grid_size"]),
        confidence=float(calibration_config["confidence"]),
        empirical_percent=[
            float(value)
            for value in calibration_config["empirical_lost_correct_percent"]
        ],
        formal_alpha=[
            float(value)
            for value in calibration_config["simultaneous_upper_bound_alpha"]
        ],
        fallback_records=fallbacks["calibration"],
        coverage_targets=[
            float(value) for value in calibration_config["coverage_targets"]
        ],
    )
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
    fit_ids = sorted(
        train_frame.loc[fit_mask, "problem_id"].astype(str).unique().tolist()
    )
    validation_ids = sorted(
        train_frame.loc[validation_mask, "problem_id"].astype(str).unique().tolist()
    )
    online_workpoints = {}
    for name, selection in calibration_config["online_workpoints"].items():
        family = str(selection["family"])
        key = str(selection["key"])
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
        "seed": args.seed,
        "dense_seed": int(config["seed"]),
        "schedule": args.schedule,
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
        "trajectory_softmin_beta": float(probe_config["trajectory_softmin_beta"]),
        "fit_split_seed": 20260803,
        "fit_fraction": 0.8,
        "scaler_fit_scope": "probe_train_fit_only",
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
            "scaler_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
            "scaler_scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
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

#!/usr/bin/env python3
"""训练四状态概率 MLP，并按固定 P(WC)-P(CW) 规则回放。"""
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
from src.final_paper_protocol import canonical_fingerprint
from src.four_state_utility_probe_v1 import (
    STATE_NAMES,
    FourStateUtilityProbe,
    four_state_targets,
    legacy_weighted_multiclass_trajectory_loss,
    multiclass_point_loss,
    predict_probabilities,
    probability_diagnostics,
    simulate_zero_utility_policy,
)
from src.legacy_empirical_probe_v4 import (
    build_features,
    fit_validation_masks,
    fit_validation_problem_ids,
    load_checkpoint_split,
    problem_batches,
)
from src.utils import atomic_json, load_yaml, seed_everything


def artifact_manifest(root: Path) -> dict[str, Any]:
    paths = sorted(root.glob("*/sample_*.pt"))
    digest = hashlib.sha256()
    fingerprints: set[str] = set()
    for path in paths:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        view = str(artifact.get("primary_replay_view_fingerprint"))
        fingerprints.add(view)
        digest.update((
            f"{path.relative_to(root)}:{path.stat().st_size}:{artifact.get('problem_id')}:"
            f"{artifact.get('dataset')}:{artifact.get('split')}:{artifact.get('dtype')}:"
            f"{artifact.get('protocol_fingerprint')}:{view}\n"
        ).encode("utf-8"))
    return {
        "root": str(root.resolve()),
        "files": len(paths),
        "artifact_identity_fingerprint": digest.hexdigest(),
        "primary_replay_view_fingerprints": sorted(fingerprints),
    }


def strip_records(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = dict(payload)
    records = result.pop("records")
    return result, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--loss",
        choices=("unweighted_ce", "legacy_weighted_ce_traj"),
        default="unweighted_ce",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("为与既有实验配对，probe seed 固定为0")

    config = load_yaml(args.config)
    dataset_config = config["datasets"][args.dataset]
    destination = args.output if args.output.is_absolute() else ROOT / args.output
    invocation = {
        "protocol_id": config["protocol_id"],
        "dataset": args.dataset,
        "raw_input": artifact_manifest(args.raw_root),
        "probe_seed": args.seed,
        "probe": config["probe"],
        "decision": config["decision"],
        "loss": args.loss,
    }
    invocation_fingerprint = canonical_fingerprint(invocation)
    marker = destination / "phase.complete"
    if args.resume and marker.is_file():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("invocation_fingerprint") == invocation_fingerprint:
            print(json.dumps({"status": "skipped_complete", "output": str(destination)}))
            return
        raise RuntimeError(f"拒绝 resume 不同指纹输出：{destination}")
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"拒绝覆盖既有输出：{destination}")
    destination.mkdir(parents=True, exist_ok=True)

    seed_everything(0)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    torch.set_num_threads(min(16, torch.get_num_threads()))

    frames: dict[str, Any] = {}
    features: dict[str, np.ndarray] = {}
    fallbacks: dict[str, list[dict[str, Any]]] = {}
    train_frame, train_hidden, capture_layers, train_fallbacks = load_checkpoint_split(
        args.raw_root / "probe_train", "sentence"
    )
    frames["probe_train"] = train_frame
    fallbacks["probe_train"] = train_fallbacks
    raw_train = build_features(train_frame, train_hidden, capture_layers, layer=20, feature_kind="full")
    del train_hidden
    fallback_ids = [row["problem_id"] for row in train_fallbacks]
    fit_ids, validation_ids = fit_validation_problem_ids(
        train_frame, args.dataset, seed=0, additional_problem_ids=fallback_ids
    )
    fit_mask, validation_mask = fit_validation_masks(
        train_frame, args.dataset, seed=0, additional_problem_ids=fallback_ids
    )
    validation_fallbacks = [row for row in train_fallbacks if row["problem_id"] in validation_ids]
    scaler = StandardScaler(copy=False)
    scaler.fit(raw_train[fit_mask])
    features["probe_train"] = scaler.transform(raw_train).astype(np.float32, copy=False)
    del raw_train

    for split in ("calibration", "heldout"):
        frame, hidden, layers, local_fallbacks = load_checkpoint_split(args.raw_root / split, "sentence")
        if layers != capture_layers:
            raise ValueError(f"{split} capture layers 与训练集不一致")
        raw = build_features(frame, hidden, layers, layer=20, feature_kind="full")
        del hidden
        frames[split] = frame
        fallbacks[split] = local_fallbacks
        features[split] = scaler.transform(raw).astype(np.float32, copy=False)
        del raw

    expected_counts = {
        "probe_train": int(dataset_config["probe_train"]),
        "calibration": int(dataset_config["calibration"]),
        "heldout": int(dataset_config["heldout"]),
    }
    for split, expected in expected_counts.items():
        actual = int(frames[split].problem_id.nunique()) + len(fallbacks[split])
        if actual != expected:
            raise ValueError(f"{args.dataset}/{split} 样本数 {actual} != {expected}")

    labels = {split: four_state_targets(frame) for split, frame in frames.items()}
    remaining = np.clip(
        (
            train_frame.dense_tokens.to_numpy(dtype=np.float32)
            - train_frame.checkpoint.to_numpy(dtype=np.float32)
        ) / np.maximum(train_frame.dense_tokens.to_numpy(dtype=np.float32), 1.0),
        0.0,
        1.0,
    ).astype(np.float32)
    width = int(features["probe_train"].shape[1])
    if width != 5126:
        raise ValueError(f"完整特征维度必须为5126，实际为{width}")
    model = FourStateUtilityProbe(width).to(device)
    probe = config["probe"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(probe["learning_rate"]),
        weight_decay=float(probe["weight_decay"]),
    )
    maximum_epochs = int(args.epochs or probe["max_epochs"])
    patience_limit = int(probe["patience"])
    rng = random.Random(0)
    history: list[dict[str, Any]] = []
    best: tuple[tuple[float, ...], int, dict[str, torch.Tensor]] | None = None
    patience = 0
    validation_frame = train_frame.loc[validation_mask].reset_index(drop=True)

    for epoch in range(maximum_epochs):
        model.train()
        losses = []
        point_losses = []
        trajectory_losses = []
        for positions, offsets in problem_batches(
            train_frame, fit_mask, int(probe["trajectory_batch_size"]), rng
        ):
            values = torch.from_numpy(features["probe_train"][positions]).to(device)
            targets = torch.from_numpy(labels["probe_train"][positions]).to(device)
            logits = model(values)
            if args.loss == "legacy_weighted_ce_traj":
                loss, point_loss, trajectory_loss = legacy_weighted_multiclass_trajectory_loss(
                    logits,
                    targets,
                    torch.from_numpy(remaining[positions]).to(device),
                    offsets,
                    beta=float(probe.get("trajectory_softmin_beta", 0.5)),
                )
            else:
                loss = multiclass_point_loss(logits, targets)
                point_loss = loss
                trajectory_loss = loss.detach() * 0.0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(probe["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            point_losses.append(float(point_loss.detach().cpu()))
            trajectory_losses.append(float(trajectory_loss.detach().cpu()))

        probabilities = predict_probabilities(
            model, features["probe_train"][validation_mask], device
        )
        diagnostics = probability_diagnostics(labels["probe_train"][validation_mask], probabilities)
        validation_policy = simulate_zero_utility_policy(
            validation_frame, probabilities, fallback_records=validation_fallbacks
        )
        # Epoch 只由 probe-train 内部验证集的四分类 NLL 选择；不使用 policy calibration/test。
        key = (-diagnostics["cross_entropy"], diagnostics["accuracy"], -float(np.mean(losses)))
        record = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "point_loss": float(np.mean(point_losses)),
            "trajectory_loss": float(np.mean(trajectory_losses)),
            "validation_cross_entropy": diagnostics["cross_entropy"],
            "validation_four_state_accuracy": diagnostics["accuracy"],
            "validation_policy_accuracy": validation_policy["accuracy"],
            "validation_policy_coverage": validation_policy["coverage"],
            "validation_policy_token_reduction": validation_policy["token_reduction"],
            "validation_policy_lost_correct_count": validation_policy["lost_correct_count"],
        }
        history.append(record)
        print(json.dumps({"dataset": args.dataset, **record}), flush=True)
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
        raise RuntimeError("没有产生可用训练 checkpoint")
    model.load_state_dict(best[2])
    probabilities = {
        split: predict_probabilities(model, values, device)
        for split, values in features.items()
    }
    probability_report = {
        split: probability_diagnostics(labels[split], probabilities[split])
        for split in frames
    }
    calibration_summary, calibration_records = strip_records(simulate_zero_utility_policy(
        frames["calibration"], probabilities["calibration"],
        include_records=True, fallback_records=fallbacks["calibration"]
    ))
    heldout_summary, heldout_records = strip_records(simulate_zero_utility_policy(
        frames["heldout"], probabilities["heldout"],
        include_records=True, fallback_records=fallbacks["heldout"]
    ))

    run_spec = {
        "dataset": args.dataset,
        "method": "four_state_utility_difference",
        "protocol_id": config["protocol_id"],
        "class_order": list(STATE_NAMES),
        "architecture": [5126, 384, 96, 4],
        "probability_transform": "softmax",
        "loss": args.loss,
        "trajectory_softmin_beta": float(probe.get("trajectory_softmin_beta", 0.5)),
        "decision_rule": "continue iff p_WC - p_CW >= 0; stop iff p_WC - p_CW < 0",
        "threshold_calibration": "none",
        "epoch_selection": "minimum_internal_validation_multiclass_cross_entropy",
        "probe_seed": 0,
        "internal_split_seed": 0,
        "schedule": "sentence",
        "layer": 20,
        "feature_kind": "full",
        "optimizer": "AdamW",
        "learning_rate": float(probe["learning_rate"]),
        "weight_decay": float(probe["weight_decay"]),
        "trajectory_batch_size": int(probe["trajectory_batch_size"]),
        "gradient_clip": float(probe["gradient_clip"]),
        "maximum_epochs": maximum_epochs,
        "patience": patience_limit,
        "scaler_fit_scope": "probe_train_internal_fit_only",
    }
    payload = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_spec": run_spec,
        "run_spec_fingerprint": canonical_fingerprint(run_spec),
        "input": invocation["raw_input"],
        "split_counts": {
            split: {
                "problems": int(frame.problem_id.nunique()) + len(fallbacks[split]),
                "scorable_problems": int(frame.problem_id.nunique()),
                "fallback_only_problems": len(fallbacks[split]),
                "checkpoints": len(frame),
                "four_state_counts": {
                    name: int((labels[split] == index).sum())
                    for index, name in enumerate(STATE_NAMES)
                },
            }
            for split, frame in frames.items()
        },
        "fit_problem_ids": sorted(fit_ids),
        "validation_problem_ids": sorted(validation_ids),
        "best_epoch": int(best[1]),
        "history": history,
        "probability_diagnostics": probability_report,
        "calibration_fixed_rule_diagnostic_only": calibration_summary,
        "heldout_fixed_rule_result": heldout_summary,
        "calibration_selected_anything": False,
    }
    atomic_torch_save({
        "status": "complete",
        "state_dict": best[2],
        "run_spec": run_spec,
        "capture_layers": capture_layers,
        "input_width": width,
        "scaler_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
        "scaler_scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
    }, destination / "probe.pt")
    atomic_torch_save({
        "status": "complete",
        "class_order": list(STATE_NAMES),
        "probabilities": {
            split: torch.from_numpy(value.astype(np.float32))
            for split, value in probabilities.items()
        },
        "problem_ids": {
            split: frame.problem_id.astype(str).tolist() for split, frame in frames.items()
        },
        "checkpoints": {
            split: frame.checkpoint.astype(int).tolist() for split, frame in frames.items()
        },
    }, destination / "probabilities.pt")
    atomic_torch_save({
        "status": "complete",
        "calibration": calibration_records,
        "heldout": heldout_records,
    }, destination / "policy_records.pt")
    atomic_json(payload, destination / "probe.json")
    atomic_json({
        "status": "complete",
        "invocation_fingerprint": invocation_fingerprint,
        "run_spec_fingerprint": payload["run_spec_fingerprint"],
        "best_epoch": int(best[1]),
        "artifacts": ["probe.json", "probe.pt", "probabilities.pt", "policy_records.pt"],
    }, marker)
    print(json.dumps({
        "status": "complete",
        "dataset": args.dataset,
        "best_epoch": int(best[1]),
        "heldout": heldout_summary,
        "output": str(destination),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

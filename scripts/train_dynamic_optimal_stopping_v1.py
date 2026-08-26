#!/usr/bin/env python3
"""训练风险约束动态最优停止器，并在calibration上冻结(lambda,mu)策略。"""
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
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from src.dynamic_optimal_stopping_v1 import (
    ContinuationValueBank,
    DYNAMIC_FEATURE_KINDS,
    DynamicLocalHeads,
    backward_value_targets,
    build_dynamic_features,
    clopper_pearson_upper,
    group_positions_offsets,
    predict_continuation_values,
    predict_local_heads,
    simulate_dynamic_policy,
    supervised_local_loss,
    one_step_value_targets,
    dense_endpoint_value_targets,
)
from src.final_paper_inference import atomic_torch_save
from src.final_paper_protocol import canonical_fingerprint
from src.legacy_empirical_probe_v4 import (
    fit_validation_masks,
    fit_validation_problem_ids,
    load_checkpoint_split,
    problem_batches,
    safe_ap_auc,
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


def local_validation(
    model: DynamicLocalHeads,
    features: np.ndarray,
    frame,
    stop_targets: np.ndarray,
    risk_targets: np.ndarray,
    remaining: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    config: dict[str, Any],
    gamma_trajectory_override: float | None = None,
) -> dict[str, float]:
    positions, offsets = group_positions_offsets(frame, mask)
    model.eval()
    with torch.no_grad():
        values = torch.from_numpy(features[positions]).to(device)
        _, stop_logits, risk_logits = model(values)
        total, parts = supervised_local_loss(
            stop_logits,
            risk_logits,
            torch.from_numpy(stop_targets[positions]).to(device),
            torch.from_numpy(risk_targets[positions]).to(device),
            torch.from_numpy(remaining[positions]).to(device),
            offsets,
            beta=float(config["trajectory_softmin_beta"]),
            gamma_risk=float(config["gamma_risk"]),
            gamma_trajectory=(
                float(config["gamma_trajectory"])
                if gamma_trajectory_override is None
                else float(gamma_trajectory_override)
            ),
        )
        stop_probability = torch.sigmoid(stop_logits).cpu().numpy()
        risk_probability = torch.sigmoid(risk_logits).cpu().numpy()
    stop_ap, stop_auc = safe_ap_auc(stop_targets[positions], stop_probability)
    risk_ap, risk_auc = safe_ap_auc(risk_targets[positions], risk_probability)
    return {
        "total": float(total.cpu()),
        "stop_bce": float(parts["stop_bce"].cpu()),
        "risk_point": float(parts["risk_point"].cpu()),
        "risk_trajectory": float(parts["risk_trajectory"].cpu()),
        "stop_ap": stop_ap,
        "stop_auc": stop_auc,
        "risk_ap": risk_ap,
        "risk_auc": risk_auc,
    }


def choose_empirical(curve: list[dict[str, Any]], dense: dict[str, Any], budget: int, epsilon: float) -> dict[str, Any]:
    feasible = [
        row for row in curve
        if row["lost_correct_count"] <= budget
        and row["accuracy"] >= dense["dense_accuracy"] - epsilon
    ]
    if not feasible:
        selected = dict(dense)
        selected.update({"selected_candidate": "dense", "dense_fallback": True})
    else:
        selected = dict(min(feasible, key=lambda row: (
            row["mean_reasoning_tokens"],
            -row["accuracy"],
            row["lost_correct_count"],
            row["candidate_index"],
        )))
        selected["dense_fallback"] = False
    selected["budget_B"] = int(budget)
    selected["accuracy_epsilon"] = float(epsilon)
    selected["selection_family"] = "empirical_B_with_accuracy_constraint"
    return selected


def choose_formal(
    curve: list[dict[str, Any]],
    dense: dict[str, Any],
    alpha: float,
    epsilon: float,
) -> dict[str, Any]:
    feasible = [
        row for row in curve
        if row["lost_correct_ucb_simultaneous95"] <= alpha
        and row["accuracy"] >= dense["dense_accuracy"] - epsilon
    ]
    if not feasible:
        selected = dict(dense)
        selected.update({
            "selected_candidate": "dense",
            "dense_fallback": True,
            "lost_correct_ucb_simultaneous95": 0.0,
        })
    else:
        selected = dict(min(feasible, key=lambda row: (
            row["mean_reasoning_tokens"],
            -row["accuracy"],
            row["lost_correct_count"],
            row["candidate_index"],
        )))
        selected["dense_fallback"] = False
    selected["alpha"] = float(alpha)
    selected["accuracy_epsilon"] = float(epsilon)
    selected["selection_family"] = "formal_simultaneous95_ucb_with_accuracy_constraint"
    return selected


def nonmonotonic_wcw_audit(frame, records_by_family: dict[str, Any]) -> dict[str, Any]:
    wcw_ids = set()
    for problem_id, group in frame.groupby("problem_id", sort=False):
        correctness = group.sort_values("checkpoint").current_success.astype(bool).tolist()
        if any((not correctness[index - 1]) and correctness[index] and (not correctness[index + 1]) for index in range(1, len(correctness) - 1)):
            wcw_ids.add(str(problem_id))
    result: dict[str, Any] = {"wcw_problem_count": len(wcw_ids), "policies": {}}
    for family, values in records_by_family.items():
        result["policies"][family] = {}
        for key, records in values.items():
            subset = [row for row in records if row["problem_id"] in wcw_ids]
            result["policies"][family][key] = {
                "wcw_problems": len(subset),
                "stopped": sum(not row["fallback"] for row in subset),
                "stopped_correct": sum((not row["fallback"]) and row["method_success"] for row in subset),
                "gained_correct_C_to_W": sum(row["transition"] == "C_to_W" for row in subset),
                "lost_correct_W_to_C": sum(row["transition"] == "W_to_C" for row in subset),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--internal-split-seed", type=int)
    parser.add_argument("--batch-seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--feature-kind", choices=DYNAMIC_FEATURE_KINDS, default="full")
    parser.add_argument(
        "--variant",
        choices=("full", "no_trajectory", "one_step_value", "dense_endpoint_value"),
        default="full",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        raise ValueError("probe seed必须是非负整数")
    internal_split_seed = args.seed if args.internal_split_seed is None else args.internal_split_seed
    batch_seed = args.seed if args.batch_seed is None else args.batch_seed
    if internal_split_seed < 0 or batch_seed < 0:
        raise ValueError("内部划分和batch seed必须是非负整数")
    config = load_yaml(args.config)
    dataset_config = config["datasets"][args.dataset]
    dynamic = config["dynamic_policy"]
    probe_config = config["probe"]
    lambdas = np.asarray([float(value) for value in dynamic["lambda_grid"] for _ in dynamic["mu_grid"]], dtype=np.float64)
    mus = np.asarray([float(value) for _ in dynamic["lambda_grid"] for value in dynamic["mu_grid"]], dtype=np.float64)
    candidates = [
        {"candidate_index": index, "lambda": float(lambdas[index]), "mu": float(mus[index])}
        for index in range(len(lambdas))
    ]
    destination = args.output if args.output.is_absolute() else ROOT / args.output
    invocation = {
        "protocol_id": config["protocol_id"],
        "dataset": args.dataset,
        "raw_input": artifact_manifest(args.raw_root),
        "probe_seed": args.seed,
        "internal_split_seed": internal_split_seed,
        "batch_seed": batch_seed,
        "probe": probe_config,
        "dynamic_policy": dynamic,
        "candidate_grid": candidates,
        "variant": args.variant,
        "feature_kind": args.feature_kind,
    }
    invocation_fingerprint = canonical_fingerprint(invocation)
    marker = destination / "phase.complete"
    if args.resume and marker.is_file():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("invocation_fingerprint") == invocation_fingerprint:
            print(json.dumps({"status": "skipped_complete", "output": str(destination)}))
            return
        raise RuntimeError(f"拒绝resume不同指纹输出：{destination}")
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"拒绝覆盖既有输出：{destination}")
    destination.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
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
    raw_train = build_dynamic_features(
        train_frame, train_hidden, capture_layers, layer=20, feature_kind=args.feature_kind
    )
    del train_hidden
    fallback_ids = [row["problem_id"] for row in train_fallbacks]
    fit_ids, validation_ids = fit_validation_problem_ids(
        train_frame, args.dataset, seed=internal_split_seed, additional_problem_ids=fallback_ids
    )
    fit_mask, validation_mask = fit_validation_masks(
        train_frame, args.dataset, seed=internal_split_seed, additional_problem_ids=fallback_ids
    )
    scaler = StandardScaler(copy=False)
    scaler.fit(raw_train[fit_mask])
    features["probe_train"] = scaler.transform(raw_train).astype(np.float32, copy=False)
    del raw_train
    for split in ("calibration", "heldout"):
        frame, hidden, layers, local_fallbacks = load_checkpoint_split(args.raw_root / split, "sentence")
        if layers != capture_layers:
            raise ValueError(f"{split} capture layer不一致")
        raw = build_dynamic_features(
            frame, hidden, layers, layer=20, feature_kind=args.feature_kind
        )
        del hidden
        frames[split] = frame
        fallbacks[split] = local_fallbacks
        features[split] = scaler.transform(raw).astype(np.float32, copy=False)
        del raw
    expected = {
        "probe_train": int(dataset_config["probe_train"]),
        "calibration": int(dataset_config["calibration"]),
        "heldout": int(dataset_config["heldout"]),
    }
    for split, count in expected.items():
        actual = int(frames[split].problem_id.nunique()) + len(fallbacks[split])
        if actual != count:
            raise ValueError(f"{args.dataset}/{split}={actual}，预期{count}")
    feature_width = int(features["probe_train"].shape[1])
    if args.feature_kind == "full" and feature_width != 5126:
        raise ValueError("完整动态方法输入维度不是5126")

    stop_targets = {
        split: frame.current_success.to_numpy(dtype=np.float32)
        for split, frame in frames.items()
    }
    risk_targets = {
        split: ((~frame.current_success.astype(bool)) & frame.dense_success.astype(bool)).to_numpy(dtype=np.float32)
        for split, frame in frames.items()
    }
    remaining = np.clip(
        (
            train_frame.dense_tokens.to_numpy(dtype=np.float32)
            - train_frame.checkpoint.to_numpy(dtype=np.float32)
        ) / np.maximum(train_frame.dense_tokens.to_numpy(dtype=np.float32), 1.0),
        0.0,
        1.0,
    ).astype(np.float32)

    local_model = DynamicLocalHeads(feature_width).to(device)
    effective_gamma_trajectory = (
        0.0 if args.variant == "no_trajectory"
        else float(probe_config["gamma_trajectory"])
    )
    optimizer = torch.optim.AdamW(
        local_model.parameters(),
        lr=float(probe_config["learning_rate"]),
        weight_decay=float(probe_config["weight_decay"]),
    )
    maximum_epochs = int(args.epochs or probe_config["max_epochs"])
    patience_limit = int(probe_config["patience"])
    rng = random.Random(batch_seed)
    local_history = []
    local_best = None
    patience = 0
    for epoch in range(maximum_epochs):
        local_model.train()
        losses = []
        part_values = {key: [] for key in ("stop_bce", "risk_point", "risk_trajectory")}
        for positions, offsets in problem_batches(
            train_frame, fit_mask, int(probe_config["trajectory_batch_size"]), rng
        ):
            values = torch.from_numpy(features["probe_train"][positions]).to(device)
            _, stop_logits, risk_logits = local_model(values)
            loss, parts = supervised_local_loss(
                stop_logits,
                risk_logits,
                torch.from_numpy(stop_targets["probe_train"][positions]).to(device),
                torch.from_numpy(risk_targets["probe_train"][positions]).to(device),
                torch.from_numpy(remaining[positions]).to(device),
                offsets,
                beta=float(probe_config["trajectory_softmin_beta"]),
                gamma_risk=float(probe_config["gamma_risk"]),
                gamma_trajectory=effective_gamma_trajectory,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), float(probe_config["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for key in part_values:
                part_values[key].append(float(parts[key].detach().cpu()))
        validation = local_validation(
            local_model,
            features["probe_train"],
            train_frame,
            stop_targets["probe_train"],
            risk_targets["probe_train"],
            remaining,
            validation_mask,
            device,
            probe_config,
            effective_gamma_trajectory,
        )
        record = {
            "epoch": epoch,
            "training_total": float(np.mean(losses)),
            **{f"training_{key}": float(np.mean(values)) for key, values in part_values.items()},
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        local_history.append(record)
        print(json.dumps({"dataset": args.dataset, "stage": "local_heads", **record}), flush=True)
        key = (-validation["total"], validation["risk_ap"], validation["stop_ap"])
        if local_best is None or key > local_best[0]:
            local_best = (
                key,
                epoch,
                {name: value.detach().cpu().clone() for name, value in local_model.state_dict().items()},
            )
            patience = 0
        else:
            patience += 1
        if patience >= patience_limit:
            break
    if local_best is None:
        raise RuntimeError("local head训练未产生checkpoint")
    local_model.load_state_dict(local_best[2])

    representations: dict[str, np.ndarray] = {}
    stop_probability: dict[str, np.ndarray] = {}
    risk_probability: dict[str, np.ndarray] = {}
    for split in frames:
        representations[split], stop_probability[split], risk_probability[split] = predict_local_heads(
            local_model, features[split], device
        )

    cost_unit = float(dynamic["cost_unit_tokens"])
    if args.variant in {"full", "no_trajectory"}:
        value_targets, oracle_state_values = backward_value_targets(
            train_frame,
            stop_probability["probe_train"],
            risk_probability["probe_train"],
            lambdas,
            mus,
            cost_unit_tokens=cost_unit,
        )
        value_target_mode = "recursive_bellman"
        cost_mode = "incremental"
    elif args.variant == "one_step_value":
        value_targets = one_step_value_targets(
            train_frame,
            stop_probability["probe_train"],
            risk_probability["probe_train"],
            mus,
        )
        oracle_state_values = value_targets
        value_target_mode = "next_checkpoint_stop_only"
        cost_mode = "incremental"
    else:
        value_targets = dense_endpoint_value_targets(train_frame, len(candidates))
        oracle_state_values = value_targets
        value_target_mode = "dense_endpoint_correctness"
        cost_mode = "remaining_to_dense"
    value_model = ContinuationValueBank(len(candidates)).to(device)
    value_optimizer = torch.optim.AdamW(
        value_model.parameters(),
        lr=float(probe_config["learning_rate"]),
        weight_decay=float(probe_config["weight_decay"]),
    )
    value_rng = random.Random(batch_seed)
    value_history = []
    value_best = None
    patience = 0
    for epoch in range(maximum_epochs):
        value_model.train()
        losses = []
        for positions, _ in problem_batches(
            train_frame, fit_mask, int(probe_config["trajectory_batch_size"]), value_rng
        ):
            prediction = value_model(torch.from_numpy(representations["probe_train"][positions]).to(device))
            target = torch.from_numpy(value_targets[positions]).to(device)
            loss = F.smooth_l1_loss(prediction, target, beta=float(probe_config["value_huber_beta"]))
            value_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), float(probe_config["gradient_clip"]))
            value_optimizer.step()
            losses.append(float(loss.detach().cpu()))
        value_model.eval()
        with torch.no_grad():
            validation_prediction = value_model(
                torch.from_numpy(representations["probe_train"][validation_mask]).to(device)
            )
            validation_target = torch.from_numpy(value_targets[validation_mask]).to(device)
            validation_loss = float(F.smooth_l1_loss(
                validation_prediction,
                validation_target,
                beta=float(probe_config["value_huber_beta"]),
            ).cpu())
            validation_mae = float(torch.mean(torch.abs(validation_prediction - validation_target)).cpu())
        record = {
            "epoch": epoch,
            "training_huber": float(np.mean(losses)),
            "validation_huber": validation_loss,
            "validation_mae": validation_mae,
        }
        value_history.append(record)
        print(json.dumps({"dataset": args.dataset, "stage": "continuation_value", **record}), flush=True)
        key = (-validation_loss, -validation_mae)
        if value_best is None or key > value_best[0]:
            value_best = (
                key,
                epoch,
                {name: value.detach().cpu().clone() for name, value in value_model.state_dict().items()},
            )
            patience = 0
        else:
            patience += 1
        if patience >= patience_limit:
            break
    if value_best is None:
        raise RuntimeError("continuation value训练未产生checkpoint")
    value_model.load_state_dict(value_best[2])
    continuation_values = {
        split: predict_continuation_values(value_model, representation, device)
        for split, representation in representations.items()
    }

    calibration_curve = []
    for candidate in candidates:
        index = candidate["candidate_index"]
        row = simulate_dynamic_policy(
            frames["calibration"],
            stop_probability["calibration"],
            risk_probability["calibration"],
            continuation_values["calibration"][:, index],
            lambda_value=candidate["lambda"],
            mu_value=candidate["mu"],
            cost_unit_tokens=cost_unit,
            fallback_records=fallbacks["calibration"],
            cost_mode=cost_mode,
        )
        row.update(candidate)
        row["lost_correct_ucb_simultaneous95"] = clopper_pearson_upper(
            row["lost_correct_count"],
            row["problems"],
            float(dynamic["formal_delta"]) / len(candidates),
        )
        calibration_curve.append(row)
    dense_calibration = simulate_dynamic_policy(
        frames["calibration"],
        stop_probability["calibration"],
        risk_probability["calibration"],
        np.zeros(len(frames["calibration"]), dtype=np.float64),
        lambda_value=0.0,
        mu_value=0.0,
        cost_unit_tokens=cost_unit,
        fallback_records=fallbacks["calibration"],
        force_dense=True,
        cost_mode=cost_mode,
    )
    dense_calibration.update({
        "selected_candidate": "dense",
        "candidate_index": None,
        "lambda": None,
        "mu": None,
        "lost_correct_ucb_simultaneous95": 0.0,
        "structural_dense_risk": 0.0,
    })
    epsilon = float(dynamic["accuracy_epsilon"])
    selected = {
        "formal_alpha": {
            str(value): choose_formal(calibration_curve, dense_calibration, float(value), epsilon)
            for value in dynamic["formal_alpha"]
        },
        "empirical_B": {
            str(int(value)): choose_empirical(calibration_curve, dense_calibration, int(value), epsilon)
            for value in dynamic["empirical_B"]
        },
    }

    heldout_frontier = []
    heldout_cache: dict[int | str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for candidate in candidates:
        index = candidate["candidate_index"]
        summary, records = strip_records(simulate_dynamic_policy(
            frames["heldout"],
            stop_probability["heldout"],
            risk_probability["heldout"],
            continuation_values["heldout"][:, index],
            lambda_value=candidate["lambda"],
            mu_value=candidate["mu"],
            cost_unit_tokens=cost_unit,
            fallback_records=fallbacks["heldout"],
            include_records=True,
            cost_mode=cost_mode,
        ))
        summary.update(candidate)
        heldout_frontier.append(summary)
        heldout_cache[index] = (summary, records)
    dense_heldout_summary, dense_heldout_records = strip_records(simulate_dynamic_policy(
        frames["heldout"],
        stop_probability["heldout"],
        risk_probability["heldout"],
        np.zeros(len(frames["heldout"]), dtype=np.float64),
        lambda_value=0.0,
        mu_value=0.0,
        cost_unit_tokens=cost_unit,
        fallback_records=fallbacks["heldout"],
        include_records=True,
        force_dense=True,
        cost_mode=cost_mode,
    ))
    heldout_cache["dense"] = (dense_heldout_summary, dense_heldout_records)
    frozen_results: dict[str, Any] = {}
    policy_records: dict[str, Any] = {}
    for family, values in selected.items():
        frozen_results[family] = {}
        policy_records[family] = {}
        for key, calibration_selection in values.items():
            index = calibration_selection.get("candidate_index")
            cache_key: int | str = "dense" if calibration_selection.get("dense_fallback") else int(index)
            heldout_summary, records = heldout_cache[cache_key]
            frozen_results[family][key] = {
                "calibration": calibration_selection,
                "heldout": heldout_summary,
            }
            policy_records[family][key] = records

    local_diagnostics = {}
    for split in frames:
        stop_ap, stop_auc = safe_ap_auc(stop_targets[split], stop_probability[split])
        risk_ap, risk_auc = safe_ap_auc(risk_targets[split], risk_probability[split])
        local_diagnostics[split] = {
            "stop_correctness_AP": stop_ap,
            "stop_correctness_AUC": stop_auc,
            "lost_correct_risk_AP": risk_ap,
            "lost_correct_risk_AUC": risk_auc,
            "mean_stop_probability": float(stop_probability[split].mean()),
            "mean_risk_probability": float(risk_probability[split].mean()),
        }

    run_spec = {
        "dataset": args.dataset,
        "protocol_id": config["protocol_id"],
        "method": "risk_constrained_dynamic_optimal_stopping",
        "variant": args.variant,
        "architecture": {
            "shared": [feature_width, 384, 96],
            "heads": ["stop_correctness", "lost_correct_risk", f"continuation_value_bank_{len(candidates)}"],
        },
        "feature_kind": args.feature_kind,
        "feature_width": feature_width,
        "decision": "stop iff p_stop - mu*p_lost_correct >= -lambda*cost + M",
        "cost_mode": cost_mode,
        "cost": "reasoning_tokens/4096",
        "short_answer_cost": 0.0,
        "value_training": value_target_mode,
        "shared_representation_frozen_during_value_fit": True,
        "joint_finetuning": False,
        "candidate_grid": candidates,
        "probe_seed": args.seed,
        "internal_split_seed": internal_split_seed,
        "batch_seed": batch_seed,
        "heldout_selection": False,
        "risk_trajectory_enabled": args.variant != "no_trajectory",
        "effective_gamma_trajectory": effective_gamma_trajectory,
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
                "stop_correct_positive_rows": int(stop_targets[split].sum()),
                "lost_correct_positive_rows": int(risk_targets[split].sum()),
            }
            for split, frame in frames.items()
        },
        "fit_problem_ids": sorted(fit_ids),
        "validation_problem_ids": sorted(validation_ids),
        "local_best_epoch": int(local_best[1]),
        "value_best_epoch": int(value_best[1]),
        "local_history": local_history,
        "value_history": value_history,
        "local_diagnostics": local_diagnostics,
        "calibration": {
            "candidate_count": len(candidates),
            "formal_delta": float(dynamic["formal_delta"]),
            "per_candidate_delta": float(dynamic["formal_delta"]) / len(candidates),
            "accuracy_epsilon": epsilon,
            "curve": calibration_curve,
            "dense_sentinel": dense_calibration,
            "selected": selected,
        },
        "frozen_policy_results": frozen_results,
        "descriptive_heldout_frontier": heldout_frontier,
        "descriptive_heldout_nonmonotonic_wcw_audit": nonmonotonic_wcw_audit(
            frames["heldout"], policy_records
        ),
        "oracle_target_value_range": [float(oracle_state_values.min()), float(oracle_state_values.max())],
    }
    atomic_torch_save({
        "status": "complete",
        "local_state_dict": local_best[2],
        "value_state_dict": value_best[2],
        "run_spec": run_spec,
        "capture_layers": capture_layers,
        "scaler_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
        "scaler_scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
        "candidate_lambdas": torch.from_numpy(lambdas.astype(np.float32)),
        "candidate_mus": torch.from_numpy(mus.astype(np.float32)),
    }, destination / "probe.pt")
    atomic_torch_save({
        "status": "complete",
        "stop_probability": {split: torch.from_numpy(value.astype(np.float32)) for split, value in stop_probability.items()},
        "risk_probability": {split: torch.from_numpy(value.astype(np.float32)) for split, value in risk_probability.items()},
        "continuation_values": {split: torch.from_numpy(value.astype(np.float32)) for split, value in continuation_values.items()},
        "problem_ids": {split: frame.problem_id.astype(str).tolist() for split, frame in frames.items()},
        "checkpoints": {split: frame.checkpoint.astype(int).tolist() for split, frame in frames.items()},
    }, destination / "predictions.pt")
    atomic_torch_save({"status": "complete", "records": policy_records}, destination / "policy_records.pt")
    atomic_json(payload, destination / "probe.json")
    atomic_json({
        "status": "complete",
        "invocation_fingerprint": invocation_fingerprint,
        "run_spec_fingerprint": payload["run_spec_fingerprint"],
        "local_best_epoch": int(local_best[1]),
        "value_best_epoch": int(value_best[1]),
        "artifacts": ["probe.json", "probe.pt", "predictions.pt", "policy_records.pt"],
    }, marker)
    print(json.dumps({
        "status": "complete",
        "dataset": args.dataset,
        "output": str(destination),
        "selected": selected,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

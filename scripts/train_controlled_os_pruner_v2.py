#!/usr/bin/env python3
"""在公共缓存上训练 Matched / Constrained OS-Pruner 受控基线。"""
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

from src.dynamic_optimal_stopping_deployable_v2 import (
    OSPrunerPolicyBank,
    build_dynamic_features,
    candidate_feasibility_counts,
    clopper_pearson_upper,
    group_positions_offsets,
    os_pruner_expected_utility,
    predict_os_stop_probabilities,
    simulate_os_pruner_policy,
    summarize_expected_os_policy,
)
from src.final_paper_inference import atomic_torch_save
from src.final_paper_protocol import canonical_fingerprint
from src.legacy_empirical_probe_v4 import (
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
    return result, result.pop("records")


def choose_empirical(
    curve: list[dict[str, Any]], dense: dict[str, Any], budget: int, epsilon: float,
) -> dict[str, Any]:
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
            -row["token_reduction"],
            -row["coverage"],
            row["candidate_index"],
        )))
        selected["dense_fallback"] = False
    selected.update({
        "budget_B": int(budget),
        "accuracy_epsilon": float(epsilon),
        "selection_family": "empirical_B_with_accuracy_constraint",
    })
    return selected


def choose_formal(
    curve: list[dict[str, Any]], dense: dict[str, Any], alpha: float, epsilon: float,
) -> dict[str, Any]:
    feasible = [
        row for row in curve
        if row["lost_correct_ucb_simultaneous95"] <= alpha
        and row["accuracy"] >= dense["dense_accuracy"] - epsilon
    ]
    if not feasible:
        selected = dict(dense)
        selected.update({
            "selected_candidate": "dense", "dense_fallback": True,
            "lost_correct_ucb_simultaneous95": 0.0,
        })
    else:
        selected = dict(min(feasible, key=lambda row: (
            row["mean_reasoning_tokens"],
            -row["token_reduction"],
            -row["coverage"],
            row["candidate_index"],
        )))
        selected["dense_fallback"] = False
    selected.update({
        "alpha": float(alpha),
        "accuracy_epsilon": float(epsilon),
        "selection_family": "formal_simultaneous95_ucb_with_accuracy_constraint",
    })
    return selected


@torch.no_grad()
def validation_loss(
    model: OSPrunerPolicyBank,
    features: np.ndarray,
    frame,
    mask: np.ndarray,
    lambdas: torch.Tensor,
    mus: torch.Tensor,
    cost_unit: float,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    positions, offsets = group_positions_offsets(frame, mask)
    model.eval()
    logits = model(torch.from_numpy(features[positions]).to(device))
    loss, parts = os_pruner_expected_utility(
        logits, frame, positions, offsets, lambdas, mus,
        cost_unit_tokens=cost_unit,
    )
    return float(loss.cpu()), {key: float(value.cpu()) for key, value in parts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--baseline", choices=("matched", "constrained"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--internal-split-seed", type=int, default=0)
    parser.add_argument("--batch-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    probe_config = config["probe"]
    dynamic = config["dynamic_policy"]
    os_config = config["os_pruner"]
    dataset_config = config["datasets"][args.dataset]
    lambda_grid = [float(value) for value in os_config["lambda_grid"]]
    mu_grid = (
        [0.0] if args.baseline == "matched"
        else [float(value) for value in os_config["constrained_mu_grid"]]
    )
    candidate_pairs = [
        (lambda_value, mu_value)
        for lambda_value in lambda_grid
        for mu_value in mu_grid
    ]
    candidates = [
        {"candidate_index": index, "lambda": lambda_value, "mu": mu_value}
        for index, (lambda_value, mu_value) in enumerate(candidate_pairs)
    ]
    lambdas_np = np.asarray([row["lambda"] for row in candidates], dtype=np.float64)
    mus_np = np.asarray([row["mu"] for row in candidates], dtype=np.float64)
    destination = args.output if args.output.is_absolute() else ROOT / args.output
    invocation = {
        "protocol_id": config["protocol_id"],
        "dataset": args.dataset,
        "baseline": args.baseline,
        "raw_input": artifact_manifest(args.raw_root),
        "probe_seed": args.seed,
        "internal_split_seed": args.internal_split_seed,
        "batch_seed": args.batch_seed,
        "policy_action_seed": int(config["seed"]["policy_action"]),
        "probe": probe_config,
        "candidate_grid": candidates,
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
    fallbacks: dict[str, list[dict[str, Any]]] = {}
    features: dict[str, np.ndarray] = {}
    train_frame, train_hidden, capture_layers, train_fallbacks = load_checkpoint_split(
        args.raw_root / "probe_train", "sentence"
    )
    frames["probe_train"] = train_frame
    fallbacks["probe_train"] = train_fallbacks
    raw_train = build_dynamic_features(
        train_frame, train_hidden, capture_layers, layer=20, feature_kind="full_no_delta"
    )
    del train_hidden
    additional = [row["problem_id"] for row in train_fallbacks]
    fit_ids, validation_ids = fit_validation_problem_ids(
        train_frame, args.dataset, seed=args.internal_split_seed,
        additional_problem_ids=additional,
    )
    fit_mask, validation_mask = fit_validation_masks(
        train_frame, args.dataset, seed=args.internal_split_seed,
        additional_problem_ids=additional,
    )
    scaler = StandardScaler(copy=False)
    scaler.fit(raw_train[fit_mask])
    features["probe_train"] = scaler.transform(raw_train).astype(np.float32, copy=False)
    for split in ("calibration", "heldout"):
        frame, hidden, layers, local_fallbacks = load_checkpoint_split(
            args.raw_root / split, "sentence"
        )
        if layers != capture_layers:
            raise ValueError(f"{split} capture layer不一致")
        raw = build_dynamic_features(
            frame, hidden, layers, layer=20, feature_kind="full_no_delta"
        )
        frames[split] = frame
        fallbacks[split] = local_fallbacks
        features[split] = scaler.transform(raw).astype(np.float32, copy=False)
    for split, expected in (
        ("probe_train", int(dataset_config["probe_train"])),
        ("calibration", int(dataset_config["calibration"])),
        ("heldout", int(dataset_config["heldout"])),
    ):
        actual = int(frames[split].problem_id.nunique()) + len(fallbacks[split])
        if actual != expected:
            raise ValueError(f"{args.dataset}/{split}={actual}，预期{expected}")
    if features["probe_train"].shape[1] != 2566:
        raise ValueError("OS-Pruner输入不是最终2566维")

    model = OSPrunerPolicyBank(2566, len(candidates)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(probe_config["learning_rate"]),
        weight_decay=float(probe_config["weight_decay"]),
    )
    lambdas = torch.from_numpy(lambdas_np.astype(np.float32)).to(device)
    mus = torch.from_numpy(mus_np.astype(np.float32)).to(device)
    maximum_epochs = int(args.epochs or probe_config["max_epochs"])
    patience_limit = int(probe_config["patience"])
    rng = random.Random(args.batch_seed)
    cost_unit = float(dynamic["cost_unit_tokens"])
    history = []
    best = None
    patience = 0
    for epoch in range(maximum_epochs):
        model.train()
        losses = []
        utilities = []
        risks = []
        for positions, offsets in problem_batches(
            train_frame, fit_mask, int(probe_config["trajectory_batch_size"]), rng
        ):
            logits = model(torch.from_numpy(features["probe_train"][positions]).to(device))
            loss, parts = os_pruner_expected_utility(
                logits, train_frame, positions, offsets, lambdas, mus,
                cost_unit_tokens=cost_unit,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(probe_config["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            utilities.append(float(parts["mean_candidate_utility"].detach().cpu()))
            risks.append(float(parts["mean_expected_lost_correct"].detach().cpu()))
        val_loss, val_parts = validation_loss(
            model, features["probe_train"], train_frame, validation_mask,
            lambdas, mus, cost_unit, device,
        )
        record = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "training_mean_candidate_utility": float(np.mean(utilities)),
            "training_mean_expected_lost_correct": float(np.mean(risks)),
            "validation_loss": val_loss,
            **{f"validation_{key}": value for key, value in val_parts.items()},
        }
        history.append(record)
        print(json.dumps({"dataset": args.dataset, "baseline": args.baseline, **record}), flush=True)
        key = (-val_loss, val_parts["mean_candidate_utility"])
        if best is None or key > best[0]:
            best = (
                key, epoch,
                {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            )
            patience = 0
        else:
            patience += 1
        if patience >= patience_limit:
            break
    if best is None:
        raise RuntimeError("OS-Pruner训练未产生checkpoint")
    model.load_state_dict(best[2])
    probabilities = {
        split: predict_os_stop_probabilities(model, features[split], device)
        for split in frames
    }

    action_seed = int(config["seed"]["policy_action"])
    calibration_curve = []
    for candidate in candidates:
        index = int(candidate["candidate_index"])
        row = simulate_os_pruner_policy(
            frames["calibration"], probabilities["calibration"][:, index],
            dataset=args.dataset, split="calibration", action_seed=action_seed,
            fallback_records=fallbacks["calibration"],
        )
        row.update(candidate)
        row.update(summarize_expected_os_policy(
            frames["calibration"], probabilities["calibration"][:, index],
            fallbacks["calibration"],
        ))
        row["lost_correct_ucb_simultaneous95"] = clopper_pearson_upper(
            int(row["lost_correct_count"]), int(row["problems"]),
            float(dynamic["formal_delta"]) / len(candidates),
        )
        calibration_curve.append(row)
    dense_calibration = simulate_os_pruner_policy(
        frames["calibration"], np.zeros(len(frames["calibration"])),
        dataset=args.dataset, split="calibration", action_seed=action_seed,
        fallback_records=fallbacks["calibration"], force_dense=True,
    )
    dense_calibration.update({
        "selected_candidate": "dense", "candidate_index": None,
        "lambda": None, "mu": None, "lost_correct_ucb_simultaneous95": 0.0,
    })
    epsilon = float(dynamic["accuracy_epsilon"])
    selected = {
        "empirical_B": {
            str(int(value)): choose_empirical(
                calibration_curve, dense_calibration, int(value), epsilon
            )
            for value in dynamic["empirical_B"]
        },
        "formal_alpha": {
            str(value): choose_formal(
                calibration_curve, dense_calibration, float(value), epsilon
            )
            for value in dynamic["formal_alpha"]
        },
    }

    selected_indices = {
        int(item["candidate_index"])
        for family in selected.values() for item in family.values()
        if not item.get("dense_fallback")
    }
    heldout_cache: dict[int | str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for index in sorted(selected_indices):
        candidate = candidates[index]
        summary, records = strip_records(simulate_os_pruner_policy(
            frames["heldout"], probabilities["heldout"][:, index],
            dataset=args.dataset, split="heldout", action_seed=action_seed,
            fallback_records=fallbacks["heldout"], include_records=True,
        ))
        summary.update(candidate)
        summary.update(summarize_expected_os_policy(
            frames["heldout"], probabilities["heldout"][:, index], fallbacks["heldout"]
        ))
        heldout_cache[index] = (summary, records)
    dense_summary, dense_records = strip_records(simulate_os_pruner_policy(
        frames["heldout"], np.zeros(len(frames["heldout"])),
        dataset=args.dataset, split="heldout", action_seed=action_seed,
        fallback_records=fallbacks["heldout"], include_records=True, force_dense=True,
    ))
    heldout_cache["dense"] = (dense_summary, dense_records)
    frozen_results: dict[str, Any] = {}
    policy_records: dict[str, Any] = {}
    for family, values in selected.items():
        frozen_results[family] = {}
        policy_records[family] = {}
        for key, calibration_selection in values.items():
            cache_key: int | str = (
                "dense" if calibration_selection.get("dense_fallback")
                else int(calibration_selection["candidate_index"])
            )
            heldout_summary, records = heldout_cache[cache_key]
            frozen_results[family][key] = {
                "calibration": calibration_selection,
                "heldout": heldout_summary,
            }
            policy_records[family][key] = records

    run_spec = {
        "dataset": args.dataset,
        "protocol_id": config["protocol_id"],
        "method": (
            "matched_os_pruner_controlled_adaptation"
            if args.baseline == "matched"
            else "constrained_os_pruner_controlled_extension"
        ),
        "baseline": args.baseline,
        "feature_kind": "full_no_delta",
        "feature_width": 2566,
        "architecture": [2566, 384, 96, len(candidates)],
        "training_objective": "differentiable first-hit expected utility",
        "lost_correct_penalty": args.baseline == "constrained",
        "rollout": "fixed-seed Bernoulli first-hit with common random numbers",
        "action_seed": action_seed,
        "candidate_grid": candidates,
        "heldout_selection": False,
        "paper_faithful": False,
        "controlled_difference_note": "same cache/features/MLP/data/calibration; no LLM finetuning",
    }
    payload = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_spec": run_spec,
        "run_spec_fingerprint": canonical_fingerprint(run_spec),
        "input": invocation["raw_input"],
        "fit_problem_ids": sorted(fit_ids),
        "validation_problem_ids": sorted(validation_ids),
        "best_epoch": int(best[1]),
        "history": history,
        "calibration": {
            "candidate_count": len(candidates),
            "curve": calibration_curve,
            "dense_sentinel": dense_calibration,
            "selected": selected,
            "candidate_feasibility": candidate_feasibility_counts(
                calibration_curve, dense_calibration["dense_accuracy"], epsilon,
                [int(value) for value in dynamic["empirical_B"]],
            ),
        },
        "frozen_policy_results": frozen_results,
        "heldout_candidate_evaluation": "calibration-selected candidates only",
    }
    atomic_torch_save({
        "status": "complete", "state_dict": best[2], "run_spec": run_spec,
        "capture_layers": capture_layers,
        "scaler_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
        "scaler_scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
        "candidate_lambdas": torch.from_numpy(lambdas_np.astype(np.float32)),
        "candidate_mus": torch.from_numpy(mus_np.astype(np.float32)),
    }, destination / "probe.pt")
    atomic_torch_save({
        "status": "complete",
        "stop_probabilities": {
            split: torch.from_numpy(values.astype(np.float32))
            for split, values in probabilities.items()
        },
        "problem_ids": {
            split: frame.problem_id.astype(str).tolist() for split, frame in frames.items()
        },
        "checkpoints": {
            split: frame.checkpoint.astype(int).tolist() for split, frame in frames.items()
        },
    }, destination / "predictions.pt")
    atomic_torch_save({"status": "complete", "records": policy_records}, destination / "policy_records.pt")
    atomic_json(payload, destination / "probe.json")
    atomic_json({
        "status": "complete", "invocation_fingerprint": invocation_fingerprint,
        "run_spec_fingerprint": payload["run_spec_fingerprint"],
        "best_epoch": int(best[1]),
        "artifacts": ["probe.json", "probe.pt", "predictions.pt", "policy_records.pt"],
    }, marker)
    print(json.dumps({
        "status": "complete", "dataset": args.dataset, "baseline": args.baseline,
        "output": str(destination), "selected": selected,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""训练更强的受控OS-Pruner：mu链独立主干，lambda由大到小渐进warm-start。

与v2共享bank基线相比，本实现避免不同mu目标竞争同一主干。每个候选仍使用
完全相同的2566维输入、MLP、fit/validation划分和calibration选择规则。
该实现标记为controlled adaptation，不声称paper-faithful reproduction。
"""
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
    summary = dict(payload)
    return summary, summary.pop("records")


def choose_empirical(
    curve: list[dict[str, Any]], dense: dict[str, Any], budget: int, epsilon: float,
) -> dict[str, Any]:
    feasible = [
        row for row in curve
        if int(row["lost_correct_count"]) <= int(budget)
        and float(row["accuracy"]) >= float(dense["dense_accuracy"]) - epsilon
    ]
    if feasible:
        selected = dict(min(feasible, key=lambda row: (
            float(row["mean_reasoning_tokens"]),
            -float(row["token_reduction"]),
            -float(row["coverage"]),
            int(row["candidate_index"]),
        )))
        selected["dense_fallback"] = False
    else:
        selected = dict(dense)
        selected.update({"selected_candidate": "dense", "candidate_index": None, "dense_fallback": True})
    selected.update({
        "selection_family": "empirical_B_with_accuracy_constraint",
        "budget_B": int(budget), "accuracy_epsilon": float(epsilon),
    })
    return selected


@torch.no_grad()
def validation_utility(
    model: OSPrunerPolicyBank,
    features: np.ndarray,
    frame,
    mask: np.ndarray,
    lambda_value: float,
    mu_value: float,
    cost_unit: float,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    positions, offsets = group_positions_offsets(frame, mask)
    model.eval()
    logits = model(torch.from_numpy(features[positions]).to(device))
    loss, parts = os_pruner_expected_utility(
        logits, frame, positions, offsets,
        torch.tensor([lambda_value], dtype=torch.float32, device=device),
        torch.tensor([mu_value], dtype=torch.float32, device=device),
        cost_unit_tokens=cost_unit,
    )
    return float(loss.cpu()), {key: float(value.cpu()) for key, value in parts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--internal-split-seed", type=int, default=0)
    parser.add_argument("--batch-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-mu-chains", type=int)
    parser.add_argument("--max-lambdas", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    probe_config = config["probe"]
    dynamic = config["dynamic_policy"]
    dataset_config = config["datasets"][args.dataset]
    lambda_grid = [float(value) for value in dynamic["lambda_grid"]]
    mu_grid = [float(value) for value in dynamic["mu_grid"]]
    if args.max_mu_chains is not None:
        mu_grid = mu_grid[:args.max_mu_chains]
    training_lambdas = sorted(lambda_grid, reverse=True)
    if args.max_lambdas is not None:
        training_lambdas = training_lambdas[:args.max_lambdas]
    active_pairs = {(value, mu) for mu in mu_grid for value in training_lambdas}
    all_pairs = [
        (lambda_value, mu_value)
        for lambda_value in lambda_grid
        for mu_value in [float(value) for value in dynamic["mu_grid"]]
        if (lambda_value, mu_value) in active_pairs
    ]
    candidates = [
        {"candidate_index": index, "lambda": pair[0], "mu": pair[1]}
        for index, pair in enumerate(all_pairs)
    ]
    pair_to_index = {(row["lambda"], row["mu"]): row["candidate_index"] for row in candidates}

    destination = args.output if args.output.is_absolute() else ROOT / args.output
    invocation = {
        "protocol_id": config["protocol_id"], "dataset": args.dataset,
        "raw_input": artifact_manifest(args.raw_root),
        "seed": args.seed, "internal_split_seed": args.internal_split_seed,
        "batch_seed": args.batch_seed, "candidate_grid": candidates,
        "training_order": "lambda descending within independent mu chain",
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

    frames = {}
    fallbacks = {}
    features = {}
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

    maximum_epochs = int(args.epochs or probe_config["max_epochs"])
    patience_limit = int(probe_config["patience"])
    cost_unit = float(dynamic["cost_unit_tokens"])
    probabilities = {
        split: np.zeros((len(frames[split]), len(candidates)), dtype=np.float64)
        for split in frames
    }
    candidate_histories = []
    candidate_states = {}
    for mu_chain_index, mu_value in enumerate(mu_grid):
        # 每个mu链具有独立主干；同一链内lambda从大到小渐进warm-start。
        seed_everything(args.seed)
        model = OSPrunerPolicyBank(2566, 1).to(device)
        for stage, lambda_value in enumerate(training_lambdas):
            candidate_index = int(pair_to_index[(lambda_value, mu_value)])
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(probe_config["learning_rate"]),
                weight_decay=float(probe_config["weight_decay"]),
            )
            rng = random.Random(args.batch_seed)
            history = []
            best = None
            patience = 0
            lambda_tensor = torch.tensor([lambda_value], dtype=torch.float32, device=device)
            mu_tensor = torch.tensor([mu_value], dtype=torch.float32, device=device)
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
                        logits, train_frame, positions, offsets,
                        lambda_tensor, mu_tensor, cost_unit_tokens=cost_unit,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(probe_config["gradient_clip"]))
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                    utilities.append(float(parts["mean_candidate_utility"].detach().cpu()))
                    risks.append(float(parts["mean_expected_lost_correct"].detach().cpu()))
                val_loss, val_parts = validation_utility(
                    model, features["probe_train"], train_frame, validation_mask,
                    lambda_value, mu_value, cost_unit, device,
                )
                record = {
                    "epoch": epoch, "training_loss": float(np.mean(losses)),
                    "training_utility": float(np.mean(utilities)),
                    "training_expected_lost_correct": float(np.mean(risks)),
                    "validation_loss": val_loss,
                    **{f"validation_{key}": value for key, value in val_parts.items()},
                }
                history.append(record)
                print(json.dumps({
                    "dataset": args.dataset, "mu_chain": mu_chain_index,
                    "stage": stage, "lambda": lambda_value, "mu": mu_value, **record,
                }), flush=True)
                key = (val_parts["mean_candidate_utility"], -val_loss)
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
                raise RuntimeError("渐进OS训练未产生checkpoint")
            model.load_state_dict(best[2])
            for split in frames:
                probabilities[split][:, candidate_index] = predict_os_stop_probabilities(
                    model, features[split], device
                )[:, 0]
            candidate_states[str(candidate_index)] = best[2]
            candidate_histories.append({
                "candidate_index": candidate_index, "lambda": lambda_value, "mu": mu_value,
                "mu_chain": mu_chain_index, "stage": stage,
                "warm_started_from_larger_lambda": stage > 0,
                "best_epoch": int(best[1]), "history": history,
            })

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
        calibration_curve.append(row)
    dense_calibration = simulate_os_pruner_policy(
        frames["calibration"], np.zeros(len(frames["calibration"]), dtype=np.float64),
        dataset=args.dataset, split="calibration", action_seed=action_seed,
        fallback_records=fallbacks["calibration"], force_dense=True,
    )
    dense_calibration.update({
        "candidate_index": None, "selected_candidate": "dense",
        "lambda": None, "mu": None,
    })
    epsilon = float(dynamic["accuracy_epsilon"])
    families = {
        "progressive_matched_os": [row for row in calibration_curve if float(row["mu"]) == 0.0],
        "progressive_constrained_os": calibration_curve,
    }
    frozen_results = {}
    policy_records = {}
    feasibility = {}
    for family, curve in families.items():
        feasibility[family] = candidate_feasibility_counts(
            curve, float(dense_calibration["dense_accuracy"]), epsilon,
            [int(value) for value in dynamic["empirical_B"]],
        )
        frozen_results[family] = {}
        policy_records[family] = {}
        for budget in dynamic["empirical_B"]:
            key = str(int(budget))
            selection = choose_empirical(curve, dense_calibration, int(budget), epsilon)
            if selection.get("dense_fallback"):
                evaluated = simulate_os_pruner_policy(
                    frames["heldout"], np.zeros(len(frames["heldout"]), dtype=np.float64),
                    dataset=args.dataset, split="heldout", action_seed=action_seed,
                    fallback_records=fallbacks["heldout"], include_records=True, force_dense=True,
                )
            else:
                index = int(selection["candidate_index"])
                evaluated = simulate_os_pruner_policy(
                    frames["heldout"], probabilities["heldout"][:, index],
                    dataset=args.dataset, split="heldout", action_seed=action_seed,
                    fallback_records=fallbacks["heldout"], include_records=True,
                )
            summary, records = strip_records(evaluated)
            frozen_results[family][key] = {"calibration": selection, "heldout": summary}
            policy_records[family][key] = records

    run_spec = {
        "dataset": args.dataset,
        "method": "controlled_progressive_independent_os_pruner",
        "paper_faithful": False,
        "architecture_per_mu_chain": [2566, 384, 96, 1],
        "independent_trunk_across_mu": True,
        "lambda_training_order": training_lambdas,
        "warm_start_within_mu_chain": True,
        "candidate_epoch_selection": "probe_train_internal_validation_expected_utility",
        "candidate_grid": candidates,
        "action_rollout": "fixed-seed Bernoulli first-hit",
        "heldout_selection": False,
    }
    payload = {
        "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "run_spec": run_spec, "run_spec_fingerprint": canonical_fingerprint(run_spec),
        "input": invocation["raw_input"],
        "fit_problem_ids": sorted(fit_ids), "validation_problem_ids": sorted(validation_ids),
        "candidate_histories": candidate_histories,
        "calibration": {"dense_sentinel": dense_calibration, "curve": calibration_curve},
        "candidate_feasibility": feasibility,
        "frozen_policy_results": frozen_results,
        "heldout_used_for_selection": False,
    }
    atomic_json(payload, destination / "probe.json")
    atomic_torch_save({
        "status": "complete", "run_spec": run_spec,
        "candidate_state_dicts": candidate_states,
        "scaler_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
        "scaler_scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
    }, destination / "probe.pt")
    atomic_torch_save({
        "status": "complete",
        "stop_probability": {
            split: torch.from_numpy(value.astype(np.float32)) for split, value in probabilities.items()
        },
        "problem_ids": {split: frame.problem_id.astype(str).tolist() for split, frame in frames.items()},
        "checkpoints": {split: frame.checkpoint.astype(int).tolist() for split, frame in frames.items()},
    }, destination / "predictions.pt")
    atomic_torch_save({"status": "complete", "records": policy_records}, destination / "policy_records.pt")
    atomic_json({
        "status": "complete", "invocation_fingerprint": invocation_fingerprint,
        "run_spec_fingerprint": payload["run_spec_fingerprint"],
        "artifacts": ["probe.json", "probe.pt", "predictions.pt", "policy_records.pt"],
    }, marker)
    print(json.dumps({
        "status": "complete", "dataset": args.dataset, "output": str(destination),
        "candidate_feasibility": feasibility,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

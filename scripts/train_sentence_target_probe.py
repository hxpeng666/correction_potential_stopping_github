#!/usr/bin/env python3
"""Matched-protocol sentence-step stopping-target ablation.

All methods use the same cached sentence checkpoints, layer-20 hidden-delta
features, scaler split, scalar MLP, and held-out evaluation.  Only the
supervision/trajectory objective and the resulting stopping-score direction
differ.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probe import (
    Probe,
    add_labels,
    category_labels,
    correction_loss,
    hidden_delta_feature,
    load_split,
    problem_batches,
    split_problem_ids,
    transition,
)
from src.utils import atomic_json, load_yaml


METHODS = ("correctness", "consistency", "last_switch", "correction")
RISK_BUDGETS = (0, 1, 2, 4, 10)
COVERAGE_TARGETS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_labels(frame)
    current = frame.current_prediction.fillna("<MISSING>").astype(str)
    dense = frame.dense_prediction.fillna("<MISSING>").astype(str)
    frame["target_correctness"] = frame.current_success.astype(bool)
    frame["target_consistency"] = current.eq(dense)
    frame["target_last_switch"] = frame.last_switch_raw.astype(bool)
    frame["target_correction"] = (
        (~frame.current_success.astype(bool)) & frame.dense_success.astype(bool)
    )
    return frame


def method_target(frame: pd.DataFrame, method: str) -> np.ndarray:
    return frame[f"target_{method}"].to_numpy(np.float32)


def direction(method: str) -> str:
    return "low" if method == "correction" else "high"


def simulate(frame: pd.DataFrame, scores: np.ndarray, method_direction: str, threshold: float) -> dict:
    scored = frame.copy()
    scored["score"] = scores
    counts = {key: 0 for key in ("W_to_C", "C_to_W", "W_to_W", "C_to_C", "fallback")}
    tokens, dense_tokens, walls, dense_walls = [], [], [], []
    for _, group in scored.groupby("problem_id", sort=False):
        ordered = group.sort_values("checkpoint")
        base = ordered.iloc[0]
        if method_direction == "high":
            eligible = ordered[ordered.score >= threshold]
        else:
            eligible = ordered[ordered.score <= threshold]
        dense_token = int(base.dense_tokens)
        dense_wall = float(base.dense_wall_ms)
        dense_tokens.append(dense_token)
        dense_walls.append(dense_wall)
        if eligible.empty:
            counts["fallback"] += 1
            tokens.append(dense_token)
            walls.append(dense_wall)
            continue
        chosen = eligible.iloc[0]
        counts[transition(chosen)] += 1
        tokens.append(min(dense_token, int(chosen.checkpoint) + int(chosen.branch_tokens)))
        walls.append(float(
            chosen.dense_prefill_cuda_ms
            + chosen.prefix_decode_cuda_ms
            + chosen.branch_wall_ms
        ))
    n = len(tokens)
    stopped = n - counts["fallback"]
    return {
        "threshold": float(threshold),
        "counts": counts,
        "coverage": float(stopped / n),
        "token_reduction": float(1.0 - np.mean(tokens) / np.mean(dense_tokens)),
        "wall_reduction": float(1.0 - np.mean(walls) / np.mean(dense_walls)),
        "accuracy_drop_pp": float(100.0 * (counts["W_to_C"] - counts["C_to_W"]) / n),
        "dangerous_stop_rate": float(counts["W_to_C"] / n),
    }


def threshold_curve(frame: pd.DataFrame, scores: np.ndarray, method_direction: str) -> list[dict]:
    values = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))
    if method_direction == "high":
        thresholds = [float(np.nextafter(values.max(), np.inf))] + [float(v) for v in values]
    else:
        thresholds = [float(np.nextafter(values.min(), -np.inf))] + [float(v) for v in values]
    return [simulate(frame, scores, method_direction, value) for value in thresholds]


def select_for_budget(curve: list[dict], budget: int) -> dict:
    feasible = [row for row in curve if row["counts"]["W_to_C"] <= budget]
    return max(feasible, key=lambda row: (row["wall_reduction"], row["token_reduction"], row["coverage"]))


def select_for_coverage(curve: list[dict], target: float) -> dict:
    return min(
        curve,
        key=lambda row: (
            abs(row["coverage"] - target),
            -row["wall_reduction"],
            row["counts"]["W_to_C"],
        ),
    )


@torch.no_grad()
def predict(model: Probe, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(values), 512):
        logits = model(torch.from_numpy(values[start:start + 512]).to(device))[:, 0]
        output.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(output)


def safe_ap_auc(truth: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(np.unique(truth)) < 2:
        return 0.0, 0.5
    return float(average_precision_score(truth, scores)), float(roc_auc_score(truth, scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--config", default="configs/qwen3_semantic_checkpoint_ablation_v1.yaml")
    parser.add_argument("--source-output", default="results/qwen3_semantic_checkpoint_ablation_v1/sentence")
    parser.add_argument("--dense-reference-output", default="results/qwen3_four_label_full_gsm8k_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=24)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = load_yaml(ROOT / args.config)
    source = ROOT / args.source_output
    dense_source = ROOT / args.dense_reference_output
    destination = ROOT / args.output
    destination.mkdir(parents=True, exist_ok=True)

    frames, hidden = {}, {}
    for split in ("train", "calibration", "heldout"):
        frame, states = load_split(source / "raw" / split, dense_source / "dense_reference" / split)
        frames[split] = add_targets(frame)
        hidden[split] = states.reshape(len(states), len(config["generation"]["capture_layers"]), -1)

    raw = {
        split: hidden_delta_feature(frames[split], hidden[split])
        for split in frames
    }
    fit_mask, validation_mask = split_problem_ids(frames["train"])
    scaler = StandardScaler().fit(raw["train"][fit_mask])
    features = {
        split: scaler.transform(values).astype(np.float32, copy=False)
        for split, values in raw.items()
    }
    labels = {split: method_target(frames[split], args.method) for split in frames}
    categories = category_labels(frames["train"])
    remaining = np.clip(
        (frames["train"].dense_tokens.to_numpy(float) - frames["train"].checkpoint.to_numpy(float))
        / np.maximum(frames["train"].dense_tokens.to_numpy(float), 1.0),
        0.0,
        1.0,
    ).astype(np.float32)

    model = Probe(features["train"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    positives = max(float(labels["train"][fit_mask].sum()), 1.0)
    negatives = max(float(fit_mask.sum()) - positives, 1.0)
    positive_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    rng = random.Random(0)
    best = None
    patience = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for positions, offsets in problem_batches(frames["train"], fit_mask, 24, rng):
            x = torch.from_numpy(features["train"][positions]).to(device)
            logits = model(x)
            if args.method == "correction":
                loss = correction_loss(
                    logits,
                    torch.from_numpy(categories[positions]).to(device),
                    torch.from_numpy(remaining[positions]).to(device),
                    offsets,
                )
            else:
                target = torch.from_numpy(labels["train"][positions]).to(device)
                loss = F.binary_cross_entropy_with_logits(
                    logits[:, 0], target, pos_weight=positive_weight
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_scores = predict(model, features["train"][validation_mask], device)
        validation_truth = labels["train"][validation_mask].astype(int)
        ap, auc = safe_ap_auc(validation_truth, validation_scores)
        validation_frame = frames["train"].loc[validation_mask].reset_index(drop=True)
        if args.method == "correction":
            curve = threshold_curve(validation_frame, validation_scores, "low")
            strict = select_for_budget(curve, 0)
            key = (strict["wall_reduction"], ap, auc)
        else:
            key = (ap, auc, -float(np.mean(losses)))
            strict = None
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "validation_ap": ap,
            "validation_auc": auc,
            "strict": strict,
        }
        history.append(record)
        print(json.dumps({"method": args.method, **record}), flush=True)
        if best is None or key > best[0]:
            best = (
                key,
                epoch,
                {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            )
            patience = 0
        else:
            patience += 1
        if patience >= 6:
            break

    assert best is not None
    model.load_state_dict(best[2])
    scores = {split: predict(model, values, device) for split, values in features.items()}
    method_direction = direction(args.method)
    calibration_curve = threshold_curve(frames["calibration"], scores["calibration"], method_direction)

    risk_policies = {}
    for budget in RISK_BUDGETS:
        frozen = select_for_budget(calibration_curve, budget)
        risk_policies[str(budget)] = {
            "calibration": frozen,
            "heldout": simulate(frames["heldout"], scores["heldout"], method_direction, frozen["threshold"]),
        }

    coverage_policies = {}
    for target in COVERAGE_TARGETS:
        frozen = select_for_coverage(calibration_curve, target)
        coverage_policies[str(int(round(100 * target)))] = {
            "calibration": frozen,
            "heldout": simulate(frames["heldout"], scores["heldout"], method_direction, frozen["threshold"]),
        }

    heldout_ap, heldout_auc = safe_ap_auc(labels["heldout"].astype(int), scores["heldout"])
    payload = {
        "status": "complete",
        "method": args.method,
        "direction": method_direction,
        "feature_kind": "layer20_hidden_delta",
        "feature_width": int(features["train"].shape[1]),
        "architecture": [int(features["train"].shape[1]), 384, 96, 1],
        "best_epoch": int(best[1]),
        "history": history,
        "heldout_label_ap": heldout_ap,
        "heldout_label_auc": heldout_auc,
        "risk_policies": risk_policies,
        "coverage_policies": coverage_policies,
    }
    atomic_json(payload, destination / "probe.json")
    torch.save(
        {"state_dict": model.state_dict(), "method": args.method, "width": features["train"].shape[1]},
        destination / "probe.pt",
    )
    with (destination / "preprocess.pkl").open("wb") as handle:
        pickle.dump({"scaler": scaler, "feature_kind": "layer20_hidden_delta"}, handle)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

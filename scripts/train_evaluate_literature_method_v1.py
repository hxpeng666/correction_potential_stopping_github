#!/usr/bin/env python3
"""Train and evaluate method-faithful LTS, LYNX, and Thought Calibration probes."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "src").is_dir():
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LITERATURE_SITE = ROOT / ".literature_reproduction_site"
if LITERATURE_SITE.is_dir():
    sys.path.insert(0, str(LITERATURE_SITE))

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import binom
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.final_paper_inference import atomic_torch_save
from src.utils import atomic_json, load_yaml, seed_everything


METHODS = ("learn_to_stop", "self_verification", "lynx", "thought_calibration")
SPLITS = ("probe_train", "calibration", "heldout")


def stable_suffix_labels(answers: list[Any]) -> np.ndarray:
    """Exact reverse-search target from the Learn-to-Stop public code."""
    normalized = ["" if answer is None else str(answer).strip() for answer in answers]
    labels: list[int] = []
    current = ""
    for answer in reversed(normalized):
        if current == "":
            current = answer
            labels.append(1)
        elif answer == current:
            labels.append(1)
        else:
            labels.append(0)
            break
    labels.extend([0] * (len(normalized) - len(labels)))
    return np.asarray(list(reversed(labels)), dtype=np.float32)


def load_artifacts(root: Path, split: str) -> list[dict[str, Any]]:
    paths = sorted((root / "cache" / split).glob("sample_*.pt"))
    artifacts = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    invalid = [value.get("problem_id") for value in artifacts if value.get("status") != "complete"]
    if invalid:
        raise RuntimeError(f"incomplete artifacts: {invalid[:10]}")
    return artifacts


def split_problem_ids(artifacts: list[dict[str, Any]], fraction: float, seed: int) -> tuple[set[str], set[str]]:
    ids = [str(value["problem_id"]) for value in artifacts]
    rng = random.Random(seed)
    rng.shuffle(ids)
    validation_size = max(1, int(len(ids) * fraction))
    return set(ids[validation_size:]), set(ids[:validation_size])


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def flatten_artifacts(
    artifacts: list[dict[str, Any]],
    label_fn: Callable[[dict[str, Any]], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[int, int]]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    problem_ids: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for value in artifacts:
        hidden = value["hidden"].float().numpy()
        if hidden.ndim != 3:
            raise ValueError(f"unexpected hidden shape: {hidden.shape}")
        # NumPy cannot infer ``-1`` when the checkpoint axis is zero.  Preserve
        # the known layer×hidden width explicitly so zero-checkpoint problems
        # remain valid dense-fallback examples in policy evaluation.
        features = hidden.reshape(hidden.shape[0], int(np.prod(hidden.shape[1:])))
        labels = label_fn(value)
        if len(features) != len(labels):
            raise ValueError(f"feature/label mismatch for {value['problem_id']}")
        xs.append(features)
        ys.append(labels.astype(np.float32))
        problem_ids.extend([str(value["problem_id"])] * len(labels))
        offsets.append((cursor, cursor + len(labels)))
        cursor += len(labels)
    width = next((x.shape[1] for x in xs if len(x)), 0)
    X = np.concatenate(xs, axis=0).astype(np.float32) if cursor else np.empty((0, width), np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32) if cursor else np.empty((0,), np.float32)
    return X, y, problem_ids, offsets


def policy_metrics(
    artifacts: list[dict[str, Any]],
    scores_by_problem: dict[str, np.ndarray],
    accept: Callable[[float], bool],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for value in artifacts:
        problem_id = str(value["problem_id"])
        scores = scores_by_problem.get(problem_id, np.empty((0,), dtype=np.float32))
        rows = value["rows"]
        selected = None
        checks = 0
        for row, score in zip(rows, scores):
            checks += 1
            if accept(float(score)):
                selected = row
                break
        dense_success = bool(value["dense"]["success"])
        dense_tokens = int(value["dense"]["reasoning_tokens"])
        if selected is None:
            success = dense_success
            prediction = value["dense"]["prediction"]
            reasoning_tokens = dense_tokens
            total_tokens = dense_tokens
            stopped = False
            checkpoint = None
            branch_tokens = 0
        else:
            success = bool(selected["current_success"])
            prediction = selected.get("current_prediction")
            reasoning_tokens = int(selected["stop_reasoning_tokens"])
            total_tokens = int(selected["stop_total_tokens"])
            stopped = True
            checkpoint = int(selected["checkpoint"])
            branch_tokens = int(selected.get("branch_tokens", 0))
        records.append({
            "problem_id": problem_id,
            "dense_success": dense_success,
            "success": success,
            "dense_prediction": value["dense"]["prediction"],
            "prediction": prediction,
            "dense_tokens": dense_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "branch_tokens": branch_tokens,
            "stopped": stopped,
            "checkpoint": checkpoint,
            "checks": checks,
            "lost_correct": bool(dense_success and not success),
            "helped": bool((not dense_success) and success),
        })
    n = len(records)
    dense_accuracy = sum(row["dense_success"] for row in records) / max(n, 1)
    accuracy = sum(row["success"] for row in records) / max(n, 1)
    dense_mean = float(np.mean([row["dense_tokens"] for row in records])) if records else 0.0
    reasoning_mean = float(np.mean([row["reasoning_tokens"] for row in records])) if records else 0.0
    total_mean = float(np.mean([row["total_tokens"] for row in records])) if records else 0.0
    summary = {
        "n": n,
        "dense_accuracy": dense_accuracy,
        "accuracy": accuracy,
        "accuracy_delta_pp": 100.0 * (accuracy - dense_accuracy),
        "dense_mean_reasoning_tokens": dense_mean,
        "mean_reasoning_tokens": reasoning_mean,
        "reasoning_token_reduction_pct": 100.0 * (dense_mean - reasoning_mean) / dense_mean if dense_mean else 0.0,
        "mean_total_tokens_including_forced_answer": total_mean,
        "total_token_reduction_pct": 100.0 * (dense_mean - total_mean) / dense_mean if dense_mean else 0.0,
        "stop_rate": sum(row["stopped"] for row in records) / max(n, 1),
        "mean_checks": float(np.mean([row["checks"] for row in records])) if records else 0.0,
        "lost_correct": int(sum(row["lost_correct"] for row in records)),
        "helped": int(sum(row["helped"] for row in records)),
    }
    return summary, records


def threshold_candidates(scores: np.ndarray, count: int) -> list[float]:
    finite = scores[np.isfinite(scores)]
    if not len(finite):
        return [float("inf")]
    values = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, count))).tolist()
    return [float("inf")] + sorted((float(x) for x in values), reverse=True) + [float("-inf")]


def score_map(artifacts: list[dict[str, Any]], flat: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    cursor = 0
    for value in artifacts:
        length = len(value["rows"])
        result[str(value["problem_id"])] = np.asarray(flat[cursor : cursor + length], dtype=np.float32)
        cursor += length
    if cursor != len(flat):
        raise ValueError("flat scores do not match artifacts")
    return result


def select_fair_budget(
    artifacts: list[dict[str, Any]],
    scores: dict[str, np.ndarray],
    candidates: list[float],
    budget: int,
) -> tuple[float, dict[str, Any]]:
    curve = []
    for threshold in candidates:
        summary, _ = policy_metrics(artifacts, scores, lambda value, t=threshold: value >= t)
        curve.append({"threshold": threshold, **summary})
    feasible = [row for row in curve if row["lost_correct"] <= budget]
    if not feasible:
        selected = curve[0]
    else:
        selected = min(
            feasible,
            key=lambda row: (
                row["mean_reasoning_tokens"],
                -row["accuracy"],
                row["mean_checks"],
                -row["threshold"],
            ),
        )
    return float(selected["threshold"]), {"selected": selected, "curve": curve}


class LSTMTagger(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return torch.sigmoid(self.fc(output)).squeeze(-1)


def collate_sequences(batch):
    features, labels = zip(*batch)
    lengths = torch.tensor([len(value) for value in features], dtype=torch.long)
    return (
        nn.utils.rnn.pad_sequence(features, batch_first=True),
        lengths,
        nn.utils.rnn.pad_sequence(labels, batch_first=True),
    )


def lts_sequences(
    artifacts: list[dict[str, Any]],
    terminal_predictions: dict[str, Any] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
    result = []
    for value in artifacts:
        hidden = value["hidden"][:, 0, :].float()
        answers = [row.get("current_prediction") for row in value["rows"]]
        problem_id = str(value["problem_id"])
        if terminal_predictions is None:
            # Exact public LTS target: the final sentence-level forced answer is the
            # reverse-search reference, not a separately sampled Dense answer.
            labels = torch.from_numpy(stable_suffix_labels(answers))
        else:
            # The paragraph transfer keeps the LTS objective but obtains z_T from the
            # full-reasoning greedy forced branch in the paired native cache.
            terminal = terminal_predictions.get(problem_id)
            labels = torch.from_numpy(stable_suffix_labels(answers + [terminal])[:-1])
        if len(hidden):
            result.append((hidden, labels, problem_id))
    return result


def predict_lstm(
    model: LSTMTagger,
    artifacts: list[dict[str, Any]],
    device: torch.device,
    terminal_predictions: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    sequences = lts_sequences(artifacts, terminal_predictions)
    result: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), 32):
            batch = [(x, y) for x, y, _problem in sequences[start : start + 32]]
            padded, lengths, _labels = collate_sequences(batch)
            scores = model(padded.to(device), lengths.to(device)).cpu()
            for index, (_x, _y, problem_id) in enumerate(sequences[start : start + 32]):
                result[problem_id] = scores[index, : int(lengths[index])].numpy()
    for value in artifacts:
        result.setdefault(str(value["problem_id"]), np.empty((0,), np.float32))
    return result


def train_lts(
    config: dict[str, Any], dataset: str, schedule: str, root: Path, output: Path, device: torch.device
) -> dict[str, Any]:
    artifacts = {split: load_artifacts(root, split) for split in SPLITS}
    terminal_predictions: dict[str, dict[str, Any]] = {}
    if schedule == "paragraph":
        paired_native = {split: load_artifacts(root.parent / "native", split) for split in SPLITS}
        for split, values in paired_native.items():
            terminal_predictions[split] = {
                str(value["problem_id"]): (
                    value["rows"][-1].get("current_prediction") if value["rows"] else None
                )
                for value in values
            }
    train_ids, validation_ids = split_problem_ids(
        artifacts["probe_train"], config["methods"]["learn_to_stop"]["validation_fraction"], 0
    )
    sequences = lts_sequences(
        artifacts["probe_train"], terminal_predictions.get("probe_train")
    )
    train_data = [(x, y) for x, y, pid in sequences if pid in train_ids]
    validation_data = [(x, y) for x, y, pid in sequences if pid in validation_ids]
    generator = torch.Generator().manual_seed(0)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=int(config["methods"]["learn_to_stop"]["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=collate_sequences,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=32, shuffle=False, collate_fn=collate_sequences
    )
    model = LSTMTagger(2560, 128, 0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    criterion = nn.BCELoss(reduction="none")
    history = []
    best = None
    for epoch in range(200):
        model.train()
        losses = []
        for padded, lengths, labels in train_loader:
            padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)
            predictions = model(padded, lengths)
            mask = torch.arange(predictions.shape[1], device=device)[None, :] < lengths[:, None]
            loss = (criterion(predictions, labels) * mask).sum() / mask.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler_lr.step()
        record = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "lr": optimizer.param_groups[0]["lr"]}
        if (epoch + 1) % 10 == 0:
            truths, predictions = [], []
            model.eval()
            with torch.no_grad():
                for padded, lengths, labels in validation_loader:
                    scores = model(padded.to(device), lengths.to(device)).cpu()
                    for index, length in enumerate(lengths):
                        truths.extend(labels[index, : int(length)].numpy().tolist())
                        predictions.extend((scores[index, : int(length)] >= 0.5).numpy().tolist())
            metric = float(f1_score(truths, predictions, zero_division=0))
            record["validation_f1"] = metric
            if best is None or metric > best[0]:
                best = (metric, epoch + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        history.append(record)
    if best is None:
        raise RuntimeError("LTS training produced no validation checkpoint")
    model.load_state_dict(best[2])
    scores = {
        split: predict_lstm(model, values, device, terminal_predictions.get(split))
        for split, values in artifacts.items()
    }
    calibration_flat = np.concatenate(list(scores["calibration"].values())) if scores["calibration"] else np.empty((0,))
    candidates = threshold_candidates(calibration_flat, int(config["fair_calibration"]["threshold_quantiles"]))
    results: dict[str, Any] = {
        "method": "learn_to_stop",
        "dataset": dataset,
        "schedule": schedule,
        "label_terminal": (
            "last_sentence_greedy_forced_answer"
            if schedule == "native"
            else "paired_full_reasoning_greedy_forced_answer"
        ),
        "best_validation_f1": best[0],
        "best_epoch": best[1],
        "history": history,
        "original_operating_points": {},
        "fair_empirical_B": {},
    }
    for threshold in config["methods"]["learn_to_stop"]["original_thresholds"]:
        cal, _ = policy_metrics(artifacts["calibration"], scores["calibration"], lambda x, t=float(threshold): x >= t)
        test, records = policy_metrics(artifacts["heldout"], scores["heldout"], lambda x, t=float(threshold): x >= t)
        results["original_operating_points"][str(threshold)] = {"threshold": threshold, "calibration": cal, "heldout": test}
        atomic_json(records, output / f"heldout_original_tau_{threshold}.json")
    for budget in config["fair_calibration"]["empirical_lost_correct_budgets"]:
        threshold, selection = select_fair_budget(
            artifacts["calibration"], scores["calibration"], candidates, int(budget)
        )
        test, records = policy_metrics(artifacts["heldout"], scores["heldout"], lambda x, t=threshold: x >= t)
        results["fair_empirical_B"][str(budget)] = {"threshold": threshold, "calibration_selection": selection["selected"], "heldout": test}
        atomic_json(records, output / f"heldout_fair_B{budget}.json")
    atomic_torch_save({"state_dict": model.state_dict(), "best_epoch": best[1], "input_dim": 2560}, output / "probe.pt")
    return results


class LynxMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class SelfVerificationProbe(nn.Module):
    def __init__(self, width: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.network = (
            nn.Linear(width, 1)
            if hidden_dim == 0
            else nn.Sequential(nn.Linear(width, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def fit_self_verification_candidate(
    X: np.ndarray,
    y: np.ndarray,
    fit_mask: np.ndarray,
    validation_mask: np.ndarray,
    hidden_dim: int,
    learning_rate: float,
    alpha: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[SelfVerificationProbe, dict[str, Any]]:
    model = SelfVerificationProbe(X.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    positives = max(1, int(y[fit_mask].sum()))
    negatives = max(1, int(fit_mask.sum() - positives))
    positive_weight = float(alpha) * negatives / positives
    indices = np.flatnonzero(fit_mask)
    rng = np.random.default_rng(0)
    best = None
    patience = 0
    history = []
    for epoch in range(200):
        rng.shuffle(indices)
        model.train()
        losses = []
        for start in range(0, len(indices), 64):
            batch = indices[start : start + 64]
            values = torch.from_numpy(X[batch]).to(device)
            target = torch.from_numpy(y[batch]).to(device)
            logits = model(values)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=torch.tensor(positive_weight, device=device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_scores = predict_point_model(model, X[validation_mask], device)
        validation_target = y[validation_mask]
        validation_loss = float(nn.functional.binary_cross_entropy(
            torch.from_numpy(np.clip(validation_scores, 1e-7, 1 - 1e-7)),
            torch.from_numpy(validation_target),
            weight=torch.where(
                torch.from_numpy(validation_target) > 0.5,
                torch.tensor(positive_weight),
                torch.tensor(1.0),
            ),
        ))
        validation_accuracy = float(accuracy_score(validation_target, validation_scores >= 0.5))
        auc, ap = safe_auc(validation_target, validation_scores)
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_auc": auc,
            "validation_ap": ap,
        })
        if best is None or validation_loss < best[0] - 1e-7:
            best = (
                validation_loss,
                epoch + 1,
                validation_accuracy,
                {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            )
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    if best is None:
        raise RuntimeError("self-verification candidate produced no model")
    model.load_state_dict(best[3])
    return model, {
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "alpha": alpha,
        "weight_decay": weight_decay,
        "positive_weight": positive_weight,
        "best_epoch": best[1],
        "validation_loss": best[0],
        "validation_accuracy": best[2],
        "history": history,
    }


def train_self_verification(
    config: dict[str, Any], dataset: str, schedule: str, root: Path, output: Path, device: torch.device
) -> dict[str, Any]:
    artifacts = {split: load_artifacts(root, split) for split in SPLITS}
    label_fn = lambda value: np.asarray([row["probe_label"] for row in value["rows"]], np.float32)
    flattened = {split: flatten_artifacts(values, label_fn) for split, values in artifacts.items()}
    X_train, y_train, problem_ids, _offsets = flattened["probe_train"]
    train_ids, validation_ids = split_problem_ids(artifacts["probe_train"], 0.2, 0)
    fit_mask = np.asarray([value in train_ids for value in problem_ids])
    validation_mask = np.asarray([value in validation_ids for value in problem_ids])
    method_config = config["methods"]["self_verification"]
    candidates = []
    for hidden_dim in method_config["hidden_dim"]:
        for learning_rate in method_config["learning_rates"]:
            for alpha in method_config["imbalance_alpha"]:
                for weight_decay in method_config["weight_decay"]:
                    model, audit = fit_self_verification_candidate(
                        X_train,
                        y_train,
                        fit_mask,
                        validation_mask,
                        int(hidden_dim),
                        float(learning_rate),
                        float(alpha),
                        float(weight_decay),
                        device,
                    )
                    candidates.append((audit, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}))
    ranked = sorted(candidates, key=lambda item: (-item[0]["validation_accuracy"], item[0]["validation_loss"]))
    top_ten = ranked[:10]
    selected_audit, selected_state = min(
        top_ten,
        key=lambda item: (item[0]["hidden_dim"], -item[0]["validation_accuracy"], item[0]["validation_loss"]),
    )
    model = SelfVerificationProbe(X_train.shape[1], int(selected_audit["hidden_dim"])).to(device)
    model.load_state_dict(selected_state)
    scores = {}
    labels = {}
    for split, (X, y, _ids, _off) in flattened.items():
        labels[split] = y
        scores[split] = score_map(artifacts[split], predict_point_model(model, X, device))
    cal_flat = np.concatenate(list(scores["calibration"].values())) if scores["calibration"] else np.empty((0,))
    fair_candidates = threshold_candidates(cal_flat, int(config["fair_calibration"]["threshold_quantiles"]))
    results: dict[str, Any] = {
        "method": "self_verification",
        "dataset": dataset,
        "schedule": schedule,
        "selected_hyperparameters": {key: value for key, value in selected_audit.items() if key != "history"},
        "grid_summary": [
            {key: value for key, value in audit.items() if key != "history"} for audit, _state in ranked
        ],
        "original_operating_points": {},
        "fair_empirical_B": {},
    }
    for threshold in method_config["original_thresholds"]:
        cal, _ = policy_metrics(artifacts["calibration"], scores["calibration"], lambda x, t=float(threshold): x >= t)
        test, records = policy_metrics(artifacts["heldout"], scores["heldout"], lambda x, t=float(threshold): x >= t)
        results["original_operating_points"][str(threshold)] = {"threshold": threshold, "calibration": cal, "heldout": test}
        atomic_json(records, output / f"heldout_original_tau_{threshold}.json")
    for budget in config["fair_calibration"]["empirical_lost_correct_budgets"]:
        threshold, selection = select_fair_budget(
            artifacts["calibration"], scores["calibration"], fair_candidates, int(budget)
        )
        test, records = policy_metrics(artifacts["heldout"], scores["heldout"], lambda x, t=threshold: x >= t)
        results["fair_empirical_B"][str(budget)] = {
            "threshold": threshold,
            "calibration_selection": selection["selected"],
            "heldout": test,
        }
        atomic_json(records, output / f"heldout_fair_B{budget}.json")
    atomic_torch_save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(X_train.shape[1]),
            "hidden_dim": int(selected_audit["hidden_dim"]),
            "selected_hyperparameters": selected_audit,
        },
        output / "probe.pt",
    )
    return results


def predict_point_model(
    model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512
) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            logits = model(torch.from_numpy(X[start : start + batch_size]).to(device))
            result.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(result) if result else np.empty((0,), np.float32)


def fit_lynx_mlp(
    X: np.ndarray, y: np.ndarray, fit_mask: np.ndarray, validation_mask: np.ndarray, device: torch.device
) -> tuple[LynxMLP, dict[str, Any]]:
    mean = X[fit_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X[fit_mask].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    X_scaled = ((X - mean) / std).astype(np.float32, copy=False)
    model = LynxMLP(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_pos = max(1, int(y[fit_mask].sum()))
    n_neg = max(1, int(fit_mask.sum() - n_pos))
    n_all = int(fit_mask.sum())
    positive_weight = n_all / (2.0 * n_pos)
    negative_weight = n_all / (2.0 * n_neg)
    indices = np.flatnonzero(fit_mask)
    rng = np.random.default_rng(42)
    best = None
    patience = 0
    history = []
    for epoch in range(500):
        rng.shuffle(indices)
        model.train()
        losses = []
        for start in range(0, len(indices), 200):
            batch = indices[start : start + 200]
            values = torch.from_numpy(X_scaled[batch]).to(device)
            target = torch.from_numpy(y[batch]).to(device)
            logits = model(values)
            point = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            weights = torch.where(target > 0.5, positive_weight, negative_weight)
            loss = (point * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_scores = predict_point_model(model, X_scaled[validation_mask], device)
        validation_accuracy = float(accuracy_score(y[validation_mask], validation_scores >= 0.5))
        validation_auc, _ap = safe_auc(y[validation_mask], validation_scores)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_accuracy": validation_accuracy, "validation_auc": validation_auc})
        key = validation_accuracy
        if best is None or key > best[0] + 1e-4:
            best = (key, epoch + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    if best is None:
        raise RuntimeError("LYNX training produced no model")
    model.load_state_dict(best[2])
    return model, {"mean": mean, "std": std, "best_validation_accuracy": best[0], "best_epoch": best[1], "history": history}


def conformal_quantile(scores: np.ndarray, delta: float) -> float:
    n = int(len(scores))
    if n <= 0:
        return float("inf")
    k = max(1, min(int(math.ceil((n + 1) * (1.0 - delta))), n))
    return float(np.sort(scores)[k - 1])


def train_lynx(
    config: dict[str, Any], dataset: str, schedule: str, root: Path, output: Path, device: torch.device
) -> dict[str, Any]:
    del device
    artifacts = {split: load_artifacts(root, split) for split in SPLITS}
    label_fn = lambda value: np.asarray([row["current_success"] for row in value["rows"]], np.float32)
    flattened = {split: flatten_artifacts(values, label_fn) for split, values in artifacts.items()}
    X_train, y_train, problem_ids, _offsets = flattened["probe_train"]
    train_ids, validation_ids = split_problem_ids(artifacts["probe_train"], 0.2, 42)
    fit_mask = np.asarray([value in train_ids for value in problem_ids])
    validation_mask = np.asarray([value in validation_ids for value in problem_ids])
    # Mirror the public LYNX trainer: StandardScaler followed by sklearn's
    # two-hidden-layer MLPClassifier and its built-in early stopping.  We retain
    # the common problem-level 80:20 split instead of leaking checkpoints from
    # one problem across train/validation.
    pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            solver="adam",
            random_state=42,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.2,
        )),
    ])
    train_labels = y_train[fit_mask].astype(np.int64)
    n_pos = max(1, int((train_labels == 1).sum()))
    n_neg = max(1, int((train_labels == 0).sum()))
    n_all = len(train_labels)
    w_pos = n_all / (2.0 * n_pos)
    w_neg = n_all / (2.0 * n_neg)
    sample_weight = np.where(train_labels == 1, w_pos, w_neg).astype(np.float32)
    pipeline.fit(X_train[fit_mask], train_labels, clf__sample_weight=sample_weight)
    validation_scores = pipeline.predict_proba(X_train[validation_mask])[:, 1]
    validation_labels = y_train[validation_mask].astype(np.int64)
    validation_auc, _ = safe_auc(validation_labels, validation_scores)
    fit = {
        "trainer": "official_sklearn_pipeline",
        "best_validation_accuracy": float(accuracy_score(validation_labels, validation_scores >= 0.5)),
        "validation_auc": validation_auc,
        "n_iter": int(pipeline.named_steps["clf"].n_iter_),
        "train_positive_weight": float(w_pos),
        "train_negative_weight": float(w_neg),
    }
    scores: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, np.ndarray] = {}
    for split, (X, y, _ids, _off) in flattened.items():
        labels[split] = y
        flat_scores = pipeline.predict_proba(X)[:, 1].astype(np.float32)
        scores[split] = score_map(artifacts[split], flat_scores)
    cal_flat = np.concatenate(list(scores["calibration"].values())) if scores["calibration"] else np.empty((0,))
    y_cal = labels["calibration"].astype(np.int64)
    conformal = {}
    results: dict[str, Any] = {"method": "lynx", "dataset": dataset, "schedule": schedule, **fit, "original_operating_points": {}, "fair_empirical_B": {}}
    paper_candidates: list[tuple[float, float, float]] = []
    for delta in config["methods"]["lynx"]["conformal_deltas"]:
        q_pos = conformal_quantile(1.0 - cal_flat[y_cal == 1], float(delta))
        q_neg = conformal_quantile(cal_flat[y_cal == 0], float(delta))
        conformal[str(delta)] = {"delta": delta, "q_pos": q_pos, "q_neg": q_neg, "n_pos": int((y_cal == 1).sum()), "n_neg": int((y_cal == 0).sum())}
        accept = lambda p, qp=q_pos, qn=q_neg: ((1.0 - p) <= qp) and not (p <= qn)
        cal, _ = policy_metrics(artifacts["calibration"], scores["calibration"], accept)
        test, records = policy_metrics(artifacts["heldout"], scores["heldout"], accept)
        results["original_operating_points"][str(delta)] = {"conformal": conformal[str(delta)], "calibration": cal, "heldout": test}
        paper_candidates.append((float(delta), q_pos, q_neg))
        atomic_json(records, output / f"heldout_conformal_delta_{delta}.json")
    # Fair selection is restricted to genuine LYNX conformal operating points.
    no_stop = {"delta": None, "q_pos": -float("inf"), "q_neg": float("inf")}
    for budget in config["fair_calibration"]["empirical_lost_correct_budgets"]:
        curve = []
        options = [(None, -float("inf"), float("inf"))] + paper_candidates
        for delta, q_pos, q_neg in options:
            accept = lambda p, qp=q_pos, qn=q_neg: ((1.0 - p) <= qp) and not (p <= qn)
            summary, _ = policy_metrics(artifacts["calibration"], scores["calibration"], accept)
            curve.append({"delta": delta, "q_pos": q_pos, "q_neg": q_neg, **summary})
        feasible = [row for row in curve if row["lost_correct"] <= int(budget)]
        selected = min(feasible, key=lambda row: (row["mean_reasoning_tokens"], -row["accuracy"], row["mean_checks"])) if feasible else curve[0]
        accept = lambda p, qp=selected["q_pos"], qn=selected["q_neg"]: ((1.0 - p) <= qp) and not (p <= qn)
        test, records = policy_metrics(artifacts["heldout"], scores["heldout"], accept)
        results["fair_empirical_B"][str(budget)] = {"calibration_selection": selected, "heldout": test}
        atomic_json(records, output / f"heldout_fair_B{budget}.json")
    atomic_torch_save(
        {"pipeline": pipeline, "input_dim": int(X_train.shape[1]), "trainer": "official_sklearn_pipeline"},
        output / "probe.pt",
    )
    atomic_json(conformal, output / "conformal.json")
    return results


def trailing_mean_scores(artifacts: list[dict[str, Any]], flat_scores: np.ndarray, window: int) -> dict[str, np.ndarray]:
    result = {}
    cursor = 0
    for value in artifacts:
        length = len(value["rows"])
        raw = flat_scores[cursor : cursor + length]
        smooth = np.asarray([
            np.mean(raw[max(0, index - window + 1) : index + 1]) for index in range(length)
        ], dtype=np.float32)
        result[str(value["problem_id"])] = smooth
        cursor += length
    return result


def fixed_sequence_ltt(
    artifacts: list[dict[str, Any]],
    scores: dict[str, np.ndarray],
    thresholds: list[float],
    risk_level: float,
    failure_probability: float,
    target: str,
) -> dict[str, Any]:
    tested = []
    last_valid = None
    for threshold in sorted(thresholds, reverse=True):
        errors = 0
        for value in artifacts:
            local = scores[str(value["problem_id"])]
            selected = next((row for row, score in zip(value["rows"], local) if score >= threshold), None)
            if selected is None:
                correct = bool(value["dense"]["success"]) if target == "supervised" else True
            else:
                correct = bool(selected["current_success"] if target == "supervised" else selected["consistency"])
            errors += int(not correct)
        n = len(artifacts)
        p_value = float(binom.cdf(errors, n, risk_level))
        row = {"threshold": threshold, "errors": errors, "n": n, "empirical_risk": errors / max(n, 1), "p_value": p_value, "rejected_null": p_value <= failure_probability}
        tested.append(row)
        if row["rejected_null"]:
            last_valid = row
        else:
            break
    return {"selected": last_valid, "tested": tested, "risk_level": risk_level, "failure_probability": failure_probability}


def train_thought(
    config: dict[str, Any], dataset: str, schedule: str, root: Path, output: Path, device: torch.device, target: str
) -> dict[str, Any]:
    del device
    artifacts = {split: load_artifacts(root, split) for split in SPLITS}
    if target == "supervised":
        label_fn = lambda value: np.asarray([row["current_success"] for row in value["rows"]], np.float32)
    else:
        label_fn = lambda value: np.asarray([row["consistency"] for row in value["rows"]], np.float32)
    flattened = {split: flatten_artifacts(values, label_fn) for split, values in artifacts.items()}
    X_train, y_train, _ids, _offsets = flattened["probe_train"]
    if not len(X_train):
        raise RuntimeError("Thought Calibration has no training checkpoints")
    components = min(int(config["methods"]["thought_calibration"]["pca_dim"]), X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=components, random_state=0)
    transformed = {"probe_train": pca.fit_transform(X_train).astype(np.float32)}
    for split in ("calibration", "heldout"):
        transformed[split] = pca.transform(flattened[split][0]).astype(np.float32)
    probe = LogisticRegression(random_state=0)
    probe.fit(transformed["probe_train"], y_train.astype(np.int64))
    window = int(config["methods"]["thought_calibration"]["probability_smoothing_window"])
    scores = {}
    for split in SPLITS:
        flat = probe.predict_proba(transformed[split])[:, 1]
        scores[split] = trailing_mean_scores(artifacts[split], flat, window)
    cal_flat = np.concatenate(list(scores["calibration"].values())) if scores["calibration"] else np.empty((0,))
    candidates = threshold_candidates(cal_flat, int(config["fair_calibration"]["threshold_quantiles"]))
    results: dict[str, Any] = {"method": "thought_calibration", "target": target, "dataset": dataset, "schedule": schedule, "pca_components": components, "learn_then_test": {}, "fair_empirical_B": {}}
    for risk in config["methods"]["thought_calibration"]["risk_levels"]:
        selection = fixed_sequence_ltt(artifacts["calibration"], scores["calibration"], candidates, float(risk), 0.05, target)
        selected = selection["selected"]
        if selected is None:
            threshold = float("inf")
            selection["fallback"] = "no certified threshold; dense fallback"
        else:
            threshold = float(selected["threshold"])
        test, records = policy_metrics(artifacts["heldout"], scores["heldout"], lambda x, t=threshold: x >= t)
        results["learn_then_test"][str(risk)] = {"selection": selection, "heldout": test}
        atomic_json(records, output / f"heldout_ltt_risk_{risk}.json")
    for budget in config["fair_calibration"]["empirical_lost_correct_budgets"]:
        threshold, selection = select_fair_budget(artifacts["calibration"], scores["calibration"], candidates, int(budget))
        test, records = policy_metrics(artifacts["heldout"], scores["heldout"], lambda x, t=threshold: x >= t)
        results["fair_empirical_B"][str(budget)] = {"threshold": threshold, "calibration_selection": selection["selected"], "heldout": test}
        atomic_json(records, output / f"heldout_fair_B{budget}.json")
    atomic_torch_save({"pca": pca, "probe": probe, "target": target}, output / "probe.pt")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--schedule", choices=("native", "paragraph"), required=True)
    parser.add_argument("--target", choices=("supervised", "consistent"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.method == "thought_calibration" and args.target is None:
        raise ValueError("Thought Calibration requires --target supervised or consistent")
    if args.method != "thought_calibration" and args.target is not None:
        raise ValueError("--target is only valid for Thought Calibration")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    root = ROOT / config["output_root"] / args.dataset / args.method / args.schedule
    suffix = f"_{args.target}" if args.target else ""
    final_output = root / f"probe{suffix}"
    complete = final_output / "phase.complete"
    if args.resume and complete.is_file():
        marker = json.loads(complete.read_text(encoding="utf-8"))
        if marker.get("status") == "complete":
            print(json.dumps({"status": "skipped_complete", "output": str(final_output)}))
            return
    if final_output.exists():
        raise RuntimeError(f"refusing to overwrite incomplete output: {final_output}")
    output = final_output.with_name(f".{final_output.name}.tmp.{os.getpid()}")
    if output.exists():
        raise RuntimeError(f"temporary output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(0)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    if args.method == "learn_to_stop":
        results = train_lts(config, args.dataset, args.schedule, root, output, device)
    elif args.method == "self_verification":
        results = train_self_verification(config, args.dataset, args.schedule, root, output, device)
    elif args.method == "lynx":
        results = train_lynx(config, args.dataset, args.schedule, root, output, device)
    else:
        results = train_thought(config, args.dataset, args.schedule, root, output, device, str(args.target))
    atomic_json(results, output / "results.json")
    marker = {"status": "complete", "dataset": args.dataset, "method": args.method, "schedule": args.schedule, "target": args.target, "results": str((final_output / "results.json").resolve())}
    atomic_json(marker, output / "phase.complete")
    os.replace(output, final_output)
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score, calibrate, and evaluate the frozen BLADE compact probe."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "src").is_dir():
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from src.utils import load_yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fingerprint(config: dict[str, Any]) -> str:
    payload = {
        "protocol_id": config["protocol_id"],
        "model": config["model"],
        "datasets": config["datasets"],
        "checkpoints": config["checkpoints"],
        "strict_clean_supervision": config["strict_clean_supervision"],
        "apls": config["apls"],
        "training": config["training"],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def collection_fingerprint(config: dict[str, Any], dataset: str) -> str:
    payload = {
        "protocol_id": config["protocol_id"],
        "source": config["source"],
        "model": config["model"],
        "common_scope": config["common_scope"],
        "dataset": dataset,
        "dataset_config": config["datasets"][dataset],
        "checkpoints": config["checkpoints"],
        "strict_clean_supervision": config["strict_clean_supervision"],
        "apls_capture": {
            "decoder_layers": config["model"]["decoder_layers"],
            "hidden_size": config["model"]["hidden_size"],
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CompactProbe(nn.Module):
    def __init__(self, input_width: int, head: list[int]):
        super().__init__()
        dimensions = [input_width, *head, 1]
        modules: list[nn.Module] = []
        for index in range(len(dimensions) - 1):
            modules.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                modules.append(nn.GELU())
        self.network = nn.Sequential(*modules)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


def load_model(config: dict[str, Any], device: torch.device):
    path = ROOT / config["output_root"] / "models" / "compact_probe.pt"
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("status") != "complete" or value.get("protocol_fingerprint") != fingerprint(config):
        raise ValueError("invalid compact probe")
    selected = [int(x) for x in value["selected_layers_zero_based"]]
    model = CompactProbe(
        len(selected) * int(config["model"]["hidden_size"]),
        [int(x) for x in config["apls"]["compact_head_dims"]],
    )
    model.load_state_dict(value["state_dict"])
    model.eval().to(device)
    return model, selected


def validate_artifact(value: dict[str, Any], config: dict[str, Any], dataset: str) -> None:
    if value.get("status") != "complete" or value.get("protocol_fingerprint") != collection_fingerprint(config, dataset):
        raise ValueError("invalid BLADE collection artifact")
    hidden = value.get("hidden", torch.empty(0))
    expected = (int(config["model"]["decoder_layers"]), int(config["model"]["hidden_size"]))
    if hidden.ndim != 3 or tuple(hidden.shape[1:]) != expected or hidden.shape[0] != len(value.get("rows", [])):
        raise ValueError(f"invalid BLADE hidden shape {tuple(hidden.shape)}")


@torch.inference_mode()
def score_artifact(model, selected, artifact, device, batch_size):
    hidden = artifact["hidden"][:, selected, :].flatten(1)
    scores = []
    for start in range(0, len(hidden), batch_size):
        value = hidden[start : start + batch_size].to(device).float()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(value)
        scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
    if len(scores) != len(artifact["rows"]):
        raise AssertionError("score count mismatch")
    return scores


def score_split(config, dataset, split, model, selected, device, resume):
    output = ROOT / config["output_root"]
    destination = output / dataset / "scores" / f"{split}.pt"
    training_fingerprint = fingerprint(config)
    if resume and destination.is_file():
        value = torch.load(destination, map_location="cpu", weights_only=False)
        if value.get("status") == "complete" and value.get("protocol_fingerprint") == training_fingerprint:
            print(json.dumps({"status": "skipped", "stage": "score", "dataset": dataset, "split": split}))
            return
        raise RuntimeError("refusing incompatible BLADE score cache")
    paths = sorted((output / dataset / "cache" / split).glob("sample_*.pt"))
    expected = int(config["datasets"][dataset][split])
    if len(paths) != expected:
        raise RuntimeError(f"{dataset}/{split}: expected {expected} artifacts, found {len(paths)}")
    trajectories = []
    batch_size = int(config["training"]["batch_size"])
    for index, path in enumerate(paths):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        validate_artifact(artifact, config, dataset)
        scores = score_artifact(model, selected, artifact, device, batch_size)
        rows = []
        for row, probability in zip(artifact["rows"], scores):
            if row.get("is_paragraph") and not (row.get("is_sentence") or row.get("is_self_doubt")):
                raise ValueError(f"paragraph-only inference checkpoint leaked into {dataset}/{split}")
            rows.append({
                "checkpoint": int(row["checkpoint"]),
                "rank": int(row["rank"]),
                "is_sentence": bool(row["is_sentence"]),
                "is_self_doubt": bool(row["is_self_doubt"]),
                "cues": list(row.get("cues", [])),
                "probability": float(probability),
                "strict_clean_label": row.get("strict_clean_label"),
                "strict_clean_ambiguous": bool(row.get("strict_clean_ambiguous", False)),
                "current_success": bool(row["current_success"]),
                "current_prediction": row.get("current_prediction"),
                "stop_reasoning_tokens": int(row["stop_reasoning_tokens"]),
                "stop_total_tokens": int(row["stop_total_tokens"]),
                "greedy_branch_tokens": int(row["greedy_branch_tokens"]),
            })
        trajectories.append({
            "dataset": dataset,
            "split": split,
            "problem_id": str(artifact["problem_id"]),
            "gold_answer": artifact["gold_answer"],
            "dense_success": bool(artifact["dense"]["success"]),
            "dense_prediction": artifact["dense"]["prediction"],
            "dense_tokens": int(artifact["dense"]["reasoning_tokens"]),
            "rows": rows,
        })
        if (index + 1) % 100 == 0:
            print(json.dumps({"stage": "score", "dataset": dataset, "split": split, "completed": index + 1}), flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    torch.save({
        "schema_version": 1,
        "status": "complete",
        "protocol_fingerprint": training_fingerprint,
        "dataset": dataset,
        "split": split,
        "selected_layers_zero_based": selected,
        "trajectories": trajectories,
        "created_at": utc_now(),
    }, temporary)
    os.replace(temporary, destination)


def conformal_quantile(values: list[float], delta: float) -> tuple[float, int]:
    if not values:
        raise ValueError("empty class in conformal calibration")
    ordered = sorted(float(x) for x in values)
    # Finite-sample split-conformal corrected empirical (1-delta) quantile.
    rank = min(len(ordered), int(math.ceil((len(ordered) + 1) * (1.0 - delta))))
    return ordered[rank - 1], rank


def calibrate(trajectories, delta):
    positive_scores = []
    negative_scores = []
    ambiguous = 0
    for trajectory in trajectories:
        for row in trajectory["rows"]:
            label = row.get("strict_clean_label")
            probability = float(row["probability"])
            if label == 1:
                positive_scores.append(1.0 - probability)
            elif label == 0:
                negative_scores.append(probability)
            else:
                ambiguous += 1
    q_positive, rank_positive = conformal_quantile(positive_scores, delta)
    q_negative, rank_negative = conformal_quantile(negative_scores, delta)
    return {
        "delta": float(delta),
        "q_positive": q_positive,
        "q_negative": q_negative,
        "positive_count": len(positive_scores),
        "negative_count": len(negative_scores),
        "ambiguous_discarded": ambiguous,
        "positive_rank_one_based": rank_positive,
        "negative_rank_one_based": rank_negative,
        "positive_probability_lower_bound": 1.0 - q_positive,
        "negative_probability_exclusion_strict_bound": q_negative,
    }


def singleton_positive(probability: float, threshold: dict[str, Any]) -> bool:
    positive_in_set = (1.0 - probability) <= float(threshold["q_positive"])
    negative_in_set = probability <= float(threshold["q_negative"])
    return bool(positive_in_set and not negative_in_set)


def apply_policy(trajectory, threshold, config):
    streak = 0
    stop = None
    checks = 0
    for row in trajectory["rows"]:
        checks += 1
        accepted = singleton_positive(float(row["probability"]), threshold)
        if bool(row["is_self_doubt"]):
            if accepted:
                stop = row
                break
            # A token position carrying both types is governed by the declared
            # self-doubt priority; a rejected doubt checkpoint neither counts as
            # nor resets a sentence confirmation.
            continue
        if bool(row["is_sentence"]):
            if accepted:
                streak += 1
                if streak >= int(config["policy"]["sentence_positive_confirmations"]):
                    stop = row
                    break
            elif bool(config["policy"]["sentence_streak_reset_on_negative_sentence"]):
                streak = 0
    if stop is None:
        success = bool(trajectory["dense_success"])
        tokens = int(trajectory["dense_tokens"])
        prediction = trajectory["dense_prediction"]
    else:
        success = bool(stop["current_success"])
        tokens = int(stop["stop_total_tokens"])
        prediction = stop["current_prediction"]
    return {
        "problem_id": trajectory["problem_id"],
        "stopped": stop is not None,
        "stop_checkpoint": int(stop["checkpoint"]) if stop else None,
        "stop_type": (
            "self_doubt" if stop and stop["is_self_doubt"] else "sentence" if stop else "dense"
        ),
        "checks": checks,
        "success": success,
        "prediction": prediction,
        "tokens": tokens,
        "dense_success": bool(trajectory["dense_success"]),
        "dense_tokens": int(trajectory["dense_tokens"]),
        "lost_correct": bool(trajectory["dense_success"] and not success),
        "helped": bool((not trajectory["dense_success"]) and success),
    }


def aes(accuracy: float, tokens: float, dense_accuracy: float, dense_tokens: float) -> float:
    if dense_tokens <= 0 or dense_accuracy <= 0:
        raise ValueError("AES baseline denominator must be positive")
    saving = (dense_tokens - tokens) / dense_tokens
    if accuracy >= dense_accuracy:
        return saving + 3.0 * (accuracy - dense_accuracy) / dense_accuracy
    return saving - 5.0 * (dense_accuracy - accuracy) / dense_accuracy


def evaluate_trajectories(trajectories, threshold, config):
    decisions = [apply_policy(value, threshold, config) for value in trajectories]
    count = len(decisions)
    accuracy = sum(value["success"] for value in decisions) / count
    dense_accuracy = sum(value["dense_success"] for value in decisions) / count
    tokens = sum(value["tokens"] for value in decisions) / count
    dense_tokens = sum(value["dense_tokens"] for value in decisions) / count
    return {
        "n": count,
        "accuracy": accuracy,
        "dense_accuracy": dense_accuracy,
        "accuracy_delta_pp": 100.0 * (accuracy - dense_accuracy),
        "average_generated_response_tokens": tokens,
        "dense_average_generated_response_tokens": dense_tokens,
        "token_reduction": (dense_tokens - tokens) / dense_tokens,
        "lost_correct": sum(value["lost_correct"] for value in decisions),
        "helped": sum(value["helped"] for value in decisions),
        "stop_count": sum(value["stopped"] for value in decisions),
        "stop_rate": sum(value["stopped"] for value in decisions) / count,
        "average_checks": sum(value["checks"] for value in decisions) / count,
        "aes": aes(accuracy, tokens, dense_accuracy, dense_tokens),
        "decisions": decisions,
    }


def calibration_and_results(config, resume):
    output = ROOT / config["output_root"]
    result_path = output / "RESULTS_ALL_DELTAS.json"
    if resume and result_path.is_file():
        value = json.loads(result_path.read_text(encoding="utf-8"))
        if value.get("status") == "complete" and value.get("protocol_fingerprint") == fingerprint(config):
            print(json.dumps({"status": "skipped", "stage": "evaluate"}))
            return
        raise RuntimeError("refusing incompatible BLADE results")
    rows = []
    detailed = {}
    primary = {}
    for dataset in config["datasets"]:
        calibration = torch.load(output / dataset / "scores" / "calibration.pt", map_location="cpu", weights_only=False)
        heldout = torch.load(output / dataset / "scores" / "heldout.pt", map_location="cpu", weights_only=False)
        if calibration.get("protocol_fingerprint") != fingerprint(config) or heldout.get("protocol_fingerprint") != fingerprint(config):
            raise ValueError("score fingerprint mismatch")
        calibration_rows = []
        heldout_rows = []
        for delta in config["calibration"]["deltas"]:
            threshold = calibrate(calibration["trajectories"], float(delta))
            calibration_metrics = evaluate_trajectories(calibration["trajectories"], threshold, config)
            heldout_metrics = evaluate_trajectories(heldout["trajectories"], threshold, config)
            summary = {
                "dataset": dataset,
                "delta": float(delta),
                **threshold,
                **{f"calibration_{key}": value for key, value in calibration_metrics.items() if key != "decisions"},
                **{f"heldout_{key}": value for key, value in heldout_metrics.items() if key != "decisions"},
            }
            rows.append(summary)
            calibration_rows.append(summary)
            heldout_rows.append(summary)
            detailed[f"{dataset}:{delta}"] = {
                "threshold": threshold,
                "calibration_decisions": calibration_metrics["decisions"],
                "heldout_decisions": heldout_metrics["decisions"],
            }
        selected = max(calibration_rows, key=lambda value: (value["calibration_aes"], -value["delta"]))
        oracle = max(heldout_rows, key=lambda value: (value["heldout_aes"], -value["delta"]))
        primary[dataset] = {
            "selection_scope": "calibration_only",
            "primary_delta": selected["delta"],
            "primary_calibration_aes": selected["calibration_aes"],
            "primary_heldout_aes": selected["heldout_aes"],
            "descriptive_test_oracle_delta": oracle["delta"],
            "descriptive_test_oracle_aes": oracle["heldout_aes"],
        }
    result = {
        "status": "complete",
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "protocol_fingerprint": fingerprint(config),
        "created_at": utc_now(),
        "method": "BLADE",
        "rows": rows,
        "primary_operating_points": primary,
        "heldout_selection_forbidden": True,
        "descriptive_test_oracle_not_used_for_claims": True,
    }
    atomic_json(result, result_path)
    atomic_json({
        "status": "complete", "protocol_fingerprint": fingerprint(config), "created_at": utc_now(), "values": detailed,
    }, output / "RESULTS_DECISIONS.json")
    fieldnames = sorted({key for row in rows for key in row})
    csv_path = output / "RESULTS_ALL_DELTAS.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    lines = ["# BLADE primary held-out results", "", "Primary delta is selected on calibration AES only.", ""]
    lines.append("| Dataset | delta | Accuracy | Acc delta (pp) | Token reduction | Lost | Helped | Stop rate | AES |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset, selection in primary.items():
        row = next(value for value in rows if value["dataset"] == dataset and value["delta"] == selection["primary_delta"])
        lines.append(
            f"| {dataset} | {row['delta']:.3f} | {100*row['heldout_accuracy']:.2f}% | "
            f"{row['heldout_accuracy_delta_pp']:+.2f} | {100*row['heldout_token_reduction']:.2f}% | "
            f"{row['heldout_lost_correct']} | {row['heldout_helped']} | {100*row['heldout_stop_rate']:.2f}% | "
            f"{row['heldout_aes']:.4f} |"
        )
    (output / "RESULTS_PRIMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("score", "evaluate"), required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "mmlu_pro"))
    parser.add_argument("--split", choices=("calibration", "heldout"))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    if args.stage == "score":
        if args.dataset is None or args.split is None or args.gpu is None:
            parser.error("score requires --dataset, --split, and --gpu")
        device = torch.device(f"cuda:{args.gpu}")
        model, selected = load_model(config, device)
        score_split(config, args.dataset, args.split, model, selected, device, args.resume)
    else:
        calibration_and_results(config, args.resume)


if __name__ == "__main__":
    main()

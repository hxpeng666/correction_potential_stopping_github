#!/usr/bin/env python3
"""Pack BLADE supervision and train the dense/APLS/compact stages.

Every stage writes a fingerprinted completion record and is safe to resume.  The
selector is intentionally one seed per invocation so independent runs can be
scheduled on whatever GPUs are safely available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "src").is_dir():
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from src.utils import load_yaml, seed_everything


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
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
    """Must remain byte-for-byte aligned with collect_blade_mixed_v1.py."""
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


def stable_unit_interval(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def clean_rows(artifact: dict[str, Any]) -> list[int]:
    return [index for index, row in enumerate(artifact["rows"]) if row.get("strict_clean_label") in (0, 1)]


def artifact_paths(config: dict[str, Any], split: str) -> list[tuple[str, Path]]:
    output = ROOT / config["output_root"]
    paths = []
    for dataset in config["training"]["combined_probe_corpus"]:
        paths.extend((str(dataset), path) for path in sorted((output / dataset / "cache" / split).glob("sample_*.pt")))
    return paths


def validate_artifact(value: dict[str, Any], config: dict[str, Any], expected_fingerprint: str) -> None:
    layers = int(config["model"]["decoder_layers"])
    width = int(config["model"]["hidden_size"])
    if value.get("status") != "complete":
        raise ValueError("incomplete BLADE collection artifact")
    if value.get("protocol_fingerprint") != expected_fingerprint:
        raise ValueError("BLADE collection fingerprint mismatch")
    hidden = value.get("hidden", torch.empty(0))
    if hidden.ndim != 3 or tuple(hidden.shape[1:]) != (layers, width):
        raise ValueError(f"unexpected hidden shape {tuple(hidden.shape)}")
    if hidden.shape[0] != len(value.get("rows", [])):
        raise ValueError("row/hidden checkpoint count mismatch")


def exact_question_split(keys: list[str], fraction: float, seed: int) -> set[str]:
    ordered = sorted(set(keys), key=lambda value: (stable_unit_interval(value, seed), value))
    count = int(round(len(ordered) * fraction))
    if not 0 < count < len(ordered):
        raise ValueError("internal validation split would be empty")
    return set(ordered[:count])


def pack(config: dict[str, Any], resume: bool) -> None:
    output = ROOT / config["output_root"]
    packed = output / "packed"
    marker = packed / "PACK_COMPLETE.json"
    protocol_fingerprint = fingerprint(config)
    if resume and marker.is_file():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("protocol_fingerprint") == protocol_fingerprint:
            print(json.dumps({"status": "skipped", "stage": "pack"}))
            return
        raise RuntimeError("refusing to reuse incompatible packed BLADE data")

    paths = artifact_paths(config, "probe_train")
    expected_questions = sum(int(config["datasets"][name]["probe_train"]) for name in config["training"]["combined_probe_corpus"])
    if len(paths) != expected_questions:
        raise RuntimeError(f"expected {expected_questions} probe artifacts, found {len(paths)}")

    question_keys = []
    counts: Counter[str] = Counter()
    labels: Counter[int] = Counter()
    for dataset, path in paths:
        value = torch.load(path, map_location="cpu", weights_only=False)
        validate_artifact(value, config, collection_fingerprint(config, dataset))
        key = f"{dataset}:{value['problem_id']}"
        question_keys.append(key)
        indices = clean_rows(value)
        counts[key] = len(indices)
        for index in indices:
            labels[int(value["rows"][index]["strict_clean_label"])] += 1
    if not labels[0] or not labels[1]:
        raise RuntimeError(f"strict-clean supervision lacks a class: {dict(labels)}")

    validation_keys = exact_question_split(
        question_keys,
        float(config["training"]["internal_validation_fraction"]),
        int(config["training"]["internal_split_seed"]),
    )
    layer_count = int(config["model"]["decoder_layers"])
    hidden_size = int(config["model"]["hidden_size"])
    split_counts = {
        "train": sum(count for key, count in counts.items() if key not in validation_keys),
        "validation": sum(count for key, count in counts.items() if key in validation_keys),
    }
    tensors = {
        name: {
            "hidden": torch.empty((count, layer_count, hidden_size), dtype=torch.float16),
            "label": torch.empty((count,), dtype=torch.int8),
        }
        for name, count in split_counts.items()
    }
    offsets = {"train": 0, "validation": 0}
    split_label_counts = {"train": Counter(), "validation": Counter()}
    split_question_counts = Counter()
    for dataset, path in paths:
        value = torch.load(path, map_location="cpu", weights_only=False)
        key = f"{dataset}:{value['problem_id']}"
        name = "validation" if key in validation_keys else "train"
        split_question_counts[name] += 1
        indices = clean_rows(value)
        if not indices:
            continue
        start = offsets[name]
        end = start + len(indices)
        tensors[name]["hidden"][start:end] = value["hidden"][indices].to(torch.float16)
        row_labels = torch.tensor(
            [int(value["rows"][index]["strict_clean_label"]) for index in indices], dtype=torch.int8
        )
        tensors[name]["label"][start:end] = row_labels
        split_label_counts[name].update(int(x) for x in row_labels.tolist())
        offsets[name] = end
    if offsets != split_counts:
        raise AssertionError(f"pack offsets do not match allocation: {offsets} != {split_counts}")

    for name, value in tensors.items():
        atomic_torch(
            {
                "schema_version": 1,
                "status": "complete",
                "protocol_fingerprint": protocol_fingerprint,
                "split": name,
                "hidden": value["hidden"],
                "label": value["label"],
            },
            packed / f"{name}.pt",
        )
    record = {
        "status": "complete",
        "stage": "pack",
        "protocol_fingerprint": protocol_fingerprint,
        "created_at": utc_now(),
        "probe_artifacts": len(paths),
        "questions": dict(split_question_counts),
        "checkpoints": split_counts,
        "labels": {name: {str(key): int(value) for key, value in counter.items()} for name, counter in split_label_counts.items()},
        "question_split_disjoint": True,
    }
    atomic_json(record, marker)
    print(json.dumps(record, indent=2))


class DenseTeacher(nn.Module):
    def __init__(self, layers: int, hidden: int, projection: int, head: list[int]):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.projection = nn.Linear(hidden, projection)
        dimensions = [layers * projection, *head, 1]
        modules: list[nn.Module] = []
        for index in range(len(dimensions) - 1):
            modules.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                modules.append(nn.GELU())
        self.head = nn.Sequential(*modules)

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        values = [torch.nn.functional.gelu(self.projection(norm(hidden[:, index]))) for index, norm in enumerate(self.norms)]
        return torch.stack(values, dim=1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(hidden)
        return self.head(encoded.flatten(1)).squeeze(-1)


class SelectionModel(nn.Module):
    def __init__(self, layers: int, projection: int, selected: int, head: list[int], temperature: float):
        super().__init__()
        self.layers = layers
        self.selected = selected
        self.temperature = temperature
        self.alpha = nn.Parameter(torch.zeros(layers))
        dimensions = [layers * projection, *head, 1]
        modules: list[nn.Module] = []
        for index in range(len(dimensions) - 1):
            modules.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                modules.append(nn.GELU())
        self.head = nn.Sequential(*modules)

    def probabilities(self) -> torch.Tensor:
        return torch.softmax(self.alpha / self.temperature, dim=0)

    def mask(self) -> torch.Tensor:
        probability = self.probabilities()
        hard = torch.zeros_like(probability)
        hard.scatter_(0, torch.topk(probability, self.selected).indices, 1.0)
        return hard + probability - probability.detach()

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        masked = encoded * self.mask().view(1, self.layers, 1)
        return self.head(masked.flatten(1)).squeeze(-1)


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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def dense_model(config: dict[str, Any]) -> DenseTeacher:
    return DenseTeacher(
        int(config["model"]["decoder_layers"]),
        int(config["model"]["hidden_size"]),
        int(config["apls"]["projection_width"]),
        [int(x) for x in config["apls"]["dense_head_dims"]],
    )


def load_packed(config: dict[str, Any], name: str) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.load(ROOT / config["output_root"] / "packed" / f"{name}.pt", map_location="cpu", weights_only=False)
    if value.get("protocol_fingerprint") != fingerprint(config):
        raise ValueError("packed BLADE data fingerprint mismatch")
    return value["hidden"], value["label"].to(torch.float32)


def batches(length: int, batch_size: int, generator: torch.Generator | None = None) -> Iterable[torch.Tensor]:
    order = torch.randperm(length, generator=generator) if generator is not None else torch.arange(length)
    for start in range(0, length, batch_size):
        yield order[start : start + batch_size]


def balanced_loss(logits: torch.Tensor, labels: torch.Tensor, positive_weight: float) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=torch.tensor(positive_weight, device=logits.device)
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.inference_mode()
def predict(model: nn.Module, features: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    result = []
    for indices in batches(len(features), batch_size):
        value = features[indices].to(device, non_blocking=True).float()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(value)
        result.append(logits.float().cpu())
    return torch.cat(result).numpy()


def binary_training_loop(
    model: nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    epochs: int,
    extra_loss=None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    positive = float(train_labels.sum().item())
    negative = float(len(train_labels) - positive)
    if positive == 0 or negative == 0:
        raise RuntimeError("training split lacks a class")
    positive_weight = negative / positive
    batch_size = int(config["training"]["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    best = {"epoch": -1, "validation_auroc": -math.inf, "state_dict": None}
    history = []
    scaler = torch.amp.GradScaler("cuda")
    for epoch in range(epochs):
        model.train()
        losses = []
        for indices in batches(len(train_features), batch_size, generator):
            x = train_features[indices].to(device, non_blocking=True).float()
            y = train_labels[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = balanced_loss(logits, y, positive_weight)
                if extra_loss is not None:
                    loss = loss + extra_loss(indices, logits)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation_logits = predict(model, validation_features, device, batch_size)
        score = float(roc_auc_score(validation_labels.numpy(), validation_logits))
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_auroc": score}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best["validation_auroc"]:
            best = {
                "epoch": epoch + 1,
                "validation_auroc": score,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
    if best["state_dict"] is None:
        raise AssertionError("no best model selected")
    return best, history


def train_dense(config: dict[str, Any], gpu: int, resume: bool) -> None:
    output = ROOT / config["output_root"] / "models"
    path = output / "dense_teacher.pt"
    protocol_fingerprint = fingerprint(config)
    if resume and path.is_file():
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("status") == "complete" and value.get("protocol_fingerprint") == protocol_fingerprint:
            print(json.dumps({"status": "skipped", "stage": "dense"}))
            return
        raise RuntimeError("refusing incompatible dense teacher")
    seed_everything(int(config["seed"]))
    train_x, train_y = load_packed(config, "train")
    val_x, val_y = load_packed(config, "validation")
    device = torch.device(f"cuda:{gpu}")
    model = dense_model(config)
    best, history = binary_training_loop(
        model, train_x, train_y, val_x, val_y, config, device,
        int(config["seed"]), int(config["training"]["epochs_dense"]),
    )
    atomic_torch({
        "schema_version": 1,
        "status": "complete",
        "stage": "dense_teacher",
        "protocol_fingerprint": protocol_fingerprint,
        "created_at": utc_now(),
        "best_epoch": best["epoch"],
        "validation_auroc": best["validation_auroc"],
        "parameter_count": parameter_count(model),
        "state_dict": best["state_dict"],
        "history": history,
    }, path)


@torch.inference_mode()
def project_split(model: DenseTeacher, hidden: torch.Tensor, device: torch.device, batch_size: int):
    encoded = []
    logits = []
    model.eval().to(device)
    for indices in batches(len(hidden), batch_size):
        value = hidden[indices].to(device).float()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            representation = model.encode(value)
            prediction = model.head(representation.flatten(1)).squeeze(-1)
        encoded.append(representation.to(torch.float16).cpu())
        logits.append(prediction.to(torch.float32).cpu())
    return torch.cat(encoded), torch.cat(logits)


def project_teacher(config: dict[str, Any], gpu: int, resume: bool) -> None:
    output = ROOT / config["output_root"]
    marker = output / "projected" / "PROJECT_COMPLETE.json"
    protocol_fingerprint = fingerprint(config)
    if resume and marker.is_file():
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value.get("status") == "complete" and value.get("protocol_fingerprint") == protocol_fingerprint:
            print(json.dumps({"status": "skipped", "stage": "project"}))
            return
        raise RuntimeError("refusing incompatible projected teacher cache")
    dense = torch.load(output / "models" / "dense_teacher.pt", map_location="cpu", weights_only=False)
    if dense.get("protocol_fingerprint") != protocol_fingerprint:
        raise ValueError("dense teacher fingerprint mismatch")
    model = dense_model(config)
    model.load_state_dict(dense["state_dict"])
    counts = {}
    for name in ("train", "validation"):
        hidden, labels = load_packed(config, name)
        encoded, logits = project_split(
            model, hidden, torch.device(f"cuda:{gpu}"), int(config["training"]["batch_size"])
        )
        atomic_torch({
            "schema_version": 1,
            "status": "complete",
            "protocol_fingerprint": protocol_fingerprint,
            "split": name,
            "encoded": encoded,
            "teacher_logits": logits,
            "label": labels.to(torch.int8),
        }, output / "projected" / f"{name}.pt")
        counts[name] = len(labels)
    atomic_json({
        "status": "complete", "stage": "project", "protocol_fingerprint": protocol_fingerprint,
        "created_at": utc_now(), "counts": counts,
    }, marker)


def load_projected(config: dict[str, Any], name: str):
    value = torch.load(ROOT / config["output_root"] / "projected" / f"{name}.pt", map_location="cpu", weights_only=False)
    if value.get("protocol_fingerprint") != fingerprint(config):
        raise ValueError("projected teacher cache fingerprint mismatch")
    return value["encoded"], value["teacher_logits"], value["label"].to(torch.float32)


def train_selector(config: dict[str, Any], gpu: int, selector_index: int, resume: bool) -> None:
    seeds = [int(x) for x in config["apls"]["selection_seeds"]]
    if not 0 <= selector_index < len(seeds):
        raise ValueError("selector index out of range")
    seed = seeds[selector_index]
    output = ROOT / config["output_root"] / "selectors"
    path = output / f"selector_{selector_index:02d}.pt"
    protocol_fingerprint = fingerprint(config)
    if resume and path.is_file():
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("status") == "complete" and value.get("protocol_fingerprint") == protocol_fingerprint:
            print(json.dumps({"status": "skipped", "stage": "selector", "index": selector_index}))
            return
        raise RuntimeError("refusing incompatible selector")
    seed_everything(seed)
    train_x, train_teacher, train_y = load_projected(config, "train")
    val_x, _val_teacher, val_y = load_projected(config, "validation")
    model = SelectionModel(
        int(config["model"]["decoder_layers"]),
        int(config["apls"]["projection_width"]),
        int(config["apls"]["selected_layers"]),
        [int(x) for x in config["apls"]["dense_head_dims"]],
        float(config["apls"]["gate_softmax_temperature"]),
    )
    device = torch.device(f"cuda:{gpu}")
    teacher_probability = torch.sigmoid(
        train_teacher / float(config["apls"]["kd_temperature"])
    )

    def kd(indices: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        target = teacher_probability[indices].to(device)
        temperature = float(config["apls"]["kd_temperature"])
        value = torch.nn.functional.binary_cross_entropy_with_logits(logits / temperature, target)
        return float(config["apls"]["kd_weight"]) * (temperature**2) * value

    best, history = binary_training_loop(
        model, train_x, train_y, val_x, val_y, config, device, seed,
        int(config["training"]["epochs_selector"]), kd,
    )
    model.load_state_dict(best["state_dict"])
    probability = model.probabilities().detach().cpu()
    order = torch.argsort(probability, descending=True)
    selected = sorted(int(x) for x in order[: int(config["apls"]["selected_layers"])].tolist())
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order))
    atomic_torch({
        "schema_version": 1,
        "status": "complete",
        "stage": "selector",
        "protocol_fingerprint": protocol_fingerprint,
        "created_at": utc_now(),
        "selector_index": selector_index,
        "seed": seed,
        "best_epoch": best["epoch"],
        "validation_auroc": best["validation_auroc"],
        "parameter_count": parameter_count(model),
        "selected_layers_zero_based": selected,
        "gate_probability": probability,
        "gate_rank_zero_based": ranks,
        "state_dict": best["state_dict"],
        "history": history,
    }, path)


def aggregate(config: dict[str, Any], resume: bool) -> None:
    output = ROOT / config["output_root"]
    path = output / "models" / "selected_layers.json"
    protocol_fingerprint = fingerprint(config)
    if resume and path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "complete" and value.get("protocol_fingerprint") == protocol_fingerprint:
            print(json.dumps({"status": "skipped", "stage": "aggregate"}))
            return
        raise RuntimeError("refusing incompatible selected-layer aggregation")
    values = []
    expected = len(config["apls"]["selection_seeds"])
    for index in range(expected):
        value = torch.load(output / "selectors" / f"selector_{index:02d}.pt", map_location="cpu", weights_only=False)
        if value.get("status") != "complete" or value.get("protocol_fingerprint") != protocol_fingerprint:
            raise ValueError(f"invalid selector {index}")
        values.append(value)
    layer_count = int(config["model"]["decoder_layers"])
    frequency = [0] * layer_count
    rank_sum = [0.0] * layer_count
    for value in values:
        for layer in value["selected_layers_zero_based"]:
            frequency[int(layer)] += 1
        ranks = value["gate_rank_zero_based"].tolist()
        for layer in range(layer_count):
            rank_sum[layer] += float(ranks[layer])
    mean_rank = [value / len(values) for value in rank_sum]
    order = sorted(range(layer_count), key=lambda layer: (-frequency[layer], mean_rank[layer], layer))
    selected = sorted(order[: int(config["apls"]["selected_layers"])])
    record = {
        "status": "complete",
        "stage": "aggregate",
        "protocol_fingerprint": protocol_fingerprint,
        "created_at": utc_now(),
        "runs": expected,
        "frequency_count": frequency,
        "frequency_fraction": [value / expected for value in frequency],
        "mean_gate_rank_zero_based": mean_rank,
        "tie_break": "descending_frequency_then_ascending_mean_gate_rank_then_layer_index",
        "selected_layers_zero_based": selected,
        "paper_reported_qwen3_4b_layers_zero_based": [19, 21, 22, 27],
    }
    atomic_json(record, path)
    print(json.dumps(record, indent=2))


def train_compact(config: dict[str, Any], gpu: int, resume: bool) -> None:
    output = ROOT / config["output_root"]
    path = output / "models" / "compact_probe.pt"
    protocol_fingerprint = fingerprint(config)
    if resume and path.is_file():
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("status") == "complete" and value.get("protocol_fingerprint") == protocol_fingerprint:
            print(json.dumps({"status": "skipped", "stage": "compact"}))
            return
        raise RuntimeError("refusing incompatible compact probe")
    seed_everything(int(config["seed"]) + 1000)
    selected_record = json.loads((output / "models" / "selected_layers.json").read_text(encoding="utf-8"))
    if selected_record.get("protocol_fingerprint") != protocol_fingerprint:
        raise ValueError("selected layers fingerprint mismatch")
    selected = [int(x) for x in selected_record["selected_layers_zero_based"]]
    train_hidden, train_y = load_packed(config, "train")
    val_hidden, val_y = load_packed(config, "validation")
    train_x = train_hidden[:, selected, :].flatten(1).contiguous()
    val_x = val_hidden[:, selected, :].flatten(1).contiguous()
    model = CompactProbe(
        len(selected) * int(config["model"]["hidden_size"]),
        [int(x) for x in config["apls"]["compact_head_dims"]],
    )
    best, history = binary_training_loop(
        model, train_x, train_y, val_x, val_y, config, torch.device(f"cuda:{gpu}"),
        int(config["seed"]) + 1000, int(config["training"]["epochs_compact"]),
    )
    atomic_torch({
        "schema_version": 1,
        "status": "complete",
        "stage": "compact_probe",
        "protocol_fingerprint": protocol_fingerprint,
        "created_at": utc_now(),
        "selected_layers_zero_based": selected,
        "best_epoch": best["epoch"],
        "validation_auroc": best["validation_auroc"],
        "parameter_count": parameter_count(model),
        "state_dict": best["state_dict"],
        "history": history,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage", choices=("pack", "dense", "project", "selector", "aggregate", "compact"), required=True
    )
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--selector-index", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    gpu_stages = {"dense", "project", "selector", "compact"}
    if args.stage in gpu_stages and args.gpu is None:
        parser.error(f"--gpu is required for {args.stage}")
    if args.stage == "selector" and args.selector_index is None:
        parser.error("--selector-index is required for selector")
    if args.stage == "pack":
        pack(config, args.resume)
    elif args.stage == "dense":
        train_dense(config, args.gpu, args.resume)
    elif args.stage == "project":
        project_teacher(config, args.gpu, args.resume)
    elif args.stage == "selector":
        train_selector(config, args.gpu, args.selector_index, args.resume)
    elif args.stage == "aggregate":
        aggregate(config, args.resume)
    elif args.stage == "compact":
        train_compact(config, args.gpu, args.resume)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score original-v2 method-axis probes on uncensored calibration/test caches.

The script never trains or changes a probe.  It groups probes by representation
and fitted feature transform so large external feature matrices are constructed
once per identical transform, then saves aligned score vectors for later LTT
calibration.  It is resumable at the split level.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from src.deepseek7b_method_exploration_v1 import (
    FeatureTransform,
    RiskProbe,
    predict_scores,
)
from src.final_paper_inference import atomic_torch_save
from src.legacy_empirical_probe_normalized_v1 import load_checkpoint_split
from src.reproducibility import (
    code_provenance,
    environment_provenance,
    strict_reproducibility,
)
from train_deepseek7b_method_exploration_v1 import (
    load_representation,
    prepare_cost_columns,
)


AXES = ("weight", "robust", "reach", "feature", "aux_feature")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform_fingerprint(state: dict[str, Any]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def model_specs(source: Path, dataset: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for axis in AXES:
        for report_path in sorted((source / "screen" / dataset / axis).glob("*/probe.json")):
            report = json.loads(report_path.read_text())
            model_path = report_path.with_name("probe.pt")
            train_scores = report_path.with_name("scores.pt")
            if report.get("status") != "complete" or not report.get("screen_only"):
                raise ValueError(f"invalid frozen screen report: {report_path}")
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            invocation = report["invocation"]
            needs_readout = "one_step" in invocation["feature_kind"]
            specs.append(
                {
                    "axis": axis,
                    "label": report_path.parent.name,
                    "model_dir": report_path.parent,
                    "report_path": report_path,
                    "model_path": model_path,
                    "train_scores_path": train_scores,
                    "report_sha256": sha256(report_path),
                    "model_sha256": sha256(model_path),
                    "train_scores_sha256": sha256(train_scores),
                    "representation_kind": invocation["representation_kind"],
                    "feature_kind": invocation["feature_kind"],
                    "readout_kind": invocation.get("readout_kind", "full"),
                    "needs_readout": needs_readout,
                    "probe_architecture": checkpoint["probe_architecture"],
                    "checkpoint": checkpoint,
                    "transform_fingerprint": transform_fingerprint(checkpoint["feature_transform"]),
                }
            )
    if len(specs) != 149:
        raise AssertionError(f"expected 149 axis models, got {len(specs)}")
    return specs


def output_path(output: Path, spec: dict[str, Any], dataset: str) -> Path:
    return output / "scores" / spec["axis"] / spec["label"] / f"{dataset}.pt"


def load_saved(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "partial",
            "source": {
                "report": str(spec["report_path"]),
                "report_sha256": spec["report_sha256"],
                "model": str(spec["model_path"]),
                "model_sha256": spec["model_sha256"],
                "probe_train_scores": str(spec["train_scores_path"]),
                "probe_train_scores_sha256": spec["train_scores_sha256"],
            },
            "scores": {},
            "keys": {},
        }
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["source"]["model_sha256"] != spec["model_sha256"]:
        raise ValueError(f"incompatible resumed score file: {path}")
    return value


def score_split(
    specs: list[dict[str, Any]],
    *,
    dataset: str,
    split_name: str,
    raw_directory: Path,
    auxiliary_directory: Path,
    output: Path,
    device: torch.device,
) -> None:
    pending = []
    for spec in specs:
        saved = load_saved(output_path(output, spec, dataset), spec)
        if split_name not in saved["scores"]:
            pending.append(spec)
    if not pending:
        return
    frame, cached_hidden, layers, fallbacks = load_checkpoint_split(raw_directory, "sentence")
    prepare_cost_columns(frame, fallbacks)
    keys = {
        "problem_ids": frame.problem_id.astype(str).tolist(),
        "checkpoints": frame.checkpoint.astype(int).tolist(),
    }
    signatures: dict[tuple[str, bool, str], list[dict[str, Any]]] = {}
    for spec in pending:
        signature = (
            str(spec["representation_kind"]),
            bool(spec["needs_readout"]),
            str(spec["readout_kind"]),
        )
        signatures.setdefault(signature, []).append(spec)
    for (representation_kind, needs_readout, readout_kind), signature_specs in signatures.items():
        representative_kind = signature_specs[0]["feature_kind"]
        if needs_readout and "one_step" not in representative_kind:
            raise AssertionError("invalid readout signature")
        hidden, readout, _ = load_representation(
            frame,
            cached_hidden,
            layers,
            layer=16,
            representation_kind=representation_kind,
            feature_kind=representative_kind,
            readout_kind=readout_kind,
            auxiliary_root=(auxiliary_directory if representation_kind != "last" or needs_readout else None),
        )
        by_transform: dict[str, list[dict[str, Any]]] = {}
        for spec in signature_specs:
            by_transform.setdefault(spec["transform_fingerprint"], []).append(spec)
        for transform_id, transform_specs in by_transform.items():
            transform = FeatureTransform.from_state_dict(
                transform_specs[0]["checkpoint"]["feature_transform"]
            )
            features = transform.transform(frame, hidden, readout)
            for spec in transform_specs:
                checkpoint = spec["checkpoint"]
                if transform_fingerprint(checkpoint["feature_transform"]) != transform_id:
                    raise AssertionError("feature-transform group changed")
                model = RiskProbe(transform.input_width, checkpoint["probe_architecture"])
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                model.to(device)
                scores = predict_scores(model, features, device, batch_size=2048)
                path = output_path(output, spec, dataset)
                saved = load_saved(path, spec)
                saved["scores"][split_name] = torch.from_numpy(scores.astype(np.float32))
                saved["keys"][split_name] = keys
                saved["updated_at"] = datetime.now(timezone.utc).isoformat()
                atomic_torch_save(saved, path)
                del model, scores
            del features
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if hidden is not cached_hidden:
            del hidden
        if readout is not None:
            del readout
        gc.collect()
    del cached_hidden, frame
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--aux-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    reproducibility = strict_reproducibility(seed=0, num_threads=1)
    code_identity = code_provenance(
        ROOT,
        (
            "scripts/score_deepseek7b_axis_external_original_v2_v1.py",
            "scripts/train_deepseek7b_method_exploration_v1.py",
            "src/deepseek7b_method_exploration_v1.py",
            "src/reproducibility.py",
        ),
    )
    source = args.source.resolve()
    output = args.output.resolve()
    data_root = args.data_root.resolve()
    auxiliary_root = args.aux_root.resolve()
    specs = model_specs(source, args.dataset)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    runtime_identity = environment_provenance(device)
    split_specs = (
        [
            (
                "calibration",
                data_root / "gsm8k/calibration",
                auxiliary_root / "gsm8k/calibration",
            ),
            (
                "heldout",
                data_root / "gsm8k/heldout",
                auxiliary_root / "gsm8k_heldout/heldout",
            ),
        ]
        if args.dataset == "gsm8k"
        else [
            (
                "calibration",
                data_root / "math/calibration",
                auxiliary_root / "math/calibration",
            ),
            (
                "heldout",
                data_root / "math500/heldout",
                auxiliary_root / "math500/heldout",
            ),
            (
                "ood",
                data_root / "aime/heldout",
                auxiliary_root / "aime/heldout",
            ),
        ]
    )
    manifest_path = output / f"{args.dataset.upper()}_SCORING_MANIFEST.json"
    atomic_json(
        {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "scientific_protocol": "uncensored_original_v2",
            "right_censoring": False,
            "dataset": args.dataset,
            "gpu": args.gpu,
            "models": len(specs),
            "splits": [name for name, _, _ in split_specs],
            "reproducibility": reproducibility,
            "code_identity": code_identity,
            "environment": runtime_identity,
            "data_root": str(data_root),
            "aux_root": str(auxiliary_root),
        },
        manifest_path,
    )
    for name, raw, auxiliary in split_specs:
        score_split(
            specs,
            dataset=args.dataset,
            split_name=name,
            raw_directory=raw,
            auxiliary_directory=auxiliary,
            output=output,
            device=device,
        )
    for spec in specs:
        path = output_path(output, spec, args.dataset)
        saved = load_saved(path, spec)
        expected = {name for name, _, _ in split_specs}
        if set(saved["scores"]) != expected:
            raise AssertionError(f"incomplete external scores: {path}")
        saved["status"] = "complete"
        saved["completed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_torch_save(saved, path)
    atomic_json(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "scientific_protocol": "uncensored_original_v2",
            "right_censoring": False,
            "dataset": args.dataset,
            "gpu": args.gpu,
            "models": len(specs),
            "splits": [name for name, _, _ in split_specs],
            "reproducibility": reproducibility,
            "code_identity": code_identity,
            "environment": runtime_identity,
            "data_root": str(data_root),
            "aux_root": str(auxiliary_root),
        },
        manifest_path,
    )
    print(json.dumps({"status": "complete", "dataset": args.dataset, "models": len(specs)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate one frozen MATH probe/calibration policy on another OOD test set."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from deepseek7b_protocol_v1 import canonical_fingerprint
from train_deepseek7b_ablation_v1 import (
    apply_semantic_answer_targets,
    artifact_manifest,
    build_dynamic_features,
    evaluate_frozen,
)
from src.final_paper_inference import atomic_torch_save
from src.legacy_empirical_probe_normalized_v1 import (
    FinalPaperProbe,
    load_checkpoint_split,
    predict_scores,
    safe_ap_auc,
    target_values,
)
from src.reproducibility import (
    code_provenance,
    environment_provenance,
    sha256_array,
    sha256_json,
    strict_reproducibility,
)
from src.utils import atomic_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("aime",), required=True)
    parser.add_argument("--source-probe", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    reproducibility = strict_reproducibility(seed=0, num_threads=1)
    project = Path(__file__).resolve().parents[1]
    code_identity = code_provenance(
        project,
        (
            "scripts/evaluate_deepseek7b_ood_v2.py",
            "scripts/train_deepseek7b_ablation_v1.py",
            "src/reproducibility.py",
            "src/legacy_empirical_probe_normalized_v1.py",
        ),
    )
    source_json_path = args.source_probe / "probe.json"
    source_pt_path = args.source_probe / "probe.pt"
    source_marker_path = args.source_probe / "phase.complete"
    for path in (source_json_path, source_pt_path, source_marker_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    heldout_manifest = artifact_manifest(args.heldout_root)
    invocation = {
        "dataset": args.dataset,
        "source_probe": str(args.source_probe.resolve()),
        "source_probe_json_sha256": sha256(source_json_path),
        "source_probe_pt_sha256": sha256(source_pt_path),
        "source_marker_sha256": sha256(source_marker_path),
        "heldout_root": str(args.heldout_root.resolve()),
        "heldout_input": heldout_manifest,
        "protocol": "same frozen MATH probe and calibration thresholds; OOD evaluation only",
        "reproducibility_protocol": reproducibility,
        "code_identity": code_identity,
    }
    invocation_fingerprint = canonical_fingerprint(invocation)
    marker_path = args.output / "phase.complete"
    if args.resume and marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("status") == "complete"
            and marker.get("invocation_fingerprint") == invocation_fingerprint
            and (args.output / "probe.json").is_file()
        ):
            print(json.dumps({"status": "skipped_complete", "output": str(args.output)}))
            return
        raise RuntimeError(f"refusing incompatible OOD resume: {args.output}")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite OOD output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    source_report = json.loads(source_json_path.read_text(encoding="utf-8"))
    source_model = torch.load(source_pt_path, map_location="cpu", weights_only=False)
    if source_report.get("status") != "complete" or source_model.get("status") != "complete":
        raise ValueError("source probe is incomplete")
    method = str(source_report["run_spec"]["method"])
    direction = str(source_report["run_spec"]["stop_direction"])
    schedule = str(source_report["run_spec"]["schedule"])
    layer = int(source_report["run_spec"]["layer"])
    feature_kind = str(source_report["run_spec"]["feature_kind"])

    frame, hidden, layers, fallbacks = load_checkpoint_split(
        args.heldout_root / "heldout", schedule
    )
    if method in {"consistency", "last_switch"}:
        frame = apply_semantic_answer_targets(frame)
    if list(source_model["capture_layers"]) != layers:
        raise ValueError("capture-layer mismatch between source probe and OOD cache")
    raw = build_dynamic_features(
        frame, hidden, layers, layer=layer, feature_kind=feature_kind
    )
    del hidden
    mean = source_model["scaler_mean"].numpy().astype(np.float32)
    scale = source_model["scaler_scale"].numpy().astype(np.float32)
    if raw.shape[1] != len(mean) or len(mean) != len(scale):
        raise ValueError("source scaler/OOD feature width mismatch")
    features = ((raw - mean) / scale).astype(np.float32, copy=False)
    del raw

    # Match the primary token-only evaluation protocol exactly.
    frame["branch_tokens"] = 0
    frame["replay_stop_wall_ms"] = frame.checkpoint.astype(float)
    frame["dense_wall_ms"] = frame.dense_tokens.astype(float)
    frame["adaptive_fallback_wall_ms"] = frame.dense_tokens.astype(float)
    for fallback in fallbacks:
        fallback["dense_wall_ms"] = float(fallback["dense_tokens"])
        fallback["adaptive_fallback_wall_ms"] = float(fallback["dense_tokens"])

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    runtime_identity = environment_provenance(device)
    model = FinalPaperProbe(int(source_model["input_width"])).to(device)
    model.load_state_dict(source_model["state_dict"])
    scores = predict_scores(model, features, device)
    labels = target_values(frame, method)
    evaluated, policy_records = evaluate_frozen(
        frame,
        scores,
        direction,
        source_report["calibration"],
        fallbacks,
    )
    heldout_ap, heldout_auc = safe_ap_auc(labels, scores)
    payload = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_spec": {
            **source_report["run_spec"],
            "dataset": args.dataset,
            "evaluation_mode": "frozen_shared_math_probe",
            "source_probe": str(args.source_probe.resolve()),
            "no_retraining": True,
            "no_recalibration": True,
        },
        "source_probe": invocation,
        "heldout_input": heldout_manifest,
        "split_counts": {
            "heldout": {
                "problems": int(frame.problem_id.nunique()) + len(fallbacks),
                "scorable_problems": int(frame.problem_id.nunique()),
                "fallback_only_problems": len(fallbacks),
                "checkpoints": len(frame),
                "positive_labels": int(labels.sum()),
                "missing_forced_answers": int(frame.current_prediction.isna().sum()),
            }
        },
        "heldout_label_ap_descriptive": heldout_ap,
        "heldout_label_auc_descriptive": heldout_auc,
        "calibration": source_report["calibration"],
        "frozen_policy_results": evaluated,
        "online_workpoints": source_report["online_workpoints"],
        "reproducibility": {
            "settings": reproducibility,
            "code": code_identity,
            "environment": runtime_identity,
            "input": {
                "row_keys_sha256": sha256_json(
                    list(
                        zip(
                            frame.problem_id.astype(str).tolist(),
                            frame.checkpoint.astype(int).tolist(),
                        )
                    )
                ),
                "features_sha256": sha256_array(features),
                "labels_sha256": sha256_array(labels),
            },
            "scores_sha256": sha256_array(scores),
        },
    }
    atomic_json(payload, args.output / "probe.json")
    atomic_torch_save(
        {
            "status": "complete",
            "scores": {"heldout": torch.from_numpy(scores.astype(np.float32))},
            "problem_ids": {"heldout": frame.problem_id.astype(str).tolist()},
            "checkpoints": {"heldout": frame.checkpoint.astype(int).tolist()},
        },
        args.output / "scores.pt",
    )
    atomic_torch_save(
        {"status": "complete", "records": policy_records},
        args.output / "policy_records.pt",
    )
    atomic_json(
        {
            "status": "complete",
            "invocation_fingerprint": invocation_fingerprint,
            "source_probe": str(args.source_probe.resolve()),
            "no_retraining": True,
            "no_recalibration": True,
            "artifacts": ["probe.json", "scores.pt", "policy_records.pt"],
        },
        marker_path,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "dataset": args.dataset,
                "source_probe": str(args.source_probe),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

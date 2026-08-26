#!/usr/bin/env python3
"""将冻结 MMLU probes/阈值原样迁移到 MMLU-Pro held-out。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_paper_inference import atomic_torch_save
from src.legacy_empirical_probe_v4 import FinalPaperProbe, build_features, load_checkpoint_split, predict_scores, safe_ap_auc, simulate_policy, target_values
from src.utils import atomic_json, load_yaml

METHODS = {
    "correctness": ("correctness", "bce"),
    "consistency": ("consistency", "bce"),
    "last_switch": ("last_switch", "bce"),
    "correction_bce": ("correction", "bce"),
    "correction_trajectory": ("correction", "bce_traj"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_paper_mmlu_pro_transfer_v1.yaml")
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(ROOT / args.config)
    source_probe_root = ROOT / config["transfer"]["source_probe_root"]
    frame, hidden, layers, fallbacks = load_checkpoint_split(args.replay_root / "heldout", "sentence")
    raw_features = build_features(frame, hidden, layers, layer=20, feature_kind="full")
    del hidden
    torch.cuda.set_device(args.gpu); device = torch.device(f"cuda:{args.gpu}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {"status":"complete","created_at":datetime.now(timezone.utc).isoformat(),"report_label":config["report_label"],
               "heldout_problems":int(frame.problem_id.nunique())+len(fallbacks),"scorable_problems":int(frame.problem_id.nunique()),
               "fallback_only_problems":len(fallbacks),"checkpoints":len(frame),"threshold_source":"frozen_MMLU_calibration_500","methods":{}}
    for output_name, (target_method, expected_loss) in METHODS.items():
        destination = args.output_root / output_name
        marker = destination / "phase.complete"
        if args.resume and marker.is_file():
            summary["methods"][output_name] = json.loads((destination / "transfer.json").read_text(encoding="utf-8")); continue
        if destination.exists() and any(destination.iterdir()): raise RuntimeError(f"拒绝覆盖：{destination}")
        destination.mkdir(parents=True, exist_ok=True)
        probe_dir = source_probe_root / output_name
        probe = torch.load(probe_dir / "probe.pt", map_location="cpu", weights_only=False)
        calibration = json.loads((probe_dir / "probe.json").read_text(encoding="utf-8"))
        run_spec = probe["run_spec"]
        if run_spec["dataset"] != "mmlu" or run_spec["method"] != target_method or run_spec["loss"] != expected_loss:
            raise ValueError(f"冻结 probe 身份错误：{output_name}")
        mean = probe["scaler_mean"].numpy(); scale = probe["scaler_scale"].numpy()
        features = ((raw_features - mean) / scale).astype(np.float32, copy=False)
        model = FinalPaperProbe(int(probe["input_width"])).to(device); model.load_state_dict(probe["state_dict"])
        scores = predict_scores(model, features, device)
        direction = run_spec["stop_direction"]
        families = {}; record_families = {}
        for family in ("empirical_B", "coverage"):
            families[family] = {}; record_families[family] = {}
            for key, frozen in calibration["calibration"][family].items():
                result = simulate_policy(frame, scores, direction, float(frozen["threshold"]), include_records=True,
                                         fallback_records=fallbacks, force_dense=bool(frozen.get("is_no_stop_sentinel", False)))
                records = result.pop("records")
                families[family][key] = {"source_MMLU_calibration": frozen, "MMLU_Pro_heldout": result}
                record_families[family][key] = records
        labels = target_values(frame, target_method); ap, auc = safe_ap_auc(labels, scores)
        payload = {"status":"complete","method":output_name,"target":target_method,"loss":expected_loss,"direction":direction,
                   "source_probe":str(probe_dir.resolve()),"source_protocol_id":run_spec["protocol_id"],
                   "threshold_source":"frozen MMLU calibration; no MMLU-Pro threshold selection","heldout_label_AP_descriptive":ap,
                   "heldout_label_AUC_descriptive":auc,"positive_labels":int(labels.sum()),"results":families}
        atomic_json(payload, destination / "transfer.json")
        atomic_torch_save({"status":"complete","scores":torch.from_numpy(scores.astype(np.float32)),"problem_ids":frame.problem_id.astype(str).tolist(),
                           "checkpoints":frame.checkpoint.astype(int).tolist(),"records":record_families}, destination / "transfer_records.pt")
        atomic_json({"status":"complete","source_probe":str(probe_dir.resolve()),"threshold_source":"MMLU calibration 500"}, marker)
        summary["methods"][output_name] = payload
        del model, features; torch.cuda.empty_cache()
    atomic_json(summary, args.output_root / "evaluation.complete"); print(json.dumps({"status":"complete","methods":list(summary["methods"]),"heldout":summary["heldout_problems"]}, indent=2))


if __name__ == "__main__":
    main()

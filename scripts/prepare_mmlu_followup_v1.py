#!/usr/bin/env python3
"""Freeze the MMLU-Pro follow-up config from GSM8K calibration selections."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from src.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm-config", required=True)
    parser.add_argument("--gsm-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gsm_config_path = Path(args.gsm_config)
    if not gsm_config_path.is_absolute():
        gsm_config_path = ROOT / gsm_config_path
    summary_path = args.gsm_summary if args.gsm_summary.is_absolute() else ROOT / args.gsm_summary
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    base = load_yaml(gsm_config_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("selection_data") != "GSM8K calibration only":
        raise RuntimeError("GSM8K calibration selection is not frozen")
    combinations = list(summary["mmlu_pro_combinations_frozen"])
    schedules = []
    for row in combinations:
        if row["schedule"] not in schedules:
            schedules.append(row["schedule"])
    digest = hashlib.sha256(json.dumps(combinations, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    config = copy.deepcopy(base)
    config["protocol_id"] = f"cps_qwen3_4b_fp16_mmlu_pro_gsm_selected_greedy_v1_{digest}"
    config["report_label"] = "MMLU-Pro validation of GSM8K-selected checkpoint-target combinations"
    config["checkpoint"]["schedules"] = schedules
    config["datasets"] = {
        "mmlu_pro": {
            "source_root": "results/final_paper_greedy_forced_mmlupro_v1/selected_common_cache/mmlu_pro/merged",
            "probe_train": 1000,
            "calibration": 500,
            "heldout": 1000,
        }
    }
    config["comparison"]["mmlu_pro_combinations"] = combinations
    config["comparison"]["selection_source"] = str(summary_path.resolve())
    config["output_root"] = "results/mmlu_pro_checkpoint_followup_v1"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    marker = {
        "status": "complete",
        "source_selection": str(summary_path.resolve()),
        "source_selection_data": "GSM8K calibration only",
        "protocol_id": config["protocol_id"],
        "schedules": schedules,
        "combinations": combinations,
        "config": str(output_path.resolve()),
    }
    (output_path.parent / "selection_marker.json").write_text(json.dumps(marker, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(marker, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

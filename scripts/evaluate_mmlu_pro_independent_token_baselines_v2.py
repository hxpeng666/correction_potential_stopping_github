#!/usr/bin/env python3
"""从独立 MMLU-Pro 公共缓存评测 Dense、Direct 与 fixed budgets（仅 token）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import torch
from evaluate_legacy_empirical_baselines_v4 import checkpoint_diagnostics, dense_direct_records, fixed_records, summarize_dense
from src.final_paper_inference import atomic_torch_save
from src.final_paper_inference import read_jsonl
from src.legacy_empirical_probe_v4 import summarize_policy_records
from src.utils import atomic_json, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_paper_mmlu_pro_independent_token_v2.yaml")
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); config = load_yaml(ROOT / args.config)
    marker = args.output_root / "phase.complete"
    if args.resume and marker.is_file(): print(json.dumps({"status":"skipped_complete"})); return
    if args.output_root.exists() and any(args.output_root.iterdir()): raise RuntimeError(f"拒绝覆盖：{args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = sorted((args.replay_root / "heldout").glob("sample_*.pt"))
    prepared = {row["problem_id"]: row for row in read_jsonl(ROOT / config["dataset"]["prepared_root"] / "heldout.jsonl")}
    if "final_test_count" in config["dataset"]:
        allowed = {pid for pid, row in prepared.items() if row.get("policy_role") in ("final_test", "heldout")}
        paths = [path for path in paths if path.stem.removeprefix("sample_") in allowed]
        expected = int(config["dataset"]["final_test_count"])
    else:
        expected = int(config["dataset"]["heldout_count"])
    if not args.smoke and len(paths) != expected: raise ValueError(f"final heldout 数量错误：{len(paths)} != {expected}")
    if not paths: raise ValueError("没有可评测 final heldout artifact")
    dense, direct = dense_direct_records(paths)
    records = {"dense":dense,"direct":direct,"fixed":{}}
    fixed_summary = {}
    for budget in config["generation"]["fixed_budgets"]:
        local = fixed_records(paths, args.replay_root / "heldout", int(budget))
        records["fixed"][str(budget)] = local; fixed_summary[str(budget)] = summarize_policy_records(local)
    payload = {"status":"complete","dataset":"mmlu_pro","report_label":config["report_label"],"latency_enabled":False,"primary_efficiency_metric":"token_reduction",
               "dense":summarize_dense(dense),"direct":summarize_policy_records(direct),"fixed":fixed_summary,
               "checkpoint_diagnostics":checkpoint_diagnostics(paths)}
    atomic_json(payload, args.output_root / "baselines.json"); atomic_torch_save({"status":"complete","records":records}, args.output_root / "baseline_records.pt")
    atomic_json({"status":"complete","artifacts":["baselines.json","baseline_records.pt"]}, marker); print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

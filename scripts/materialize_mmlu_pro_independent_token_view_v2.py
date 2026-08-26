#!/usr/bin/env python3
"""把旧800训练缓存与新1700缓存统一物化为 token-only 公共 replay view。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_paper_inference import atomic_torch_save, read_jsonl
from src.final_paper_protocol import canonical_fingerprint
from src.utils import atomic_json, load_yaml

TOKEN_LABEL = "token-only replay；并发采集耗时不作为论文指标"


def generation_semantics(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": config["model"]["id"], "revision": config["model"]["revision"],
        "metadata_fingerprint": config["model"]["metadata_fingerprint"],
        "dtype": config["model"]["dtype"], "attention_backend": config["model"]["attention_backend"],
        "enable_thinking": config["model"]["enable_thinking"], "generation": config["generation"],
        "checkpoint_protocol": config["checkpoint_protocol"], "prompt": config["prompt"],
        "sample_seed_semantics": "task_seed(20260803,mmlu_pro,heldout,problem_id,kind_or_checkpoint)",
        "demonstrations": "official validation, category-specific 5-shot, no reasoning",
    }


def tokenized_copy(source: dict[str, Any], record: dict[str, Any], destination: Path, config: dict[str, Any], view_fingerprint: str) -> dict[str, Any]:
    if source.get("status") != "complete" or source.get("dtype") != "float16":
        raise ValueError(f"缓存状态或 dtype 不兼容：{source.get('problem_id')}")
    if str(source["problem_id"]) != str(record["problem_id"]):
        raise ValueError("sample ID 与 prepared record 错位")
    source_record = source.get("record", {})
    for key in ("question", "choices", "answer", "category", "option_count"):
        if source_record.get(key) != record.get(key):
            raise ValueError(f"缓存语义字段不一致：{record['problem_id']} / {key}")
    if int(source.get("seed", -1)) != int(config["seed"]["global"]):
        raise ValueError(f"缓存 global seed 不一致：{record['problem_id']}")
    dense_tokens = int(source["dense"]["reasoning_tokens"])
    dense = dict(source["dense"])
    dense.update({
        "original_collection_wall_ms": dense.get("wall_ms"),
        "wall_ms": float(dense_tokens), "replay_wall_ms": float(dense_tokens),
        "adaptive_fallback_wall_ms": float(dense_tokens),
        "sentence_adaptive_fallback_wall_ms": float(dense_tokens),
        "fixed_adaptive_fallback_wall_ms": float(dense_tokens),
        "latency_available": False, "cost_proxy": "generated_token_count",
    })
    direct = dict(source["direct"])
    direct_tokens = int(direct["generated_tokens"])
    direct.update({
        "original_collection_wall_ms": direct.get("wall_ms"),
        "wall_ms": float(direct_tokens), "replay_wall_ms": float(direct_tokens),
        "latency_available": False, "cost_proxy": "generated_token_count",
    })
    rows = []
    for original in source["rows"]:
        row = dict(original)
        checkpoint, branch_tokens = int(row["checkpoint"]), int(row["branch_tokens"])
        used = float(min(dense_tokens, checkpoint + branch_tokens))
        row.update({
            "dense_wall_ms": float(dense_tokens), "dense_reference_wall_ms": float(dense_tokens),
            "adaptive_fallback_wall_ms": float(dense_tokens),
            "sentence_adaptive_fallback_wall_ms": float(dense_tokens),
            "fixed_adaptive_fallback_wall_ms": float(dense_tokens),
            "dense_prefill_cuda_ms": 0.0, "prefix_decode_cuda_ms": float(checkpoint),
            "branch_wall_ms": float(branch_tokens), "replay_stop_wall_ms": used,
            "sentence_replay_stop_wall_ms": used, "fixed_replay_stop_wall_ms": used,
            "fixed_adaptive_replay_stop_wall_ms": used,
            "replay_latency_label": TOKEN_LABEL, "latency_available": False,
        })
        rows.append(row)
    hidden = source.get("hidden")
    if not torch.is_tensor(hidden) or int(hidden.shape[0]) != len(rows) or not torch.isfinite(hidden).all():
        raise ValueError(f"hidden/row 完整性失败：{record['problem_id']}")
    checkpoints = [int(row["checkpoint"]) for row in rows]
    if len(checkpoints) != len(set(checkpoints)) or checkpoints != sorted(checkpoints):
        raise ValueError(f"checkpoint 重复或乱序：{record['problem_id']}")
    replay = dict(source)
    replay.update({
        "protocol_id": config["protocol_id"], "protocol_fingerprint": view_fingerprint,
        "source_protocol_id": source.get("protocol_id"), "source_protocol_fingerprint": source.get("protocol_fingerprint"),
        "record": record, "split": record["policy_role"], "dense": dense, "direct": direct, "rows": rows,
        "source_dense_artifact": str(destination.resolve()), "source_common_cache_artifact": str(source.get("source_common_cache_artifact", "")),
        "latency_label": TOKEN_LABEL, "latency_available": False, "cost_proxy": "generated_token_count",
        "primary_replay_view_fingerprint": view_fingerprint,
        "replay_view_created_at": datetime.now(timezone.utc).isoformat(),
    })
    return replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_paper_mmlu_pro_independent_token_v2.yaml")
    parser.add_argument("--new-merged-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    config = load_yaml(ROOT / args.config)
    old_config = load_yaml(ROOT / "configs/final_paper_mmlu_pro_learnstop_style_v1.yaml")
    if generation_semantics(old_config) != generation_semantics(config):
        raise ValueError("旧800缓存的模型/prompt/生成/checkpoint语义与新协议不一致，禁止复用")
    prepared = ROOT / config["dataset"]["prepared_root"]
    old_root = ROOT / config["dataset"]["old_replay_root"]
    new_root = args.new_merged_root / "heldout"
    semantics = generation_semantics(config)
    view_fingerprint = canonical_fingerprint({"semantics": semantics, "split_fingerprint": json.loads((ROOT / config["output_root"] / config["dataset"]["split_manifest"]).read_text(encoding="utf-8"))["fingerprint"], "cost": "token_count_only"})
    counts = {"probe_train": 0, "calibration": 0, "heldout": 0, "old_reused": 0, "new": 0, "missing": 0, "skipped": 0}
    missing = []
    source_fingerprints = set()
    for role in ("probe_train", "calibration", "heldout"):
        rows = read_jsonl(prepared / f"{role}.jsonl")
        target = args.output_root / role
        target.mkdir(parents=True, exist_ok=True)
        for record in rows:
            problem_id = str(record["problem_id"])
            destination = target / f"sample_{problem_id}.pt"
            if args.resume and destination.is_file():
                previous = torch.load(destination, map_location="cpu", weights_only=False)
                if previous.get("status") == "complete" and previous.get("primary_replay_view_fingerprint") == view_fingerprint:
                    counts[role] += 1; counts["skipped"] += 1; continue
                raise RuntimeError(f"拒绝 resume 不兼容目标：{destination}")
            if destination.exists():
                raise RuntimeError(f"拒绝覆盖：{destination}")
            old_path = old_root / f"sample_{problem_id}.pt"
            new_path = new_root / f"sample_{problem_id}.pt"
            if record.get("reused_from_mmlu_pro_800"):
                source_path, source_kind = old_path, "old_reused"
            else:
                source_path, source_kind = new_path, "new"
            if not source_path.is_file():
                missing.append(problem_id); counts["missing"] += 1
                continue
            source = torch.load(source_path, map_location="cpu", weights_only=False)
            source_fingerprints.add(str(source.get("protocol_fingerprint")))
            atomic_torch_save(tokenized_copy(source, record, destination, config, view_fingerprint), destination)
            counts[role] += 1; counts[source_kind] += 1
    expected = {"probe_train": 1000, "calibration": 500, "heldout": 1000}
    complete = all(counts[key] == value for key, value in expected.items()) and not missing
    if not complete and not args.allow_partial:
        raise RuntimeError(f"公共缓存尚不完整：counts={counts}, missing={len(missing)}")
    audit = {
        "status": "complete" if complete else "partial", "protocol_id": config["protocol_id"],
        "view_fingerprint": view_fingerprint, "generation_semantics": semantics, "counts": counts,
        "expected": expected, "source_protocol_fingerprints": sorted(source_fingerprints),
        "missing_ids": missing, "old_cache_role_restriction": "probe_train only",
        "latency": {"enabled": False, "reason": "同卡多副本并发；只报告 token reduction", "proxy_fields_not_reportable_as_latency": True},
    }
    atomic_json(audit, args.output_root / ("materialize.complete" if complete else "materialize.partial.json"))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

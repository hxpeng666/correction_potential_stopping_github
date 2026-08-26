#!/usr/bin/env python3
"""审计当前动态主方法特征消融的缓存、训练与冻结评测完整性。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.dynamic_optimal_stopping_v1 import DYNAMIC_FEATURE_KINDS, summarize_token_records
from src.legacy_empirical_probe_v4 import load_checkpoint_split
from src.utils import atomic_json


MAIN_FEATURE = "full_no_delta"
DATASETS = {
    "gsm8k": "results/final_paper_primary_v1/main_float16_seed20260803/replay_view/gsm8k",
    "mmlu_pro": "results/final_paper_mmlu_pro_independent_token_v2_train1000_cal500_test1000/replay_view_token",
}
EXPECTED = {"gsm8k": (1000, 500, 1319), "mmlu_pro": (1000, 500, 1000)}


def finite_tree(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else True
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def sample_audit(root: Path, dataset: str) -> dict[str, Any]:
    expected = EXPECTED[dataset]
    ids: dict[str, set[str]] = {}
    protocols: set[str] = set()
    views: set[str] = set()
    dtypes: set[str] = set()
    attention: set[str] = set()
    seeds: set[int] = set()
    schedule_failures: list[str] = []
    tensor_failures: list[str] = []
    counts: dict[str, int] = {}
    rows_total = 0
    for split, expected_count in zip(("probe_train", "calibration", "heldout"), expected):
        paths = sorted((root / split).glob("sample_*.pt"))
        counts[split] = len(paths)
        if len(paths) != expected_count:
            raise AssertionError(f"{dataset}/{split}数量{len(paths)} != {expected_count}")
        ids[split] = set()
        for path in paths:
            item = torch.load(path, map_location="cpu", weights_only=False)
            problem_id = str(item["problem_id"])
            if problem_id in ids[split]:
                raise AssertionError(f"{dataset}/{split}重复ID {problem_id}")
            ids[split].add(problem_id)
            if str(item.get("split")) != split:
                raise AssertionError(f"{path}内部split错误")
            if str(item.get("dataset")) != dataset:
                raise AssertionError(f"{path}内部dataset错误")
            protocols.add(str(item.get("protocol_fingerprint")))
            views.add(str(item.get("primary_replay_view_fingerprint")))
            dtypes.add(str(item.get("dtype")))
            attention.add(str(item.get("attention_backend")))
            seeds.add(int(item.get("seed")))
            hidden = item["hidden"]
            rows = item["rows"]
            if hidden.ndim != 3 or len(rows) != hidden.shape[0] or not torch.isfinite(hidden.float()).all():
                tensor_failures.append(problem_id)
            rows_total += len(rows)
            token_count = int(item["dense_content_tokens"])
            if token_count != len(item["dense"]["content_tokens"]):
                tensor_failures.append(problem_id)
            cps = item["checkpoint_protocol"]
            minimum, maximum, gap = int(cps["minimum"]), int(cps["maximum"]), int(cps["sentence_minimum_gap"])
            checkpoints = sorted(int(x) for x in item["schedules"]["sentence"])
            if any(x < minimum or x > min(maximum, token_count) for x in checkpoints):
                schedule_failures.append(problem_id)
            if any(b - a < gap for a, b in zip(checkpoints, checkpoints[1:])):
                schedule_failures.append(problem_id)
    if any(ids[a] & ids[b] for a in ids for b in ids if a < b):
        raise AssertionError(f"{dataset} split存在交集")
    if tensor_failures or schedule_failures:
        raise AssertionError(f"{dataset}缓存失败 tensor={tensor_failures[:3]} schedule={schedule_failures[:3]}")
    return {
        "counts": counts,
        "unique_ids": {key: len(value) for key, value in ids.items()},
        "split_disjoint": True,
        "rows_total": rows_total,
        "protocol_fingerprints": sorted(protocols),
        "primary_replay_view_fingerprints": sorted(views),
        "dtypes": sorted(dtypes),
        "attention_backends": sorted(attention),
        "global_generation_seeds": sorted(seeds),
        "hidden_rows_aligned_and_finite": True,
        "dense_content_token_lengths_match": True,
        "sentence_schedule_valid": True,
    }


def records_audit(records: list[dict[str, Any]], expected_n: int) -> None:
    if len(records) != expected_n:
        raise AssertionError(f"策略分母{len(records)} != {expected_n}")
    ids = [str(row["problem_id"]) for row in records]
    if len(set(ids)) != expected_n:
        raise AssertionError("策略记录ID重复")
    summary = summarize_token_records(records)
    if abs(summary["accuracy"] - (summary["dense_accuracy"] + (summary["gained_correct_count"] - summary["lost_correct_count"]) / expected_n)) > 1e-12:
        raise AssertionError("准确率与transition计数恒等式失败")
    for row in records:
        if row["fallback"]:
            if row["method_success"] != row["dense_success"] or row["method_tokens"] != row["dense_tokens"]:
                raise AssertionError("fallback未严格等于Dense")
        elif int(row["checkpoint"]) > int(row["dense_tokens"]):
            raise AssertionError("停止点晚于Dense")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", nargs="+", default=list(DYNAMIC_FEATURE_KINDS))
    args = parser.parse_args()
    report: dict[str, Any] = {"status": "complete", "datasets": {}, "probes": {}}
    for dataset, raw_value in DATASETS.items():
        raw = ROOT / raw_value
        report["datasets"][dataset] = sample_audit(raw, dataset)
        expected_n = EXPECTED[dataset][2]
        expected_ids = None
        for feature in args.features:
            probe_root = args.feature_root / feature / dataset / "probe"
            probe = json.loads((probe_root / "probe.json").read_text(encoding="utf-8"))
            predictions = torch.load(probe_root / "predictions.pt", map_location="cpu", weights_only=False)
            records_payload = torch.load(probe_root / "policy_records.pt", map_location="cpu", weights_only=False)["records"]
            if probe["run_spec"]["feature_kind"] != feature:
                raise AssertionError(f"{dataset}/{feature} feature_kind错误")
            if probe["run_spec"]["online_observability"] != "current z_t only; no next checkpoint/dense remainder":
                raise AssertionError(f"{dataset}/{feature}未来信息声明错误")
            if probe["run_spec"]["heldout_selection"] is not False:
                raise AssertionError(f"{dataset}/{feature}使用heldout选点")
            fit, validation = set(probe["fit_problem_ids"]), set(probe["validation_problem_ids"])
            if fit & validation or len(fit) != 800 or len(validation) != 200:
                raise AssertionError(f"{dataset}/{feature}内部划分错误")
            if not finite_tree(predictions):
                raise AssertionError(f"{dataset}/{feature}预测包含NaN/Inf")
            ids = predictions["problem_ids"]["heldout"]
            checkpoints = predictions["checkpoints"]["heldout"]
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise AssertionError(f"{dataset}/{feature} heldout ID/行顺序不配对")
            for key in ("stop_probability", "risk_probability"):
                if len(predictions[key]["heldout"]) != len(ids):
                    raise AssertionError(f"{dataset}/{feature}/{key}行数错误")
            if predictions["q_continue_values"]["heldout"].shape != (len(ids), 48):
                raise AssertionError(f"{dataset}/{feature} Q bank维度错误")
            if len(checkpoints) != len(ids):
                raise AssertionError(f"{dataset}/{feature} checkpoint行数错误")
            for family, values in records_payload.items():
                for _, records in values.items():
                    records_audit(records, expected_n)
            dense = records_payload["empirical_B"]["0"] if feature == MAIN_FEATURE and dataset == "gsm8k" else None
            if dense is not None:
                summary = summarize_token_records(dense)
                if summary["coverage"] != 0 or summary["token_reduction"] != 0 or summary["fallback"] != expected_n:
                    raise AssertionError("GSM8K主方法Dense sentinel失败")
            coverage_payload = torch.load(
                args.feature_root / feature / dataset / "coverage" / "policy_records.pt",
                map_location="cpu", weights_only=False,
            )["records"]
            for records in coverage_payload.values():
                records_audit(records, expected_n)
            report["probes"][f"{dataset}/{feature}"] = {
                "run_spec_fingerprint": probe["run_spec_fingerprint"],
                "feature_width": probe["run_spec"]["feature_width"],
                "fit_problems": len(fit),
                "validation_problems": len(validation),
                "prediction_rows": len(ids),
                "predictions_finite": True,
                "current_state_only": True,
                "heldout_used_for_selection": False,
                "policy_records_valid": True,
            }
    parity: dict[str, Any] = {}
    for dataset in DATASETS:
        base = ROOT / "results/final_paper_dynamic_deployable_os_frontier_v2/dynamic/full" / dataset / "predictions.pt"
        rerun = ROOT / "results/final_paper_dynamic_deployable_feature_ablation_v3/hardware_parity_a100_1/full_no_delta" / dataset / "predictions.pt"
        parity[dataset] = {
            "a100_0_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
            "a100_1_sha256": hashlib.sha256(rerun.read_bytes()).hexdigest(),
        }
        parity[dataset]["byte_identical"] = parity[dataset]["a100_0_sha256"] == parity[dataset]["a100_1_sha256"]
        if not parity[dataset]["byte_identical"]:
            raise AssertionError(f"{dataset} A100复现不一致")
    report.update({
        "features": list(args.features),
        "feature_dataset_probes": len(args.features) * len(DATASETS),
        "a100_parity": parity,
        "all_checks_passed": True,
    })
    atomic_json(report, args.output)
    print(json.dumps({"status": "complete", "output": str(args.output), "probes": len(report["probes"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()

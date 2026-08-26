from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "splits/mmlu_pro_independent_split.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_split_is_disjoint_same_source_and_exact_size():
    manifest = json.loads(SPLIT.read_text())
    roles = {key: set(manifest["role_ids"][key]) for key in ("probe_train", "calibration", "heldout")}
    assert list(map(len, roles.values())) == [1000, 500, 1000]
    assert not roles["probe_train"] & roles["calibration"]
    assert not roles["probe_train"] & roles["heldout"]
    assert not roles["calibration"] & roles["heldout"]
    assert manifest["source"]["train"] == manifest["source"]["calibration"] == manifest["source"]["heldout"] == "official test"
    assert manifest["calibration_vs_heldout_total_variation"]["category"] <= 0.05
    assert manifest["calibration_vs_heldout_total_variation"]["length_bin"] <= 0.05


def test_old_800_only_enter_probe_train():
    manifest = json.loads(SPLIT.read_text())
    assert manifest["counts"]["old_cache_reused_in_probe_train_only"] == 800
    assert manifest["leakage_audit"]["old_800_roles"] == ["probe_train"]
    assert manifest["leakage_audit"]["pairwise_overlap"] == 0


def test_missing_answers_are_not_consistent_and_last_switch_includes_dense():
    core = load_module("legacy", ROOT / "src/legacy_empirical_probe_v4.py")
    frame = pd.DataFrame([
        {"problem_id": "x", "checkpoint": 64, "current_prediction": None, "dense_prediction": None, "current_success": False, "dense_success": False},
        {"problem_id": "y", "checkpoint": 64, "current_prediction": "A", "dense_prediction": "B", "current_success": True, "dense_success": False},
    ])
    result = core.add_targets(frame)
    assert not bool(result.loc[0, "target_consistency"])
    assert not bool(result.loc[1, "target_last_switch"])


def test_token_budget_selection_and_dense_sentinel():
    trainer = load_module("token_trainer", ROOT / "scripts/train_mmlu_pro_independent_token_v2.py")
    curve = [
        {"lost_correct_count": 0, "mean_reasoning_and_answer_tokens": 100.0, "coverage": 0.0, "threshold": -1.0, "is_no_stop_sentinel": True},
        {"lost_correct_count": 1, "mean_reasoning_and_answer_tokens": 60.0, "coverage": 0.5, "threshold": 0.2, "is_no_stop_sentinel": False},
        {"lost_correct_count": 2, "mean_reasoning_and_answer_tokens": 40.0, "coverage": 0.8, "threshold": 0.4, "is_no_stop_sentinel": False},
    ]
    assert trainer.select_empirical_budget_token(curve, 0)["is_no_stop_sentinel"]
    assert trainer.select_empirical_budget_token(curve, 1)["threshold"] == 0.2
    assert trainer.select_empirical_budget_token(curve, 2)["threshold"] == 0.4


def test_first_hit_low_and_high_direction():
    core = load_module("legacy_first_hit", ROOT / "src/legacy_empirical_probe_v4.py")
    frame = pd.DataFrame([
        {"problem_id":"x","checkpoint":64,"current_success":False,"dense_success":True,"dense_tokens":200,"branch_tokens":2,"dense_wall_ms":200.0,"adaptive_fallback_wall_ms":200.0,"replay_stop_wall_ms":66.0,"dense_prefill_cuda_ms":0.0,"prefix_decode_cuda_ms":64.0,"branch_wall_ms":2.0,"current_prediction":"A","dense_prediction":"B","gold_answer":"B"},
        {"problem_id":"x","checkpoint":96,"current_success":True,"dense_success":True,"dense_tokens":200,"branch_tokens":2,"dense_wall_ms":200.0,"adaptive_fallback_wall_ms":200.0,"replay_stop_wall_ms":98.0,"dense_prefill_cuda_ms":0.0,"prefix_decode_cuda_ms":96.0,"branch_wall_ms":2.0,"current_prediction":"B","dense_prediction":"B","gold_answer":"B"},
    ])
    low = core.simulate_policy(frame, np.asarray([0.1, 0.0]), "low", 0.2, include_records=True)
    high = core.simulate_policy(frame, np.asarray([0.9, 1.0]), "high", 0.8, include_records=True)
    assert low["records"][0]["checkpoint"] == 64
    assert high["records"][0]["checkpoint"] == 64
    sentinel = core.simulate_policy(frame, np.asarray([0.1, 0.0]), "low", -1.0, include_records=True, force_dense=True)
    assert sentinel["coverage"] == 0.0 and sentinel["token_reduction"] == 0.0

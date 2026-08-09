from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_scope", ROOT / "scripts/prepare_final_paper_single_seed_scope.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_single_seed_scope_counts_and_invariants():
    gsm = json.loads((ROOT / "splits/gsm8k_split.json").read_text(encoding="utf-8"))
    mmlu = json.loads((ROOT / "splits/mmlu_split.json").read_text(encoding="utf-8"))
    scope = MODULE.build_scope(gsm, mmlu, 20260803)
    assert scope["datasets"]["gsm8k"]["probe_train_count"] == 2000
    assert scope["datasets"]["gsm8k"]["calibration_count"] == 1000
    assert scope["datasets"]["gsm8k"]["heldout_count"] == 1319
    assert scope["datasets"]["mmlu"]["probe_train_count"] == 2000
    assert scope["datasets"]["mmlu"]["calibration_count"] == 1000
    assert scope["datasets"]["mmlu"]["heldout_count"] == 1000
    counts = scope["datasets"]["mmlu"]["heldout_subject_counts"]
    assert len(counts) == 57
    assert set(counts.values()) == {17, 18}
    assert sum(counts.values()) == 1000


def test_single_seed_scope_is_deterministic_and_disjoint():
    gsm = json.loads((ROOT / "splits/gsm8k_split.json").read_text(encoding="utf-8"))
    mmlu = json.loads((ROOT / "splits/mmlu_split.json").read_text(encoding="utf-8"))
    first = MODULE.build_scope(gsm, mmlu, 20260803)
    second = MODULE.build_scope(gsm, mmlu, 20260803)
    assert first == second
    for details in first["datasets"].values():
        probe = set(details["probe_train_problem_ids"])
        calibration = set(details["calibration_problem_ids"])
        heldout = set(details["heldout_problem_ids"])
        assert not probe & calibration
        assert not probe & heldout
        assert not calibration & heldout

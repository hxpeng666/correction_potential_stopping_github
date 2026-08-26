#!/usr/bin/env python3
"""Validate the portable scientific release without model weights or GPUs."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_DIRS = (ROOT / "src", ROOT / "scripts", ROOT / "tests", ROOT / "configs")
FORBIDDEN_TEXT = (
    "/mnt/" + "hbnas/",
    "/var/" + "folders/",
    "correction_potential_stopping/" + "results/",
)
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".safetensors", ".log", ".pid", ".incomplete")
REQUIRED_FILES = (
    "scripts/deepseek7b_protocol_v1.py",
    "scripts/collect_deepseek7b_paragraph_v1.py",
    "scripts/train_deepseek7b_ablation_v1.py",
    "scripts/evaluate_deepseek7b_ood_v2.py",
    "scripts/collect_deepseek7b_fixed_budget_frontier_v1.py",
    "scripts/collect_literature_method_data_v1.py",
    "scripts/train_evaluate_literature_method_v1.py",
    "src/legacy_empirical_probe_normalized_v1.py",
    "tests/test_normalized_softmin_v1.py",
    "configs/gsm8k_full_checkpoint_schedule_ablation_v1.yaml",
    "configs/literature_methods_qwen3_4b_strict_v2.yaml",
    "configs/deepseek7b_main_v2.yaml",
    "configs/deepseek7b_main_v3_selective32k.yaml",
)


def load_yaml(relative: str) -> dict:
    value = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{relative} is not a mapping")
    return value


def validate_protocols() -> None:
    qwen = load_yaml("configs/gsm8k_full_checkpoint_schedule_ablation_v1.yaml")
    deepseek = load_yaml("configs/deepseek7b_main_v2.yaml")
    extension = load_yaml("configs/deepseek7b_main_v3_selective32k.yaml")

    assert qwen["generation"] == {
        **qwen["generation"],
        "dense_temperature": 0.6,
        "dense_top_p": 0.95,
        "dense_top_k": 20,
        "dense_max_new_tokens": 4096,
        "forced_answer_strategy": "greedy_argmax",
        "forced_answer_do_sample": False,
        "force_answer_max_new_tokens": 16,
    }
    assert qwen["checkpoint"]["range_filter"] == "none"
    assert qwen["features"]["primary_width"] == 2566
    assert qwen["features"]["layer_zero_based"] == 20

    for config, budget in ((deepseek, 13000), (extension, 32768)):
        generation = config["generation"]
        assert generation["dense_max_new_tokens"] == budget
        assert generation["temperature"] == 0.6
        assert generation["top_p"] == 0.95
        assert generation["top_k"] == 20
        assert generation["do_sample"] is True
        assert generation["forced_answer_strategy"] == "greedy_argmax"
        assert generation["forced_answer_do_sample"] is False
        assert generation["force_answer_max_new_tokens"] == 48
        assert config["checkpoint"]["schedule"] == "paragraph"
        assert config["checkpoint"]["range_filter"] == "none"
        assert config["checkpoint"]["zero_checkpoint_policy"] == "dense_fallback"
        assert config["features"] == {
            "primary_kind": "full_no_delta",
            "primary_width": 3590,
            "layer_zero_based": 16,
        }
        assert config["probe"]["trajectory_softmin_beta"] == 0.5
        assert config["probe"]["trajectory_weight"] == 1.0
        assert config["data"]["gsm8k"] == {
            "probe_train": 1000,
            "calibration": 500,
            "heldout": 1319,
        }
        assert config["data"]["math"]["per_category"] == {
            "probe_train": 200,
            "calibration": 100,
        }
        assert config["data"]["math500"]["heldout"] == 500
        assert config["data"]["aime"]["heldout"] == 30

    assert extension["selective_dense_extension"]["source_dense_max_new_tokens"] == 13000
    assert extension["selective_dense_extension"]["require_first_13000_token_identity"] is True


def validate_tree() -> dict[str, int]:
    for relative in REQUIRED_FILES:
        assert (ROOT / relative).is_file(), f"missing required file: {relative}"

    counts = {"python": 0, "yaml": 0, "json": 0}
    for base in SCIENTIFIC_DIRS:
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT)
            assert path.suffix not in FORBIDDEN_SUFFIXES, f"generated artifact included: {relative}"
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
                counts["python"] += 1
            elif path.suffix in (".yaml", ".yml"):
                yaml.safe_load(path.read_text(encoding="utf-8"))
                counts["yaml"] += 1
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
                counts["json"] += 1
            if path.suffix in (".py", ".yaml", ".yml", ".json"):
                text = path.read_text(encoding="utf-8")
                for forbidden in FORBIDDEN_TEXT:
                    assert forbidden not in text, f"machine-specific path in {relative}: {forbidden}"
    return counts


def main() -> None:
    counts = validate_tree()
    validate_protocols()
    print(json.dumps({"status": "complete", **counts}, indent=2))


if __name__ == "__main__":
    main()

from pathlib import Path

import yaml

from scripts.collect_qwen3_14b_deterministic_ood_v1 import DATA_LAYOUT


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_dataset_layout_and_counts() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/qwen3_14b_deterministic_ood13k_v1.yaml").read_text()
    )
    assert DATA_LAYOUT == (
        ("gsm8k", "probe_train"),
        ("gsm8k", "calibration"),
        ("gsm8k", "heldout"),
        ("math", "probe_train"),
        ("math", "calibration"),
        ("math500", "heldout"),
        ("aime", "heldout"),
    )
    total = (
        sum(config["data"]["gsm8k"].values())
        + config["data"]["math"]["probe_train"]
        + config["data"]["math"]["calibration"]
        + config["data"]["math500"]["heldout"]
        + config["data"]["aime"]["heldout"]
    )
    assert total == 5449


def test_current_scientific_protocol_is_frozen() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/qwen3_14b_deterministic_ood13k_v1.yaml").read_text()
    )
    assert config["seed"] == 0
    assert config["generation"] == {
        "dense_max_new_tokens": 13000,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "do_sample": True,
        "cap_hit_grader": "forced_answer_at_exact_13k_prefix",
        "force_answer_max_new_tokens": 48,
        "forced_answer_strategy": "greedy_argmax",
        "forced_answer_do_sample": False,
        "force_answer_suffix": "\n</think>\n\n\\boxed{",
    }
    assert config["checkpoint"]["schedule"] == "paragraph"
    assert config["features"]["layer_zero_based"] == 20
    assert config["calibration"]["primary_calibrator"] == "trajectory_envelope_ltt"
    assert config["calibration"]["empirical_budget_B_used"] is False

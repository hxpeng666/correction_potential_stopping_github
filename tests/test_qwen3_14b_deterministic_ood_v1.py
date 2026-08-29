from pathlib import Path

import yaml

from scripts.collect_qwen3_14b_deterministic_ood_v1 import (
    DATA_LAYOUT,
    tokenize_prompt_like_deepseek,
)
from scripts.run_qwen3_14b_deterministic_ood_v1 import FORMAL_WORKERS


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
    assert config["seed"] == 20260820
    assert config["reproducibility"]["dense_rollout_base_seed"] == 20260820
    assert config["prompt_tokenization"] == {
        "render": "apply_chat_template_tokenize_false_add_generation_prompt_true",
        "encode": "tokenizer_default_add_special_tokens",
        "reference": "scripts/collect_deepseek7b_paragraph_v1.py",
    }
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


def test_dense_collection_contract_matches_frozen_deepseek_rollout() -> None:
    qwen = yaml.safe_load(
        (ROOT / "configs/qwen3_14b_deterministic_ood13k_v1.yaml").read_text()
    )
    deepseek = yaml.safe_load(
        (ROOT / "configs/deepseek7b_main_v2.yaml").read_text()
    )
    assert qwen["seed"] == deepseek["seed"] == 20260820
    for key in (
        "dense_max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "do_sample",
        "force_answer_max_new_tokens",
        "forced_answer_strategy",
        "forced_answer_do_sample",
        "force_answer_suffix",
    ):
        assert qwen["generation"][key] == deepseek["generation"][key]
    for key in ("schedule", "boundary_regex", "range_filter", "zero_checkpoint_policy"):
        assert qwen["checkpoint"][key] == deepseek["checkpoint"][key]
    assert qwen["data"]["gsm8k"] == deepseek["data"]["gsm8k"]
    for key in ("probe_train", "calibration", "categories", "per_category"):
        assert qwen["data"]["math"][key] == deepseek["data"]["math"][key]
    assert qwen["data"]["math500"] == deepseek["data"]["math500"]


def test_prompt_tokenizer_uses_deepseek_default_special_token_path() -> None:
    class Result:
        input_ids = "ids"

    class Recorder:
        def __init__(self) -> None:
            self.kwargs = None

        def __call__(self, _text, **kwargs):
            self.kwargs = kwargs
            return Result()

    tokenizer = Recorder()
    assert tokenize_prompt_like_deepseek(tokenizer, "prompt") == "ids"
    assert tokenizer.kwargs == {"return_tensors": "pt"}


def test_three_worker_shards_are_exact_and_disjoint() -> None:
    assert FORMAL_WORKERS == (
        (0, "formal_gpu0_replica0", 0, 3),
        (1, "formal_gpu1_replica0", 1, 3),
        (1, "formal_gpu1_replica1", 2, 3),
    )
    assert {worker[2] for worker in FORMAL_WORKERS} == set(range(3))

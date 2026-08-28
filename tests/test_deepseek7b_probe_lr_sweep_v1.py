from scripts.summarize_deepseek7b_probe_lr_sweep_v1 import (
    forbidden_key_paths,
    lr_tag,
    select_shared_learning_rate,
)


def test_learning_rate_tags_are_stable() -> None:
    assert lr_tag(0.000025) == "0p000025"
    assert lr_tag(0.0004) == "0p0004"


def test_forbidden_empirical_calibration_fields_are_detected() -> None:
    payload = {
        "history": [{"validation_ap": 0.8}],
        "legacy_empirical_B_diagnostic": {},
        "nested": {"validation_B0": {"token_reduction": 0.3}},
    }
    assert forbidden_key_paths(payload) == [
        "legacy_empirical_B_diagnostic",
        "nested.validation_B0",
    ]


def test_threshold_free_payload_has_no_forbidden_fields() -> None:
    payload = {
        "history": [{"validation_ap": 0.8}],
        "internal_validation": {
            "label_ap": 0.8,
            "deployment_calibration": "not_performed_on_internal_validation",
        },
    }
    assert forbidden_key_paths(payload) == []


def test_shared_lr_selection_uses_fixed_seed_validation_only() -> None:
    rows = []
    for grader in ("original_13k_parser", "forced_answer_at_cap"):
        for dataset in ("gsm8k", "math"):
            for method in ("bce", "bce_trajectory"):
                for index, learning_rate in enumerate(
                    (0.000025, 0.00005, 0.0001, 0.0002, 0.0004)
                ):
                    rows.append(
                        {
                            "grader": grader,
                            "dataset": dataset,
                            "method": method,
                            "learning_rate": learning_rate,
                            "validation_objective": 1.0 + abs(index - 2),
                        }
                    )
    selected = select_shared_learning_rate(rows)
    assert selected["selected_learning_rate"] == 0.0001
    assert selected["training_seed"] == 0
    assert selected["additional_seed_confirmation"] is False
    assert selected["selection_uses_calibration"] is False

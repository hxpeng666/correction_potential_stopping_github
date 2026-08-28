from scripts.summarize_deepseek7b_probe_lr_sweep_v1 import (
    forbidden_key_paths,
    lr_tag,
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

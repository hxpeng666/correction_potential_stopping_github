from scripts.train_deepseek7b_ablation_v1 import model_selection_key


def test_corrected_correction_selection_ignores_b0_token_reduction() -> None:
    common = {
        "method": "correction",
        "validation_ap": 0.8,
        "validation_auc": 0.9,
        "training_loss": 0.4,
    }
    left = model_selection_key(
        "label_ap", validation_b0_token_reduction=0.1, **common
    )
    right = model_selection_key(
        "label_ap", validation_b0_token_reduction=0.9, **common
    )
    assert left == right


def test_legacy_correction_selection_uses_b0_token_reduction() -> None:
    common = {
        "method": "correction",
        "validation_ap": 0.8,
        "validation_auc": 0.9,
        "training_loss": 0.4,
    }
    assert model_selection_key(
        "legacy_b0_token", validation_b0_token_reduction=0.9, **common
    ) > model_selection_key(
        "legacy_b0_token", validation_b0_token_reduction=0.1, **common
    )


def test_target_baselines_remain_threshold_free() -> None:
    common = {
        "method": "correctness",
        "validation_ap": 0.8,
        "validation_auc": 0.9,
        "training_loss": 0.4,
    }
    assert model_selection_key(
        "legacy_b0_token", validation_b0_token_reduction=None, **common
    ) == model_selection_key(
        "label_ap", validation_b0_token_reduction=None, **common
    )

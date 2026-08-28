from scripts.train_deepseek7b_method_exploration_v1 import model_selection_key


def test_label_ap_selection_ignores_internal_b0_token_reduction() -> None:
    common = {
        "validation_ap": 0.8,
        "validation_auc": 0.9,
        "validation_objective": 0.4,
    }
    label_key_a = model_selection_key(
        "label_ap", validation_b0_token_reduction=0.1, **common
    )
    label_key_b = model_selection_key(
        "label_ap", validation_b0_token_reduction=0.9, **common
    )
    assert label_key_a == label_key_b


def test_legacy_selection_uses_internal_b0_token_reduction() -> None:
    common = {
        "validation_ap": 0.8,
        "validation_auc": 0.9,
        "validation_objective": 0.4,
    }
    assert model_selection_key(
        "legacy_b0_token", validation_b0_token_reduction=0.9, **common
    ) > model_selection_key(
        "legacy_b0_token", validation_b0_token_reduction=0.1, **common
    )


def test_validation_objective_is_minimized() -> None:
    common = {
        "validation_ap": 0.8,
        "validation_auc": 0.9,
        "validation_b0_token_reduction": 0.5,
    }
    assert model_selection_key(
        "validation_objective", validation_objective=0.2, **common
    ) > model_selection_key(
        "validation_objective", validation_objective=0.3, **common
    )


def test_threshold_free_selection_accepts_no_b0_replay() -> None:
    assert model_selection_key(
        "label_ap",
        validation_ap=0.8,
        validation_auc=0.9,
        validation_objective=0.4,
        validation_b0_token_reduction=None,
    ) == (0.8, 0.9, -0.4)

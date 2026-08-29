import pytest

from scripts.run_qwen3_14b_vllm_full_v1 import (
    SELF_REPRODUCIBILITY_POLICY,
    validate_risk_gate,
    worker_capacity,
)


def valid_payload():
    config = {
        "determinism_gate": {
            "problems": [{"problem_id": "math"}, {"problem_id": "gsm"}]
        }
    }
    result = {
        "self_reproducibility_accepted": True,
        "same_gpu_exact": {"math": True, "gsm": True},
        "cross_gpu_exact": {"math": True, "gsm": True},
        "baseline_equivalent": False,
    }
    report = {
        "status": "complete",
        "acceptance_policy": {"name": SELF_REPRODUCIBILITY_POLICY},
        "accepted_profiles": ["full_apc_b2"],
        "recommended_profile": "full_apc_b2",
        "results": {"full_apc_b2": result},
    }
    return config, report, result


def test_self_reproducible_profile_can_differ_from_baseline():
    config, report, result = valid_payload()
    assert validate_risk_gate(config, report, "full_apc_b2") is result


@pytest.mark.parametrize("field", ["same_gpu_exact", "cross_gpu_exact"])
def test_missing_or_nonexact_repeat_fails_closed(field):
    config, report, _ = valid_payload()
    report["results"]["full_apc_b2"][field]["math"] = False
    with pytest.raises(RuntimeError):
        validate_risk_gate(config, report, "full_apc_b2")


def test_incomplete_problem_coverage_fails_closed():
    config, report, _ = valid_payload()
    del report["results"]["full_apc_b2"]["cross_gpu_exact"]["gsm"]
    with pytest.raises(RuntimeError):
        validate_risk_gate(config, report, "full_apc_b2")


def test_worker_capacity_reserves_40gib_per_replica():
    assert worker_capacity(81920, 22948, 40960, 2) == 1
    assert worker_capacity(81920, 14798, 40960, 2) == 1
    assert worker_capacity(81920, 0, 40960, 2) == 2

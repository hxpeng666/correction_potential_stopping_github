from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_scheduler_has_an_explicit_per_gpu_bound() -> None:
    source = (ROOT / "scripts/run_deepseek7b_method_axes_original_v2_v1.py").read_text()
    assert '"--per-gpu-parallel"' in source
    assert "threading.BoundedSemaphore(args.per_gpu_parallel)" in source
    assert '"per_gpu_parallel": args.per_gpu_parallel' in source


def test_formal_runner_gates_same_gpu_concurrency_before_training() -> None:
    source = (ROOT / "scripts/run_deepseek7b_deterministic_method_axes_v1.py").read_text()
    gate_position = source.index('state["completed"].append("same_gpu_concurrency_gate")')
    training_position = source.index('state["status"] = "training_101_core_axes"')
    assert gate_position < training_position
    assert 'f"SERIAL_VS_{name.upper()}.json"' in source
    assert "CONCURRENT_PEER_AUDIT.json" in source


def test_auxiliary_scheduler_keeps_a_bounded_parallel_range() -> None:
    source = (ROOT / "scripts/run_deepseek7b_aux_feature_original_v2_v1.py").read_text()
    assert "if not 1 <= args.parallel <= 8" in source
    assert '"parallel": args.parallel' in source

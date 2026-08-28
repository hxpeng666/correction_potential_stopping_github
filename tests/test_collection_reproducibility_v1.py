from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_deterministic_collection_pair_v1",
    ROOT / "scripts/audit_deterministic_collection_pair_v1.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def artifact(gpu: int, uuid: str) -> dict:
    return {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": "test",
        "protocol_fingerprint": "fingerprint",
        "primary_replay_view_fingerprint": "fingerprint:paragraph",
        "dataset": "gsm8k",
        "split": "heldout",
        "problem_id": "sample",
        "dtype": "bfloat16",
        "seed": 7,
        "problem_seed": 11,
        "actual_checkpoint_schedule": "paragraph",
        "checkpoint_protocol": {"schedule": "paragraph"},
        "capture_layers": [16],
        "rows": [
            {
                "checkpoint": 4,
                "current_prediction": "2",
                "dense_wall_ms": float(gpu + 1),
                "branch_collection_wall_ms": float(gpu + 2),
                "producer_gpu": gpu,
            }
        ],
        "hidden": torch.tensor([[[1.0, 2.0]]], dtype=torch.float16),
        "record": {"question": "1+1"},
        "gold_answer": "2",
        "prompt_text": "prompt",
        "prompt_tokens": 2,
        "dense": {
            "tokens": [1, 2],
            "content_tokens": [1, 2],
            "text": "2",
            "entropies_top20": [0.1, 0.2],
            "prediction": "2",
            "success": True,
            "reasoning_tokens": 2,
            "reached_max_tokens": False,
            "wall_ms": float(gpu + 3),
        },
        "dense_generation": {"do_sample": True},
        "forced_answer_decoding": {"strategy": "greedy_argmax"},
        "trajectory": {"paragraphs": [2]},
        "schedule_checkpoints": [4],
        "model_audit": {"path": f"/machine/{gpu}", "hidden_size": 2},
        "reproducibility": {
            "settings": {"protocol_id": "strict"},
            "runtime_lock": {"path": f"/machine/{gpu}/lock", "lock_id": "lock", "sha256": "x"},
            "environment": {
                "python": "same",
                "gpu": {
                    "logical_index": gpu,
                    "uuid": uuid,
                    "name": "NVIDIA A100 80GB PCIe",
                },
            },
            "code": {"git": {"commit": "abc"}, "source_sha256": {"x": "y"}},
        },
    }


def test_collection_scientific_payload_ignores_only_operational_metadata() -> None:
    left = artifact(0, "GPU-0")
    right = artifact(1, "GPU-1")
    assert MODULE.scientific_payload(left) == MODULE.scientific_payload(right)
    assert torch.equal(left["hidden"], right["hidden"])


def test_collection_scientific_payload_detects_changed_answer() -> None:
    left = artifact(0, "GPU-0")
    right = artifact(1, "GPU-1")
    right["rows"][0]["current_prediction"] = "3"
    assert MODULE.scientific_payload(left) != MODULE.scientific_payload(right)

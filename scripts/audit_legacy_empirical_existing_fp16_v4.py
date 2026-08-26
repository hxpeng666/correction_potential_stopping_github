#!/usr/bin/env python3
"""Fail-closed audit of selected existing FP16 semantic/replay artifacts."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import atomic_json

EXPECTED = {
    "gsm8k": {"probe_train": 1000, "calibration": 500, "heldout": 1319},
    "mmlu": {"probe_train": 1000, "calibration": 500, "heldout": 1000},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scope = json.loads((args.run_root / "SCOPE_MANIFEST.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    checks: dict[str, Any] = {}
    split_sets: dict[str, dict[str, set[str]]] = {}
    fingerprints: dict[str, set[str]] = {"gsm8k": set(), "mmlu": set()}
    for dataset in ("gsm8k", "mmlu"):
        checks[dataset] = {}; split_sets[dataset] = {}
        for split, expected_count in EXPECTED[dataset].items():
            expected_ids = [str(value) for value in scope["datasets"][dataset][f"{split}_problem_ids"]]
            paths = sorted((args.run_root / "replay_view" / dataset / split).glob("sample_*.pt"))
            observed_ids = {path.stem.removeprefix("sample_") for path in paths}
            split_sets[dataset][split] = set(expected_ids)
            if len(expected_ids) != expected_count or len(set(expected_ids)) != expected_count:
                errors.append(f"scope count/duplicate mismatch {dataset}/{split}")
            if observed_ids != set(expected_ids):
                errors.append(f"view missing/extra IDs {dataset}/{split}: observed={len(observed_ids)} expected={expected_count}")
            sources: Counter[str] = Counter(); subjects: Counter[str] = Counter()
            checkpoints = no_sentence = maxed = 0
            for path in paths:
                try:
                    value = torch.load(path, map_location="cpu", weights_only=False)
                except Exception as error:
                    errors.append(f"cannot load {path}: {error!r}"); continue
                problem_id = path.stem.removeprefix("sample_")
                if value.get("status") != "complete" or str(value.get("problem_id")) != problem_id:
                    errors.append(f"status/ID mismatch {path}")
                if value.get("dtype") != "float16" or value.get("attention_backend") != "sdpa":
                    errors.append(f"dtype/backend mismatch {path}")
                if int(value.get("seed", -1)) != 20260803:
                    errors.append(f"generation seed mismatch {path}")
                fingerprints[dataset].add(str(value.get("protocol_fingerprint")))
                if value.get("latency_label") != "A100 single-request replay-estimated latency":
                    errors.append(f"latency label mismatch {path}")
                if float(value.get("checkpoint_cost_mean_ms", -1.0)) != 0.0:
                    errors.append(f"non-legacy checkpoint overhead {path}")
                layers = [int(item) for item in value.get("capture_layers", [])]
                if 20 not in layers:
                    errors.append(f"layer 20 absent {path}")
                rows = value.get("rows", []); hidden = value.get("hidden")
                if not torch.is_tensor(hidden) or tuple(hidden.shape) != (len(rows), len(layers), 2560):
                    errors.append(f"row/vector mismatch {path}")
                elif not bool(torch.isfinite(hidden).all()):
                    errors.append(f"hidden NaN/Inf {path}")
                row_positions = [int(row.get("checkpoint", -1)) for row in rows]
                if len(row_positions) != len(set(row_positions)):
                    errors.append(f"duplicate checkpoint {path}")
                sentence = [int(item) for item in value.get("schedules", {}).get("sentence", [])]
                sentence_rows = [int(row["checkpoint"]) for row in rows if "sentence" in row.get("checkpoint_schedules", [])]
                if sentence != sentence_rows:
                    errors.append(f"sentence schedule mismatch {path}")
                if any(position < 64 or position > 768 for position in sentence):
                    errors.append(f"sentence position out of range {path}")
                if any(right - left < 8 for left, right in zip(sentence, sentence[1:])):
                    errors.append(f"sentence gap violation {path}")
                dense = value.get("dense", {}); declared = int(dense.get("reasoning_tokens", -1))
                if declared != len(dense.get("tokens", [])):
                    errors.append(f"Dense token mismatch {path}")
                for key in ("logps", "margins", "entropies_top20"):
                    if len(dense.get(key, [])) != declared:
                        errors.append(f"Dense {key} mismatch {path}")
                if not math.isfinite(float(dense.get("wall_ms", float("nan")))):
                    errors.append(f"Dense invalid replay wall {path}")
                for row in rows:
                    required = ("current_success", "dense_success", "current_prediction", "dense_prediction", "branch_tokens", "prefix_mean_entropy_tail8")
                    if any(key not in row for key in required):
                        errors.append(f"missing forced-answer/label field {path}"); break
                    if not math.isfinite(float(row["prefix_mean_entropy_tail8"])):
                        errors.append(f"invalid entropy {path}"); break
                record = value.get("record", {})
                source = str(record.get("source_split", record.get("split", "official_train" if split != "heldout" else "official_test")))
                sources[source] += 1
                if record.get("subject") is not None:
                    subjects[str(record["subject"])] += 1
                checkpoints += len(sentence); no_sentence += int(not sentence)
                maxed += int(bool(dense.get("reached_max_tokens")))
                if len(errors) >= 200:
                    break
            checks[dataset][split] = {
                "expected": expected_count, "observed": len(paths),
                "source_counts": dict(sorted(sources.items())),
                "subject_counts": dict(sorted(subjects.items())),
                "subjects_covered": len(subjects), "sentence_checkpoints": checkpoints,
                "no_sentence_checkpoint": no_sentence, "reached_4096": maxed,
            }
        names = ("probe_train", "calibration", "heldout")
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                overlap = split_sets[dataset][left] & split_sets[dataset][right]
                if overlap:
                    errors.append(f"data leakage {dataset}/{left}/{right}: {len(overlap)}")
        if len(fingerprints[dataset]) != 1:
            errors.append(f"protocol fingerprint not unique for {dataset}: {sorted(fingerprints[dataset])}")
    for split in ("probe_train", "calibration", "heldout"):
        if checks["mmlu"][split]["subjects_covered"] != 57:
            errors.append(f"MMLU {split} does not cover 57 subjects")
    report = {
        "status": "passed" if not errors else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope_fingerprint": scope["scope_fingerprint"],
        "generation_reused": True, "model_generation_performed": False,
        "dtype_override": "float16", "seed_override": "existing deterministic per-task seed",
        "protocol_fingerprints": {key: sorted(value) for key, value in fingerprints.items()},
        "checks": checks,
        "mmlu_distribution_shift": {
            "present": True,
            "reason": "probe_train/calibration are auxiliary_train while heldout is official test",
            "test_used_for_threshold_selection": False,
        },
        "errors": errors[:200], "error_count": len(errors),
    }
    atomic_json(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

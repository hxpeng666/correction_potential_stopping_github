#!/usr/bin/env python3
"""在非 test 配对清单上验证增量边界调度与冻结 sentence schedule 等价。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.final_paper_protocol import checkpoint_schedules, semantic_boundaries
from src.utils import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("heldout_used") or any(row["split"] == "heldout" for row in manifest["samples"]):
        raise ValueError("边界等价性审计不得使用 heldout")
    tokenizer = Qwen2TokenizerFast.from_pretrained(ROOT / "models/Qwen3-4B", local_files_only=True)
    schedule_mismatches = []
    decoded_text_mismatches = []
    for item in manifest["samples"]:
        import torch
        artifact = torch.load(item["source_fp16_artifact"], map_location="cpu", weights_only=False)
        tokens = list(artifact["dense"]["content_tokens"][:768])
        pieces = [tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False) for token in tokens]
        joined = "".join(pieces)
        full = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        offsets, total = [], 0
        for piece in pieces:
            offsets.append((total, total + len(piece))); total += len(piece)
        semantic = semantic_boundaries(joined, offsets)
        incremental = checkpoint_schedules(semantic, len(tokens), minimum=64, maximum=768, sentence_gap=8)["sentence"]
        frozen = [int(value) for value in artifact["schedules"]["sentence"] if int(value) <= len(tokens)]
        detail = {"dataset": item["dataset"], "split": item["split"], "sample_id": item["sample_id"], "decoded_text_equal": joined == full, "incremental": incremental, "frozen": frozen}
        if joined != full:
            decoded_text_mismatches.append(detail)
        if incremental != frozen:
            schedule_mismatches.append(detail)
    payload = {"status": "passed" if not schedule_mismatches else "failed", "samples": len(manifest["samples"]), "heldout_used": False, "algorithm": "decode each new token once, append to rolling text, apply exact semantic regex and gap rules", "schedule_mismatch_count": len(schedule_mismatches), "schedule_mismatches": schedule_mismatches[:20], "decoded_surface_mismatch_count": len(decoded_text_mismatches), "decoded_surface_mismatches": decoded_text_mismatches[:20], "decision_rule": "checkpoint positions must match exactly; harmless decoded-surface differences are retained as warnings"}
    atomic_json(payload, args.output)
    print(json.dumps({k: payload[k] for k in ("status", "samples", "schedule_mismatch_count", "decoded_surface_mismatch_count")}, indent=2))
    raise SystemExit(0 if not schedule_mismatches else 2)


if __name__ == "__main__":
    main()

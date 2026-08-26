#!/usr/bin/env python3
"""复用 baseline 评测并修正 replay-v3 的统一延迟口径说明。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.utils import atomic_json
import evaluate_final_paper_baselines as entry


def output_argument() -> Path:
    try:
        value = Path(sys.argv[sys.argv.index("--output") + 1])
    except (ValueError, IndexError):
        raise SystemExit("必须提供 --output")
    return value if value.is_absolute() else ROOT / value


if __name__ == "__main__":
    destination = output_argument()
    entry.main()
    path = destination / "baselines.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["timing_note"] = (
        "所有延迟均为 A100 single-request replay-estimated latency；"
        "2080 Ti 分支时间、并发门控时间和高速缓存收集时间全部排除。"
    )
    value["latency_label"] = "A100 single-request replay-estimated latency"
    atomic_json(value, path)

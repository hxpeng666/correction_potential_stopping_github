#!/usr/bin/env python3
"""Download GSM8K and write the JSONL layout expected by the collectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def write_jsonl(rows, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"question": row["question"], "answer": row["answer"]}, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/gsm8k")
    args = parser.parse_args()
    output = Path(args.output)
    dataset = load_dataset("openai/gsm8k", "main")
    write_jsonl(dataset["train"], output / "train.jsonl")
    write_jsonl(dataset["test"], output / "test.jsonl")
    print(json.dumps({
        "status": "complete",
        "train": len(dataset["train"]),
        "test": len(dataset["test"]),
        "output": str(output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

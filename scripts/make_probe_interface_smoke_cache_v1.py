#!/usr/bin/env python3
"""Build a non-scientific interface-only cache for probe smoke tests."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from src.final_paper_inference import atomic_torch_save


NAMES = [
    "boundary",
    "preboundary_nonblank",
    "last4_noncontrol_mean",
    "last8_noncontrol_mean",
    "sentence_mean",
    "paragraph_mean",
    "last8_noncontrol_ln_mean",
    "paragraph_ln_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--calibration", type=int, default=50)
    parser.add_argument("--heldout", type=int, default=50)
    args = parser.parse_args()
    counts = {
        "probe_train": args.train,
        "calibration": args.calibration,
        "heldout": args.heldout,
    }
    for split, count in counts.items():
        paths = sorted((args.source_root / split).glob("sample_*.pt"))[:count]
        destination = args.output_root / split
        destination.mkdir(parents=True, exist_ok=True)
        for path in paths:
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            hidden = artifact["hidden"].float()
            repeated = hidden.repeat(1, len(NAMES), 1).to(torch.float16)
            rows = [dict(row) for row in artifact["rows"]]
            for row in rows:
                row["sampling_pmax"] = 0.5
                row["sampling_probability_gap"] = 0.25
            if len(hidden):
                prompt = F.layer_norm(hidden[0, 0], (hidden.shape[-1],)).to(torch.float16)
            else:
                prompt = torch.zeros(2560, dtype=torch.float16)
            artifact.update(
                {
                    "rows": rows,
                    "hidden": repeated,
                    "representation_names": NAMES,
                    "prompt_hidden_ln": prompt,
                    "token_pooling_fingerprint": "INTERFACE_SMOKE_ONLY_NOT_SCIENTIFIC",
                }
            )
            atomic_torch_save(artifact, destination / path.name)


if __name__ == "__main__":
    main()

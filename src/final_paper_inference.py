"""最终论文完整推理、检查点分支和在线运行共用的推理工具。"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers.cache_utils import DynamicCache

from src.final_paper_protocol import (
    build_mmlu_five_shot_prompt,
    parse_mcq_answer,
)
from src.outcome_verifier import exact_success, gold_answer, predicted_answer
from src.qwen3_reasoning import sample_token


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_example_seed(seed: int, problem_id: str, salt: str = "") -> int:
    digest = hashlib.sha256(f"{seed}:{problem_id}:{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def demonstrations_by_subject(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        result.setdefault(row["subject"], []).append(row)
    invalid = {subject: len(rows) for subject, rows in result.items() if len(rows) != 5}
    if invalid:
        raise ValueError(f"invalid five-shot demonstrations: {invalid}")
    return result


def prompt_messages(
    dataset: str,
    record: dict[str, Any],
    config: dict[str, Any],
    demonstrations: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, str]]:
    if dataset == "gsm8k":
        user = config["prompt"]["user_template"].format(question=record["question"])
    elif dataset == "mmlu":
        if demonstrations is None or record["subject"] not in demonstrations:
            raise KeyError(f"missing demonstrations for {record['subject']}")
        user = build_mmlu_five_shot_prompt(
            record["subject"],
            demonstrations[record["subject"]],
            record["question"],
            record["choices"],
        )
    else:
        raise ValueError(dataset)
    return [
        {"role": "system", "content": config["prompt"]["system"]},
        {"role": "user", "content": user},
    ]


def render_prompt(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def gold_for(dataset: str, record: dict[str, Any]) -> str | None:
    if dataset == "gsm8k":
        return gold_answer(record["answer"])
    if dataset == "mmlu":
        value = str(record["answer"]).strip().upper()
        return value if value in {"A", "B", "C", "D"} else None
    raise ValueError(dataset)


def prediction_for(dataset: str, text: str) -> str | None:
    if dataset == "gsm8k":
        return predicted_answer(text)
    if dataset == "mmlu":
        return parse_mcq_answer(text)
    raise ValueError(dataset)


def success_for(dataset: str, gold: str | None, prediction: str | None) -> bool:
    if dataset == "gsm8k":
        return exact_success(gold, prediction)
    return gold is not None and prediction is not None and gold == prediction


def resolved_generation(config: dict[str, Any], maximum: int) -> dict[str, Any]:
    generation = config["generation"]
    return {
        "max_new_tokens": int(maximum),
        "temperature": float(generation["temperature"]),
        "top_p": float(generation["top_p"]),
        "top_k": int(generation["top_k"]),
    }


def artifact_complete(path: Path, problem_id: str) -> bool:
    if not path.is_file():
        return False
    value = torch.load(path, map_location="cpu", weights_only=False)
    return value.get("status") == "complete" and value.get("problem_id") == problem_id


class PositionHiddenCapture:
    """在一次教师强制前向传播中捕获解码器层的指定位置。"""

    def __init__(self, model, layer_indices: list[int]):
        layers = model.model.layers
        if min(layer_indices) < 0 or max(layer_indices) >= len(layers):
            raise IndexError(f"capture layers {layer_indices} incompatible with {len(layers)}")
        self.layer_indices = list(layer_indices)
        self.positions: torch.Tensor | None = None
        self.values: dict[int, torch.Tensor] = {}
        self.enabled = False
        self.handles = [
            layers[index].register_forward_hook(self._hook(index))
            for index in self.layer_indices
        ]

    def _hook(self, index: int):
        def capture(_module, _inputs, output):
            if not self.enabled or self.positions is None:
                return
            value = output[0] if isinstance(output, tuple) else output
            self.values[index] = value[0].index_select(0, self.positions).detach()
        return capture

    def begin(self, positions: Iterable[int], device: torch.device) -> None:
        self.values.clear()
        self.positions = torch.tensor(list(positions), dtype=torch.long, device=device)
        self.enabled = True

    def finish_cpu(self) -> torch.Tensor:
        self.enabled = False
        missing = [index for index in self.layer_indices if index not in self.values]
        if missing:
            raise RuntimeError(f"hooks did not capture layers {missing}")
        result = torch.stack(
            [self.values[index] for index in self.layer_indices], dim=1
        ).float().cpu()
        self.values.clear()
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def branch_from_legacy_cache(
    model,
    tokenizer,
    legacy_cache,
    *,
    prefix_context: int,
    suffix_ids: torch.Tensor,
    generation: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    cache = DynamicCache.from_legacy_cache(
        tuple(
            (key[:, :, :prefix_context, :], value[:, :, :prefix_context, :])
            for key, value in legacy_cache
        )
    )
    generator = torch.Generator(device=suffix_ids.device).manual_seed(seed)
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    torch.cuda.synchronize()
    started = time.perf_counter()
    mask = torch.ones(
        (1, prefix_context + suffix_ids.shape[1]),
        dtype=torch.long,
        device=suffix_ids.device,
    )
    output = model(
        input_ids=suffix_ids,
        attention_mask=mask,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    past = output.past_key_values
    token = sample_token(
        output.logits,
        generator,
        generation["temperature"],
        generation["top_k"],
        generation["top_p"],
    )
    tokens = [token]
    maximum = int(generation["max_new_tokens"])
    while len(tokens) < maximum and tokens[-1] not in eos:
        token_tensor = torch.tensor([[tokens[-1]]], dtype=torch.long, device=suffix_ids.device)
        mask = torch.ones(
            (1, prefix_context + suffix_ids.shape[1] + len(tokens)),
            dtype=torch.long,
            device=suffix_ids.device,
        )
        output = model(
            input_ids=token_tensor,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        tokens.append(
            sample_token(
                output.logits,
                generator,
                generation["temperature"],
                generation["top_k"],
                generation["top_p"],
            )
        )
    torch.cuda.synchronize()
    generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    suffix_text = tokenizer.decode(suffix_ids[0], skip_special_tokens=True)
    return {
        "tokens": tokens,
        "generated_text": generated_text,
        "text": suffix_text + generated_text,
        "wall_ms": 1000.0 * (time.perf_counter() - started),
    }

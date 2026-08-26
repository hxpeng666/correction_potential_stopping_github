"""Frozen protocol helpers for the DeepSeek-R1-Distill-Qwen-7B experiment."""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import transformers.utils as _transformers_utils
import transformers.utils.import_utils as _transformers_import_utils

_transformers_utils.is_flash_attn_2_available = lambda: False
_transformers_import_utils.is_flash_attn_2_available = lambda: False
os.environ.setdefault("USE_TIMM", "0")
os.environ.setdefault("USE_TORCHVISION", "0")

from transformers import AutoConfig, AutoTokenizer, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache


PARAGRAPH = re.compile(r"\n\s*\n+")
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def canonical_fingerprint(config: dict[str, Any]) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def inspect_model(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    index_path = path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"incomplete local model at {path}")
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if config.model_type != "qwen2" or "Qwen2ForCausalLM" not in (config.architectures or []):
        raise ValueError(f"expected Qwen2ForCausalLM, got {config.model_type}/{config.architectures}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing model shards: {missing}")
    return {
        "path": str(path.resolve()),
        "model_type": config.model_type,
        "architectures": list(config.architectures or []),
        "layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "max_position_embeddings": int(config.max_position_embeddings),
        "weight_shards": shards,
        "metadata_fingerprint": hashlib.sha256(
            config_path.read_bytes() + index_path.read_bytes()
        ).hexdigest(),
    }


def load_model(path: Path, device: torch.device):
    audit = inspect_model(path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    model = Qwen2ForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, audit


class CheckpointHiddenCapture:
    def __init__(self, model, layer: int):
        layers = model.model.layers
        if not 0 <= layer < len(layers):
            raise IndexError(f"layer {layer} incompatible with {len(layers)} layers")
        self.layer = layer
        self.enabled = False
        self.value: torch.Tensor | None = None
        self.handle = layers[layer].register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        if self.enabled:
            value = output[0] if isinstance(output, tuple) else output
            self.value = value[:, -1, :].detach()

    def begin(self) -> None:
        self.value = None
        self.enabled = True

    def finish_cpu(self) -> torch.Tensor:
        self.enabled = False
        if self.value is None:
            raise RuntimeError("hidden-state hook captured nothing")
        result = self.value[0].float().cpu()
        self.value = None
        return result

    def close(self) -> None:
        self.handle.remove()


def render_prompt(tokenizer, question: str) -> str:
    instruction = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question.strip() + "\n\n" + instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def entropy_top20(logits: torch.Tensor) -> float:
    top = logits[0, -1].float().topk(20).values
    probability = torch.softmax(top, dim=-1)
    return float(-(probability * probability.clamp_min(1e-12).log()).sum())


def sample_token(
    logits: torch.Tensor,
    generator: torch.Generator,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    values = logits[0, -1].float() / temperature
    if top_k > 0:
        cutoff = values.topk(min(top_k, values.numel())).values[-1]
        values = values.masked_fill(values < cutoff, -torch.inf)
    if top_p < 1.0:
        ordered, indices = torch.sort(values, descending=True)
        probabilities = torch.softmax(ordered, dim=-1)
        remove = probabilities.cumsum(dim=-1) - probabilities > top_p
        ordered = ordered.masked_fill(remove, -torch.inf)
        values = torch.full_like(values, -torch.inf).scatter(0, indices, ordered)
    return int(torch.multinomial(torch.softmax(values, dim=-1), 1, generator=generator))


def generate_dense(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    torch.cuda.synchronize(input_ids.device)
    started = time.perf_counter()
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
        return_dict=True,
    )
    past = output.past_key_values
    tokens: list[int] = []
    entropies: list[float] = []
    while len(tokens) < max_new_tokens:
        entropies.append(entropy_top20(output.logits))
        token = sample_token(output.logits, generator, temperature, top_p, top_k)
        tokens.append(token)
        if token in eos or len(tokens) >= max_new_tokens:
            break
        current = torch.tensor([[token]], dtype=torch.long, device=input_ids.device)
        mask = torch.ones((1, input_ids.shape[1] + len(tokens)), dtype=torch.long, device=input_ids.device)
        output = model(
            input_ids=current,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
    torch.cuda.synchronize(input_ids.device)
    return {
        "tokens": tokens,
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "entropies_top20": entropies,
        "wall_ms": 1000.0 * (time.perf_counter() - started),
        "reached_max_tokens": len(tokens) >= max_new_tokens,
    }


def prefill_token_cache(
    model,
    token_ids: torch.Tensor,
    *,
    chunk_size: int = 512,
):
    """Teacher-force a known prefix without materializing vocabulary logits for it."""
    if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] <= 0:
        raise ValueError(f"expected non-empty [1, length] token_ids, got {tuple(token_ids.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    past = None
    last_hidden = None
    for start in range(0, int(token_ids.shape[1]), chunk_size):
        end = min(start + chunk_size, int(token_ids.shape[1]))
        current = token_ids[:, start:end]
        mask = torch.ones((1, end), dtype=torch.long, device=token_ids.device)
        output = model.model(
            input_ids=current,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1:, :]
        del output
    if past is None or last_hidden is None:
        raise RuntimeError("prefix prefill produced no cache")
    return past, last_hidden


def advance_sampling_generator(
    generator: torch.Generator,
    *,
    steps: int,
    device: torch.device,
) -> None:
    """Advance the dedicated CUDA multinomial generator by an exact call count."""
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    probabilities = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)
    for _ in range(steps):
        torch.multinomial(probabilities, 1, generator=generator)


def replay_sampled_prefix_cache(
    model,
    input_ids: torch.Tensor,
    source_tokens: list[int],
):
    """Rebuild the exact autoregressive KV path while skipping logits and sampling."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] <= 0:
        raise ValueError(f"expected non-empty [1, length] input_ids, got {tuple(input_ids.shape)}")
    if not source_tokens:
        raise ValueError("source_tokens must be non-empty")
    prompt = model.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
        return_dict=True,
    )
    past = prompt.past_key_values
    last_hidden = prompt.last_hidden_state[:, -1:, :]
    del prompt
    prompt_length = int(input_ids.shape[1])
    for index, token in enumerate(source_tokens, start=1):
        current = torch.tensor([[token]], dtype=torch.long, device=input_ids.device)
        mask = torch.ones(
            (1, prompt_length + index),
            dtype=torch.long,
            device=input_ids.device,
        )
        output = model.model(
            input_ids=current,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1:, :]
        del output
    return past, last_hidden


def generate_dense_from_prefix(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    source_dense: dict[str, Any],
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, Any]:
    """Continue a capped sampled rollout while preserving its exact RNG stream."""
    source_tokens = [int(token) for token in source_dense["tokens"]]
    source_entropies = [float(value) for value in source_dense["entropies_top20"]]
    if not source_dense.get("reached_max_tokens"):
        raise ValueError("incremental continuation requires a source rollout that reached its cap")
    if not source_tokens or len(source_tokens) >= max_new_tokens:
        raise ValueError(
            f"source length must be in [1, target): source={len(source_tokens)} target={max_new_tokens}"
        )
    if len(source_entropies) != len(source_tokens):
        raise ValueError(
            f"source entropy/token mismatch: {len(source_entropies)} != {len(source_tokens)}"
        )

    device = input_ids.device
    generator = torch.Generator(device=device).manual_seed(seed)
    advance_sampling_generator(generator, steps=len(source_tokens), device=device)

    torch.cuda.synchronize(device)
    prefill_started = time.perf_counter()
    past, last_hidden = replay_sampled_prefix_cache(
        model,
        input_ids,
        source_tokens,
    )
    logits = model.lm_head(last_hidden)
    del last_hidden
    torch.cuda.synchronize(device)
    prefill_wall_ms = 1000.0 * (time.perf_counter() - prefill_started)

    tokens = list(source_tokens)
    entropies = list(source_entropies)
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    torch.cuda.synchronize(device)
    continuation_started = time.perf_counter()
    while len(tokens) < max_new_tokens:
        entropies.append(entropy_top20(logits))
        token = sample_token(logits, generator, temperature, top_p, top_k)
        tokens.append(token)
        if token in eos or len(tokens) >= max_new_tokens:
            break
        current = torch.tensor([[token]], dtype=torch.long, device=device)
        mask = torch.ones(
            (1, int(input_ids.shape[1]) + len(tokens)),
            dtype=torch.long,
            device=device,
        )
        output = model.model(
            input_ids=current,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        logits = model.lm_head(output.last_hidden_state[:, -1:, :])
        del output
    torch.cuda.synchronize(device)
    continuation_wall_ms = 1000.0 * (time.perf_counter() - continuation_started)
    source_wall_ms = float(source_dense["wall_ms"])
    return {
        "tokens": tokens,
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "entropies_top20": entropies,
        "wall_ms": source_wall_ms + continuation_wall_ms,
        "reached_max_tokens": len(tokens) >= max_new_tokens,
        "incremental_resume": {
            "source_tokens": len(source_tokens),
            "prefix_replay_mode": "tokenwise_teacher_forcing_without_lm_head_or_sampling",
            "prefill_wall_ms": prefill_wall_ms,
            "source_dense_wall_ms": source_wall_ms,
            "continuation_decode_wall_ms": continuation_wall_ms,
            "rng_fast_forward_multinomial_calls": len(source_tokens),
        },
    }


def reasoning_end(tokenizer, tokens: list[int]) -> tuple[int, str]:
    pattern = list(tokenizer("</think>", add_special_tokens=False).input_ids)
    for start in range(0, len(tokens) - len(pattern) + 1):
        if tokens[start : start + len(pattern)] == pattern:
            return start, "first_think_close"
    return len(tokens), "full_dense_content"


def paragraph_checkpoints(tokenizer, tokens: list[int]) -> tuple[list[int], dict[str, Any]]:
    upper, source = reasoning_end(tokenizer, tokens)
    limited = tokens[:upper]
    if not limited:
        return [], {"reasoning_end": upper, "reasoning_end_source": source}
    text = tokenizer.decode(limited, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded.input_ids) == limited:
        ends = [int(end) for _start, end in encoded.offset_mapping]
    else:
        ends = [
            len(tokenizer.decode(limited[:end], skip_special_tokens=False, clean_up_tokenization_spaces=False))
            for end in range(1, len(limited) + 1)
        ]
    positions: set[int] = set()
    for match in PARAGRAPH.finditer(text):
        index = bisect.bisect_left(ends, match.end())
        if index < len(ends):
            positions.add(index + 1)
    return sorted(positions), {
        "reasoning_end": upper,
        "reasoning_end_source": source,
        "paragraph_checkpoints": len(positions),
        "range_filter": "none",
    }


def greedy_branch(
    model,
    tokenizer,
    base_cache,
    *,
    prefix_context: int,
    suffix_ids: torch.Tensor,
    maximum: int,
) -> dict[str, Any]:
    cache = DynamicCache.from_legacy_cache(base_cache.to_legacy_cache())
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    started = time.perf_counter()
    mask = torch.ones((1, prefix_context + suffix_ids.shape[1]), dtype=torch.long, device=suffix_ids.device)
    output = model(
        input_ids=suffix_ids,
        attention_mask=mask,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    past = output.past_key_values
    tokens = [int(torch.argmax(output.logits[0, -1].float()).item())]
    while len(tokens) < maximum and tokens[-1] not in eos:
        current = torch.tensor([[tokens[-1]]], dtype=torch.long, device=suffix_ids.device)
        mask = torch.ones((1, prefix_context + suffix_ids.shape[1] + len(tokens)), dtype=torch.long, device=suffix_ids.device)
        output = model(
            input_ids=current,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        tokens.append(int(torch.argmax(output.logits[0, -1].float()).item()))
    generated = tokenizer.decode(tokens, skip_special_tokens=True)
    suffix = tokenizer.decode(suffix_ids[0], skip_special_tokens=True)
    return {
        "tokens": tokens,
        "generated_text": generated,
        "text": suffix + generated,
        "wall_ms": 1000.0 * (time.perf_counter() - started),
    }


def last_boxed(text: str) -> str | None:
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(starts):
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            return text[match.end() : index - 1].strip()
    unbraced = re.findall(r"\\boxed\s+([^\s$.,]+)", text)
    if unbraced:
        return unbraced[-1].strip()
    return None


def normalize_math(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().replace("$", "").replace("\\!", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace(" ", "")
    text = text.replace("^{\\circ}", "").replace("^\\circ", "")
    if text.startswith("\\text{") and text.endswith("}"):
        text = text[6:-1]
    if text.endswith("."):
        text = text[:-1]
    return text


def numeric_value(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.replace(",", "").strip()
    try:
        if text.startswith("\\frac{"):
            match = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
            if match:
                return float(match.group(1)) / float(match.group(2))
        if "/" in text and text.count("/") == 1:
            left, right = text.split("/")
            return float(left) / float(right)
        return float(text)
    except (ValueError, ZeroDivisionError):
        return None


def _parse_digits(value: Any) -> float | None:
    text = str(value).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        if text.endswith("%"):
            text = text[:-1].rstrip("\\")
            try:
                return float(text) / 100.0
            except ValueError:
                return None
    return None


def _symbolic_equal_unbounded(left: str, right: str) -> bool:
    """Symbolic fallback adapted from Qwen2.5-Math's official grader."""
    try:
        import sympy
        from sympy import N, simplify
        from sympy.parsing.sympy_parser import parse_expr
        from sympy.parsing.latex import parse_latex
        from latex2sympy2 import latex2sympy
    except Exception:
        return False

    def parse(value: str):
        for function in (parse_latex, parse_expr, latex2sympy):
            for candidate in (value.replace("\\\\", "\\"), value):
                try:
                    return function(candidate)
                except Exception:
                    pass
        return value

    a, b = parse(left), parse(right)
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass
    try:
        if abs(a.lhs - a.rhs).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass
    try:
        if math.isclose(float(N(a)), float(N(b)), rel_tol=1e-4):
            return True
    except Exception:
        pass
    try:
        if a.shape == b.shape:
            return bool(a.applyfunc(lambda x: round(x, 3)).equals(b.applyfunc(lambda x: round(x, 3))))
    except Exception:
        pass
    return False


def _symbolic_equal(left: str, right: str) -> bool:
    """Use the protocol's dataset-standard symbolic comparator without a time cutoff."""
    return _symbolic_equal_unbounded(left, right)


def math_equal(predicted: str, reference: str) -> bool:
    """Dataset-standard MATH equivalence, aligned to Qwen2.5-Math grader."""
    if predicted is None or reference is None:
        return False
    prediction_text = str(predicted).strip()
    reference_text = str(reference).strip()
    if prediction_text.lower() == reference_text.lower():
        return True

    prediction_number = _parse_digits(prediction_text)
    reference_number = _parse_digits(reference_text)
    if prediction_number is not None and reference_number is not None:
        return any(
            math.isclose(prediction_number, candidate, rel_tol=1e-4)
            for candidate in (reference_number / 100.0, reference_number, reference_number * 100.0)
        )
    if not prediction_text:
        return False

    pred_str, ref_str = prediction_text, reference_text
    if (
        (pred_str.startswith("[") and pred_str.endswith("]") and not ref_str.startswith("("))
        or (pred_str.startswith("(") and pred_str.endswith(")") and not ref_str.startswith("["))
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    stripped_prediction = pred_str
    stripped_reference = ref_str
    for symbol in ("{", "}", "(", ")"):
        stripped_prediction = stripped_prediction.replace(symbol, "")
        stripped_reference = stripped_reference.replace(symbol, "")
    if stripped_prediction.lower() == stripped_reference.lower():
        return True

    if (
        re.match(r"(\(|\[).+(\)|\])", prediction_text)
        and re.match(r"(\(|\[).+(\)|\])", reference_text)
    ):
        prediction_parts = prediction_text[1:-1].split(",")
        reference_parts = reference_text[1:-1].split(",")
        if len(prediction_parts) == len(reference_parts) and all(
            math_equal(left, right)
            for left, right in zip(prediction_parts, reference_parts)
        ):
            return True

    matrix_beginnings = ("\\begin{pmatrix}", "\\begin{bmatrix}")
    matrix_endings = ("\\end{pmatrix}", "\\end{bmatrix}")
    if (
        prediction_text.startswith(matrix_beginnings)
        and prediction_text.endswith(matrix_endings)
        and reference_text.startswith(matrix_beginnings)
        and reference_text.endswith(matrix_endings)
    ):
        def matrix_rows(value: str) -> list[list[str]]:
            body = re.sub(r"^\\begin\{[pb]matrix\}|\\end\{[pb]matrix\}$", "", value)
            return [[cell.strip() for cell in row.split("&")] for row in body.split("\\\\") if row.strip()]

        prediction_rows = matrix_rows(prediction_text)
        reference_rows = matrix_rows(reference_text)
        if len(prediction_rows) == len(reference_rows) and all(
            len(left) == len(right)
            and all(math_equal(a, b) for a, b in zip(left, right))
            for left, right in zip(prediction_rows, reference_rows)
        ):
            return True

    if prediction_text.count("=") == 1 and reference_text.count("=") == 1:
        left = prediction_text.split("=")
        right = reference_text.split("=")
        prediction_equation = f"{left[0].strip()} - ({left[1].strip()})"
        reference_equation = f"{right[0].strip()} - ({right[1].strip()})"
        if _symbolic_equal(prediction_equation, reference_equation) or _symbolic_equal(
            f"-({prediction_equation})", reference_equation
        ):
            return True
    elif prediction_text.count("=") == 1 and len(prediction_text.split("=")[0].strip()) <= 2:
        if math_equal(prediction_text.split("=")[1], reference_text):
            return True
    elif reference_text.count("=") == 1 and len(reference_text.split("=")[0].strip()) <= 2:
        if math_equal(prediction_text, reference_text.split("=")[1]):
            return True

    return _symbolic_equal(prediction_text, reference_text)


def prediction(dataset: str, text: str) -> str | None:
    boxed = last_boxed(text)
    if boxed is not None:
        return normalize_math(boxed)
    matches = NUMBER.findall(text)
    return normalize_math(matches[-1]) if matches else None


def success(dataset: str, gold: str | None, predicted: str | None) -> bool:
    gold_norm = normalize_math(gold)
    pred_norm = normalize_math(predicted)
    if gold_norm is None or pred_norm is None:
        return False
    if dataset in {"gsm8k", "aime"}:
        gold_number = numeric_value(gold_norm)
        pred_number = numeric_value(pred_norm)
        return gold_number is not None and pred_number is not None and math.isclose(
            gold_number, pred_number, rel_tol=1e-9, abs_tol=1e-9
        )
    return math_equal(pred_norm, gold_norm)


def stable_seed(seed: int, problem_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{problem_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def tail_mean(values: list[float], end: int, width: int = 8) -> float:
    local = values[max(0, end - width) : end]
    return float(np.mean(local)) if local else 0.0

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Qwen3 是纯文本模型；关闭可选的视觉组件自动发现，可避免导入无关的
# torchvision/timm 安装。注意力后端由实验配置显式指定（论文协议使用 SDPA）。
os.environ.setdefault("USE_TIMM", "0")
os.environ.setdefault("USE_TORCHVISION", "0")

from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


def inspect_qwen3(path: Path) -> dict[str, Any]:
    """只有 *path* 指向完整的本地 Qwen3 检查点时才允许继续。"""
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing local Qwen3 config: {config_path}")
    config = Qwen3Config.from_pretrained(path, local_files_only=True)
    if config.model_type != "qwen3" or "Qwen3ForCausalLM" not in (config.architectures or []):
        raise ValueError(f"expected Qwen3ForCausalLM, found {config.model_type}/{config.architectures}")
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing weight index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (path / name).is_file() or (path / name).stat().st_size == 0]
    actual_bytes = sum((path / name).stat().st_size for name in shards if (path / name).is_file())
    expected_bytes = int(index.get("metadata", {}).get("total_size", 0))
    if missing or actual_bytes < expected_bytes:
        raise FileNotFoundError(
            f"incomplete Qwen3 weights: missing={missing}, actual={actual_bytes}, expected={expected_bytes}"
        )
    revision_fingerprint = hashlib.sha256(config_path.read_bytes() + index_path.read_bytes()).hexdigest()
    return {
        "path": str(path.resolve()),
        "model_type": config.model_type,
        "architectures": config.architectures,
        "layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "weight_shards": shards,
        "weight_bytes": actual_bytes,
        "metadata_fingerprint": revision_fingerprint,
    }


def load_qwen3(path: Path, device: torch.device, dtype_name: str = "float16",
               attention_backend: str = "sdpa"):
    audit = inspect_qwen3(path)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    tokenizer = Qwen2TokenizerFast.from_pretrained(path, local_files_only=True)
    model = Qwen3ForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        attn_implementation=attention_backend,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, audit


class CheckpointHiddenCapture:
    """可开关的钩子：仅在请求的解码步骤保留张量。"""

    def __init__(self, model, layer_indices: list[int]):
        layers = model.model.layers
        if min(layer_indices) < 0 or max(layer_indices) >= len(layers):
            raise IndexError(f"capture layers {layer_indices} incompatible with {len(layers)} layers")
        self.layer_indices = list(layer_indices)
        self.enabled = False
        self.values: dict[int, torch.Tensor] = {}
        self.handles = [layers[index].register_forward_hook(self._hook(index)) for index in layer_indices]

    def _hook(self, index: int):
        def capture(_module, _inputs, output):
            if self.enabled:
                value = output[0] if isinstance(output, tuple) else output
                self.values[index] = value[:, -1, :].detach()
        return capture

    def begin(self) -> None:
        self.values.clear()
        self.enabled = True

    def finish_cpu(self) -> torch.Tensor:
        self.enabled = False
        missing = [index for index in self.layer_indices if index not in self.values]
        if missing:
            raise RuntimeError(f"hooks did not capture layers {missing}")
        return torch.stack([self.values[index][0].float().cpu() for index in self.layer_indices])

    def finish_device(self) -> torch.Tensor:
        self.enabled = False
        missing = [index for index in self.layer_indices if index not in self.values]
        if missing:
            raise RuntimeError(f"hooks did not capture layers {missing}")
        return torch.cat([self.values[index][0].float() for index in self.layer_indices])

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def confidence_stats(logits: torch.Tensor) -> tuple[float, float, float]:
    values = logits[0, -1].float()
    top = values.topk(20)
    log_z = torch.logsumexp(values, dim=-1)
    logp = float(top.values[0] - log_z)
    margin = float(top.values[0] - top.values[1])
    probabilities = torch.softmax(top.values, dim=-1)
    entropy20 = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return logp, margin, entropy20


def sample_token(logits: torch.Tensor, generator: torch.Generator, temperature: float,
                 top_k: int, top_p: float) -> int:
    values = logits[0, -1].float() / temperature
    if top_k > 0:
        cutoff = values.topk(min(top_k, values.numel())).values[-1]
        values = values.masked_fill(values < cutoff, -torch.inf)
    if top_p < 1.0:
        sorted_values, sorted_indices = torch.sort(values, descending=True)
        sorted_probabilities = torch.softmax(sorted_values, dim=-1)
        remove = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities > top_p
        sorted_values = sorted_values.masked_fill(remove, -torch.inf)
        values = torch.full_like(values, -torch.inf).scatter(0, sorted_indices, sorted_values)
    probabilities = torch.softmax(values, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator))


def _timed_forward(model, *, measure_timing: bool = True, **kwargs):
    if not measure_timing:
        return model(**kwargs), float("nan")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = model(**kwargs)
    end.record()
    end.synchronize()
    return output, float(start.elapsed_time(end))


@dataclass
class Trace:
    tokens: list[int]
    hidden: dict[int, torch.Tensor]
    logps: list[float]
    margins: list[float]
    entropies: list[float]
    prefill_cuda_ms: float
    decode_cuda_ms: list[float]
    wall_ms: float


def generate_trace(model, tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                   generation: dict[str, Any], seed: int,
                   capture: CheckpointHiddenCapture | None = None,
                   checkpoints: list[int] | None = None,
                   measure_timing: bool = True) -> Trace:
    checkpoints = sorted(set(checkpoints or []))
    checkpoint_set = set(checkpoints)
    generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    eos_ids = tokenizer.eos_token_id
    eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    started = time.perf_counter() if measure_timing else None
    output, prefill_ms = _timed_forward(
        model,
        measure_timing=measure_timing,
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    past = output.past_key_values
    token = sample_token(output.logits, generator, generation["temperature"],
                         generation["top_k"], generation["top_p"])
    logp, margin, entropy = confidence_stats(output.logits)
    tokens = [token]
    logps, margins, entropies = [logp], [margin], [entropy]
    decode_ms: list[float] = []
    hidden: dict[int, torch.Tensor] = {}
    max_new_tokens = int(generation["max_new_tokens"])
    while len(tokens) < max_new_tokens and tokens[-1] not in eos:
        at_checkpoint = len(tokens) in checkpoint_set and capture is not None
        if at_checkpoint:
            capture.begin()
        token_tensor = torch.tensor([[tokens[-1]]], dtype=input_ids.dtype, device=input_ids.device)
        total_length = input_ids.shape[1] + len(tokens)
        mask = torch.ones((1, total_length), dtype=attention_mask.dtype, device=input_ids.device)
        output, elapsed = _timed_forward(
            model,
            measure_timing=measure_timing,
            input_ids=token_tensor,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        if at_checkpoint:
            hidden[len(tokens)] = capture.finish_cpu()
        past = output.past_key_values
        token = sample_token(output.logits, generator, generation["temperature"],
                             generation["top_k"], generation["top_p"])
        logp, margin, entropy = confidence_stats(output.logits)
        tokens.append(token)
        logps.append(logp)
        margins.append(margin)
        entropies.append(entropy)
        if measure_timing:
            decode_ms.append(elapsed)
    if measure_timing:
        torch.cuda.synchronize()
    wall_ms = (
        1000.0 * (time.perf_counter() - started)
        if started is not None
        else float("nan")
    )
    return Trace(tokens, hidden, logps, margins, entropies, prefill_ms, decode_ms,
                 wall_ms)


def tail_mean(values: list[float], end: int, width: int = 8) -> float:
    part = values[max(0, end - width):end]
    return sum(part) / len(part) if part else math.nan

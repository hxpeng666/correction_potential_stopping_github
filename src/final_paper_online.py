"""用于最终论文计时的、无需答案探针的真实在线句子级停止实现。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.final_paper_probe import (
    FinalPaperProbe,
    build_online_feature,
)
from src.final_paper_protocol import BOUNDARY
from src.qwen3_reasoning import (
    CheckpointHiddenCapture,
    sample_token,
)


def load_probe_bundle(path: Path, device: torch.device) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("status") != "complete":
        raise ValueError(f"incomplete probe: {path}")
    model = FinalPaperProbe(int(artifact["input_width"])).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return {
        **artifact,
        "model": model,
        "scaler_mean_numpy": artifact["scaler_mean"].numpy().astype(np.float32),
        "scaler_scale_numpy": artifact["scaler_scale"].numpy().astype(np.float32),
    }


def _ends_at_sentence_boundary(tokenizer, tokens: list[int]) -> bool:
    text = tokenizer.decode(
        tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return any(match.end() == len(text) for match in BOUNDARY.finditer(text))

def _entropy_top20(logits: torch.Tensor) -> torch.Tensor:
    values = logits[0, -1].float().topk(20).values
    probabilities = torch.softmax(values, dim=-1)
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum()




def _generate_suffix(
    model,
    tokenizer,
    past,
    *,
    prefix_context: int,
    suffix_ids: torch.Tensor,
    generation: dict[str, Any],
    generator: torch.Generator,
) -> tuple[list[int], Any]:
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    mask = torch.ones(
        (1, prefix_context + suffix_ids.shape[1]),
        dtype=torch.long,
        device=suffix_ids.device,
    )
    output = model(
        input_ids=suffix_ids,
        attention_mask=mask,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
    )
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
        token_tensor = torch.tensor(
            [[tokens[-1]]], dtype=torch.long, device=suffix_ids.device
        )
        mask = torch.ones(
            (1, prefix_context + suffix_ids.shape[1] + len(tokens)),
            dtype=torch.long,
            device=suffix_ids.device,
        )
        output = model(
            input_ids=token_tensor,
            attention_mask=mask,
            past_key_values=output.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        tokens.append(
            sample_token(
                output.logits,
                generator,
                generation["temperature"],
                generation["top_k"],
                generation["top_p"],
            )
        )
    return tokens, output.past_key_values


def generate_online_stopped(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    dense_generation: dict[str, Any],
    branch_generation: dict[str, Any],
    suffix_ids: torch.Tensor,
    probe_bundle: dict[str, Any],
    threshold: float,
    direction: str,
    checkpoint_protocol: dict[str, Any],
    dense_seed: int,
    branch_seed_for_checkpoint,
) -> dict[str, Any]:
    """解码一条轨迹，并且只在句子边界运行仅依赖隐藏状态的停止器。"""
    device = input_ids.device
    layer = int(probe_bundle["run_spec"]["layer"])
    feature_kind = str(probe_bundle["run_spec"]["feature_kind"])
    probe = probe_bundle["model"]
    scaler_mean = probe_bundle["scaler_mean_numpy"]
    scaler_scale = probe_bundle["scaler_scale_numpy"]
    capture = CheckpointHiddenCapture(model, [layer])
    dense_generator = torch.Generator(device=device).manual_seed(dense_seed)
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    minimum = int(checkpoint_protocol["minimum"])
    maximum_checkpoint = int(checkpoint_protocol["maximum"])
    minimum_gap = int(checkpoint_protocol["sentence_minimum_gap"])
    previous_hidden: np.ndarray | None = None
    previous_checkpoint = 0
    checkpoint_records: list[dict[str, Any]] = []
    stopper_overhead_ms = 0.0
    boundary_overhead_ms = 0.0
    entropy_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    past = output.past_key_values
    token = sample_token(
        output.logits,
        dense_generator,
        dense_generation["temperature"],
        dense_generation["top_k"],
        dense_generation["top_p"],
    )
    entropy_start = torch.cuda.Event(enable_timing=True)
    entropy_end = torch.cuda.Event(enable_timing=True)
    entropy_start.record()
    entropy = _entropy_top20(output.logits)
    entropy_end.record()
    entropy_events.append((entropy_start, entropy_end))
    tokens = [token]
    entropies = [entropy]
    stopped = False
    stop_checkpoint = None
    branch_tokens: list[int] = []
    branch_text = ""

    while (
        len(tokens) < int(dense_generation["max_new_tokens"])
        and tokens[-1] not in eos
    ):
        checkpoint = len(tokens)
        boundary_started = time.perf_counter()
        is_boundary = (
            minimum <= checkpoint <= maximum_checkpoint
            and checkpoint - previous_checkpoint >= minimum_gap
            and _ends_at_sentence_boundary(tokenizer, tokens)
        )
        boundary_overhead_ms += 1000.0 * (time.perf_counter() - boundary_started)
        if is_boundary:
            capture.begin()
        token_tensor = torch.tensor(
            [[tokens[-1]]], dtype=torch.long, device=device
        )
        mask = torch.ones(
            (1, input_ids.shape[1] + len(tokens)),
            dtype=attention_mask.dtype,
            device=device,
        )
        output = model(
            input_ids=token_tensor,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        if is_boundary:
            overhead_started = time.perf_counter()
            current_hidden = capture.finish_cpu()[0].numpy().astype(np.float32)
            entropy_tail8 = float(torch.stack(
                entropies[max(0, checkpoint - 8):checkpoint]
            ).mean().cpu())
            feature = build_online_feature(
                current_hidden,
                previous_hidden,
                checkpoint,
                previous_checkpoint,
                entropy_tail8,
                feature_kind,
            )
            standardized = (
                (feature - scaler_mean[None, :])
                / scaler_scale[None, :]
            ).astype(np.float32, copy=False)
            with torch.inference_mode():
                score = float(
                    torch.sigmoid(
                        probe(torch.from_numpy(standardized).to(device))
                    )[0].cpu()
                )
            torch.cuda.synchronize()
            stopper_overhead_ms += 1000.0 * (time.perf_counter() - overhead_started)
            eligible = score >= threshold if direction == "high" else score <= threshold
            checkpoint_records.append(
                {
                    "checkpoint": checkpoint,
                    "score": score,
                    "eligible": bool(eligible),
                    "entropy_tail8": entropy_tail8,
                }
            )
            previous_hidden = current_hidden
            previous_checkpoint = checkpoint
            if eligible:
                branch_generator = torch.Generator(device=device).manual_seed(
                    int(branch_seed_for_checkpoint(checkpoint))
                )
                branch_tokens, _past = _generate_suffix(
                    model,
                    tokenizer,
                    past,
                    prefix_context=int(input_ids.shape[1]) + checkpoint,
                    suffix_ids=suffix_ids,
                    generation=branch_generation,
                    generator=branch_generator,
                )
                generated_text = tokenizer.decode(
                    branch_tokens, skip_special_tokens=True
                )
                suffix_text = tokenizer.decode(
                    suffix_ids[0], skip_special_tokens=True
                )
                branch_text = suffix_text + generated_text
                stopped = True
                stop_checkpoint = checkpoint
                break
        token = sample_token(
            output.logits,
            dense_generator,
            dense_generation["temperature"],
            dense_generation["top_k"],
            dense_generation["top_p"],
        )
        entropy_start = torch.cuda.Event(enable_timing=True)
        entropy_end = torch.cuda.Event(enable_timing=True)
        entropy_start.record()
        entropy = _entropy_top20(output.logits)
        entropy_end.record()
        entropy_events.append((entropy_start, entropy_end))
        tokens.append(token)
        entropies.append(entropy)

    capture.close()
    torch.cuda.synchronize()
    entropy_cuda_ms = float(sum(
        start.elapsed_time(end) for start, end in entropy_events
    ))
    wall_ms = 1000.0 * (time.perf_counter() - started)
    if stopped:
        prediction_text = branch_text
        reasoning_tokens = int(stop_checkpoint)
        total_generated_tokens = reasoning_tokens + len(branch_tokens)
    else:
        prediction_text = tokenizer.decode(tokens, skip_special_tokens=True)
        reasoning_tokens = len(tokens)
        total_generated_tokens = len(tokens)
    return {
        "stopped": stopped,
        "fallback": not stopped,
        "stop_checkpoint": stop_checkpoint,
        "reasoning_tokens": reasoning_tokens,
        "final_answer_tokens": len(branch_tokens),
        "total_generated_tokens": total_generated_tokens,
        "reasoning_text": tokenizer.decode(
            tokens[:reasoning_tokens], skip_special_tokens=True
        ),
        "answer_text": prediction_text,
        "branch_tokens": branch_tokens,
        "wall_ms": wall_ms,
        "stopper_overhead_ms": stopper_overhead_ms + boundary_overhead_ms + entropy_cuda_ms,
        "entropy_cuda_ms": entropy_cuda_ms,
        "feature_mlp_overhead_ms": stopper_overhead_ms,
        "boundary_overhead_ms": boundary_overhead_ms,
        "checkpoints_evaluated": checkpoint_records,
        "reached_max_tokens": (
            (not stopped)
            and len(tokens) >= int(dense_generation["max_new_tokens"])
        ),
    }



def generate_online_dense(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    generation: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """与策略匹配的批大小 1 完整推理计时，不进行逐令牌同步。"""
    generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    token = sample_token(
        output.logits,
        generator,
        generation["temperature"],
        generation["top_k"],
        generation["top_p"],
    )
    tokens = [token]
    while len(tokens) < int(generation["max_new_tokens"]) and tokens[-1] not in eos:
        token_tensor = torch.tensor(
            [[tokens[-1]]], dtype=torch.long, device=input_ids.device
        )
        mask = torch.ones(
            (1, input_ids.shape[1] + len(tokens)),
            dtype=attention_mask.dtype,
            device=input_ids.device,
        )
        output = model(
            input_ids=token_tensor,
            attention_mask=mask,
            past_key_values=output.past_key_values,
            use_cache=True,
            return_dict=True,
        )
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
    wall_ms = 1000.0 * (time.perf_counter() - started)
    return {
        "tokens": tokens,
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "reasoning_tokens": len(tokens),
        "total_generated_tokens": len(tokens),
        "wall_ms": wall_ms,
        "reached_max_tokens": len(tokens) >= int(generation["max_new_tokens"]),
        "stopper_overhead_ms": 0.0,
        "fallback": False,
        "stopped": False,
        "stop_checkpoint": None,
    }

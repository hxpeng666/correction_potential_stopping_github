"""不可变公共轨迹缓存的设备无关辅助函数。

本模块有意不包含分布式调度器或文件系统任务队列，只定义科学实验所需的
缓存约定、确定性采样种子、协议指纹以及句子与固定检查点构造方法。
"""
from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from src.final_paper_protocol import BOUNDARY, canonical_fingerprint
from src.qwen3_reasoning import inspect_qwen3
from src.utils import load_yaml


BRANCH_DIRECT = -1


def task_seed(
    global_seed: int,
    dataset: str,
    split: str,
    sample_id: str,
    checkpoint: int | str,
) -> int:
    """返回由预注册五元组键唯一确定的任务级随机种子。"""
    payload = f"{global_seed}:{dataset}:{split}:{sample_id}:{checkpoint}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def protocol_fingerprint(
    config_path: Path,
    split_manifest: Path,
    model_root: Path,
) -> str:
    """为复用产物前必须一致的所有字段计算联合指纹。"""
    config = load_yaml(config_path)
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    model = inspect_qwen3(model_root)
    protected = {
        "protocol_id": config["protocol_id"],
        "seed": config["seed"],
        "dataset": config["dataset"],
        "model": config["model"],
        "generation": config["generation"],
        "checkpoint_protocol": config["checkpoint_protocol"],
        "prompt": config["prompt"],
        "split_fingerprint": manifest["fingerprint"],
        "model_metadata_fingerprint": model["metadata_fingerprint"],
    }
    return canonical_fingerprint(protected)


def artifact_matches(path: Path, *, problem_id: str, fingerprint: str) -> bool:
    """失败即中止：只断点复用来自完全相同协议的完整产物。"""
    if not path.is_file():
        return False
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return (
        artifact.get("status") == "complete"
        and str(artifact.get("problem_id")) == str(problem_id)
        and artifact.get("protocol_fingerprint") == fingerprint
    )


def raw_semantic_boundaries(
    tokenizer,
    token_ids: list[int],
    upper: int,
) -> tuple[list[int], str]:
    """将英文换行和标点边界映射回准确的令牌索引。"""
    limited = token_ids[:upper]
    text = tokenizer.decode(
        limited,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if list(encoded.input_ids) == limited:
        token_ends = [int(end) for _start, end in encoded.offset_mapping]
    else:
        # 极少数情况下，分词器往返转换不一致。前缀解码速度较慢，
        # 但能够保留到原始生成令牌序列的映射。
        token_ends = []
        for end in range(1, len(limited) + 1):
            prefix = tokenizer.decode(
                limited[:end],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            token_ends.append(len(prefix))
    checkpoints: set[int] = set()
    for match in BOUNDARY.finditer(text):
        position = bisect.bisect_left(token_ends, match.end())
        if position < len(token_ends):
            checkpoints.add(position + 1)
    return sorted(checkpoints), text


def schedules_for_trace(
    tokenizer,
    content_ids: list[int],
    *,
    minimum: int,
    maximum: int,
    sentence_gap: int,
    fixed: Iterable[int],
) -> tuple[dict[str, list[int]], str]:
    """只构造预注册的句子级与固定位置检查计划。"""
    upper = min(int(maximum), len(content_ids))
    semantic, decoded = (
        raw_semantic_boundaries(tokenizer, content_ids, upper)
        if upper
        else ([], "")
    )
    sentence: list[int] = []
    previous = 0
    for checkpoint in semantic:
        if minimum <= checkpoint <= upper and checkpoint - previous >= sentence_gap:
            sentence.append(checkpoint)
            previous = checkpoint
    fixed_values = [
        int(value) for value in fixed if minimum <= int(value) <= upper
    ]
    return {"sentence": sentence, "fixed": fixed_values}, decoded


def tail_mean(values: list[float], end: int, width: int = 8) -> float:
    local = values[max(0, end - width):end]
    return float(sum(local) / len(local)) if local else float("nan")


def cache_paths(cache_root: Path, split: str, problem_id: str) -> dict[str, Path]:
    return {
        "dense": cache_root / "dense" / split / f"sample_{problem_id}.pt",
        "branches": cache_root / "branches" / split / problem_id,
        "merged": cache_root / "merged" / split / f"sample_{problem_id}.pt",
    }


def branch_path(
    cache_root: Path,
    split: str,
    problem_id: str,
    checkpoint: int,
) -> Path:
    name = (
        "direct.pt"
        if checkpoint == BRANCH_DIRECT
        else f"checkpoint_{checkpoint:04d}.pt"
    )
    return cache_root / "branches" / split / problem_id / name

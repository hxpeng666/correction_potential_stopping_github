"""MMLU-Pro 的动态选项、严格答案解析与五样例提示协议。"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any


CHOICE_LETTERS = tuple("ABCDEFGHIJ")


def valid_letters(option_count: int) -> tuple[str, ...]:
    if option_count < 2 or option_count > len(CHOICE_LETTERS):
        raise ValueError(f"MMLU-Pro 选项数必须在 2 到 10 之间，实际为 {option_count}")
    return CHOICE_LETTERS[:option_count]


def answer_letter(answer: int | str, option_count: int) -> str:
    allowed = valid_letters(option_count)
    if isinstance(answer, str):
        value = answer.strip().upper()
        if value in allowed:
            return value
        if value in CHOICE_LETTERS:
            raise ValueError(f"答案 {value} 超出当前 {option_count} 个选项")
        answer = int(value)
    index = int(answer)
    if index < 0 or index >= option_count:
        raise ValueError(f"答案索引 {index} 超出当前 {option_count} 个选项")
    return allowed[index]


def parse_answer(text: str | None, option_count: int) -> str | None:
    """只解析最后一个显式 boxed 或 Final answer，并限制在当前有效选项内。"""
    if not text:
        return None
    allowed = set(valid_letters(option_count))
    boxed = re.findall(r"\\boxed\s*\{\s*([A-Ja-j])\s*\}", text)
    if boxed:
        value = boxed[-1].upper()
        return value if value in allowed else None
    final = re.findall(r"Final\s+answer\s*[:=]\s*\(?\s*([A-Ja-j])\s*\)?", text, re.I)
    if final:
        value = final[-1].upper()
        return value if value in allowed else None
    return None


def format_item(
    question: str,
    options: Sequence[str],
    answer: int | str | None = None,
) -> str:
    letters = valid_letters(len(options))
    lines = [str(question).strip()]
    lines.extend(f"{letter}. {option}" for letter, option in zip(letters, options))
    lines.append(
        "Answer:"
        if answer is None
        else f"Answer: {answer_letter(answer, len(options))}"
    )
    return "\n".join(lines)


def build_five_shot_prompt(
    category: str,
    demonstrations: Sequence[dict[str, Any]],
    question: str,
    options: Sequence[str],
) -> str:
    if len(demonstrations) != 5:
        raise ValueError(f"MMLU-Pro 每类必须恰好使用 5 条 validation 演示，实际为 {len(demonstrations)}")
    sections = [
        f"The following are multiple choice questions (with answers) about {category}."
    ]
    for row in demonstrations:
        sections.append(format_item(row["question"], row["choices"], row["answer"]))
    sections.append(format_item(question, options, None))
    return "\n\n".join(sections)


def demonstrations_by_category(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(str(row["category"]), []).append(dict(row))
    invalid = {category: len(values) for category, values in result.items() if len(values) != 5}
    if invalid:
        raise ValueError(f"MMLU-Pro 五样例数量错误：{invalid}")
    return result

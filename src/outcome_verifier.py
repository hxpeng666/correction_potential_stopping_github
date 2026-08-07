from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


NUMBER = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/[1-9]\d*)?")


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().replace("$", "").replace(",", "")
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            number = Decimal(numerator) / Decimal(denominator)
        else:
            number = Decimal(text)
        normalized = format(number.normalize(), "f")
        return "0" if normalized in {"-0", "+0"} else normalized
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def gold_answer(solution: str) -> str | None:
    marker = solution.rsplit("####", 1)[-1]
    matches = NUMBER.findall(marker)
    return normalize_number(matches[-1]) if matches else None


def predicted_answer(generation: str) -> str | None:
    patterns = [
        r"(?is)final\s+answer\s*(?:is|:)?\s*([^\n]+)",
        r"(?is)answer\s*(?:is|:|=)\s*([^\n]+)",
        r"\\boxed\{([^}]+)\}",
        r"####\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, generation)
        if matches:
            numbers = NUMBER.findall(matches[-1])
            if numbers:
                return normalize_number(numbers[-1])
    matches = NUMBER.findall(generation)
    return normalize_number(matches[-1]) if matches else None


def exact_success(gold: str | None, prediction: str | None) -> bool:
    return gold is not None and prediction is not None and gold == prediction


def input_features(question: str, prompt_tokens: int) -> dict[str, float]:
    lower = question.lower()
    return {
        "prompt_tokens": float(prompt_tokens),
        "question_chars": float(len(question)),
        "number_count": float(len(NUMBER.findall(question))),
        "sentence_count": float(max(1, len(re.findall(r"[.!?]", question)))),
        "operation_word_count": float(sum(lower.count(word) for word in
                                           ("each", "total", "left", "more", "less", "times", "per", "percent"))),
        "question_mark_count": float(question.count("?")),
    }

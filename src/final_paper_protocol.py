"""Final-paper protocol constants, prompts, parsers, and checkpoint schedules."""
from __future__ import annotations

import bisect
import hashlib
import json
import re
from typing import Any, Iterable, Sequence

MMLU_SUBJECTS = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics", "econometrics",
    "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history",
    "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology",
    "world_religions",
)

MMLU_CATEGORIES = {
    "STEM": {
        "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning",
    },
    "Humanities": {
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    },
    "Social Sciences": {
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology", "human_sexuality",
        "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy",
    },
    "Other": {
        "anatomy", "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology",
    },
}

CHOICE_LETTERS = ("A", "B", "C", "D")
BOXED_MCQ = re.compile(r"\\boxed\s*\{\s*([ABCD])\s*\}", re.IGNORECASE)
FINAL_MCQ = re.compile(
    r"(?im)\bfinal\s+answer\s*(?::|is\b)\s*(?:option\s*)?([ABCD])\b"
)
BOUNDARY = re.compile(r"\n+|[.!?;]+(?:[\"')\]]*)?(?=\s|$)")


def parse_mcq_answer(text: str | None) -> str | None:
    """Parse only an explicit boxed answer or explicit Final answer declaration."""
    if not text:
        return None
    boxed = BOXED_MCQ.findall(text)
    if boxed:
        return boxed[-1].upper()
    final = FINAL_MCQ.findall(text)
    return final[-1].upper() if final else None


def normalize_question(text: str) -> str:
    return " ".join(str(text).casefold().split())


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def answer_letter(answer: int | str) -> str:
    if isinstance(answer, str):
        value = answer.strip().upper()
        if value in CHOICE_LETTERS:
            return value
        answer = int(value)
    index = int(answer)
    if index < 0 or index >= len(CHOICE_LETTERS):
        raise ValueError(f"invalid MMLU answer index: {answer}")
    return CHOICE_LETTERS[index]


def mmlu_category(subject: str) -> str:
    matches = [name for name, subjects in MMLU_CATEGORIES.items() if subject in subjects]
    if len(matches) != 1:
        raise KeyError(f"subject has invalid category membership: {subject} -> {matches}")
    return matches[0]


def format_mmlu_item(
    question: str, choices: Sequence[str], answer: str | None = None
) -> str:
    if len(choices) != 4:
        raise ValueError(f"MMLU item must have four choices, found {len(choices)}")
    lines = [str(question).strip()]
    lines.extend(f"{letter}. {choice}" for letter, choice in zip(CHOICE_LETTERS, choices))
    lines.append("Answer:" if answer is None else f"Answer: {answer}")
    return "\n".join(lines)


def build_mmlu_five_shot_prompt(
    subject: str,
    demonstrations: Sequence[dict[str, Any]],
    question: str,
    choices: Sequence[str],
) -> str:
    if len(demonstrations) != 5:
        raise ValueError(f"standard MMLU prompt requires exactly five demos, found {len(demonstrations)}")
    readable = subject.replace("_", " ")
    sections = [
        f"The following are multiple choice questions (with answers) about {readable}."
    ]
    for row in demonstrations:
        sections.append(
            format_mmlu_item(
                row["question"], row["choices"], answer_letter(row["answer"])
            )
        )
    sections.append(format_mmlu_item(question, choices, None))
    return "\n\n".join(sections)


def semantic_boundaries(text: str, offsets: Sequence[tuple[int, int]]) -> list[int]:
    token_ends = [int(end) for _start, end in offsets]
    result: set[int] = set()
    for match in BOUNDARY.finditer(text):
        position = bisect.bisect_left(token_ends, match.end())
        if position < len(token_ends):
            result.add(position + 1)
    return sorted(result)


def checkpoint_schedules(
    semantic: Iterable[int],
    content_tokens: int,
    *,
    minimum: int = 64,
    maximum: int = 768,
    sentence_gap: int = 8,
    hybrid_minimum_gap: int = 32,
    hybrid_maximum_gap: int = 128,
    fixed: Sequence[int] = (64, 96, 128, 256, 512, 768),
) -> dict[str, list[int]]:
    upper = min(int(maximum), int(content_tokens))
    fixed_values = [int(x) for x in fixed if minimum <= int(x) <= upper]
    semantic_values = sorted(set(int(x) for x in semantic))
    sentence: list[int] = []
    last = 0
    for checkpoint in semantic_values:
        if minimum <= checkpoint <= upper and checkpoint - last >= sentence_gap:
            sentence.append(checkpoint)
            last = checkpoint
    hybrid: list[int] = []
    semantic_set = set(semantic_values)
    last = 0
    for checkpoint in range(minimum, upper + 1):
        if checkpoint - last < hybrid_minimum_gap:
            continue
        if checkpoint in semantic_set or checkpoint - last >= hybrid_maximum_gap:
            hybrid.append(checkpoint)
            last = checkpoint
    return {"fixed": fixed_values, "sentence": sentence, "hybrid": hybrid}

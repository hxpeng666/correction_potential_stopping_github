#!/usr/bin/env python3
"""MMLU-Pro 独立协议单元测试。"""
from src.mmlu_pro_protocol import answer_letter, format_item, parse_answer, valid_letters


def main() -> None:
    assert valid_letters(4) == ("A", "B", "C", "D")
    assert valid_letters(10)[-1] == "J"
    assert answer_letter(8, 9) == "I"
    assert answer_letter("j", 10) == "J"
    assert parse_answer(r"analysis A; \boxed{B}; later \boxed{I}", 9) == "I"
    assert parse_answer("Final answer: (J)", 10) == "J"
    assert parse_answer(r"\boxed{J}", 9) is None
    assert parse_answer("Reasoning mentions A and ends with D", 4) is None
    assert parse_answer(None, 4) is None
    assert parse_answer("", 4) is None
    formatted = format_item("Q?", ["x", "y", "z", "w"], "C")
    assert formatted.endswith("Answer: C") and "D. w" in formatted
    try:
        valid_letters(11)
    except ValueError:
        pass
    else:
        raise AssertionError("11 个选项必须被拒绝")
    print("MMLU-Pro 协议单元测试：全部通过")


if __name__ == "__main__":
    main()

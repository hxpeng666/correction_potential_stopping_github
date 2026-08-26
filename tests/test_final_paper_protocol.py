from __future__ import annotations

import unittest

from src.final_paper_protocol import (
    MMLU_SUBJECTS,
    build_mmlu_five_shot_prompt,
    checkpoint_schedules,
    mmlu_category,
    parse_mcq_answer,
)


class McqParserTest(unittest.TestCase):
    def test_last_boxed_has_priority(self):
        self.assertEqual(parse_mcq_answer(r"First \\boxed{A}; corrected \\boxed{ c }"), "C")

    def test_final_answer_fallback(self):
        self.assertEqual(parse_mcq_answer("Reasoning mentions A and B.\nFinal answer: d"), "D")
        self.assertEqual(parse_mcq_answer("Final answer is option B"), "B")

    def test_does_not_take_arbitrary_last_letter(self):
        self.assertIsNone(parse_mcq_answer("A seems plausible, but I considered B and C."))
        self.assertIsNone(parse_mcq_answer("Answer: A"))
        self.assertIsNone(parse_mcq_answer(r"\\boxed{E}"))

    def test_missing(self):
        self.assertIsNone(parse_mcq_answer(None))
        self.assertIsNone(parse_mcq_answer(""))


class ProtocolTest(unittest.TestCase):
    def test_all_subjects_have_one_category(self):
        self.assertEqual(len(MMLU_SUBJECTS), 57)
        self.assertEqual({mmlu_category(subject) for subject in MMLU_SUBJECTS},
                         {"STEM", "Humanities", "Social Sciences", "Other"})

    def test_prompt_is_five_shot(self):
        demos = [
            {"question": f"q{i}", "choices": ["x", "y", "z", "w"], "answer": i % 4}
            for i in range(5)
        ]
        prompt = build_mmlu_five_shot_prompt(
            "abstract_algebra", demos, "target", ["a", "b", "c", "d"]
        )
        self.assertEqual(prompt.count("Answer:"), 6)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_checkpoint_schedules(self):
        values = checkpoint_schedules([63, 64, 70, 72, 80, 769], 900)
        self.assertEqual(values["fixed"], [64, 96, 128, 256, 512, 768])
        self.assertEqual(values["sentence"], [64, 72, 80])
        self.assertTrue(all(64 <= x <= 768 for x in values["hybrid"]))


if __name__ == "__main__":
    unittest.main()

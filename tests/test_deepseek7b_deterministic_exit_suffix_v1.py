from __future__ import annotations

import numpy as np

from scripts.analyze_deepseek7b_deterministic_exit_suffix_v1 import exit_summary
from scripts.prepare_deepseek7b_deterministic_exit_suffix_v1 import select_checkpoints
from scripts.recalibrate_deepseek7b_method_exploration_ltt_v1 import ReplayData


def test_select_checkpoints_is_label_blind_and_spans_trajectory() -> None:
    rows = [{"checkpoint": value, "current_success": value % 2 == 0} for value in range(10, 110, 10)]
    assert select_checkpoints(rows, 6) == [10, 30, 50, 60, 80, 100]


def test_exit_summary_four_states() -> None:
    data = ReplayData(
        problem_ids=["a", "b", "c", "d"],
        row_problem_ids=["a", "a", "b", "b", "c", "c", "d", "d"],
        row_checkpoints=np.asarray([10, 20] * 4),
        row_current_success=np.asarray([True, True, True, True, False, False, False, False]),
        groups=[(0, 2, True, 40), (2, 4, False, 40), (4, 6, True, 40), (6, 8, False, 40)],
        fallbacks=[],
    )
    scores = np.asarray([0.1, 0.9] * 4)
    result = exit_summary(data, scores, threshold=0.2, sentinel=False)
    assert result["exits"] == 4
    assert result["transitions"]["C_to_C"]["count"] == 1
    assert result["transitions"]["C_to_W"]["count"] == 1
    assert result["transitions"]["W_to_C"]["count"] == 1
    assert result["transitions"]["W_to_W"]["count"] == 1


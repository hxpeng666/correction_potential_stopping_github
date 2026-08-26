from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.final_paper_replay_cache import (
    claim_task,
    ensure_task,
    finish_task,
    queue_counts,
    task_seed,
)


class ReplayCacheTests(unittest.TestCase):
    def test_task_seed_uses_every_key_component(self):
        base = task_seed(20260803, "gsm8k", "probe_train", "sample", 64)
        variants = {
            task_seed(20260804, "gsm8k", "probe_train", "sample", 64),
            task_seed(20260803, "mmlu", "probe_train", "sample", 64),
            task_seed(20260803, "gsm8k", "calibration", "sample", 64),
            task_seed(20260803, "gsm8k", "probe_train", "other", 64),
            task_seed(20260803, "gsm8k", "probe_train", "sample", 96),
        }
        self.assertNotIn(base, variants)
        self.assertEqual(len(variants), 5)
        self.assertEqual(base, task_seed(20260803, "gsm8k", "probe_train", "sample", 64))

    def test_filesystem_queue_claim_and_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = {
                "kind": "dense",
                "dataset": "gsm8k",
                "split": "probe_train",
                "problem_id": "x",
                "checkpoint": "dense",
            }
            self.assertTrue(ensure_task(root, payload))
            self.assertFalse(ensure_task(root, payload))
            self.assertEqual(queue_counts(root, "dense")["pending"], 1)
            claimed = claim_task(root, "dense", "worker")
            self.assertIsNotNone(claimed)
            task, path = claimed
            self.assertEqual(task["problem_id"], "x")
            self.assertEqual(queue_counts(root, "dense")["claimed"], 1)
            finish_task(path, task, root, "done")
            counts = queue_counts(root, "dense")
            self.assertEqual(counts["pending"], 0)
            self.assertEqual(counts["claimed"], 0)
            self.assertEqual(counts["done"], 1)

    def test_successful_branch_uses_artifact_as_completion_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = {
                "kind": "branch",
                "dataset": "mmlu",
                "split": "heldout",
                "problem_id": "x",
                "checkpoint": 64,
            }
            self.assertTrue(ensure_task(root, payload))
            task, path = claim_task(root, "branch", "worker")
            finish_task(path, task, root, "done")
            counts = queue_counts(root, "branch")
            self.assertEqual(counts["pending"], 0)
            self.assertEqual(counts["claimed"], 0)
            self.assertEqual(counts["done"], 0)


if __name__ == "__main__":
    unittest.main()

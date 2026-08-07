import tempfile
import unittest
from pathlib import Path

import torch

from src.final_paper_cache import artifact_matches, branch_path, task_seed


class CacheContractTests(unittest.TestCase):
    def test_seed_depends_on_every_key_field(self):
        base = (20260803, "gsm8k", "calibration", "sample-1", 64)
        reference = task_seed(*base)
        variants = [
            (20260804, *base[1:]),
            (base[0], "mmlu", *base[2:]),
            (*base[:2], "heldout", *base[3:]),
            (*base[:3], "sample-2", base[4]),
            (*base[:4], 96),
        ]
        self.assertTrue(all(task_seed(*value) != reference for value in variants))
        self.assertEqual(reference, task_seed(*base))

    def test_resume_is_fingerprint_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pt"
            torch.save(
                {
                    "status": "complete",
                    "problem_id": "x",
                    "protocol_fingerprint": "abc",
                },
                path,
            )
            self.assertTrue(
                artifact_matches(path, problem_id="x", fingerprint="abc")
            )
            self.assertFalse(
                artifact_matches(path, problem_id="x", fingerprint="other")
            )

    def test_direct_and_checkpoint_paths_do_not_collide(self):
        root = Path("cache")
        direct = branch_path(root, "heldout", "x", -1)
        checkpoint = branch_path(root, "heldout", "x", 64)
        self.assertNotEqual(direct, checkpoint)


if __name__ == "__main__":
    unittest.main()

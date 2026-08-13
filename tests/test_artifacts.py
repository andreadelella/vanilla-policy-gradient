import tempfile
import unittest
from pathlib import Path

import numpy as np

from vpg.artifacts import load_config, save_training_rewards


class ArtifactTests(unittest.TestCase):
    def test_reward_archive_keeps_seed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_training_rewards(
                directory,
                [[1.0, 2.0], [3.0, 4.0]],
                [23, 24],
            )
            with np.load(path) as archive:
                np.testing.assert_array_equal(archive["seeds"], [23, 24])
                self.assertEqual(archive["rewards"].shape, (2, 2))

    def test_reward_archive_rejects_missing_seed_labels(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            save_training_rewards(directory, [[1.0], [2.0]], [23])

    def test_load_config_reads_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "config.json")
            path.write_text('{"gamma": 0.99}', encoding="utf-8")
            self.assertEqual(load_config(path), {"gamma": 0.99})


if __name__ == "__main__":
    unittest.main()

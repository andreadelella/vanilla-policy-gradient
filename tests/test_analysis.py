import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import (
    load_reward_file,
    load_reward_files,
    main,
    resolve_reward_path,
)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def save_npz(self, name, rewards, seeds):
        path = self.root / name
        np.savez(path, rewards=np.asarray(rewards), seeds=np.asarray(seeds))
        return path

    def test_directory_resolution_prefers_npz(self):
        np.save(self.root / "training_rewards.npy", np.arange(4))
        expected = self.save_npz("training_rewards.npz", [[1, 2]], [23])
        self.assertEqual(resolve_reward_path(self.root), expected.resolve())

    def test_loading_promotes_one_dimensional_rewards(self):
        path = self.root / "training_rewards.npy"
        np.save(path, np.array([1.0, 2.0, 3.0]))
        data = load_reward_file(path)
        self.assertEqual(data.curves.shape, (1, 3))
        self.assertEqual(data.labels, ["training_rewards"])

    def test_selected_files_are_combined(self):
        first = self.save_npz("seed23.npz", [[1, 2, 3]], [23])
        second = self.save_npz("seed24.npz", [[4, 5, 6]], [24])
        data = load_reward_files([first, second])
        np.testing.assert_array_equal(data.seeds, [23, 24])
        self.assertEqual(data.curves.shape, (2, 3))

    def test_mismatched_lengths_are_rejected(self):
        first = self.save_npz("short.npz", [[1, 2]], [1])
        second = self.save_npz("long.npz", [[1, 2, 3]], [2])
        with self.assertRaisesRegex(ValueError, "equal length"):
            load_reward_files([first, second])

    def test_all_cli_modes_create_output(self):
        first = self.save_npz("first.npz", [[1, 2, 3], [2, 3, 4]], [1, 2])
        second = self.save_npz("second.npz", [[3, 2, 1]], [3])

        commands = (
            ["single", str(first), "--output", str(self.root / "single.png")],
            [
                "ci",
                str(first),
                str(second),
                "--output",
                str(self.root / "ci.png"),
                "--title",
                "Selected seeds",
            ],
            [
                "compare",
                "--run",
                "first",
                str(first),
                "--run",
                "second",
                str(second),
                "--output",
                str(self.root / "compare.png"),
                "--title",
                "Different runs",
            ],
        )
        for command in commands:
            with self.subTest(command=command[0]):
                self.assertEqual(main(command), 0)
                self.assertTrue(Path(command[command.index("--output") + 1]).is_file())


if __name__ == "__main__":
    unittest.main()

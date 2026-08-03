import tempfile
import unittest
from pathlib import Path

import numpy as np

from vpg.analysis import (
    discover_seed_files,
    load_reward_file,
    load_reward_files,
    main,
    resolve_reward_inputs,
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

    def make_multiseed_run(self, seeds, n_iters=3, nested_only=()):
        """Write a multiseed run: per-seed files plus a partial aggregate.

        Mirrors train.py's layout. `nested_only` seeds get just the seed<N>/
        copy, as happens when a seed is interrupted before completing.
        """
        for offset, seed in enumerate(seeds):
            curve = [[float(seed) + i for i in range(n_iters)]]
            if seed not in nested_only:
                self.save_npz(f"training_rewards_seed{seed}.npz", curve, [seed])
            nested = self.root / f"seed{seed}"
            nested.mkdir(exist_ok=True)
            np.savez(
                nested / "training_rewards.npz",
                rewards=np.asarray(curve),
                seeds=np.asarray([seed]),
            )
        # Aggregate covering only the final seed, as a resumed run leaves behind.
        last = seeds[-1]
        self.save_npz(
            "training_rewards.npz",
            [[float(last) + i for i in range(n_iters)]],
            [last],
        )

    def test_directory_loads_every_seed_not_just_the_aggregate(self):
        self.make_multiseed_run([23, 24, 25])
        data = load_reward_files([self.root])
        np.testing.assert_array_equal(data.seeds, [23, 24, 25])
        self.assertEqual(data.curves.shape, (3, 3))

    def test_completed_per_seed_file_wins_over_nested_copy(self):
        self.make_multiseed_run([23])
        # The nested copy is written periodically and may be stale/truncated,
        # so the top-level completed file must be the one that is loaded.
        resolved = resolve_reward_inputs(self.root)
        self.assertEqual(
            [p.name for p in resolved], ["training_rewards_seed23.npz"]
        )

    def test_interrupted_seed_falls_back_to_nested_copy(self):
        self.make_multiseed_run([23, 24], nested_only=(24,))
        seed_files = discover_seed_files(self.root)
        self.assertEqual(
            [p.parent.name + "/" + p.name for p in seed_files],
            [f"{self.root.name}/training_rewards_seed23.npz", "seed24/training_rewards.npz"],
        )

    def test_aggregate_is_used_when_no_per_seed_files_exist(self):
        self.save_npz("training_rewards.npz", [[1, 2], [3, 4]], [7, 8])
        data = load_reward_files([self.root])
        np.testing.assert_array_equal(data.seeds, [7, 8])

    def test_duplicate_inputs_are_loaded_once(self):
        self.make_multiseed_run([23, 24])
        explicit = self.root / "training_rewards_seed23.npz"
        data = load_reward_files([self.root, explicit])
        np.testing.assert_array_equal(data.seeds, [23, 24])

    def test_truncate_clips_unequal_seed_lengths(self):
        self.save_npz("training_rewards_seed23.npz", [[1, 2, 3, 4]], [23])
        self.save_npz("training_rewards_seed24.npz", [[5, 6]], [24])
        data = load_reward_files([self.root], truncate=True)
        self.assertEqual(data.curves.shape, (2, 2))
        np.testing.assert_array_equal(data.curves, [[1, 2], [5, 6]])

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

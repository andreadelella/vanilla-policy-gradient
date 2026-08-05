"""Tests for Step 2 stochastic categorical-bandit training."""

import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from exploration.categorical_bandit.algorithms import (
    AlgorithmSpec,
    apply_update,
    exact_expected_direction,
    policy_log_probabilities,
    raw_update_direction,
    recenter_logits,
)
from exploration.categorical_bandit.environment import BanditBatch, generate_paired_bandits
from exploration.categorical_bandit.experiment import TrainingConfig, run_training_unit
from exploration.categorical_bandit.identity import analytic_barrier_gradient
from exploration.categorical_bandit.reporting import curve_mean_ci


class BanditGenerationTests(unittest.TestCase):
    def test_prescribed_unique_optimum_and_gap(self):
        bandits = generate_paired_bandits(20, 10, 23)
        sorted_means = torch.sort(bandits.mean_rewards, dim=1, descending=True).values
        torch.testing.assert_close(sorted_means[:, 0], torch.ones(20, dtype=torch.float64))
        torch.testing.assert_close(
            sorted_means[:, 1], torch.full((20,), 0.9, dtype=torch.float64)
        )
        torch.testing.assert_close(
            sorted_means[:, 0] - sorted_means[:, 1],
            torch.full((20,), 0.1, dtype=torch.float64),
        )
        self.assertTrue(bool(torch.all(sorted_means[:, 2] < 0.9)))
        self.assertTrue(torch.equal(torch.argmax(bandits.mean_rewards, dim=1), bandits.optimal_actions))

    def test_identical_seed_reuses_paired_means(self):
        first = generate_paired_bandits(4, 10, 77)
        second = generate_paired_bandits(4, 10, 77)
        self.assertTrue(torch.equal(first.mean_rewards, second.mean_rewards))
        self.assertTrue(torch.equal(first.optimal_actions, second.optimal_actions))


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64)
        self.action = torch.tensor([1], dtype=torch.long)
        self.reward = torch.tensor([1.7], dtype=torch.float64)
        self.alpha = 0.05

    def _score(self):
        probability = torch.softmax(self.logits, dim=1)
        score = -probability
        score[0, 1] += 1.0
        return probability, score

    def test_every_one_step_formula(self):
        probability, score = self._score()
        selected_log_probability = torch.log_softmax(self.logits, dim=1)[0, 1]
        cases = [
            (
                AlgorithmSpec("sgb", "sgb", self.alpha),
                self.reward[:, None] * score,
            ),
            (
                AlgorithmSpec(
                    "entropy", "entropy_sgb", self.alpha, entropy_coefficient=0.2
                ),
                (self.reward - 0.2 * selected_log_probability)[:, None] * score,
            ),
            (
                AlgorithmSpec("npg", "npg", self.alpha),
                torch.tensor(
                    [[0.0, self.reward.item() / probability[0, 1].item(), 0.0]],
                    dtype=torch.float64,
                ),
            ),
            (
                AlgorithmSpec("lb", "lb_sgb", self.alpha, eta=100.0),
                self.reward[:, None] * score
                + 0.01 * (torch.ones_like(probability) - 3.0 * probability),
            ),
        ]
        for spec, expected_direction in cases:
            with self.subTest(spec=spec.kind):
                torch.testing.assert_close(
                    raw_update_direction(self.logits, self.action, self.reward, spec),
                    expected_direction,
                    atol=1e-14,
                    rtol=1e-14,
                )
                torch.testing.assert_close(
                    apply_update(self.logits, self.action, self.reward, spec),
                    recenter_logits(self.logits + self.alpha * expected_direction),
                    atol=1e-14,
                    rtol=1e-14,
                )

    def test_zero_barrier_coefficient_is_exact_sgb(self):
        sgb = AlgorithmSpec("sgb", "sgb", self.alpha)
        lb = AlgorithmSpec("lb", "lb_sgb", self.alpha, eta=math.inf)
        self.assertTrue(
            torch.equal(
                apply_update(self.logits, self.action, self.reward, sgb),
                apply_update(self.logits, self.action, self.reward, lb),
            )
        )

    def test_barrier_direction_matches_step_one_oracle(self):
        lb = AlgorithmSpec("lb", "lb_sgb", self.alpha, eta=7.0)
        zero_reward = torch.zeros_like(self.reward)
        direction = raw_update_direction(self.logits, self.action, zero_reward, lb)
        expected = analytic_barrier_gradient(self.logits[0]) / 7.0
        torch.testing.assert_close(direction[0], expected, atol=1e-14, rtol=1e-14)

    def test_exact_expected_sgb_and_lb_gradients(self):
        means = torch.tensor([[1.0, 0.5, -0.2]], dtype=torch.float64)
        probability = torch.softmax(self.logits, dim=1)
        objective = (probability * means).sum(dim=1, keepdim=True)
        expected_sgb = probability * (means - objective)
        sgb = AlgorithmSpec("sgb", "sgb", self.alpha)
        torch.testing.assert_close(
            exact_expected_direction(self.logits, means, sgb),
            expected_sgb,
            atol=1e-14,
            rtol=1e-14,
        )
        lb = AlgorithmSpec("lb", "lb_sgb", self.alpha, eta=11.0)
        expected_lb = expected_sgb + (1.0 / 11.0) * (1.0 - 3.0 * probability)
        torch.testing.assert_close(
            exact_expected_direction(self.logits, means, lb),
            expected_lb,
            atol=1e-14,
            rtol=1e-14,
        )

    def test_gauge_recentering_preserves_probabilities(self):
        shifted = self.logits + 123.0
        torch.testing.assert_close(
            policy_log_probabilities(shifted).exp(),
            policy_log_probabilities(recenter_logits(shifted)).exp(),
            atol=1e-14,
            rtol=1e-14,
        )


class RunnerTests(unittest.TestCase):
    @staticmethod
    def _config(algorithm, *, runs=3, horizon=4, interval=2, reward_std=1.0):
        return TrainingConfig(
            preset="test",
            num_actions=3,
            num_runs=runs,
            horizon=horizon,
            record_interval=interval,
            reward_std=reward_std,
            collapse_threshold=1e-12,
            seed=1234,
            algorithm=algorithm,
        )

    def test_initial_metrics_shapes_and_ci_axis(self):
        bandits = generate_paired_bandits(3, 3, 12)
        config = self._config(AlgorithmSpec("sgb", "sgb", 0.01))
        result = run_training_unit(config, bandits)
        self.assertEqual(result.steps.tolist(), [0, 2, 4])
        for values in result.metrics.values():
            self.assertEqual(values.shape, (3, 3))
        np.testing.assert_allclose(result.metrics["minimum_probability"][:, 0], 1.0 / 3.0)
        np.testing.assert_allclose(result.metrics["normalized_log_fisher_volume"][:, 0], 0.0, atol=1e-15)
        np.testing.assert_allclose(result.metrics["entropy"][:, 0], math.log(3.0))
        mean, lower, upper, counts = curve_mean_ci(result.metrics["minimum_probability"])
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(counts[0], 3)
        self.assertAlmostEqual(lower[0], upper[0])

    def test_deterministic_replay(self):
        bandits = generate_paired_bandits(3, 3, 12)
        config = self._config(AlgorithmSpec("lb", "lb_sgb", 0.01, eta=1000.0))
        first = run_training_unit(config, bandits)
        second = run_training_unit(config, bandits)
        for metric in first.metrics:
            np.testing.assert_array_equal(first.metrics[metric], second.metrics[metric])
        np.testing.assert_array_equal(first.final_probabilities, second.final_probabilities)

    def test_nonfinite_npg_is_recorded_not_clipped(self):
        bandits = BanditBatch(
            mean_rewards=torch.full((2, 3), 1e308, dtype=torch.float64),
            optimal_actions=torch.zeros(2, dtype=torch.long),
            second_best_actions=torch.ones(2, dtype=torch.long),
        )
        config = self._config(
            AlgorithmSpec("npg", "npg", 1.0), runs=2, horizon=1, interval=1, reward_std=0.0
        )
        result = run_training_unit(config, bandits)
        self.assertTrue(bool(result.failed.all()))
        np.testing.assert_array_equal(result.failure_steps, np.ones(2, dtype=np.int64))
        self.assertTrue(np.isnan(result.metrics["optimal_arm_probability"][:, 1]).all())
        self.assertTrue(np.isnan(result.final_probabilities).all())


class SmokeCliTests(unittest.TestCase):
    def test_smoke_artifacts_resume_and_incompatible_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "smoke"
            base = [
                sys.executable,
                "-m",
                "exploration.categorical_bandit.run_training",
                "--preset",
                "smoke",
                "--output-dir",
                str(output),
                "--device",
                "cpu",
            ]
            first = subprocess.run(base, capture_output=True, text=True, timeout=120)
            self.assertEqual(first.returncode, 0, msg=first.stderr + first.stdout)
            self.assertTrue((output / "config.json").exists())
            self.assertTrue((output / "paired_bandits_K0010.npz").exists())
            self.assertEqual(len(list(output.glob("K0010_alpha0p01__*.npz"))), 4)
            self.assertTrue((output / "run_status.csv").exists())
            self.assertTrue((output / "summary.csv").exists())
            self.assertTrue((output / "paired_final_differences.csv").exists())
            self.assertTrue((output / "K0010_alpha0p01" / "performance.png").exists())
            self.assertTrue((output / "K0010_alpha0p01" / "geometry.png").exists())
            self.assertTrue(
                (output / "K0010_alpha0p01" / "positive_fisher_eigenspectra.png").exists()
            )

            resumed = subprocess.run(base + ["--resume"], capture_output=True, text=True, timeout=120)
            self.assertEqual(resumed.returncode, 0, msg=resumed.stderr + resumed.stdout)
            self.assertIn("skipped", resumed.stdout)

            incompatible = subprocess.run(
                base + ["--resume", "--seed", "999"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(incompatible.returncode, 1)
            self.assertIn("incompatible configuration", incompatible.stderr)


if __name__ == "__main__":
    unittest.main()

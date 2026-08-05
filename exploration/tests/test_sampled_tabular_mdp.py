"""Tests for Step 4 finite-batch sampled tabular geometry."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from exploration.sampled_tabular_mdp.estimators import (
    exact_finite_batch_moments,
    sampled_conditional_gradient,
    sampled_empirical_fisher,
    sampled_reward_gradient,
)
from exploration.sampled_tabular_mdp.experiment import SampledTrainingConfig, train_sampled
from exploration.sampled_tabular_mdp.sampling import sample_batch
from exploration.sampled_tabular_mdp.verify import run_verification
from exploration.tabular_mdp.geometry import barrier_gradients, pooled_fisher
from exploration.tabular_mdp.model import TwoStepTrap, probabilities_from_reduced_logits


class SampledEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.phi = torch.tensor((0.4, -0.7, -0.2, 0.8), dtype=torch.float64)

    def test_complete_verification(self):
        result = run_verification()
        self.assertTrue(result.passed, msg={name: passed for name, passed in result.checks.items() if not passed})

    def test_batch_shapes_counts_and_replay(self):
        generator1 = torch.Generator().manual_seed(23)
        generator2 = torch.Generator().manual_seed(23)
        first = sample_batch(self.phi.expand(3, 4), 8, generator=generator1)
        second = sample_batch(self.phi.expand(3, 4), 8, generator=generator2)
        self.assertEqual(first.actions.shape, (3, 8, 2))
        self.assertTrue(torch.equal(first.actions, second.actions))
        torch.testing.assert_close(first.m, 8 + first.k1)
        self.assertTrue(torch.equal(first.mask[..., 1], first.actions[..., 0] == 1))

    def test_exact_ratio_moments_and_convergence(self):
        pi0, _ = probabilities_from_reduced_logits(self.phi)
        target = pi0[1] / (1 + pi0[1])
        small = exact_finite_batch_moments(self.phi, 4)
        large = exact_finite_batch_moments(self.phi, 4096)
        self.assertNotAlmostEqual(float(small.mu1_mean), float(target), places=10)
        self.assertLess(abs(float(large.mu1_mean - target)), abs(float(small.mu1_mean - target)))
        self.assertAlmostEqual(float(small.zero_s1_probability), float((1 - pi0[1]).pow(4)), places=14)

    def test_reward_and_fisher_population_targets(self):
        mdp = TwoStepTrap()
        moments = exact_finite_batch_moments(self.phi, 7)
        torch.testing.assert_close(moments.reward_mean, mdp.exact_reward_gradient(self.phi), atol=1e-14, rtol=1e-14)
        self.assertGreater(torch.linalg.matrix_norm(moments.fisher_mean - pooled_fisher(self.phi)).item(), 1e-8)

    def test_zero_state_batch_has_no_state1_protection_or_fisher(self):
        uniforms = torch.tensor(((0.01, 0.2), (0.99, 0.4), (0.02, 0.6)), dtype=torch.float64)
        batch = sample_batch(self.phi, 3, uniforms=uniforms)
        self.assertEqual(int(batch.k1), 0)
        gradient = sampled_conditional_gradient(self.phi, batch)
        fisher = sampled_empirical_fisher(self.phi, batch)
        torch.testing.assert_close(gradient[2:], torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(fisher[2:, 2:], torch.zeros((2, 2), dtype=torch.float64))
        self.assertEqual(float(torch.linalg.slogdet(fisher).sign), 0.0)

    def test_sample_validation(self):
        with self.assertRaises(ValueError):
            sample_batch(self.phi, 0)
        with self.assertRaises(ValueError):
            sample_batch(self.phi, 2, uniforms=torch.ones((2, 2), dtype=torch.float64))
        with self.assertRaises(ValueError):
            sample_batch(self.phi, 2, reward_noise_std=-1)


class SampledTrainingTests(unittest.TestCase):
    def setUp(self):
        self.phi = torch.tensor((0.4, -0.7, -0.2, 0.8), dtype=torch.float64)

    def test_shapes_finiteness_and_deterministic_replay(self):
        config = SampledTrainingConfig("detached_conditional_sampled", "adverse", 4, 5, updates=6, record_interval=2)
        first = train_sampled(config)
        second = train_sampled(config)
        self.assertEqual(first.phi.shape, (4, 5, 4))
        self.assertTrue(first.finite.all())
        np.testing.assert_array_equal(first.phi, second.phi)
        for key in first.metrics:
            np.testing.assert_array_equal(first.metrics[key], second.metrics[key])

    def test_zero_beta_reproduces_reward_only(self):
        reward = SampledTrainingConfig("reward_only", "adverse", 8, 4, beta=0, updates=8, record_interval=1)
        barrier = SampledTrainingConfig("detached_conditional_sampled", "adverse", 8, 4, beta=0, updates=8, record_interval=1)
        np.testing.assert_array_equal(train_sampled(reward).phi, train_sampled(barrier).phi)

    def test_oracle_and_sampled_targets_are_distinct(self):
        generator = torch.Generator().manual_seed(7)
        batch = sample_batch(self.phi.expand(16, 4), 4, generator=generator)
        sampled = sampled_conditional_gradient(self.phi.expand(16, 4), batch)
        oracle = barrier_gradients(self.phi.expand(16, 4)).detached_conditional
        self.assertGreater(torch.linalg.vector_norm(sampled - oracle, dim=-1).max().item(), 1e-8)

    def test_temporary_barrier_schedule_and_identical_prefix(self):
        fixed = SampledTrainingConfig(
            "detached_conditional_sampled", "adverse", 8, 4,
            beta=0.2, updates=4, record_interval=1,
        )
        handoff = SampledTrainingConfig(
            "detached_conditional_sampled", "adverse", 8, 4,
            beta=0.2, beta_after=0.0, handoff_update=2,
            updates=4, record_interval=1,
        )
        fixed_result = train_sampled(fixed)
        handoff_result = train_sampled(handoff)
        np.testing.assert_array_equal(fixed_result.phi[:3], handoff_result.phi[:3])
        np.testing.assert_array_equal(
            handoff_result.metrics["effective_beta"][1:, 0],
            np.asarray((0.2, 0.2, 0.0, 0.0)),
        )
        self.assertGreater(
            np.max(np.abs(fixed_result.phi[-1] - handoff_result.phi[-1])), 0.0
        )

    def test_schedule_validation(self):
        with self.assertRaises(ValueError):
            SampledTrainingConfig(
                "detached_conditional_sampled", "adverse", 8, 4,
                beta=0.2, beta_after=0.0, updates=4,
            )
        with self.assertRaises(ValueError):
            SampledTrainingConfig(
                "detached_conditional_sampled", "adverse", 8, 4,
                beta=0.2, beta_after=0.0, handoff_update=4, updates=4,
            )


class SampledCliTests(unittest.TestCase):
    def test_verify_and_smoke_artifacts_resume(self):
        verify = subprocess.run([sys.executable, "-m", "exploration.sampled_tabular_mdp.verify"], capture_output=True, text=True, timeout=120)
        self.assertEqual(verify.returncode, 0, msg=verify.stderr + verify.stdout)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "step4"
            command = [sys.executable, "-m", "exploration.sampled_tabular_mdp.run_experiment", "--preset", "smoke", "--output-dir", str(output)]
            first = subprocess.run(command, capture_output=True, text=True, timeout=240)
            self.assertEqual(first.returncode, 0, msg=first.stderr + first.stdout)
            for name in ("config.json", "verification.json", "audit.json", "audit.csv", "summary.csv", "paired_differences.csv", "run_status.csv"):
                self.assertTrue((output / name).exists(), name)
            self.assertTrue((output / "plots" / "audit" / "zero_s1.png").exists())
            self.assertTrue(any((output / "plots" / "training").rglob("performance_and_visitation.png")))
            resumed = subprocess.run(command + ["--resume"], capture_output=True, text=True, timeout=240)
            self.assertEqual(resumed.returncode, 0, msg=resumed.stderr + resumed.stdout)
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["preset"], "smoke")
            incompatible_command = [sys.executable, "-m", "exploration.sampled_tabular_mdp.run_experiment", "--preset", "pilot", "--output-dir", str(output), "--resume"]
            incompatible = subprocess.run(incompatible_command, capture_output=True, text=True, timeout=120)
            self.assertEqual(incompatible.returncode, 1)
            self.assertIn("incompatible configuration", incompatible.stderr)

    def test_handoff_smoke_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "handoff"
            command = [
                sys.executable, "-m", "exploration.sampled_tabular_mdp.run_handoff",
                "--preset", "smoke", "--output-dir", str(output),
            ]
            first = subprocess.run(command, capture_output=True, text=True, timeout=240)
            self.assertEqual(first.returncode, 0, msg=first.stderr + first.stdout)
            for name in (
                "config.json", "verification.json", "summary.csv",
                "paired_differences.csv", "checkpoint_summary.csv",
                "post_handoff_changes.csv", "handoff_pairwise.csv", "run_status.csv",
            ):
                self.assertTrue((output / name).exists(), name)
            self.assertTrue(
                (output / "plots" / "adverse" / "performance_and_handoff.png").exists()
            )
            resumed = subprocess.run(
                command + ["--resume"], capture_output=True, text=True, timeout=240
            )
            self.assertEqual(resumed.returncode, 0, msg=resumed.stderr + resumed.stdout)


if __name__ == "__main__":
    unittest.main()

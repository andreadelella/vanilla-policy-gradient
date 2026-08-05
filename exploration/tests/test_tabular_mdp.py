"""Tests for Step 3 exact two-state tabular geometry."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from exploration.tabular_mdp.experiment import ExactTrainingConfig, magnitude_matched_betas, train_exact
from exploration.tabular_mdp.geometry import barrier_gradients, barrier_terms, pooled_fisher
from exploration.tabular_mdp.model import TwoStepTrap, phi_from_q_and_good, probabilities_from_reduced_logits, transition_pool_weights
from exploration.tabular_mdp.verify import run_verification


class ExactModelTests(unittest.TestCase):
    def setUp(self):
        self.phi = torch.tensor((0.4, -0.7, -0.2, 0.8), dtype=torch.float64)

    def test_complete_verification(self):
        result = run_verification()
        self.assertTrue(result.passed, msg={key: value for key, value in result.checks.items() if not value})

    def test_transition_pool_convention_and_return(self):
        mdp = TwoStepTrap()
        pi0, _ = probabilities_from_reduced_logits(self.phi)
        mu0, mu1 = transition_pool_weights(self.phi)
        expected = torch.tensor((1.0, pi0[1]), dtype=torch.float64)
        expected /= expected.sum()
        torch.testing.assert_close(torch.stack((mu0, mu1)), expected, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(mdp.exact_return(self.phi), mdp.enumerated_return(self.phi), atol=1e-14, rtol=1e-14)

    def test_determinant_and_gradient_decompositions(self):
        terms = barrier_terms(self.phi)
        gradients = barrier_gradients(self.phi)
        sign, logdet = torch.linalg.slogdet(pooled_fisher(self.phi))
        self.assertEqual(sign.item(), 1.0)
        torch.testing.assert_close(0.5 * logdet, terms.full, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(terms.full, terms.uniform + terms.visit, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(
            gradients.complete_weighted,
            gradients.detached_conditional + gradients.weighted_state_term,
            atol=1e-14,
            rtol=1e-14,
        )
        torch.testing.assert_close(
            gradients.full_pooled_fisher,
            gradients.uniform_action + gradients.visitation_only,
            atol=1e-14,
            rtol=1e-14,
        )
        self.assertGreater(torch.linalg.vector_norm(gradients.weighted_state_term).item(), 1e-8)

    def test_grid_constructor(self):
        q = torch.tensor((0.02, 0.25, 0.9), dtype=torch.float64)
        good = torch.tensor((0.9, 0.5, 0.02), dtype=torch.float64)
        phi = phi_from_q_and_good(q, good)
        pi0, pi1 = probabilities_from_reduced_logits(phi)
        torch.testing.assert_close(pi0[:, 1], q, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(pi1[:, 0], good, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(pi0[:, 0], pi0[:, 2], atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(pi1[:, 1], pi1[:, 2], atol=1e-14, rtol=1e-14)


class ExactTrainingTests(unittest.TestCase):
    def test_deterministic_replay_shapes_and_initial_metrics(self):
        config = ExactTrainingConfig("full_pooled_fisher", 0.05, 0.1, 5, "test")
        first = train_exact(config, (0, 0, 0, 0))
        second = train_exact(config, (0, 0, 0, 0))
        self.assertEqual(first.phi.shape, (6, 1, 4))
        for key in first.metrics:
            np.testing.assert_array_equal(first.metrics[key], second.metrics[key])
        self.assertAlmostEqual(first.metrics["q"][0, 0], 1 / 3)
        self.assertTrue(first.finite.all())

    def test_magnitude_matching(self):
        initial = torch.tensor((2, -2, -2, 2), dtype=torch.float64)
        mdp = TwoStepTrap()
        target = torch.linalg.vector_norm(mdp.exact_reward_gradient(initial)).item()
        betas = magnitude_matched_betas(initial)
        gradients = barrier_gradients(initial)
        mapping = {
            "detached_conditional": gradients.detached_conditional,
            "complete_weighted": gradients.complete_weighted,
            "uniform_action": gradients.uniform_action,
            "visitation_only": gradients.visitation_only,
            "full_pooled_fisher": gradients.full_pooled_fisher,
        }
        for method, beta in betas.items():
            self.assertAlmostEqual(beta * torch.linalg.vector_norm(mapping[method]).item(), target, places=13)


class TabularCliTests(unittest.TestCase):
    def test_verify_cli_and_experiment_artifacts_resume(self):
        verify = subprocess.run([sys.executable, "-m", "exploration.tabular_mdp.verify"], capture_output=True, text=True, timeout=120)
        self.assertEqual(verify.returncode, 0, msg=verify.stderr + verify.stdout)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "step3"
            command = [sys.executable, "-m", "exploration.tabular_mdp.run_experiment", "--smoke", "--output-dir", str(output)]
            first = subprocess.run(command, capture_output=True, text=True, timeout=180)
            self.assertEqual(first.returncode, 0, msg=first.stderr + first.stdout)
            for name in ("config.json", "verification.json", "summary.csv", "run_status.csv", "magnitude_matched_betas.json"):
                self.assertTrue((output / name).exists())
            self.assertTrue((output / "plots" / "main" / "adverse_beta_0p1" / "gradient_decompositions.png").exists())
            self.assertTrue((output / "plots" / "basin" / "full_pooled_fisher.png").exists())
            resumed = subprocess.run(command + ["--resume"], capture_output=True, text=True, timeout=180)
            self.assertEqual(resumed.returncode, 0, msg=resumed.stderr + resumed.stdout)
            self.assertIn("Completed", resumed.stdout)
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["smoke"])
            incompatible = subprocess.run([sys.executable, "-m", "exploration.tabular_mdp.run_experiment", "--output-dir", str(output), "--resume"], capture_output=True, text=True, timeout=120)
            self.assertEqual(incompatible.returncode, 1)
            self.assertIn("incompatible configuration", incompatible.stderr)


if __name__ == "__main__":
    unittest.main()

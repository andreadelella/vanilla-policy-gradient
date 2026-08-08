"""Tests for the exact joint state-action Fisher validation stage."""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from exploration.joint_state_action_fisher.definitions import (
    joint_score,
    joint_state_action_probabilities,
    policy_score,
    q_gradient,
    state_distribution_score,
)
from exploration.joint_state_action_fisher.geometry import (
    joint_logdet_closed_form,
    joint_logdet_from_matrix,
    joint_logdet_gradient_analytic,
    joint_state_action_fisher_decomposed,
    joint_state_action_fisher_enumerated,
    joint_visitation_contribution,
    pooled_policy_fisher_closed_form,
    pooled_policy_fisher_enumerated,
    pooled_policy_logdet,
    state_distribution_fisher_closed_form,
    state_distribution_fisher_enumerated,
)
from exploration.joint_state_action_fisher.exact_two_state import (
    ExactRunConfig,
    magnitude_matched_betas,
    regularizer_gradient,
    train_exact,
    vector_field_rows,
)
from exploration.joint_state_action_fisher.verify_identity import run_verification
from exploration.tabular_mdp.geometry import barrier_terms, pooled_fisher
from exploration.tabular_mdp.model import TwoStepTrap, phi_from_q_and_good, transition_pool_weights


class JointFisherIdentityTests(unittest.TestCase):
    def setUp(self):
        self.phi = torch.tensor((0.4, -0.7, -0.2, 0.8), dtype=torch.float64)

    def test_complete_declared_suite(self):
        result = run_verification()
        failures = {
            case.name: [name for name, passed in case.checks.items() if not passed]
            for case in result.cases
            if not case.passed
        }
        self.assertTrue(result.passed, msg=failures)

    def test_distribution_and_score_shapes(self):
        rho = joint_state_action_probabilities(self.phi)
        self.assertEqual(rho.shape, (2, 3))
        self.assertAlmostEqual(float(rho.sum()), 1.0, places=14)
        self.assertEqual(q_gradient(self.phi).shape, (4,))
        for state in range(2):
            self.assertEqual(state_distribution_score(self.phi, state).shape, (4,))
            for action in range(3):
                self.assertEqual(policy_score(self.phi, state, action).shape, (4,))
                self.assertEqual(joint_score(self.phi, state, action).shape, (4,))

    def test_three_fisher_constructions(self):
        torch.testing.assert_close(
            pooled_policy_fisher_closed_form(self.phi),
            pooled_policy_fisher_enumerated(self.phi),
            atol=1e-14,
            rtol=1e-14,
        )
        torch.testing.assert_close(
            pooled_policy_fisher_closed_form(self.phi), pooled_fisher(self.phi), atol=1e-14, rtol=1e-14
        )
        torch.testing.assert_close(
            state_distribution_fisher_closed_form(self.phi),
            state_distribution_fisher_enumerated(self.phi),
            atol=1e-14,
            rtol=1e-14,
        )
        torch.testing.assert_close(
            joint_state_action_fisher_decomposed(self.phi),
            joint_state_action_fisher_enumerated(self.phi),
            atol=1e-14,
            rtol=1e-14,
        )

    def test_logdet_gradient_is_differentiable(self):
        work = self.phi.clone().requires_grad_(True)
        matrix_gradient = torch.autograd.grad(joint_logdet_from_matrix(work), work)[0]
        work = self.phi.clone().requires_grad_(True)
        scalar_gradient = torch.autograd.grad(joint_logdet_closed_form(work), work)[0]
        analytic = joint_logdet_gradient_analytic(self.phi)
        torch.testing.assert_close(matrix_gradient, analytic, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(scalar_gradient, analytic, atol=1e-12, rtol=1e-12)

    def test_near_boundary_remains_finite(self):
        for q in (1e-2, 1e-4, 1e-6, 0.99):
            phi = phi_from_q_and_good(q, 0.9)
            self.assertTrue(bool(torch.isfinite(joint_state_action_fisher_decomposed(phi)).all()))
            self.assertTrue(bool(torch.isfinite(joint_logdet_closed_form(phi))))

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            joint_state_action_probabilities((0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            policy_score(self.phi, 2, 0)
        with self.assertRaises(ValueError):
            policy_score(self.phi, 0, 3)
        with self.assertRaises(ValueError):
            state_distribution_score(torch.zeros((2, 4), dtype=torch.float64), 0)


class JointFisherCliTests(unittest.TestCase):
    def test_cli_writes_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "identity"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "exploration.joint_state_action_fisher.verify_identity",
                    "--output-dir",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)
            self.assertIn("all_passed=True", completed.stdout)
            for name in ("identity_cases.csv", "verification.json", "manifest.json"):
                self.assertTrue((output / name).exists())
            verification = json.loads((output / "verification.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(verification["all_passed"])
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["scientific_role"], "algebraic_verification")

    def test_step2_smoke_cli_resume_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "step2"
            command = [
                sys.executable,
                "-m",
                "exploration.joint_state_action_fisher.run_exact_two_state",
                "--smoke",
                "--output-dir",
                str(output),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
            self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)
            for name in (
                "config.json",
                "manifest.json",
                "checkpoints.csv",
                "endpoints.csv",
                "gradient_decomposition.csv",
                "summary.csv",
                "vector_field.csv",
                "mechanism_summary.json",
            ):
                self.assertTrue((output / name).exists(), msg=name)
            self.assertTrue((output / "plots" / "vector_fields" / "objective_vector_fields.png").exists())
            resumed = subprocess.run(command + ["--resume"], capture_output=True, text=True, timeout=60)
            self.assertEqual(resumed.returncode, 0, msg=resumed.stdout + resumed.stderr)
            self.assertIn("Completed compatible run", resumed.stdout)


class ExactStep2Tests(unittest.TestCase):
    def setUp(self):
        self.phi = torch.tensor((0.4, -0.7, -0.2, 0.8), dtype=torch.float64)

    @staticmethod
    def _autograd(function, phi):
        work = phi.clone().requires_grad_(True)
        return torch.autograd.grad(function(work), work)[0]

    def test_regularizer_gradients_against_declared_scalars(self):
        conditional = self._autograd(
            lambda x: transition_pool_weights(x)[0].detach() * barrier_terms(x).b0
            + transition_pool_weights(x)[1].detach() * barrier_terms(x).b1,
            self.phi,
        )
        pooled = self._autograd(pooled_policy_logdet, self.phi)
        joint = self._autograd(joint_logdet_closed_form, self.phi)
        state = self._autograd(joint_visitation_contribution, self.phi)
        correction = self._autograd(lambda x: 0.5 * torch.log(transition_pool_weights(x)[0]), self.phi)
        expected = {
            "statewise_conditional_barrier": conditional,
            "pooled_policy_logdet": pooled,
            "joint_state_action_logdet": joint,
            "state_distribution_only": state,
            "joint_correction_only": correction,
        }
        for method, target in expected.items():
            torch.testing.assert_close(regularizer_gradient(self.phi, method), target, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(joint, pooled + correction, atol=1e-12, rtol=1e-12)

    def test_exact_training_replay_and_magnitude_matching(self):
        config = ExactRunConfig("joint_state_action_logdet", "test", "adverse", 0.05, 0.1, 10)
        first = train_exact(config)
        second = train_exact(config)
        self.assertTrue(first.finite)
        self.assertEqual(len(first.checkpoints), len(second.checkpoints))
        for left, right in zip(first.checkpoints, second.checkpoints):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                if isinstance(left[key], float) and math.isnan(left[key]):
                    self.assertTrue(math.isnan(right[key]), msg=key)
                else:
                    self.assertEqual(left[key], right[key], msg=key)
        self.assertEqual(len(first.checkpoints), 11)
        reward_norm = torch.linalg.vector_norm(
            TwoStepTrap().exact_reward_gradient(torch.tensor((2.0, -2.0, -2.0, 2.0), dtype=torch.float64))
        ).item()
        coefficients = magnitude_matched_betas()
        initial = torch.tensor((2.0, -2.0, -2.0, 2.0), dtype=torch.float64)
        for method, beta in coefficients.items():
            contribution = beta * torch.linalg.vector_norm(regularizer_gradient(initial, method)).item()
            self.assertAlmostEqual(contribution, reward_norm, places=13)

    def test_joint_direction_is_not_only_a_rescaling(self):
        rows = [row for row in vector_field_rows(grid_size=5) if row["method"] == "reward_only"]
        self.assertTrue(any(float(row["cosine_pooled_joint_regularizer"]) < 0.999999 for row in rows))
        self.assertTrue(all(float(row["joint_correction_dq_dt"]) < 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()

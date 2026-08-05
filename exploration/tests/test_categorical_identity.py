from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from exploration.categorical_bandit.identity import (
    analytic_barrier_gradient,
    categorical_scores,
    fisher_closed_form,
    fisher_from_score_expectation,
    log_barrier,
    reduced_fisher,
    verify_identity_case,
)
from exploration.categorical_bandit.verify import build_cases


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ExactCategoricalIdentityTest(unittest.TestCase):
    def test_uniform_two_action_policy_has_known_fisher(self) -> None:
        logits = torch.zeros(2, dtype=torch.float64)
        probabilities = torch.softmax(logits, dim=0)
        fisher = fisher_closed_form(probabilities)
        expected = torch.tensor(
            [[0.25, -0.25], [-0.25, 0.25]],
            dtype=torch.float64,
        )

        torch.testing.assert_close(fisher, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            torch.linalg.eigvalsh(fisher),
            torch.tensor([0.0, 0.5], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

        sign, reduced_logdet = torch.linalg.slogdet(reduced_fisher(fisher, 1))
        self.assertEqual(float(sign), 1.0)
        expected_logdet = torch.log(torch.tensor(0.25, dtype=torch.float64))
        self.assertAlmostEqual(float(reduced_logdet), float(expected_logdet), places=15)
        torch.testing.assert_close(
            analytic_barrier_gradient(logits),
            torch.zeros(2, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_exact_score_expectation_matches_closed_form(self) -> None:
        for name, logits in build_cases():
            with self.subTest(case=name):
                probabilities = torch.softmax(logits, dim=0)
                scores = categorical_scores(probabilities)
                expected_score = (probabilities.unsqueeze(1) * scores).sum(dim=0)
                torch.testing.assert_close(
                    expected_score,
                    torch.zeros_like(expected_score),
                    rtol=0.0,
                    atol=1e-12,
                )
                torch.testing.assert_close(
                    fisher_from_score_expectation(probabilities),
                    fisher_closed_form(probabilities),
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_bartlett_structure_and_all_standard_cases(self) -> None:
        for name, logits in build_cases():
            with self.subTest(case=name):
                result = verify_identity_case(name, logits)
                self.assertTrue(result.passed, result.to_dict())
                self.assertLessEqual(result.expected_score_error, 1e-12)
                self.assertLessEqual(result.bartlett_error, 1e-12)
                self.assertLessEqual(result.symmetry_error, 1e-12)
                self.assertLessEqual(result.null_residual, 1e-12)
                self.assertGreaterEqual(result.minimum_eigenvalue, -1e-12)
                self.assertEqual(result.numerical_rank, result.action_count - 1)

    def test_determinant_identity_is_independent_of_reference_action(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(23)
        cases = (
            torch.tensor([-0.7, 0.2, 1.1], dtype=torch.float64),
            torch.randn(10, dtype=torch.float64, generator=generator),
        )
        for logits in cases:
            probabilities = torch.softmax(logits, dim=0)
            fisher = fisher_closed_form(probabilities)
            target = torch.log_softmax(logits, dim=0).sum()
            for reference_action in range(logits.numel()):
                with self.subTest(
                    action_count=logits.numel(),
                    reference_action=reference_action,
                ):
                    sign, actual = torch.linalg.slogdet(
                        reduced_fisher(fisher, reference_action)
                    )
                    self.assertEqual(float(sign), 1.0)
                    torch.testing.assert_close(actual, target, rtol=1e-10, atol=1e-10)

    def test_barrier_gradient_hessian_and_finite_difference(self) -> None:
        logits = torch.tensor([-1.2, -0.1, 0.3, 1.4], dtype=torch.float64)
        result = verify_identity_case("derivatives", logits)

        self.assertLessEqual(result.gradient_error, 1e-12)
        self.assertLessEqual(result.gradient_sum_residual, 1e-12)
        self.assertLessEqual(result.hessian_error, 1e-11)
        self.assertLessEqual(result.finite_difference_error, 1e-7)

        autograd_logits = logits.clone().requires_grad_(True)
        autograd_gradient = torch.autograd.grad(
            log_barrier(autograd_logits), autograd_logits
        )[0]
        torch.testing.assert_close(
            autograd_gradient,
            analytic_barrier_gradient(logits),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_near_boundary_case_remains_finite(self) -> None:
        logits = torch.linspace(-8.0, 8.0, 10, dtype=torch.float64)
        result = verify_identity_case("near_boundary", logits)

        self.assertTrue(result.passed, result.to_dict())
        for key, value in result.to_dict().items():
            if isinstance(value, float):
                self.assertTrue(torch.isfinite(torch.tensor(value)), key)

    def test_validation_rejects_malformed_inputs(self) -> None:
        with self.assertRaises(ValueError):
            verify_identity_case("too_small", [0.0])
        with self.assertRaises(ValueError):
            verify_identity_case("nonfinite", [0.0, float("nan")])
        with self.assertRaises(ValueError):
            verify_identity_case("bad_reference", [0.0, 1.0], reference_action=2)
        with self.assertRaises(ValueError):
            categorical_scores([0.2, 0.2])
        with self.assertRaises(ValueError):
            categorical_scores([1.0, 0.0])


class CategoricalIdentityCliTest(unittest.TestCase):
    def test_cli_prints_table_and_writes_valid_optional_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            json_path = Path(temporary_directory) / "nested" / "identity.json"
            command = [
                sys.executable,
                "-m",
                "exploration.categorical_bandit.verify",
                "--json-output",
                str(json_path),
            ]
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("k2_uniform", completed.stdout)
            self.assertIn("k10_near_boundary", completed.stdout)
            self.assertIn("ALL CHECKS PASSED", completed.stdout)
            self.assertTrue(json_path.is_file())

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["dtype"], "float64")
            self.assertEqual(payload["seed"], 23)
            self.assertEqual(len(payload["cases"]), 5)
            self.assertTrue(payload["all_passed"])
            self.assertTrue(all(case["passed"] for case in payload["cases"]))


if __name__ == "__main__":
    unittest.main()

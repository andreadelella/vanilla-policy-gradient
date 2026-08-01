from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from fisher_analysis.run_fisher_analysis import (
    analyze_fisher,
    compute_empirical_fisher,
    parameter_layout,
)
from vpg.policy import GaussianPolicy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class EmpiricalFisherTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.policy = GaussianPolicy(
            state_dim=2,
            action_dim=1,
            hidden_sizes=(),
        ).double()
        self.states = torch.tensor(
            [[0.2, -0.5], [1.0, 0.3], [-0.7, 0.4]],
            dtype=torch.float64,
        )
        self.actions = torch.tensor(
            [[0.1], [-0.4], [0.8]],
            dtype=torch.float64,
        )

    def test_fisher_matches_manual_autograd_scores(self) -> None:
        actual = compute_empirical_fisher(
            self.policy,
            self.states,
            self.actions,
            score_batch_size=2,
        )

        parameters = list(self.policy.parameters())
        scores = []
        for state, action in zip(self.states, self.actions):
            mean, std = self.policy(state.unsqueeze(0))
            log_probability = Normal(mean.squeeze(0), std).log_prob(action).sum()
            gradients = torch.autograd.grad(log_probability, parameters)
            scores.append(torch.cat([gradient.reshape(-1) for gradient in gradients]))
        score_matrix = torch.stack(scores)
        expected = score_matrix.T @ score_matrix / len(scores)

        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_shape_spectrum_metrics_and_policy_immutability(self) -> None:
        initial = {
            name: value.detach().clone()
            for name, value in self.policy.state_dict().items()
        }
        fisher = compute_empirical_fisher(
            self.policy,
            self.states,
            self.actions,
        )
        dimension = sum(parameter.numel() for parameter in self.policy.parameters())
        self.assertEqual(fisher.shape, (dimension, dimension))
        torch.testing.assert_close(fisher, fisher.T, rtol=0.0, atol=1e-14)

        eigenvalues, metrics, rank_tolerance, _ = analyze_fisher(
            fisher,
            sample_count=len(self.states),
        )
        self.assertTrue(np.all(eigenvalues[:-1] >= eigenvalues[1:]))
        self.assertGreaterEqual(float(eigenvalues[-1]), -1e-12)
        self.assertAlmostEqual(
            float(torch.trace(fisher)),
            float(np.sum(eigenvalues)),
            places=11,
        )
        self.assertEqual(metrics["matrix_dimension"], dimension)
        self.assertEqual(metrics["sample_count"], len(self.states))
        self.assertLessEqual(metrics["numerical_rank"], len(self.states))
        self.assertGreater(rank_tolerance, 0.0)

        layout = parameter_layout(self.policy)
        self.assertEqual(layout[-1]["stop"], dimension)
        for name, value in self.policy.state_dict().items():
            self.assertTrue(torch.equal(value, initial[name]))
        self.assertTrue(all(parameter.grad is None for parameter in self.policy.parameters()))

    def test_rank_metrics_match_known_diagonal_spectrum(self) -> None:
        fisher = np.diag([4.0, 1.0, 0.0])
        eigenvalues, metrics, _, _ = analyze_fisher(fisher, sample_count=5)

        np.testing.assert_array_equal(eigenvalues, np.asarray([4.0, 1.0, 0.0]))
        self.assertEqual(metrics["numerical_rank"], 2)
        self.assertAlmostEqual(metrics["trace"], 5.0)
        self.assertAlmostEqual(metrics["positive_condition_number"], 4.0)
        self.assertAlmostEqual(metrics["stable_rank"], 17.0 / 16.0)
        expected_effective_rank = np.exp(
            -(0.8 * np.log(0.8) + 0.2 * np.log(0.2))
        )
        self.assertAlmostEqual(metrics["effective_rank"], expected_effective_rank)
        self.assertEqual(metrics["components_90"], 2)
        self.assertEqual(metrics["components_95"], 2)
        self.assertEqual(metrics["components_99"], 2)


class CliSmokeTest(unittest.TestCase):
    def test_cartpole_cli_writes_valid_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "results"
            environment = dict(os.environ)
            environment["MPLCONFIGDIR"] = str(Path(temporary_directory) / "mpl")
            command = [
                sys.executable,
                "-m",
                "fisher_analysis.run_fisher_analysis",
                "--env-id",
                "CartPole-v1",
                "--widths",
                "2",
                "--depth",
                "2",
                "--iterations",
                "1",
                "--n-envs",
                "2",
                "--trajectories-per-env",
                "1",
                "--horizon",
                "8",
                "--seed",
                "11",
                "--output-dir",
                str(output_dir),
            ]
            subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

            expected_files = [
                "config.json",
                "iteration_stats.csv",
                "summary.csv",
                "fisher_width_2.npz",
                "raw_eigenspectrum.png",
                "trace_normalized_eigenspectrum.png",
                "cumulative_explained_trace.png",
                "checkpoints/policy_width_2.pt",
            ]
            for relative_path in expected_files:
                self.assertTrue((output_dir / relative_path).is_file(), relative_path)

            with (output_dir / "config.json").open(encoding="utf-8") as handle:
                config = json.load(handle)
            self.assertEqual(config["parameter_dimensions"]["2"], 22)
            self.assertEqual(config["damping"], 0.0)
            self.assertEqual(
                config["vectorization"],
                "gymnasium.vector.AsyncVectorEnv",
            )

            with (output_dir / "summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 1)
            self.assertEqual(int(summary[0]["matrix_dimension"]), 22)
            self.assertEqual(int(summary[0]["sample_count"]), 16)

            with np.load(output_dir / "fisher_width_2.npz") as archive:
                fisher = archive["fisher"]
                eigenvalues = archive["eigenvalues"]
                self.assertEqual(fisher.shape, (22, 22))
                self.assertTrue(np.allclose(fisher, fisher.T))
                self.assertTrue(np.all(eigenvalues[:-1] >= eigenvalues[1:]))
                self.assertAlmostEqual(
                    float(np.trace(fisher)),
                    float(np.sum(eigenvalues)),
                    places=10,
                )


if __name__ == "__main__":
    unittest.main()

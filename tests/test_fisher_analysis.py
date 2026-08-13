import unittest

import numpy as np
import torch

from fisher_analysis.fisher import analyze_fisher, compute_empirical_fisher
from vpg.policy import GaussianPolicy


class FisherAnalysisTests(unittest.TestCase):
    def test_streaming_fisher_matches_analytic_result(self):
        policy = GaussianPolicy(
            1,
            1,
            hidden_sizes=(),
            init_log_std=0.0,
            learn_std=False,
        ).to(dtype=torch.float64)
        with torch.no_grad():
            policy.mean_net[0].weight.zero_()
            policy.mean_net[0].bias.zero_()

        states = np.array([[1.0], [-1.0]])
        actions = np.array([[1.0], [-1.0]])
        one_at_a_time = compute_empirical_fisher(
            policy,
            states,
            actions,
            score_batch_size=1,
        )
        one_batch = compute_empirical_fisher(
            policy,
            states,
            actions,
            score_batch_size=2,
        )

        torch.testing.assert_close(one_at_a_time, torch.eye(2, dtype=torch.float64))
        torch.testing.assert_close(one_at_a_time, one_batch)
        self.assertTrue(all(parameter.grad is None for parameter in policy.parameters()))

    def test_spectral_metrics_use_only_numerically_positive_directions(self):
        matrix = np.diag([4.0, 1.0, 0.0])
        eigenvalues, metrics, rank_tolerance, _ = analyze_fisher(
            matrix,
            sample_count=10,
        )

        np.testing.assert_allclose(eigenvalues, [4.0, 1.0, 0.0])
        self.assertEqual(metrics["numerical_rank"], 2)
        self.assertEqual(metrics["positive_condition_number"], 4.0)
        self.assertGreater(rank_tolerance, 0.0)

    def test_non_symmetric_and_indefinite_matrices_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "not symmetric"):
            analyze_fisher(np.array([[1.0, 1.0], [0.0, 1.0]]), sample_count=1)
        with self.assertRaisesRegex(AssertionError, "positive semidefinite"):
            analyze_fisher(np.diag([1.0, -1.0]), sample_count=1)


if __name__ == "__main__":
    unittest.main()

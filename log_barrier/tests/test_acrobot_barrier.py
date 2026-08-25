import unittest

import torch

from log_barrier.acrobot.barrier import analytic_logit_gradient, categorical_log_barrier
from log_barrier.acrobot.training import AcrobotConfig, regularization_coefficient


class AcrobotBarrierTests(unittest.TestCase):
    def test_analytic_gradient(self):
        logits = torch.tensor(((0.2, -0.7, 1.1), (-0.3, 0.4, 0.8)), dtype=torch.float64, requires_grad=True)
        barrier, diagnostics = categorical_log_barrier(logits)
        actual = torch.autograd.grad(barrier, logits)[0]
        torch.testing.assert_close(actual, analytic_logit_gradient(logits), atol=1e-12, rtol=1e-12)
        self.assertEqual(diagnostics.state_count, 2)
        self.assertEqual(diagnostics.action_count, 3)

    def test_uniform_is_stationary(self):
        logits = torch.zeros((4, 3), dtype=torch.float64)
        torch.testing.assert_close(analytic_logit_gradient(logits), torch.zeros_like(logits), atol=1e-15, rtol=0.0)

    def test_fixed_coefficient(self):
        self.assertEqual(regularization_coefficient(AcrobotConfig("reward_only", 1)), 0.0)
        self.assertEqual(regularization_coefficient(AcrobotConfig("log_barrier", 1, beta=2.0)), 2.0)
        self.assertEqual(regularization_coefficient(AcrobotConfig("fisher_logdet", 1)), 0.0)

    def test_fisher_method_resolves_to_identifiable_policy(self):
        config = AcrobotConfig("fisher_logdet", 1)
        self.assertEqual(config.effective_policy_parameterization, "reference")
        config.validate()

    def test_fisher_method_rejects_standard_softmax_coordinates(self):
        config = AcrobotConfig(
            "fisher_logdet",
            1,
            policy_parameterization="standard",
        )
        with self.assertRaisesRegex(ValueError, "identifiable reference-logit"):
            config.validate()

    def test_fisher_collection_size_must_match_worker_count(self):
        config = AcrobotConfig(
            "fisher_logdet",
            1,
            fisher_episodes_per_update=10,
            fisher_parallel_envs=4,
        )
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            config.validate()

    def test_fisher_score_backend_is_validated(self):
        config = AcrobotConfig("fisher_logdet", 1, fisher_score_backend="invalid")
        with self.assertRaisesRegex(ValueError, "fisher_score_backend"):
            config.validate()


if __name__ == "__main__":
    unittest.main()

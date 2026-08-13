import unittest

import gymnasium as gym
import numpy as np
import torch

from vpg.policy import GaussianPolicy, MLPSoftmaxPolicy, build_policy


class PolicyTests(unittest.TestCase):
    def test_gaussian_log_probability_sums_action_dimensions(self):
        policy = GaussianPolicy(3, 2, hidden_sizes=(4,), init_log_std=-0.5)
        states = torch.zeros(5, 3)
        actions = torch.zeros(5, 2)
        actual = policy.log_prob(states, actions)
        expected = policy.distribution(states).log_prob(actions).sum(-1)

        self.assertEqual(actual.shape, (5,))
        torch.testing.assert_close(actual, expected)

    def test_fixed_standard_deviation_is_a_buffer(self):
        policy = GaussianPolicy(2, 1, hidden_sizes=(), learn_std=False)
        self.assertNotIn("log_std", dict(policy.named_parameters()))
        self.assertIn("log_std", dict(policy.named_buffers()))

    def test_discrete_policy_returns_one_log_probability_per_state(self):
        policy = MLPSoftmaxPolicy(4, 2, hidden_sizes=(8,))
        states = torch.zeros(3, 4)
        actions = np.array([0, 1, 0])
        self.assertEqual(policy.log_prob(states, actions).shape, (3,))

    def test_factory_matches_environment_action_space(self):
        continuous = gym.make("Pendulum-v1")
        discrete = gym.make("CartPole-v1")
        try:
            cfg = {"hidden_sizes": [4, 4], "learn_std": True}
            self.assertIsInstance(build_policy(cfg, continuous), GaussianPolicy)
            self.assertIsInstance(build_policy(cfg, discrete), MLPSoftmaxPolicy)
        finally:
            continuous.close()
            discrete.close()


if __name__ == "__main__":
    unittest.main()

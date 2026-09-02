import unittest

import gymnasium as gym
import torch

from fisher_log_barrier.lunar_lander_continuous import (
    ACTION_DIM,
    ENV_ID,
    STATE_DIM,
    LunarLanderContinuousConfig,
    _build_policy,
    _make_env,
    _set_policy_gradients,
)


class LunarLanderContinuousTests(unittest.TestCase):
    def test_environment_is_native_continuous_lunar_lander(self):
        config = LunarLanderContinuousConfig()
        env = _make_env(config)()
        try:
            self.assertEqual(env.spec.id, ENV_ID)
            self.assertEqual(env.observation_space.shape, (STATE_DIM,))
            self.assertIsInstance(env.action_space, gym.spaces.Box)
            self.assertEqual(env.action_space.shape, (ACTION_DIM,))
        finally:
            env.close()

    def test_established_policy_and_fisher_batch(self):
        config = LunarLanderContinuousConfig()
        parameter_count = sum(
            parameter.numel() for parameter in _build_policy(config).parameters()
        )

        self.assertEqual(parameter_count, 452)
        self.assertGreaterEqual(config.fisher_trajectory_count, parameter_count)
        self.assertEqual(config.to_dict()["mode"], "fixed")

    def test_fixed_mode_applies_configured_beta(self):
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        reward_loss = (parameter - 1.0).square().sum()
        fisher_surrogate = parameter.square().sum()

        metrics = _set_policy_gradients(
            (parameter,),
            reward_loss,
            fisher_surrogate,
            0.25,
            None,
        )

        self.assertAlmostEqual(parameter.grad.item(), 1.0)
        self.assertEqual(metrics["effective_beta"], 0.25)
        self.assertFalse(metrics["clipped"])

    def test_clipped_mode_caps_applied_gradient(self):
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        reward_loss = (parameter - 1.0).square().sum()
        fisher_surrogate = parameter.square().sum()

        metrics = _set_policy_gradients(
            (parameter,),
            reward_loss,
            fisher_surrogate,
            1.0,
            0.2,
        )

        self.assertAlmostEqual(parameter.grad.item(), 1.6)
        self.assertAlmostEqual(metrics["effective_beta"], 0.1)
        self.assertAlmostEqual(metrics["barrier_to_reward_ratio"], 0.2)
        self.assertTrue(metrics["clipped"])


if __name__ == "__main__":
    unittest.main()

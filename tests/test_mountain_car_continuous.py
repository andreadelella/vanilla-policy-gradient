import unittest

import torch

from fisher_log_barrier.mountain_car_continuous import (
    MountainCarContinuousConfig,
    _build_policy,
    _make_env,
    _set_policy_gradients,
)


class MountainCarContinuousTests(unittest.TestCase):
    def test_established_configuration_and_policy_size(self):
        config = MountainCarContinuousConfig()

        self.assertEqual(config.hidden_sizes, (2, 2))
        self.assertEqual(config.reward_trajectory_count, 32)
        self.assertEqual(config.fisher_trajectory_count, 256)
        self.assertEqual(config.reward_trajectories_per_worker, 2)
        self.assertEqual(config.fisher_trajectories_per_worker, 16)
        self.assertEqual(
            sum(parameter.numel() for parameter in _build_policy(config).parameters()),
            16,
        )

    def test_native_environment_uses_500_step_horizon(self):
        config = MountainCarContinuousConfig(horizon=500)
        env = _make_env(config)()
        try:
            env.reset(seed=101)
            decision_steps = 0
            truncated = False
            while not truncated:
                _, _, terminated, truncated, _ = env.step([0.0])
                self.assertFalse(terminated)
                decision_steps += 1
        finally:
            env.close()

        self.assertEqual(decision_steps, 500)

    def test_config_records_independent_fisher_batch(self):
        config = MountainCarContinuousConfig()

        self.assertEqual(config.to_dict()["mode"], "fixed")
        self.assertNotEqual(
            config.reward_trajectory_count,
            config.fisher_trajectory_count,
        )

    def test_config_records_clipped_mode(self):
        config = MountainCarContinuousConfig(
            clip_ratio=0.2,
        )

        self.assertEqual(config.to_dict()["mode"], "clipped")

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
        self.assertEqual(metrics["clip_scale"], 1.0)
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

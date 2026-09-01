import unittest

from fisher_log_barrier.continuous_mountain_car_experiment import (
    ContinuousMountainCarConfig,
    gradient_balanced_beta,
)


class AdaptiveFisherGradientRatioTests(unittest.TestCase):
    def test_gradient_balanced_beta_hits_requested_ratio(self):
        reward_norm = 12.0
        unscaled_fisher_norm = 30.0

        beta = gradient_balanced_beta(reward_norm, unscaled_fisher_norm, 0.05)

        self.assertAlmostEqual(beta, 0.02)
        self.assertAlmostEqual(
            beta * unscaled_fisher_norm / reward_norm,
            0.05,
        )

    def test_gradient_balanced_beta_disables_zero_reward_norm(self):
        self.assertEqual(gradient_balanced_beta(0.0, 3.0, 0.05), 0.0)

    def test_gradient_balanced_beta_disables_zero_fisher_norm(self):
        self.assertEqual(gradient_balanced_beta(3.0, 0.0, 0.05), 0.0)

    def test_target_gradient_ratio_is_validated(self):
        with self.assertRaisesRegex(
            ValueError,
            "target_fisher_gradient_ratio",
        ):
            ContinuousMountainCarConfig(
                target_fisher_gradient_ratio=-0.01
            ).validate()


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import torch

from vpg.data_collection import Trajectory
from vpg.gpomdp import compute_discounted_returns_matrix, trajectories_to_tensors


def trajectory(states, actions, rewards):
    return Trajectory(
        states=[np.asarray(state, dtype=np.float32) for state in states],
        actions=[np.asarray(action, dtype=np.float32) for action in actions],
        rewards=list(rewards),
        dones=[False] * (len(rewards) - 1) + [True],
    )


class DiscountedReturnTests(unittest.TestCase):
    def test_known_reward_to_go(self):
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        expected = torch.tensor([[2.75, 3.5, 3.0]])
        torch.testing.assert_close(
            compute_discounted_returns_matrix(rewards, 0.5), expected
        )

    def test_recursive_and_vectorized_backends_agree(self):
        generator = torch.Generator().manual_seed(4)
        rewards = torch.randn(3, 1000, generator=generator)
        for gamma in (0.0, 0.5, 0.95, 0.999, 1.0):
            with self.subTest(gamma=gamma):
                recursive = compute_discounted_returns_matrix(
                    rewards, gamma, "recursive"
                )
                vectorized = compute_discounted_returns_matrix(
                    rewards, gamma, "vectorized"
                )
                torch.testing.assert_close(
                    recursive, vectorized, rtol=1e-4, atol=2e-4
                )

    def test_vectorized_backend_rejects_unsafe_range(self):
        with self.assertRaisesRegex(ValueError, "numerically unsafe"):
            compute_discounted_returns_matrix(
                torch.ones(1, 1000), 0.1, "vectorized"
            )

    def test_invalid_gamma_and_backend_are_rejected(self):
        with self.assertRaises(ValueError):
            compute_discounted_returns_matrix(torch.ones(1, 2), 1.1)
        with self.assertRaises(ValueError):
            compute_discounted_returns_matrix(torch.ones(1, 2), 0.9, "other")

    def test_variable_length_trajectories_are_masked(self):
        trajectories = [
            trajectory([[1], [2]], [[0.1], [0.2]], [1, 2]),
            trajectory([[3]], [[0.3]], [3]),
        ]
        _, _, rewards, mask = trajectories_to_tensors(trajectories)
        torch.testing.assert_close(rewards, torch.tensor([[1.0, 2.0], [3.0, 0.0]]))
        torch.testing.assert_close(mask, torch.tensor([[1.0, 1.0], [1.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()

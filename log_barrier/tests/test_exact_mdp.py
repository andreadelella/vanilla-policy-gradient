import unittest

import torch

from log_barrier.exact_mdp.geometry import (
    enumerated_joint_fisher,
    joint_fisher,
    policy_fisher,
    state_fisher,
)
from log_barrier.exact_mdp.model import DTYPE, ThreeStateChain, discounted_occupancy
from log_barrier.exact_mdp.training import ExactTrainingConfig, checkpoint_updates, train


class ExactMdpTests(unittest.TestCase):
    def setUp(self):
        self.phi = torch.tensor((0.3, -0.7, -0.2, 0.8, 0.5, -0.4), dtype=DTYPE)

    def test_return_gradient(self):
        value = self.phi.clone().requires_grad_(True)
        expected = torch.autograd.grad(ThreeStateChain().exact_return(value), value)[0]
        torch.testing.assert_close(ThreeStateChain().reward_gradient(self.phi), expected, atol=1e-12, rtol=1e-12)

    def test_discounted_occupancy(self):
        occupancy = discounted_occupancy(self.phi)
        self.assertAlmostEqual(float(occupancy.sum()), 1.0, places=14)
        self.assertTrue(bool((occupancy > 0.0).all()))

    def test_joint_fisher_two_independent_constructions(self):
        joint = joint_fisher(self.phi)
        torch.testing.assert_close(joint, policy_fisher(self.phi) + state_fisher(self.phi), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(joint, enumerated_joint_fisher(self.phi), atol=1e-12, rtol=1e-12)
        self.assertGreater(float(torch.linalg.eigvalsh(joint)[0]), 0.0)

    def test_decile_checkpoints(self):
        self.assertEqual(checkpoint_updates(2000), tuple(range(0, 2001, 200)))
        result = train(ExactTrainingConfig("reward_only", updates=10))
        self.assertEqual(len(result.trajectory), 11)
        self.assertEqual(len(result.spectra), 22)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from fisher_log_barrier.loss1 import (
    FisherLogDetDomainError,
    estimate_trajectory_fisher_inverse,
    trajectory_fisher_logdet_surrogate,
)
from fisher_log_barrier.policy import ReferenceMLPSoftmaxPolicy
from vpg.data_collection import Trajectory


def _exact_logdet_gradient(theta: torch.Tensor, mu: float) -> torch.Tensor:
    logits = torch.cat((theta, theta.new_zeros(1)))
    log_probabilities = torch.log_softmax(logits, dim=0)
    probabilities = log_probabilities.exp()
    scores = torch.stack(
        [
            torch.autograd.grad(value, theta, create_graph=True, retain_graph=True)[0]
            for value in log_probabilities
        ]
    )
    fisher = torch.einsum("a,ai,aj->ij", probabilities, scores, scores)
    margin = fisher - mu * torch.eye(2, dtype=theta.dtype)
    return torch.autograd.grad(torch.linalg.slogdet(margin).logabsdet, theta)[0]


def _loss1_gradient(theta: torch.Tensor, mu: float, detach_inverse: bool) -> torch.Tensor:
    logits = torch.cat((theta, theta.new_zeros(1)))
    log_probabilities = torch.log_softmax(logits, dim=0)
    probabilities = log_probabilities.exp()
    scores = torch.stack(
        [
            torch.autograd.grad(value, theta, create_graph=True, retain_graph=True)[0]
            for value in log_probabilities
        ]
    )
    fisher = torch.einsum("a,ai,aj->ij", probabilities, scores, scores)
    inverse = torch.linalg.inv(fisher - mu * torch.eye(2, dtype=theta.dtype))
    if detach_inverse:
        inverse = inverse.detach()
    b_values = torch.einsum("ai,ij,aj->a", scores, inverse, scores) - mu * torch.trace(inverse)
    surrogate = (
        probabilities.detach()
        * (log_probabilities * b_values.detach() + b_values)
    ).sum()
    return torch.autograd.grad(surrogate, theta)[0]


class ReducedCategoricalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor((0.3, -0.2), dtype=torch.float64))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        logits = torch.cat((self.theta, self.theta.new_zeros(1)))
        return logits.expand(states.shape[0], -1)

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return Categorical(logits=self.forward(states)).log_prob(actions.long())


def _one_step_trajectory(action: int, state: float = 0.0) -> Trajectory:
    return Trajectory(
        states=[np.array((state,), dtype=np.float64)],
        actions=[np.array(action)],
        rewards=[0.0],
        dones=[True],
    )


class FisherLogDetLoss1Tests(unittest.TestCase):
    def test_exact_surrogate_gradient_matches_logdet_gradient(self):
        mu = 0.01
        exact_theta = torch.tensor((0.4, -0.35), dtype=torch.float64, requires_grad=True)
        surrogate_theta = exact_theta.detach().clone().requires_grad_(True)
        exact = _exact_logdet_gradient(exact_theta, mu)
        surrogate = _loss1_gradient(surrogate_theta, mu, detach_inverse=True)
        torch.testing.assert_close(surrogate, exact, atol=1e-10, rtol=1e-10)

    def test_inverse_stop_gradient_is_necessary(self):
        mu = 0.01
        exact_theta = torch.tensor((0.4, -0.35), dtype=torch.float64, requires_grad=True)
        incorrect_theta = exact_theta.detach().clone().requires_grad_(True)
        exact = _exact_logdet_gradient(exact_theta, mu)
        incorrect = _loss1_gradient(incorrect_theta, mu, detach_inverse=False)
        self.assertGreater(float(torch.max(torch.abs(incorrect - exact))), 1e-5)

    def test_full_rank_trajectory_surrogate_is_differentiable(self):
        policy = ReducedCategoricalPolicy()
        trajectories = [_one_step_trajectory(action) for action in range(3)]
        surrogate, diagnostics = trajectory_fisher_logdet_surrogate(
            policy,
            trajectories,
            mu=0.01,
        )
        gradient = torch.autograd.grad(surrogate, policy.theta)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertEqual(diagnostics.parameter_count, 2)
        self.assertEqual(diagnostics.trajectory_count, 3)
        self.assertEqual(diagnostics.rank, 2)

    def test_rank_deficient_batch_fails_strict_domain(self):
        policy = ReducedCategoricalPolicy()
        trajectories = [_one_step_trajectory(0), _one_step_trajectory(0)]
        with self.assertRaises(FisherLogDetDomainError) as caught:
            trajectory_fisher_logdet_surrogate(policy, trajectories, mu=1e-4)
        error = caught.exception
        self.assertEqual(error.parameter_count, 2)
        self.assertEqual(error.trajectory_count, 2)
        self.assertLess(error.rank, error.parameter_count)
        self.assertIn("F_hat - mu I > 0", str(error))

    def test_reference_policy_fisher_is_estimated_in_float64_without_graph(self):
        policy = ReferenceMLPSoftmaxPolicy(1, 3, hidden_sizes=())
        fisher_trajectories = [
            _one_step_trajectory(action, state)
            for state in (-1.0, 0.0, 1.0)
            for action in range(3)
        ]
        estimate = estimate_trajectory_fisher_inverse(
            policy,
            fisher_trajectories,
            mu=1e-4,
        )
        self.assertEqual(estimate.parameter_count, 4)
        self.assertEqual(estimate.rank, 4)
        self.assertEqual(estimate.inverse_margin.dtype, torch.float64)
        self.assertFalse(estimate.inverse_margin.requires_grad)

    def test_separate_fisher_and_gradient_batches_are_reported(self):
        policy = ReferenceMLPSoftmaxPolicy(1, 3, hidden_sizes=())
        fisher_trajectories = [
            _one_step_trajectory(action, state)
            for state in (-1.0, 0.0, 1.0)
            for action in range(3)
        ]
        gradient_trajectories = [
            _one_step_trajectory(0, -0.5),
            _one_step_trajectory(2, 0.5),
        ]
        surrogate, diagnostics = trajectory_fisher_logdet_surrogate(
            policy,
            gradient_trajectories,
            fisher_trajectories=fisher_trajectories,
            mu=1e-4,
        )
        gradients = torch.autograd.grad(surrogate, tuple(policy.parameters()))
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))
        self.assertTrue(diagnostics.separate_fisher_batch)
        self.assertEqual(diagnostics.trajectory_count, 9)
        self.assertEqual(diagnostics.gradient_trajectory_count, 2)


if __name__ == "__main__":
    unittest.main()

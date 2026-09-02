import unittest

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from fisher_log_barrier.loss1 import (
    FisherLogDetDomainError,
    compute_trajectory_fisher,
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

    def test_public_fisher_matrix_matches_score_outer_products(self):
        policy = ReducedCategoricalPolicy()
        trajectories = [_one_step_trajectory(action) for action in range(3)]

        fisher, scores = compute_trajectory_fisher(policy, trajectories)

        self.assertEqual(fisher.dtype, torch.float64)
        self.assertEqual(scores.dtype, torch.float64)
        self.assertFalse(fisher.requires_grad)
        self.assertFalse(scores.requires_grad)
        torch.testing.assert_close(fisher, scores.T @ scores / len(trajectories))

    def test_separate_fisher_and_gradient_batches_are_supported(self):
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
        self.assertEqual(diagnostics.trajectory_count, 9)

    def test_vmap_matches_loop_surrogate_and_gradient(self):
        loop_policy = ReferenceMLPSoftmaxPolicy(1, 3, hidden_sizes=()).double()
        vmap_policy = ReferenceMLPSoftmaxPolicy(1, 3, hidden_sizes=()).double()
        vmap_policy.load_state_dict(loop_policy.state_dict())
        fisher_trajectories = [
            _one_step_trajectory(action, state)
            for state in (-1.0, 0.0, 1.0)
            for action in range(3)
        ]
        gradient_trajectories = [
            _one_step_trajectory(0, -0.5),
            Trajectory(
                states=[
                    np.array((0.25,), dtype=np.float64),
                    np.array((0.75,), dtype=np.float64),
                ],
                actions=[np.array(1), np.array(2)],
                rewards=[0.0, 0.0],
                dones=[False, True],
            ),
        ]

        def value_and_gradient(policy, backend):
            surrogate, diagnostics = trajectory_fisher_logdet_surrogate(
                policy,
                gradient_trajectories,
                fisher_trajectories=fisher_trajectories,
                score_backend=backend,
                mu=1e-4,
            )
            gradients = torch.autograd.grad(surrogate, tuple(policy.parameters()))
            flat_gradient = torch.cat([value.reshape(-1) for value in gradients])
            return surrogate.detach(), flat_gradient, diagnostics

        loop_value, loop_gradient, loop_diagnostics = value_and_gradient(loop_policy, "loop")
        vmap_value, vmap_gradient, vmap_diagnostics = value_and_gradient(vmap_policy, "vmap")
        torch.testing.assert_close(vmap_value, loop_value, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(vmap_gradient, loop_gradient, atol=1e-10, rtol=1e-10)
        self.assertEqual(loop_diagnostics.score_backend, "loop")
        self.assertEqual(vmap_diagnostics.score_backend, "vmap")

    def test_vmap_handles_short_trajectories_for_full_acrobot_policy(self):
        policy = ReferenceMLPSoftmaxPolicy(6, 3, hidden_sizes=(8, 8))
        trajectories = []
        for trajectory_index in range(8):
            states = [
                np.full(6, trajectory_index + timestep / 10.0, dtype=np.float32)
                for timestep in range(1 + trajectory_index % 3)
            ]
            actions = [
                np.array((trajectory_index + timestep) % 3)
                for timestep in range(len(states))
            ]
            trajectories.append(
                Trajectory(
                    states=states,
                    actions=actions,
                    rewards=[0.0] * len(states),
                    dones=[False] * (len(states) - 1) + [True],
                )
            )
        with self.assertRaises(FisherLogDetDomainError) as caught:
            estimate_trajectory_fisher_inverse(
                policy,
                trajectories,
                mu=0.0,
                score_backend="vmap",
            )
        self.assertEqual(caught.exception.parameter_count, 146)
        self.assertLessEqual(caught.exception.rank, 8)


if __name__ == "__main__":
    unittest.main()

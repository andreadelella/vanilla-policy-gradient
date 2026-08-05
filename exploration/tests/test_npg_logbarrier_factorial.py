"""Focused algebra and invariance tests for the NPG × barrier stage."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from exploration.neural_discrete_log_barrier.training import NeuralTrainingConfig, build_seeded_policy
from exploration.npg_logbarrier_factorial.acrobot import AcrobotFactorialConfig
from exploration.npg_logbarrier_factorial.exact_two_state import (
    ExactFactorialConfig,
    run_one as run_exact_one,
)
from exploration.npg_logbarrier_factorial.fisher_validation import (
    DTYPE,
    categorical_enumerated_fisher,
    categorical_kl_hessian,
    gaussian_analytic_fisher,
    gaussian_kl_hessian,
    parameter_layout,
    parameter_vector,
    sampled_categorical_fisher,
)
from exploration.npg_logbarrier_factorial.natural_step import (
    flatten_parameters,
    set_parameters,
    target_kl_natural_step,
)
from exploration.npg_logbarrier_factorial.sampled_two_state import SampledFactorialConfig
from exploration.npg_logbarrier_factorial.run_experiment import DEFAULT_ROOT
from exploration.sampled_tabular_mdp.estimators import (
    sampled_conditional_gradient,
    sampled_empirical_fisher,
    sampled_reward_gradient,
)
from exploration.sampled_tabular_mdp.sampling import sample_batch
from exploration.tabular_mdp.model import as_phi
from vpg.data_collection import Trajectory
from vpg.policy import GaussianPolicy, MLPSoftmaxPolicy


class FisherValidationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.states = torch.tensor(
            [[-0.4, 0.2], [0.7, -1.1], [0.1, 0.9]], dtype=DTYPE
        )

    def test_categorical_enumeration_matches_kl_hessian(self):
        policy = MLPSoftmaxPolicy(2, 3, hidden_sizes=(3,)).to(DTYPE)
        fisher = categorical_enumerated_fisher(policy, self.states)
        hessian = categorical_kl_hessian(policy, self.states)
        self.assertTrue(torch.allclose(fisher, hessian, atol=1e-9, rtol=1e-9))

    def test_gaussian_analytic_matches_kl_hessian(self):
        policy = GaussianPolicy(2, 2, hidden_sizes=(3,), init_log_std=-0.3).to(DTYPE)
        fisher = gaussian_analytic_fisher(policy, self.states)
        hessian = gaussian_kl_hessian(policy, self.states)
        self.assertTrue(torch.allclose(fisher, hessian, atol=1e-9, rtol=1e-9))

    def test_sampled_categorical_converges_and_does_not_mutate(self):
        policy = MLPSoftmaxPolicy(2, 3, hidden_sizes=(2,)).to(DTYPE)
        before = parameter_vector(policy).clone()
        target = categorical_enumerated_fisher(policy, self.states)
        low = sampled_categorical_fisher(
            policy, self.states, 1,
            torch.Generator(device="cpu").manual_seed(81),
        )
        high = sampled_categorical_fisher(
            policy, self.states, 4096,
            torch.Generator(device="cpu").manual_seed(82),
        )
        self.assertLess(float((high - target).norm()), float((low - target).norm()))
        self.assertTrue(torch.equal(before, parameter_vector(policy)))

    def test_parameter_order_matches_named_parameter_order(self):
        policy = MLPSoftmaxPolicy(2, 3, hidden_sizes=(3,)).to(DTYPE)
        layout = parameter_layout(policy)
        self.assertEqual(layout.names, tuple(name for name, _ in policy.named_parameters()))
        self.assertEqual(sum(layout.counts), parameter_vector(policy).numel())

    def test_raw_and_executed_actions_are_distinct_fields(self):
        raw = np.asarray([1.7], dtype=np.float32)
        executed = np.asarray([1.0], dtype=np.float32)
        trajectory = Trajectory(
            states=[np.zeros(2, dtype=np.float32)], actions=[raw], rewards=[0.0],
            dones=[True], executed_actions=[executed],
        )
        self.assertAlmostEqual(float(trajectory.actions[0][0]), 1.7, places=6)
        self.assertEqual(float(trajectory.executed_actions[0][0]), 1.0)


class NaturalStepTests(unittest.TestCase):
    def test_target_kl_scaling_reconstructs_prediction(self):
        fisher = torch.diag(torch.tensor([0.5, 2.0], dtype=DTYPE))
        gradient = torch.tensor([1.0, -0.5], dtype=DTYPE)
        result = target_kl_natural_step(
            gradient, fisher, damping=0.1, target_kl=0.003
        )
        self.assertTrue(result.valid)
        reconstructed = 0.5 * torch.dot(result.step, fisher @ result.step)
        self.assertAlmostEqual(float(reconstructed), 0.003, places=12)
        self.assertAlmostEqual(result.predicted_kl, 0.003, places=12)

    def test_nonpositive_quadratic_is_invalid_without_abs_repair(self):
        fisher = torch.zeros((2, 2), dtype=DTYPE)
        result = target_kl_natural_step(
            torch.ones(2, dtype=DTYPE), fisher, damping=0.1, target_kl=0.001
        )
        self.assertFalse(result.valid)
        self.assertEqual(
            result.invalid_reason,
            "nonpositive_or_nonfinite_undamped_quadratic_form",
        )

    def test_regularized_gradient_is_preconditioned_as_one_object(self):
        fisher = torch.diag(torch.tensor([1.0, 3.0], dtype=DTYPE))
        reward = torch.tensor([1.0, 0.0], dtype=DTYPE)
        barrier = torch.tensor([0.0, 2.0], dtype=DTYPE)
        beta = 0.4
        result = target_kl_natural_step(
            reward + beta * barrier, fisher, damping=0.2, target_kl=0.001
        )
        expected_direction = torch.linalg.solve(
            fisher + 0.2 * torch.eye(2, dtype=DTYPE), reward + beta * barrier
        )
        self.assertTrue(torch.allclose(result.direction, expected_direction))


class ScheduleAndSamplingTests(unittest.TestCase):
    def test_beta_is_exactly_zero_after_handoff(self):
        exact = ExactFactorialConfig(
            "exact_npg_logbarrier_handoff", "adverse", 0.01
        )
        sampled = SampledFactorialConfig(
            "sampled_npg_logbarrier_handoff", "adverse", 32, 2
        )
        acrobot = AcrobotFactorialConfig("npg_logbarrier_handoff", 999)
        self.assertGreater(exact.beta_at(exact.handoff_update - 1), 0.0)
        self.assertEqual(exact.beta_at(exact.handoff_update), 0.0)
        self.assertEqual(sampled.beta_at(sampled.handoff_update), 0.0)
        self.assertEqual(acrobot.beta_at(acrobot.handoff_update), 0.0)

    def test_missing_s1_has_no_oracle_reward_barrier_or_fisher_block(self):
        phi = as_phi((2.0, -2.0, -2.0, 2.0))
        uniforms = torch.zeros((8, 2), dtype=DTYPE)
        batch = sample_batch(phi, 8, uniforms=uniforms)
        self.assertEqual(int(batch.k1), 0)
        reward = sampled_reward_gradient(phi, batch)
        barrier = sampled_conditional_gradient(phi, batch)
        fisher = sampled_empirical_fisher(phi, batch)
        self.assertTrue(torch.equal(reward[2:], torch.zeros(2, dtype=DTYPE)))
        self.assertTrue(torch.equal(barrier[2:], torch.zeros(2, dtype=DTYPE)))
        self.assertTrue(torch.equal(fisher[2:, 2:], torch.zeros((2, 2), dtype=DTYPE)))

    def test_npg_methods_share_fisher_inputs_and_only_objective_differs(self):
        reward = SampledFactorialConfig("sampled_npg_reward_only", "uniform", 4, 2)
        barrier = SampledFactorialConfig("sampled_npg_logbarrier_handoff", "uniform", 4, 2)
        self.assertEqual(reward.n_trajectories, barrier.n_trajectories)
        self.assertEqual(reward.damping, barrier.damping)
        self.assertEqual(reward.target_kl, barrier.target_kl)
        self.assertEqual(reward.base_seed, barrier.base_seed)

    def test_checkpoint_roundtrip_reproduces_probabilities(self):
        config = NeuralTrainingConfig(
            environment="Acrobot-v1", method="gpomdp_reward_only", seed=987,
            hidden_sizes=(8, 8), updates=1, batch_steps=1, horizon=500,
        )
        policy, _ = build_seeded_policy(config)
        states = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        before = policy.distribution(states).probs.detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(policy.state_dict(), path)
            restored, _ = build_seeded_policy(config)
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            after = restored.distribution(states).probs.detach()
        self.assertTrue(torch.equal(before, after))

    def test_exact_fixed_and_handoff_match_while_both_are_active(self):
        common = dict(
            initialization="adverse", damping=0.01, updates=4,
            handoff_update=2, record_interval=1,
        )
        handoff_rows, _ = run_exact_one(ExactFactorialConfig(
            method="exact_pg_logbarrier_handoff", **common
        ))
        fixed_rows, _ = run_exact_one(ExactFactorialConfig(
            method="exact_pg_logbarrier_fixed", **common
        ))
        handoff = {row["update"]: row for row in handoff_rows}
        fixed = {row["update"]: row for row in fixed_rows}
        for update in (0, 1, 2):
            for key in ("return", "q", "pi1_good"):
                self.assertAlmostEqual(handoff[update][key], fixed[update][key], places=14)
        self.assertNotAlmostEqual(handoff[3]["return"], fixed[3]["return"], places=14)

    def test_exact_natural_run_logs_realized_and_predicted_kl(self):
        rows, endpoint = run_exact_one(ExactFactorialConfig(
            method="exact_npg_reward_only", initialization="uniform",
            damping=0.01, updates=2, handoff_update=1, record_interval=1,
        ))
        self.assertTrue(endpoint["finite"])
        for row in rows[1:]:
            self.assertIn("realized_kl", row)
            self.assertIn("predicted_kl", row)
            self.assertGreaterEqual(row["realized_kl"], 0.0)

    def test_new_results_use_an_isolated_output_root(self):
        self.assertEqual(
            DEFAULT_ROOT.as_posix(),
            "exploration/results/npg_logbarrier_factorial",
        )
        self.assertNotIn("neural_discrete_log_barrier", DEFAULT_ROOT.parts)


if __name__ == "__main__":
    unittest.main()

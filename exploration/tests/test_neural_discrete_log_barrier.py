from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from exploration.neural_discrete_log_barrier.barrier import (
    analytic_logit_gradient,
    categorical_log_barrier,
)
from exploration.neural_discrete_log_barrier.fisher import (
    action_enumerated_score_matrix,
    fisher_spectrum_from_scores,
    state_bank_hash,
)
from exploration.neural_discrete_log_barrier.training import (
    NeuralTrainingConfig,
    build_seeded_policy,
    checkpoint_updates,
    restore_policy,
    train_policy,
)
from exploration.neural_discrete_log_barrier.baseline import baseline_config


class CategoricalBarrierTest(unittest.TestCase):
    def test_value_gradient_and_finite_difference(self) -> None:
        torch.manual_seed(3)
        logits = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        actual, diagnostics = categorical_log_barrier(logits)
        expected = torch.log_softmax(logits, -1).sum() / 12.0
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-14)
        gradient, = torch.autograd.grad(actual, logits)
        torch.testing.assert_close(gradient, analytic_logit_gradient(logits.detach()), rtol=0.0, atol=1e-13)
        direction = torch.randn_like(logits); direction /= direction.norm()
        epsilon = 1e-6
        plus, _ = categorical_log_barrier(logits.detach() + epsilon * direction)
        minus, _ = categorical_log_barrier(logits.detach() - epsilon * direction)
        finite = (plus - minus) / (2 * epsilon)
        analytic = (gradient * direction).sum()
        torch.testing.assert_close(finite, analytic, rtol=0.0, atol=1e-9)
        self.assertEqual(diagnostics.action_count, 3)

    def test_handoff_is_exactly_zero(self) -> None:
        config = NeuralTrainingConfig(
            environment="CartPole-v1", method="gpomdp_logbarrier_handoff", seed=1,
            beta=2.5, handoff_fraction=0.25, updates=20,
        )
        self.assertEqual(config.handoff_update, 5)
        self.assertEqual(config.beta_at_update(4), 2.5)
        self.assertEqual(config.beta_at_update(5), 0.0)


class FisherTest(unittest.TestCase):
    def test_dense_and_gram_nonzero_spectra_match(self) -> None:
        policy = torch.nn.Linear(2, 2, bias=True).to(torch.float64)
        # Linear has no policy helper; wrap it minimally.
        class Wrapper(torch.nn.Module):
            def __init__(self, layer): super().__init__(); self.layer = layer
            def forward(self, x): return self.layer(x)
        wrapped = Wrapper(policy)
        states = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
        scores, actions = action_enumerated_score_matrix(wrapped, states)
        result = fisher_spectrum_from_scores(scores, state_count=3, action_count=actions)
        dense = scores.T @ scores / 3.0
        expected = torch.linalg.eigvalsh(dense)
        threshold = result.metrics.eigenvalue_threshold
        expected = expected[expected > threshold].flip(0)
        torch.testing.assert_close(result.eigenvalues, expected, rtol=1e-10, atol=1e-12)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(dense).min().detach()), -1e-12)
        self.assertAlmostEqual(float(expected.sum().detach()), result.metrics.trace, places=11)

    def test_state_bank_hash_is_content_based(self) -> None:
        states = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.assertEqual(state_bank_hash(states), state_bank_hash(states.copy()))
        changed = states.copy(); changed[0, 0] += 1
        self.assertNotEqual(state_bank_hash(states), state_bank_hash(changed))


class TrainingArchiveTest(unittest.TestCase):
    def test_complete_episode_update_mode_honors_updates_and_episode_count(self) -> None:
        config = NeuralTrainingConfig(
            environment="CartPole-v1",
            method="gpomdp_reward_only",
            seed=38,
            updates=3,
            batch_steps=10,
            horizon=500,
            center_returns=True,
            collector_mode="complete_episodes_by_update",
            parallel_environments=2,
            episodes_per_update=2,
            evaluation_episodes=1,
            checkpoint_fractions=(0.0, 1.0),
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            summary = train_policy(config, directory)
            self.assertEqual(summary["actual_optimizer_updates"], 3)
            self.assertEqual(summary["actual_training_episodes"], 6)
            self.assertEqual(summary["declared_training_episodes"], 6)
            self.assertEqual(summary["budget_unit"], "complete_training_episodes_and_optimizer_updates")
            rows = (directory / "training.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 4)
            self.assertTrue((directory / "checkpoints/checkpoint_update_000003.pt").exists())

    def test_acrobot_baseline_protocol_is_episode_based(self) -> None:
        config = baseline_config(seed=7, learning_rate=1e-3, updates=300)
        self.assertEqual(config.collector_mode, "complete_episodes_by_update")
        self.assertEqual(config.episodes_per_update, 8)
        self.assertEqual(config.total_training_episodes, 2400)
        self.assertTrue(config.center_returns)
        self.assertFalse(config.normalize_returns)

    def test_complete_episode_mode_uses_exact_step_budget(self) -> None:
        config = NeuralTrainingConfig(
            environment="CartPole-v1",
            method="gpomdp_reward_only",
            seed=39,
            updates=1,
            batch_steps=40,
            horizon=25,
            center_returns=True,
            collector_mode="complete_episodes",
            parallel_environments=2,
            evaluation_episodes=1,
            checkpoint_fractions=(0.0, 1.0),
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            summary = train_policy(config, directory)
            rows = (directory / "training.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(int(rows[-1].split(",")[1]), 40)
            self.assertEqual(summary["total_environment_steps"], 40)
            self.assertTrue(any((directory / "checkpoints").glob("checkpoint_step_*_target_*.pt")))

    def test_paired_initial_weights_and_checkpoint_restoration(self) -> None:
        fixed = NeuralTrainingConfig(
            environment="CartPole-v1", method="gpomdp_logbarrier_fixed", seed=44,
            beta=0.1, updates=1, batch_steps=8, horizon=10, evaluation_episodes=1,
            checkpoint_fractions=(0.0, 1.0),
        )
        handoff = replace(fixed, method="gpomdp_logbarrier_handoff", handoff_fraction=0.5)
        first, first_id = build_seeded_policy(fixed)
        second, second_id = build_seeded_policy(handoff)
        self.assertEqual(first_id, second_id)
        state = torch.randn(5, 4)
        torch.testing.assert_close(first(state), second(state))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            summary = train_policy(fixed, directory)
            restored = restore_policy(fixed, directory / "checkpoints" / "checkpoint_update_000001.pt")
            saved = torch.load(directory / "checkpoints" / "checkpoint_update_000001.pt", weights_only=True)
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, saved[name])
            resumed = train_policy(fixed, directory)
            self.assertEqual(resumed["initial_weight_identifier"], summary["initial_weight_identifier"])
            with self.assertRaises(FileExistsError):
                train_policy(replace(fixed, beta=0.2), directory)
            self.assertTrue(summary["finite"])

    def test_fixed_and_handoff_are_identical_before_switch(self) -> None:
        base = NeuralTrainingConfig(
            environment="CartPole-v1", method="gpomdp_logbarrier_fixed", seed=51,
            beta=0.05, updates=4, batch_steps=12, horizon=12, evaluation_episodes=1,
            checkpoint_fractions=(0.0, 0.5, 1.0),
        )
        temporary = replace(base, method="gpomdp_logbarrier_handoff", handoff_fraction=0.75)
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            train_policy(base, root / "fixed")
            train_policy(temporary, root / "temporary")
            fixed_state = torch.load(root / "fixed/checkpoints/checkpoint_update_000002.pt", weights_only=True)
            temporary_state = torch.load(root / "temporary/checkpoints/checkpoint_update_000002.pt", weights_only=True)
            for name in fixed_state:
                torch.testing.assert_close(fixed_state[name], temporary_state[name], rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()

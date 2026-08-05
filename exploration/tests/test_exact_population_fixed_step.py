"""Mathematics and implementation tests for the exact fixed-step NPG experiment.

These tests verify identities and implementation invariants. They deliberately
do not assert that exact NPG escapes the adverse initialization; that is a
scientific observation reported by the experiment, not a unit-test contract.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import unittest
from pathlib import Path

import torch

from exploration.npg_logbarrier_factorial import exact_population_fixed_step as fixed_step
from exploration.npg_logbarrier_factorial.exact_population_fixed_step import (
    DAMPING_CONTROLS,
    INITIALIZATIONS,
    METHODS,
    PRIMARY_DAMPING,
    PRIMARY_METHOD,
    FixedStepConfig,
    analytic_natural_direction,
    attenuation_factors,
    fixed_step_direction,
    predicted_log_odds_increments,
    realized_log_odds_increments,
    run_one,
    state_value_one,
)
from exploration.tabular_mdp.geometry import (
    enumerated_reduced_fisher,
    pooled_fisher,
    reduced_categorical_fisher,
)
from exploration.tabular_mdp.model import (
    DTYPE,
    TwoStepTrap,
    as_phi,
    probabilities_from_reduced_logits,
    transition_pool_weights,
)


def _interior_policies() -> list[torch.Tensor]:
    """Deterministic spread of interior policies, including tiny q."""
    values = []
    for a in (-6.0, -2.0, -0.5, 0.0, 0.5, 2.0, 6.0):
        for b in (-5.0, -1.0, 0.0, 1.0, 5.0):
            values.append(as_phi((a, b, b, a)))
            values.append(as_phi((b, a, a, b)))
    values.extend(as_phi(value) for value in INITIALIZATIONS.values())
    return values


class ExactReturnTests(unittest.TestCase):
    def test_return_equals_five_trajectory_enumeration(self):
        mdp = TwoStepTrap()
        for phi in _interior_policies():
            paths = mdp.trajectories(phi)
            self.assertEqual(len(paths), 5)
            enumerated = sum(float(path.probability) * path.reward for path in paths)
            self.assertAlmostEqual(enumerated, float(mdp.exact_return(phi)), places=12)

    def test_value_one_matches_definition(self):
        for phi in _interior_policies():
            _, pi1 = probabilities_from_reduced_logits(phi)
            expected = float(pi1[0]) + 0.2 * float(pi1[1])
            self.assertAlmostEqual(expected, float(state_value_one(phi)), places=12)


class RewardGradientTests(unittest.TestCase):
    def test_closed_form_gradient_matches_autograd(self):
        mdp = TwoStepTrap()
        for phi in _interior_policies():
            variable = phi.detach().clone().requires_grad_(True)
            mdp.exact_return(variable).backward()
            closed = mdp.exact_reward_gradient(phi)
            self.assertTrue(torch.allclose(variable.grad, closed, atol=1e-11, rtol=1e-11))

    def test_closed_form_gradient_matches_central_differences(self):
        mdp = TwoStepTrap()
        step = 1e-6
        for phi in _interior_policies()[:12]:
            closed = mdp.exact_reward_gradient(phi)
            for index in range(4):
                shift = torch.zeros(4, dtype=DTYPE)
                shift[index] = step
                numeric = float(
                    (mdp.exact_return(phi + shift) - mdp.exact_return(phi - shift)) / (2 * step)
                )
                self.assertAlmostEqual(numeric, float(closed[index]), places=6)


class PooledFisherTests(unittest.TestCase):
    def test_statewise_fisher_equals_action_score_enumeration(self):
        for phi in _interior_policies():
            pi0, pi1 = probabilities_from_reduced_logits(phi)
            for probabilities in (pi0, pi1):
                self.assertTrue(
                    torch.allclose(
                        reduced_categorical_fisher(probabilities),
                        enumerated_reduced_fisher(probabilities),
                        atol=1e-12,
                        rtol=1e-12,
                    )
                )

    def test_pooled_fisher_block_structure(self):
        for phi in _interior_policies():
            pi0, pi1 = probabilities_from_reduced_logits(phi)
            mu0, mu1 = transition_pool_weights(phi)
            fisher = pooled_fisher(phi)
            self.assertTrue(
                torch.allclose(fisher[:2, :2], mu0 * reduced_categorical_fisher(pi0), atol=1e-12)
            )
            self.assertTrue(
                torch.allclose(fisher[2:, 2:], mu1 * reduced_categorical_fisher(pi1), atol=1e-12)
            )
            self.assertTrue(torch.allclose(fisher[:2, 2:], torch.zeros(2, 2, dtype=DTYPE)))
            self.assertTrue(torch.allclose(fisher[2:, :2], torch.zeros(2, 2, dtype=DTYPE)))

    def test_pooled_fisher_is_symmetric_positive_definite(self):
        for phi in _interior_policies():
            fisher = pooled_fisher(phi)
            self.assertTrue(torch.allclose(fisher, fisher.T, atol=1e-14))
            self.assertGreater(float(torch.linalg.eigvalsh(fisher)[0]), 0.0)


class DirectionIdentityTests(unittest.TestCase):
    def test_undamped_solve_equals_analytic_direction(self):
        mdp = TwoStepTrap()
        for phi in _interior_policies():
            gradient = mdp.exact_reward_gradient(phi)
            result = fixed_step_direction(gradient, pooled_fisher(phi), damping=PRIMARY_DAMPING)
            self.assertTrue(result.valid)
            analytic = analytic_natural_direction(phi)
            self.assertTrue(torch.allclose(result.direction, analytic, atol=1e-9, rtol=1e-9))

    def test_analytic_direction_equals_declared_closed_form(self):
        for phi in _interior_policies():
            pi0, _ = probabilities_from_reduced_logits(phi)
            q = float(pi0[1])
            v1 = float(state_value_one(phi))
            expected = (1.0 + q) * torch.tensor([0.5, v1, 1.0, 0.2], dtype=DTYPE)
            self.assertTrue(
                torch.allclose(analytic_natural_direction(phi), expected, atol=1e-12, rtol=1e-12)
            )

    def test_downstream_direction_survives_vanishing_q(self):
        """As q -> 0+ the downstream block stays near (1 + q) * (1.0, 0.2)."""
        mdp = TwoStepTrap()
        for exponent in range(3, 13):
            q = 10.0 ** (-exponent)
            pi0 = torch.tensor([0.5 * (1 - q), q, 0.5 * (1 - q)], dtype=DTYPE)
            phi = torch.cat(
                (
                    torch.log(pi0[:2]) - torch.log(pi0[2:3]),
                    torch.tensor([0.3, -0.4], dtype=DTYPE),
                )
            )
            gradient = mdp.exact_reward_gradient(phi)
            result = fixed_step_direction(gradient, pooled_fisher(phi), damping=PRIMARY_DAMPING)
            self.assertTrue(result.valid)
            downstream = result.direction[2:]
            expected = (1.0 + q) * torch.tensor([1.0, 0.2], dtype=DTYPE)
            self.assertTrue(torch.allclose(downstream, expected, atol=1e-6, rtol=1e-6))
            self.assertGreater(float(downstream[0]), 0.9)


class LogOddsIdentityTests(unittest.TestCase):
    def test_log_odds_increments_match_prediction(self):
        mdp = TwoStepTrap()
        eta = 0.05
        for phi in _interior_policies():
            gradient = mdp.exact_reward_gradient(phi)
            result = fixed_step_direction(gradient, pooled_fisher(phi), damping=PRIMARY_DAMPING)
            realized = realized_log_odds_increments(phi, phi + eta * result.direction)
            predicted = predicted_log_odds_increments(phi, eta)
            for name in ("explore_safe", "good_medium", "good_reference"):
                self.assertAlmostEqual(
                    predicted[f"predicted_delta_log_odds_{name}"],
                    realized[f"realized_delta_log_odds_{name}"],
                    places=9,
                )

    def test_explore_increment_changes_sign_at_threshold(self):
        mdp = TwoStepTrap()
        eta = 0.05
        for good, sign in ((0.9, 1.0), (0.05, -1.0)):
            pi1 = torch.tensor([good, 0.5 * (1 - good), 0.5 * (1 - good)], dtype=DTYPE)
            phi = torch.cat(
                (
                    torch.tensor([0.1, -0.2], dtype=DTYPE),
                    torch.log(pi1[:2]) - torch.log(pi1[2:3]),
                )
            )
            value1 = float(state_value_one(phi))
            self.assertEqual(sign > 0, value1 > 0.5)
            gradient = mdp.exact_reward_gradient(phi)
            result = fixed_step_direction(gradient, pooled_fisher(phi), damping=PRIMARY_DAMPING)
            realized = realized_log_odds_increments(phi, phi + eta * result.direction)
            self.assertEqual(
                sign > 0, realized["realized_delta_log_odds_explore_safe"] > 0
            )


class DampingTests(unittest.TestCase):
    def test_attenuation_matches_eigendecomposition(self):
        for phi in _interior_policies():
            for damping in DAMPING_CONTROLS:
                eigenvalues = torch.linalg.eigvalsh(pooled_fisher(phi))
                factors = attenuation_factors(phi, damping)
                for index, value in enumerate(eigenvalues):
                    expected = float(value) / (float(value) + damping)
                    self.assertAlmostEqual(factors[f"attenuation_{index}"], expected, places=12)

    def test_statewise_attenuation_matches_pooled_spectrum(self):
        """Pooled eigenvalues are exactly the mu-scaled statewise eigenvalues."""
        for phi in _interior_policies():
            pi0, pi1 = probabilities_from_reduced_logits(phi)
            mu0, mu1 = transition_pool_weights(phi)
            pooled = sorted(float(v) for v in torch.linalg.eigvalsh(pooled_fisher(phi)))
            statewise = sorted(
                [float(mu0 * v) for v in torch.linalg.eigvalsh(reduced_categorical_fisher(pi0))]
                + [float(mu1 * v) for v in torch.linalg.eigvalsh(reduced_categorical_fisher(pi1))]
            )
            for left, right in zip(pooled, statewise):
                self.assertAlmostEqual(left, right, places=12)

    def test_damped_direction_is_attenuated_relative_to_undamped(self):
        mdp = TwoStepTrap()
        phi = as_phi(INITIALIZATIONS["adverse"])
        gradient = mdp.exact_reward_gradient(phi)
        fisher = pooled_fisher(phi)
        undamped = fixed_step_direction(gradient, fisher, damping=PRIMARY_DAMPING)
        for damping in DAMPING_CONTROLS:
            damped = fixed_step_direction(gradient, fisher, damping=damping)
            self.assertTrue(damped.valid)
            self.assertLess(float(damped.direction.norm()), float(undamped.direction.norm()))


class NumericalValidityTests(unittest.TestCase):
    """The identity must hold wherever the run reports the solve as trustworthy."""

    def test_identity_holds_throughout_the_trusted_region(self):
        for initialization in INITIALIZATIONS:
            config = FixedStepConfig(
                method=PRIMARY_METHOD, initialization=initialization, updates=2000
            )
            _, endpoint = run_one(config)
            self.assertLess(
                endpoint["worst_undamped_direction_relative_error"],
                1e-5,
                f"{initialization}: solve deviates from the analytic direction while "
                "still flagged trustworthy",
            )

    def test_trust_flag_latches_once_the_metric_collapses(self):
        config = FixedStepConfig(
            method=PRIMARY_METHOD, initialization="adverse", updates=2000, record_interval=10
        )
        rows, endpoint = run_one(config)
        boundary = endpoint["first_update_fisher_not_positive_definite"]
        if boundary is None:
            self.skipTest("metric never became degenerate in this horizon")
        trusted_updates = [
            row["update"] for row in rows if row.get("solve_trustworthy") is True
        ]
        self.assertTrue(
            all(update <= boundary for update in trusted_updates),
            "solve_trustworthy re-enabled after the degeneracy boundary",
        )

    def test_damped_controls_stay_well_conditioned(self):
        for damping in DAMPING_CONTROLS:
            config = FixedStepConfig(
                method=PRIMARY_METHOD, initialization="adverse",
                damping=damping, updates=500,
            )
            _, endpoint = run_one(config)
            self.assertTrue(endpoint["finite"])


class DeterminismTests(unittest.TestCase):
    def test_repeated_runs_are_identical(self):
        config = FixedStepConfig(
            method=PRIMARY_METHOD, initialization="adverse", updates=120,
            handoff_update=60, record_interval=10,
        )
        first_rows, first_endpoint = run_one(config)
        second_rows, second_endpoint = run_one(config)
        self.assertEqual(first_endpoint, second_endpoint)
        self.assertEqual(len(first_rows), len(second_rows))
        for left, right in zip(first_rows, second_rows):
            self.assertEqual(left, right)

    def test_every_method_runs_and_stays_finite_valued(self):
        for method in METHODS:
            config = FixedStepConfig(
                method=method, initialization="adverse", updates=60, handoff_update=30,
                record_interval=10,
            )
            rows, endpoint = run_one(config)
            self.assertTrue(rows)
            self.assertIn("final_return", endpoint)
            for row in rows:
                for key in ("return", "q", "v1", "pi1_good"):
                    self.assertFalse(
                        torch.isnan(torch.tensor(row[key], dtype=DTYPE)),
                        f"{method} produced NaN in {key}",
                    )


class PurityTests(unittest.TestCase):
    FORBIDDEN_MODULES = (
        "gym", "gymnasium", "vpg.data_collection", "vpg.gpomdp",
        "exploration.sampled_tabular_mdp", "exploration.neural_discrete_log_barrier",
    )
    FORBIDDEN_CALLS = (
        "manual_seed", "rand", "randn", "randint", "multinomial", "sample",
        "Generator", "default_rng", "seed",
    )

    def _module_source(self):
        return Path(inspect.getfile(fixed_step)).read_text(encoding="utf-8")

    def test_exact_runner_imports_no_sampling_or_rollout_module(self):
        tree = ast.parse(self._module_source())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for name in imported:
            for forbidden in self.FORBIDDEN_MODULES:
                self.assertFalse(
                    name == forbidden or name.startswith(forbidden + "."),
                    f"exact runner must not import {name}",
                )

    def test_exact_runner_calls_no_random_or_seeding_function(self):
        tree = ast.parse(self._module_source())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                attribute = node.func.attr if isinstance(node.func, ast.Attribute) else None
                name = node.func.id if isinstance(node.func, ast.Name) else None
                for forbidden in self.FORBIDDEN_CALLS:
                    self.assertNotEqual(attribute, forbidden)
                    self.assertNotEqual(name, forbidden)

    def test_config_exposes_no_seed_field(self):
        self.assertNotIn("seed", FixedStepConfig.__dataclass_fields__)

    def test_primary_method_uses_no_damping_and_no_target_kl(self):
        config = FixedStepConfig(method=PRIMARY_METHOD, initialization="adverse")
        self.assertEqual(config.damping, 0.0)
        self.assertNotIn("target_kl", FixedStepConfig.__dataclass_fields__)

    def test_exact_runner_does_not_import_the_target_kl_step(self):
        tree = ast.parse(self._module_source())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotIn("natural_step", node.module)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    self.assertNotIn("target_kl", alias.name)


class ExistingOutputsUnchangedTests(unittest.TestCase):
    """The new experiment must not touch prior result archives."""

    ROOT = Path("exploration/results/npg_logbarrier_factorial")
    GUARDED = (
        "exact_two_state/exact_checkpoints.csv",
        "exact_two_state/exact_endpoints.csv",
        "exact_two_state/manifest.json",
        "fisher_validation/validation_result.json",
        "sampled_two_state/pilot/manifest.json",
    )

    def test_existing_target_kl_and_sampled_outputs_are_untouched(self):
        present = [path for path in self.GUARDED if (self.ROOT / path).exists()]
        if not present:
            self.skipTest("no existing archives available to guard")
        before = {path: self._digest(self.ROOT / path) for path in present}
        config = FixedStepConfig(
            method=PRIMARY_METHOD, initialization="adverse", updates=50,
            handoff_update=25, record_interval=10,
        )
        run_one(config)
        for path in present:
            self.assertEqual(before[path], self._digest(self.ROOT / path), f"{path} changed")

    def test_exact_two_state_module_is_not_monkeypatched(self):
        from exploration.npg_logbarrier_factorial import exact_two_state, natural_step

        self.assertIs(
            exact_two_state.target_kl_natural_step, natural_step.target_kl_natural_step
        )

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()

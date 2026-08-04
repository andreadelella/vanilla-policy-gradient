"""Deterministic verification for the Step 4 finite-batch estimators."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path

import torch

from exploration.tabular_mdp.geometry import barrier_gradients
from exploration.tabular_mdp.model import DTYPE, TwoStepTrap, probabilities_from_reduced_logits

from .estimators import exact_finite_batch_moments, sampled_conditional_gradient, sampled_empirical_fisher, sampled_reward_gradient
from .experiment import SampledTrainingConfig, train_sampled
from .sampling import SampledBatch, sample_batch


TOLERANCES = {"algebraic": 1e-12, "finite_difference": 1e-7, "eigenvalue": -1e-12}


@dataclass(frozen=True)
class VerificationResult:
    checks: dict[str, bool]
    residuals: dict[str, float]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": self.checks, "residuals": self.residuals}


def _batch_from_paths(paths: tuple[int, ...], mdp: TwoStepTrap) -> SampledBatch:
    trajectories = mdp.trajectories(torch.tensor((0.4, -0.7, -0.2, 0.8), dtype=DTYPE))
    n = len(paths)
    actions = torch.full((n, 2), -1, dtype=torch.int64)
    rewards = torch.zeros((n, 2), dtype=DTYPE)
    mask = torch.zeros((n, 2), dtype=torch.bool)
    for index, path_index in enumerate(paths):
        trajectory = trajectories[path_index]
        length = len(trajectory.actions)
        actions[index, :length] = torch.tensor(trajectory.actions)
        mask[index, :length] = True
        rewards[index, length - 1] = trajectory.reward
    k1 = mask[:, 1].sum()
    return SampledBatch(actions, rewards, mask, k1, torch.tensor(n, dtype=torch.int64) + k1)


def _enumerated_batch_expectations(phi: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    mdp = TwoStepTrap()
    trajectories = mdp.trajectories(phi)
    conditional = torch.zeros(4, dtype=DTYPE)
    fisher = torch.zeros((4, 4), dtype=DTYPE)
    for indices in itertools.product(range(len(trajectories)), repeat=n):
        probability = torch.ones((), dtype=DTYPE)
        for index in indices:
            probability = probability * trajectories[index].probability
        batch = _batch_from_paths(indices, mdp)
        conditional += probability * sampled_conditional_gradient(phi, batch)
        fisher += probability * sampled_empirical_fisher(phi, batch)
    return conditional, fisher


def _autograd_processed_gradient(phi: torch.Tensor, batch: SampledBatch) -> torch.Tensor:
    value = phi.detach().clone().requires_grad_(True)
    zeros = torch.zeros(1, dtype=DTYPE)
    logits0 = torch.cat((value[:2], zeros))
    logits1 = torch.cat((value[2:], zeros))
    logp0 = torch.log_softmax(logits0, dim=-1)[batch.actions[:, 0]]
    safe1 = batch.actions[:, 1].clamp(min=0)
    logp1 = torch.log_softmax(logits1, dim=-1)[safe1]
    returns1 = batch.rewards[:, 1]
    returns0 = batch.rewards[:, 0] + returns1
    returns = torch.stack((returns0, returns1), dim=-1)
    mask = batch.mask.to(DTYPE)
    valid = returns[batch.mask]
    processed = (returns - valid.mean()) / (valid.std() + 1e-8)
    objective = (processed[:, 0] * logp0 + processed[:, 1] * logp1 * mask[:, 1]).mean()
    return torch.autograd.grad(objective, value)[0]


def run_verification() -> VerificationResult:
    phi = torch.tensor((0.4, -0.7, -0.2, 0.8), dtype=DTYPE)
    mdp = TwoStepTrap()
    checks: dict[str, bool] = {}
    residuals: dict[str, float] = {}

    uniforms = torch.tensor(
        ((0.05, 0.1), (0.55, 0.2), (0.95, 0.8), (0.55, 0.9)), dtype=DTYPE
    )
    first = sample_batch(phi, 4, uniforms=uniforms)
    second = sample_batch(phi, 4, uniforms=uniforms)
    checks["sample_shapes"] = first.actions.shape == (4, 2) and first.mask.shape == (4, 2)
    checks["deterministic_replay"] = torch.equal(first.actions, second.actions) and torch.equal(first.rewards, second.rewards)
    checks["pooled_count"] = int(first.m) == 4 + int(first.k1)

    moments = exact_finite_batch_moments(phi, 2)
    enum_conditional, enum_fisher = _enumerated_batch_expectations(phi, 2)
    residuals["conditional_enumeration"] = float(torch.max(torch.abs(enum_conditional - moments.conditional_mean)))
    residuals["fisher_enumeration"] = float(torch.max(torch.abs(enum_fisher - moments.fisher_mean)))
    residuals["reward_unbiased"] = float(torch.max(torch.abs(moments.reward_mean - mdp.exact_reward_gradient(phi))))
    checks["conditional_enumeration"] = residuals["conditional_enumeration"] <= TOLERANCES["algebraic"]
    checks["fisher_enumeration"] = residuals["fisher_enumeration"] <= TOLERANCES["algebraic"]
    checks["reward_unbiased"] = residuals["reward_unbiased"] <= TOLERANCES["algebraic"]
    q = probabilities_from_reduced_logits(phi)[0][1]
    residuals["zero_probability"] = abs(float(moments.zero_s1_probability - (1 - q).pow(2)))
    checks["zero_probability"] = residuals["zero_probability"] <= TOLERANCES["algebraic"]
    checks["finite_ratio_bias"] = abs(float(moments.mu1_mean - q / (1 + q))) > 1e-8
    small_bias = abs(float(exact_finite_batch_moments(phi, 4).mu1_mean - q / (1 + q)))
    large_bias = abs(float(exact_finite_batch_moments(phi, 4096).mu1_mean - q / (1 + q)))
    checks["ratio_convergence"] = large_bias < small_bias

    no_reach_uniforms = torch.tensor(((0.01, 0.2), (0.99, 0.4), (0.02, 0.6)), dtype=DTYPE)
    no_reach = sample_batch(phi, 3, uniforms=no_reach_uniforms)
    no_reach_fisher = sampled_empirical_fisher(phi, no_reach)
    no_reach_barrier = sampled_conditional_gradient(phi, no_reach)
    sign, _ = torch.linalg.slogdet(no_reach_fisher)
    checks["zero_s1_batch"] = int(no_reach.k1) == 0
    checks["zero_s1_fisher_block"] = bool((no_reach_fisher[2:, 2:] == 0).all())
    checks["zero_s1_barrier_block"] = bool((no_reach_barrier[2:] == 0).all())
    checks["rank_deficient_logdet"] = float(sign) == 0.0

    processed_explicit = sampled_reward_gradient(phi, first, center_returns=True, normalize_returns=True)
    processed_autograd = _autograd_processed_gradient(phi, first)
    residuals["processed_gradient"] = float(torch.max(torch.abs(processed_explicit - processed_autograd)))
    checks["processed_gradient"] = residuals["processed_gradient"] <= TOLERANCES["algebraic"]

    exact_cond = barrier_gradients(phi).detached_conditional
    large_moment = exact_finite_batch_moments(phi, 4096)
    residuals["conditional_large_n"] = float(torch.linalg.vector_norm(large_moment.conditional_mean - exact_cond))
    checks["conditional_large_n"] = residuals["conditional_large_n"] < 1e-4

    reward_cfg = SampledTrainingConfig("reward_only", "adverse", 4, 3, beta=0.0, updates=5, record_interval=1)
    sampled_cfg = SampledTrainingConfig("detached_conditional_sampled", "adverse", 4, 3, beta=0.0, updates=5, record_interval=1)
    reward_run = train_sampled(reward_cfg)
    sampled_run = train_sampled(sampled_cfg)
    checks["zero_beta_replay"] = np_array_equal(reward_run.phi, sampled_run.phi)
    checks["finite_short_training"] = bool(reward_run.finite.all() and sampled_run.finite.all())
    return VerificationResult(checks, residuals)


def np_array_equal(left, right) -> bool:
    import numpy as np

    return bool(np.array_equal(left, right, equal_nan=True))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    result = run_verification()
    for name, passed in result.checks.items():
        print(f"{name:32s} {'PASS' if passed else 'FAIL'}")
    print(f"all_passed={result.passed}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

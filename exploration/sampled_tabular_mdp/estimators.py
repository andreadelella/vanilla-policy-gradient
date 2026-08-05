"""Finite-batch reward, barrier, and Fisher estimators with exact moments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from exploration.tabular_mdp.geometry import (
    barrier_gradients,
    pooled_fisher,
    reduced_categorical_fisher,
    reduced_scores,
)
from exploration.tabular_mdp.model import DTYPE, TwoStepTrap, as_phi, probabilities_from_reduced_logits

from .sampling import SampledBatch


def _gather_scores(probabilities: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    scores = reduced_scores(probabilities)
    expanded = scores.unsqueeze(-3).expand(actions.shape + (3, 2))
    index = actions.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(actions.shape + (1, 2))
    return torch.gather(expanded, -2, index).squeeze(-2)


def reward_to_go(batch: SampledBatch) -> torch.Tensor:
    g1 = batch.rewards[..., 1]
    g0 = batch.rewards[..., 0] + g1
    return torch.stack((g0, g1), dim=-1)


def _processed_returns(
    batch: SampledBatch,
    *,
    center_returns: bool,
    normalize_returns: bool,
) -> torch.Tensor:
    returns = reward_to_go(batch)
    mask = batch.mask.to(DTYPE)
    if not center_returns and not normalize_returns:
        return returns
    count = mask.sum(dim=(-2, -1))
    mean = (returns * mask).sum(dim=(-2, -1)) / count
    centered = returns - mean[..., None, None]
    if center_returns:
        returns = centered
    if normalize_returns:
        if bool((count <= 1).any()):
            raise ValueError("return normalization requires at least two valid transitions")
        variance = (centered.square() * mask).sum(dim=(-2, -1)) / (count - 1)
        returns = returns / (torch.sqrt(variance)[..., None, None] + 1e-8)
    return returns


def sampled_reward_gradient(
    phi,
    batch: SampledBatch,
    *,
    center_returns: bool = False,
    normalize_returns: bool = False,
) -> torch.Tensor:
    """Trajectory-mean reward-to-go REINFORCE gradient."""

    value = as_phi(phi)
    pi0, pi1 = probabilities_from_reduced_logits(value)
    action0, action1 = batch.actions[..., 0], batch.actions[..., 1]
    score0 = _gather_scores(pi0, action0)
    score1 = _gather_scores(pi1, action1)
    returns = _processed_returns(
        batch, center_returns=center_returns, normalize_returns=normalize_returns
    )
    reaches = batch.mask[..., 1].to(DTYPE)
    grad0 = (returns[..., 0, None] * score0).mean(dim=-2)
    grad1 = (returns[..., 1, None] * score1 * reaches[..., None]).mean(dim=-2)
    return torch.cat((grad0, grad1), dim=-1)


def sampled_conditional_gradient(phi, batch: SampledBatch) -> torch.Tensor:
    """Pooled-state categorical barrier gradient with random count weights."""

    value = as_phi(phi)
    pi0, pi1 = probabilities_from_reduced_logits(value)
    gb0 = 1.0 - 3.0 * pi0[..., :2]
    gb1 = 1.0 - 3.0 * pi1[..., :2]
    n = torch.full_like(batch.m, batch.n_trajectories, dtype=DTYPE)
    denominator = batch.m.to(DTYPE)
    mu0_hat = n / denominator
    mu1_hat = batch.k1.to(DTYPE) / denominator
    return torch.cat((mu0_hat[..., None] * gb0, mu1_hat[..., None] * gb1), dim=-1)


def sampled_entropy_gradient(phi, batch: SampledBatch) -> torch.Tensor:
    """Pooled-state entropy gradient, weighted exactly as the barrier's is.

    Companion to :func:`sampled_conditional_gradient`. Same random visitation
    weights ``mu0_hat``/``mu1_hat``, so an entropy arm and a barrier arm differ
    only in the per-state force and not in how states are counted -- which is what
    makes the two a controlled comparison of functional form.

    For ``H = -sum_a p_a log p_a`` on reduced logits (the last logit pinned to 0),
    ``dH/dz_j = -p_j (log p_j + H)`` for the free coordinates ``j``. Written out
    analytically rather than via autograd to match the closed form the barrier
    estimator uses; verified against autograd to ~1e-16.

    Note the contrast with the barrier's ``1 - 3 p``, which is monotone in ``p``
    and keeps pushing as ``p -> 0``. This form carries a factor of ``p``, so it
    *vanishes* as ``p -> 0``: it abandons an action once that action is nearly
    dead. That is the whole substantive difference between the two arms.
    """

    value = as_phi(phi)
    pi0, pi1 = probabilities_from_reduced_logits(value)

    def per_state(probabilities: torch.Tensor) -> torch.Tensor:
        entropy = -(probabilities * torch.log(probabilities)).sum(dim=-1, keepdim=True)
        free = probabilities[..., :2]
        return -free * (torch.log(free) + entropy)

    n = torch.full_like(batch.m, batch.n_trajectories, dtype=DTYPE)
    denominator = batch.m.to(DTYPE)
    mu0_hat = n / denominator
    mu1_hat = batch.k1.to(DTYPE) / denominator
    return torch.cat(
        (mu0_hat[..., None] * per_state(pi0), mu1_hat[..., None] * per_state(pi1)),
        dim=-1,
    )


def sampled_empirical_fisher(phi, batch: SampledBatch) -> torch.Tensor:
    """Undamped pooled score outer product, exactly matching S^T S / M."""

    value = as_phi(phi)
    pi0, pi1 = probabilities_from_reduced_logits(value)
    score0 = _gather_scores(pi0, batch.actions[..., 0])
    score1 = _gather_scores(pi1, batch.actions[..., 1])
    reaches = batch.mask[..., 1].to(DTYPE)
    outer0 = torch.einsum("...ni,...nj->...ij", score0, score0)
    outer1 = torch.einsum("...n,...ni,...nj->...ij", reaches, score1, score1)
    shape = outer0.shape[:-2] + (4, 4)
    result = torch.zeros(shape, dtype=DTYPE)
    result[..., :2, :2] = outer0
    result[..., 2:, 2:] = outer1
    return result / batch.m.to(DTYPE)[..., None, None]


@dataclass(frozen=True)
class FiniteBatchMoments:
    n_trajectories: int
    zero_s1_probability: torch.Tensor
    mu1_mean: torch.Tensor
    mu1_variance: torch.Tensor
    conditional_mean: torch.Tensor
    conditional_covariance: torch.Tensor
    reward_mean: torch.Tensor
    reward_covariance: torch.Tensor
    fisher_mean: torch.Tensor


def _binomial_probabilities(n: int, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.arange(n + 1, dtype=DTYPE)
    log_choose = torch.lgamma(torch.tensor(n + 1.0, dtype=DTYPE)) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    log_probability = log_choose + k * torch.log(q) + (n - k) * torch.log1p(-q)
    return k, torch.exp(log_probability)


def _single_trajectory_reward_moments(phi, mdp: TwoStepTrap) -> tuple[torch.Tensor, torch.Tensor]:
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    scores0, scores1 = reduced_scores(pi0), reduced_scores(pi1)
    contributions: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    for action0 in (0, 2):
        reward = mdp.safe_reward if action0 == 0 else 0.0
        contributions.append(torch.cat((reward * scores0[action0], torch.zeros(2, dtype=DTYPE))))
        probabilities.append(pi0[action0])
    for action1, reward in enumerate(mdp.state1_rewards):
        contributions.append(torch.cat((reward * scores0[1], reward * scores1[action1])))
        probabilities.append(pi0[1] * pi1[action1])
    values = torch.stack(contributions)
    weights = torch.stack(probabilities)
    mean = (weights[:, None] * values).sum(dim=0)
    centered = values - mean
    covariance = torch.einsum("a,ai,aj->ij", weights, centered, centered)
    return mean, covariance


def exact_finite_batch_moments(
    phi,
    n_trajectories: int,
    *,
    mdp: TwoStepTrap | None = None,
) -> FiniteBatchMoments:
    """Exact moments under the declared finite pooled-transition convention."""

    if n_trajectories < 1:
        raise ValueError("n_trajectories must be positive")
    mdp = mdp or TwoStepTrap()
    value = as_phi(phi)
    if value.ndim != 1:
        raise ValueError("exact moments require one policy")
    pi0, pi1 = probabilities_from_reduced_logits(value)
    q = pi0[1]
    k, weights = _binomial_probabilities(n_trajectories, q)
    mu1_values = k / (n_trajectories + k)
    mu1_mean = (weights * mu1_values).sum()
    mu1_variance = (weights * (mu1_values - mu1_mean).square()).sum()

    gb0 = 1.0 - 3.0 * pi0[:2]
    gb1 = 1.0 - 3.0 * pi1[:2]
    conditional_values = torch.cat(
        (
            (1.0 - mu1_values)[:, None] * gb0[None, :],
            mu1_values[:, None] * gb1[None, :],
        ),
        dim=-1,
    )
    conditional_mean = (weights[:, None] * conditional_values).sum(dim=0)
    conditional_centered = conditional_values - conditional_mean
    conditional_covariance = torch.einsum(
        "k,ki,kj->ij", weights, conditional_centered, conditional_centered
    )

    score0 = reduced_scores(pi0)
    outer0 = torch.einsum("ai,aj->aij", score0, score0)
    nonreach = (pi0[0] * outer0[0] + pi0[2] * outer0[2]) / (1.0 - q)
    f1 = reduced_categorical_fisher(pi1)
    fisher_values = []
    for count in k:
        count_value = float(count.item())
        numerator0 = count_value * outer0[1] + (n_trajectories - count_value) * nonreach
        denominator = n_trajectories + count_value
        matrix = torch.zeros((4, 4), dtype=DTYPE)
        matrix[:2, :2] = numerator0 / denominator
        matrix[2:, 2:] = count_value * f1 / denominator
        fisher_values.append(matrix)
    fisher_mean = torch.einsum("k,kij->ij", weights, torch.stack(fisher_values))

    reward_mean, reward_covariance_single = _single_trajectory_reward_moments(value, mdp)
    return FiniteBatchMoments(
        n_trajectories,
        (1.0 - q).pow(n_trajectories),
        mu1_mean,
        mu1_variance,
        conditional_mean,
        conditional_covariance,
        reward_mean,
        reward_covariance_single / n_trajectories,
        fisher_mean,
    )


def population_targets(phi) -> dict[str, torch.Tensor]:
    value = as_phi(phi)
    mdp = TwoStepTrap()
    return {
        "reward_gradient": mdp.exact_reward_gradient(value),
        "conditional_gradient": barrier_gradients(value).detached_conditional,
        "pooled_fisher": pooled_fisher(value),
    }

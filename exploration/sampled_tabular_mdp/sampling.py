"""Exact-policy trajectory sampling for the Step 4 two-state MDP."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from exploration.tabular_mdp.model import DTYPE, TwoStepTrap, as_phi, probabilities_from_reduced_logits


@dataclass(frozen=True)
class SampledBatch:
    """A padded batch of two-step trajectories.

    The final axis of ``actions``, ``rewards``, and ``mask`` is time. Invalid
    second actions are stored as ``-1`` and must always be interpreted with the
    mask. Leading dimensions index independent policies/seeds and the
    penultimate dimension indexes trajectories.
    """

    actions: torch.Tensor
    rewards: torch.Tensor
    mask: torch.Tensor
    k1: torch.Tensor
    m: torch.Tensor

    @property
    def n_trajectories(self) -> int:
        return int(self.actions.shape[-2])

    @property
    def batch_shape(self) -> torch.Size:
        return self.actions.shape[:-2]


def _inverse_cdf(probabilities: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    cumulative = probabilities.cumsum(dim=-1)
    return (uniforms.unsqueeze(-1) > cumulative.unsqueeze(-2)).sum(dim=-1).to(torch.int64)


def sample_batch(
    phi,
    n_trajectories: int,
    *,
    generator: torch.Generator | None = None,
    uniforms: torch.Tensor | None = None,
    terminal_noise: torch.Tensor | None = None,
    reward_noise_std: float = 0.0,
    mdp: TwoStepTrap | None = None,
) -> SampledBatch:
    """Sample a padded batch with one optional terminal reward-noise draw.

    Providing ``uniforms`` makes common-random-number comparisons explicit.
    It must have shape ``batch_shape + (N, 2)``. The two entries select the
    actions at s0 and s1; the latter is generated even when s1 is not reached.
    """

    if n_trajectories < 1:
        raise ValueError("n_trajectories must be positive")
    if not torch.isfinite(torch.tensor(reward_noise_std)) or reward_noise_std < 0:
        raise ValueError("reward_noise_std must be finite and nonnegative")
    mdp = mdp or TwoStepTrap()
    value = as_phi(phi)
    pi0, pi1 = probabilities_from_reduced_logits(value)
    batch_shape = value.shape[:-1]
    random_shape = batch_shape + (n_trajectories, 2)
    if uniforms is None:
        uniforms = torch.rand(random_shape, dtype=DTYPE, generator=generator)
    else:
        uniforms = torch.as_tensor(uniforms, dtype=DTYPE, device="cpu")
        if uniforms.shape != random_shape:
            raise ValueError(f"uniforms must have shape {tuple(random_shape)}")
        if not bool(((uniforms >= 0) & (uniforms < 1)).all()):
            raise ValueError("uniforms must lie in [0, 1)")

    action0 = _inverse_cdf(pi0, uniforms[..., 0])
    action1_draw = _inverse_cdf(pi1, uniforms[..., 1])
    reaches = action0 == 1
    action1 = torch.where(reaches, action1_draw, torch.full_like(action1_draw, -1))
    actions = torch.stack((action0, action1), dim=-1)
    mask = torch.stack((torch.ones_like(reaches), reaches), dim=-1)

    if terminal_noise is None:
        if reward_noise_std == 0:
            terminal_noise = torch.zeros(batch_shape + (n_trajectories,), dtype=DTYPE)
        else:
            terminal_noise = reward_noise_std * torch.randn(
                batch_shape + (n_trajectories,), dtype=DTYPE, generator=generator
            )
    else:
        terminal_noise = torch.as_tensor(terminal_noise, dtype=DTYPE, device="cpu")
        if terminal_noise.shape != batch_shape + (n_trajectories,):
            raise ValueError("terminal_noise has an incompatible shape")
        if not bool(torch.isfinite(terminal_noise).all()):
            raise ValueError("terminal_noise must be finite")

    rewards0 = torch.zeros_like(terminal_noise)
    rewards0 = torch.where(action0 == 0, torch.full_like(rewards0, mdp.safe_reward), rewards0)
    rewards1_table = torch.tensor(mdp.state1_rewards, dtype=DTYPE)
    safe_action1 = action1_draw.clamp(min=0)
    rewards1 = torch.where(reaches, rewards1_table[safe_action1], torch.zeros_like(rewards0))
    rewards0 = rewards0 + torch.where(reaches, torch.zeros_like(terminal_noise), terminal_noise)
    rewards1 = rewards1 + torch.where(reaches, terminal_noise, torch.zeros_like(terminal_noise))
    rewards = torch.stack((rewards0, rewards1), dim=-1)

    k1 = reaches.sum(dim=-1)
    m = torch.full_like(k1, n_trajectories) + k1
    return SampledBatch(actions, rewards, mask, k1, m)

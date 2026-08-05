"""Stationary Gaussian categorical-bandit instances for Step 2."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BanditBatch:
    """A batch of paired stationary bandit instances."""

    mean_rewards: torch.Tensor
    optimal_actions: torch.Tensor
    second_best_actions: torch.Tensor

    @property
    def num_runs(self) -> int:
        return int(self.mean_rewards.shape[0])

    @property
    def num_actions(self) -> int:
        return int(self.mean_rewards.shape[1])

    def to(self, device: torch.device | str) -> "BanditBatch":
        """Move the instance batch without changing its values."""
        return BanditBatch(
            mean_rewards=self.mean_rewards.to(device=device),
            optimal_actions=self.optimal_actions.to(device=device),
            second_best_actions=self.second_best_actions.to(device=device),
        )


def generate_paired_bandits(
    num_runs: int,
    num_actions: int,
    seed: int,
    *,
    best_mean: float = 1.0,
    gap: float = 0.1,
    lower_mean: float = -1.0,
) -> BanditBatch:
    """Generate deterministic paired instances with a unique prescribed gap.

    Each row has one arm of mean ``best_mean``, one of mean
    ``best_mean - gap``, and all other means sampled strictly below the
    second-best value.  A separate random permutation hides their positions.
    """
    if num_runs < 1:
        raise ValueError("num_runs must be positive")
    if num_actions < 2:
        raise ValueError("num_actions must be at least two")
    if gap <= 0.0:
        raise ValueError("gap must be positive")
    second_mean = best_mean - gap
    if lower_mean >= second_mean:
        raise ValueError("lower_mean must be below the second-best mean")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    rows = []
    optimal = []
    second_best = []
    for _ in range(num_runs):
        if num_actions > 2:
            # rand is in [0, 1), hence these values are strictly below second_mean.
            remaining = lower_mean + (second_mean - lower_mean) * torch.rand(
                num_actions - 2, generator=generator, dtype=torch.float64
            )
            ordered = torch.cat(
                (
                    torch.tensor([best_mean, second_mean], dtype=torch.float64),
                    remaining,
                )
            )
        else:
            ordered = torch.tensor([best_mean, second_mean], dtype=torch.float64)
        permutation = torch.randperm(num_actions, generator=generator)
        row = ordered[permutation]
        rows.append(row)
        optimal.append(int(torch.argmax(row).item()))
        second_best.append(int(torch.argsort(row, descending=True)[1].item()))

    return BanditBatch(
        mean_rewards=torch.stack(rows),
        optimal_actions=torch.tensor(optimal, dtype=torch.long),
        second_best_actions=torch.tensor(second_best, dtype=torch.long),
    )

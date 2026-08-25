"""Identifiable categorical policies for trajectory-Fisher experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class ReferenceMLPSoftmaxPolicy(nn.Module):
    """Categorical MLP with one fixed reference logit.

    A categorical distribution with ``K`` actions needs only ``K - 1`` logits.
    Fixing the final logit to zero removes the common-logit null direction from
    the policy coordinates without reducing the represented distribution family.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (32,),
    ) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        if action_dim < 2:
            raise ValueError("reference-logit policies require at least two actions")

        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_sizes:
            if hidden_dim < 1:
                raise ValueError("hidden sizes must be positive")
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.Tanh()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim - 1))
        self.net = nn.Sequential(*layers)
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        learned_logits = self.net(state)
        reference_logit = torch.zeros_like(learned_logits[..., :1])
        return torch.cat((learned_logits, reference_logit), dim=-1)

    def distribution(self, state: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(state))

    def sample_action_tensor(self, state: torch.Tensor) -> torch.Tensor:
        return self.distribution(state).sample()

    def sample_action(self, state: torch.Tensor):
        return self.sample_action_tensor(state).cpu().numpy()

    def log_prob(self, state: torch.Tensor, action) -> torch.Tensor:
        action_tensor = torch.as_tensor(
            action,
            dtype=torch.long,
            device=state.device,
        )
        return self.distribution(state).log_prob(action_tensor)

import torch
from torch import nn
from torch.distributions import Categorical


class LinearSoftmaxPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(state_dim, action_dim, bias=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.linear(state)
        return torch.softmax(logits, dim=-1)

    def sample_action(self, state: torch.Tensor) -> int:
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return int(action.item())

    def log_prob(self, state: torch.Tensor, action: int) -> torch.Tensor:
        probs = self.forward(state)
        dist = Categorical(probs)

        action_tensor = torch.tensor(
            action,
            dtype=torch.long,
            device=state.device,
        )

        return dist.log_prob(action_tensor)
    

class MLPSoftmaxPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.net(state)
        return torch.softmax(logits, dim=-1)

    def sample_action(self, state: torch.Tensor) -> int:
        probs = self.forward(state)
        dist = Categorical(probs)
        return int(dist.sample().item())

    def log_prob(self, state: torch.Tensor, action: int) -> torch.Tensor:
        probs = self.forward(state)
        dist = Categorical(probs)

        action_tensor = torch.tensor(
            action,
            dtype=torch.long,
            device=state.device,
        )

        return dist.log_prob(action_tensor)
import torch
from torch import nn
from torch.distributions import Categorical
from torch.distributions import Normal

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
    
class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes=(64, 64),
        init_log_std: float = -0.5,
        learn_std: bool = True,
    ):
        super().__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.mean_net = nn.Sequential(*layers)

        initial_log_std = torch.ones(action_dim) * init_log_std

        if learn_std:
            self.log_std = nn.Parameter(initial_log_std)
        else:
            self.register_buffer("log_std", initial_log_std)

    def forward(self, state: torch.Tensor):
        mean = self.mean_net(state)
        std = torch.exp(self.log_std)
        return mean, std

    def sample_action(self, state: torch.Tensor):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        return action.cpu().numpy()

    def log_prob(self, state: torch.Tensor, action) -> torch.Tensor:
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        action_tensor = torch.tensor(
            action,
            dtype=torch.float32,
            device=state.device,
        )

        return dist.log_prob(action_tensor).sum(dim=-1)
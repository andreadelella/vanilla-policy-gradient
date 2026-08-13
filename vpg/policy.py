import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical, Normal


class LinearSoftmaxPolicy(nn.Module):
    """Linear categorical policy for discrete action spaces."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(state_dim, action_dim, bias=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state)

    def distribution(self, state: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(state))

    def sample_action_tensor(self, state: torch.Tensor) -> torch.Tensor:
        return self.distribution(state).sample()

    def sample_action(self, state: torch.Tensor):
        return self.sample_action_tensor(state).cpu().numpy()

    def log_prob(self, state: torch.Tensor, action) -> torch.Tensor:
        dist = self.distribution(state)
        action_tensor = torch.as_tensor(
            action,
            dtype=torch.long,
            device=state.device,
        )
        return dist.log_prob(action_tensor)


class MLPSoftmaxPolicy(nn.Module):
    """Tanh MLP categorical policy for discrete action spaces."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(32,)):
        super().__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def distribution(self, state: torch.Tensor) -> Categorical:
        # Categorical(logits=...) uses log_softmax internally, which is more
        # numerically stable than softmax -> log(probs) for large logit spreads.
        return Categorical(logits=self.forward(state))

    def sample_action_tensor(self, state: torch.Tensor) -> torch.Tensor:
        return self.distribution(state).sample()

    def sample_action(self, state: torch.Tensor):
        return self.sample_action_tensor(state).cpu().numpy()

    def log_prob(self, state: torch.Tensor, action) -> torch.Tensor:
        dist = self.distribution(state)
        action_tensor = torch.as_tensor(
            action,
            dtype=torch.long,
            device=state.device,
        )
        return dist.log_prob(action_tensor)


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy with an MLP mean and state-independent log std."""

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

        initial_log_std = torch.full(
            (action_dim,),
            float(init_log_std),
            dtype=torch.float32,
        )

        if learn_std:
            self.log_std = nn.Parameter(initial_log_std)
        else:
            # Fixed exploration noise; kept as a buffer so it moves with the model.
            self.register_buffer("log_std", initial_log_std)

    def forward(self, state: torch.Tensor):
        mean = self.mean_net(state)
        std = torch.exp(self.log_std)
        return mean, std

    def distribution(self, state: torch.Tensor) -> Normal:
        mean, std = self.forward(state)
        return Normal(mean, std)

    def sample_action_tensor(self, state: torch.Tensor) -> torch.Tensor:
        return self.distribution(state).sample()

    def sample_action(self, state: torch.Tensor):
        return self.sample_action_tensor(state).cpu().numpy()

    def log_prob(self, state: torch.Tensor, action) -> torch.Tensor:
        dist = self.distribution(state)
        action_tensor = torch.as_tensor(
            action,
            dtype=state.dtype,
            device=state.device,
        )
        # Sum over action dimensions: assumes each coordinate is an independent Gaussian.
        return dist.log_prob(action_tensor).sum(dim=-1)


def build_policy(cfg: dict, env) -> nn.Module:
    """Construct the policy that matches `cfg` for `env`'s observation/action spaces.

    Shared by training (train.py) and checkpoint replay (video.record_checkpoint_video)
    so both build the exact same architecture from a config dict.
    """
    state_dim = env.observation_space.shape[0]

    if isinstance(env.action_space, gym.spaces.Box):
        return GaussianPolicy(
            state_dim=state_dim,
            action_dim=env.action_space.shape[0],
            hidden_sizes=tuple(cfg["hidden_sizes"]),
            init_log_std=cfg.get("init_log_std", -0.5),
            learn_std=cfg.get("learn_std", True),
        )

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        if cfg.get("policy", "mlp") == "linear":
            return LinearSoftmaxPolicy(state_dim=state_dim, action_dim=action_dim)
        return MLPSoftmaxPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=tuple(cfg["hidden_sizes"]),
        )

    raise ValueError(f"Unsupported action space: {env.action_space}")

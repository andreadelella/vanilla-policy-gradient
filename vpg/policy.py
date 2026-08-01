import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical, Normal

# Comment key: M says what the function does. A says how it works and why.


class LinearSoftmaxPolicy(nn.Module):
    # Linear policy for discrete action spaces. Useful as a simple baseline;
    # expressiveness is limited to linear state-action mappings.
    #M: Creates a simple linear policy for discrete actions.
    #A: Uses one linear layer that maps a state directly to one score per action.
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(state_dim, action_dim, bias=False)

    #M: Calculates one raw score for every possible action.
    #A: Passes the state through the linear layer without applying softmax.
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state)  # raw logits

    #M: Converts the action scores into a discrete probability distribution.
    #A: Categorical normalizes the logits so actions can be sampled or evaluated.
    def distribution(self, state: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(state))

    #M: Samples a discrete action and keeps it as a PyTorch tensor.
    #A: Builds the categorical distribution and asks it to sample.
    def sample_action_tensor(self, state: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(state)
        return dist.sample()

    #M: Samples a discrete action and returns it as a NumPy value.
    #A: Samples a tensor, moves it to CPU, and converts it for Gymnasium.
    def sample_action(self, state: torch.Tensor):
        action = self.sample_action_tensor(state)
        return action.cpu().numpy()

    #M: Returns the log probability of a chosen action at a given state.
    #A: Converts the action to an integer tensor and evaluates the categorical distribution.
    def log_prob(self, state: torch.Tensor, action) -> torch.Tensor:
        dist = self.distribution(state)

        action_tensor = torch.as_tensor(
            action,
            dtype=torch.long,
            device=state.device,
        )

        return dist.log_prob(action_tensor)


class MLPSoftmaxPolicy(nn.Module):
    # MLP policy for discrete action spaces (e.g. CartPole).
    # Outputs a categorical distribution over actions via a softmax over logits.
    #M: Creates a neural-network policy for discrete actions.
    #A: Builds hidden linear layers with Tanh, followed by one score per action.
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

    #M: Calculates one raw score for every possible action.
    #A: Passes the state through the full neural network.
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)  # raw logits

    #M: Converts the network's action scores into a discrete distribution.
    #A: Categorical handles normalization directly from the logits.
    def distribution(self, state: torch.Tensor) -> Categorical:
        # Categorical(logits=...) uses log_softmax internally, which is more
        # numerically stable than softmax -> log(probs) for large logit spreads.
        return Categorical(logits=self.forward(state))

    #M: Samples a discrete action and keeps it as a PyTorch tensor.
    #A: Builds the categorical distribution and asks it to sample.
    def sample_action_tensor(self, state: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(state)
        return dist.sample()

    #M: Samples a discrete action and returns it as a NumPy value.
    #A: Samples a tensor, moves it to CPU, and converts it for Gymnasium.
    def sample_action(self, state: torch.Tensor):
        action = self.sample_action_tensor(state)
        return action.cpu().numpy()

    #M: Returns the log probability of a chosen action at a given state.
    #A: Converts the action to an integer tensor and evaluates the categorical distribution.
    def log_prob(self, state: torch.Tensor, action) -> torch.Tensor:
        dist = self.distribution(state)

        action_tensor = torch.as_tensor(
            action,
            dtype=torch.long,
            device=state.device,
        )

        return dist.log_prob(action_tensor)


class GaussianPolicy(nn.Module):
    # Used for continuous action spaces (e.g. MuJoCo locomotion tasks).
    # The mean is state-dependent (output of the MLP); log_std is state-independent
    # (a single learnable vector shared across all states). This is the standard
    # parameterization for simple policy gradient implementations.
    #M: Creates a neural-network policy for continuous actions.
    #A: Uses a network for the action mean and one shared log standard deviation vector.
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

    #M: Calculates the Gaussian mean and standard deviation for a state.
    #A: Uses the network for the mean and exponentiates log_std to keep std positive.
    def forward(self, state: torch.Tensor):
        mean = self.mean_net(state)
        std = torch.exp(self.log_std)
        return mean, std

    #M: Builds the continuous action distribution for a state.
    #A: Uses independent Normal distributions with the calculated mean and standard deviation.
    def distribution(self, state: torch.Tensor) -> Normal:
        mean, std = self.forward(state)
        return Normal(mean, std)

    #M: Samples a continuous action and keeps it as a PyTorch tensor.
    #A: Builds the Normal distribution and asks it to sample.
    def sample_action_tensor(self, state: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(state)
        return dist.sample()

    #M: Samples a continuous action and returns it as a NumPy array.
    #A: Samples a tensor, moves it to CPU, and converts it for Gymnasium.
    def sample_action(self, state: torch.Tensor):
        action = self.sample_action_tensor(state)
        return action.cpu().numpy()

    #M: Returns the joint log probability of a continuous action.
    #A: Finds each action coordinate's log probability and adds them together.
    def log_prob(self, state: torch.Tensor, action) -> torch.Tensor:
        dist = self.distribution(state)

        action_tensor = torch.as_tensor(
            action,
            dtype=torch.float32,
            device=state.device,
        )

        # Sum over action dimensions: assumes each coordinate is an independent Gaussian.
        return dist.log_prob(action_tensor).sum(dim=-1)


#M: Creates the correct policy type for an environment and configuration.
#A: Uses Gaussian for continuous actions and linear or MLP softmax for discrete actions.
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

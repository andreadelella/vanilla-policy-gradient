from abc import ABC, abstractmethod

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

# This helper is for single-environment experiments.
# The main training loop uses AsyncVectorEnv instead.


class BaseEnv(ABC):
    """Small single-environment wrapper used by auxiliary experiments."""

    def __init__(self, env_id: str, horizon: int = 0, gamma: float = 0.99):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")

        self.env_id = env_id
        self.horizon = horizon
        self.gamma = gamma
        self.time = 0

        self.env = gym.make(env_id)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.state_dim = self.observation_space.shape[0]

        if isinstance(self.action_space, Box):
            self.continuous_env = True
            self.action_dim = self.action_space.shape[0]
        elif isinstance(self.action_space, Discrete):
            self.continuous_env = False
            self.action_dim = self.action_space.n
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}")

    def reset(self, seed=None):
        self.time = 0
        return self.env.reset(seed=seed)

    def step(self, action):
        # The caller must clip continuous actions when needed.
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.time += 1

        if self.horizon > 0 and self.time >= self.horizon:
            truncated = True

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def sample_action(self):
        return self.action_space.sample()

    def sample_state(self):
        return self.observation_space.sample()

    @abstractmethod
    def set_state(self, state):
        pass


class GymEnvWrapper(BaseEnv):
    def set_state(self, state):
        raise NotImplementedError(
            "Generic Gymnasium environments do not expose a standard set_state method."
        )

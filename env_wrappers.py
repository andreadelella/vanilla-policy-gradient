from abc import ABC, abstractmethod

import gymnasium as gym
from gymnasium.spaces import Box, Discrete


class BaseEnv(ABC):
    def __init__(self, env_id: str, horizon: int = 0, gamma: float = 0.99, clip: bool = True):
        assert 0.0 <= gamma <= 1.0

        self.env_id = env_id
        self.horizon = horizon
        self.gamma = gamma
        self.clip = clip
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
        if self.clip and isinstance(self.action_space, Box):
            action = action.clip(self.action_space.low, self.action_space.high)

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
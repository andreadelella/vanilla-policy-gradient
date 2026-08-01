from abc import ABC, abstractmethod

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

# Comment key: M says what the function does. A says how it works and why.

# This helper is for single-environment experiments.
# The main training loop uses AsyncVectorEnv instead.


class BaseEnv(ABC):
    #M: Creates a simple wrapper around one Gymnasium environment.
    #A: Stores its spaces, dimensions, horizon, and discount setting.
    def __init__(self, env_id: str, horizon: int = 0, gamma: float = 0.99):
        assert 0.0 <= gamma <= 1.0

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

    #M: Starts a new episode.
    #A: Resets the step counter and passes the optional seed to Gymnasium.
    def reset(self, seed=None):
        self.time = 0
        return self.env.reset(seed=seed)

    #M: Sends one action to the environment.
    #A: Tracks elapsed steps and marks the episode truncated at the chosen horizon.
    def step(self, action):
        # The caller must clip continuous actions when needed.
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.time += 1

        if self.horizon > 0 and self.time >= self.horizon:
            truncated = True

        return obs, reward, terminated, truncated, info

    #M: Returns the environment's current rendered frame.
    #A: Passes the request directly to Gymnasium.
    def render(self):
        return self.env.render()

    #M: Releases the environment's resources.
    #A: Calls Gymnasium's close method.
    def close(self):
        self.env.close()

    #M: Samples a random valid action.
    #A: Uses the environment's action space.
    def sample_action(self):
        return self.action_space.sample()

    #M: Samples an example observation.
    #A: Uses the environment's observation space.
    def sample_state(self):
        return self.observation_space.sample()

    #M: Defines the interface for manually replacing environment state.
    #A: Leaves the implementation to a specific environment wrapper.
    @abstractmethod
    def set_state(self, state):
        pass


class GymEnvWrapper(BaseEnv):
    #M: Handles requests to manually set a generic Gymnasium state.
    #A: Raises an error because Gymnasium has no common set_state method.
    def set_state(self, state):
        raise NotImplementedError(
            "Generic Gymnasium environments do not expose a standard set_state method."
        )

import unittest

import gymnasium as gym
import numpy as np
import torch

from vpg.data_collection import collect_parallel_trajectories


class OneStepVectorEnv:
    def __init__(self):
        self.num_envs = 1
        self.single_action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )
        self.received_actions = []

    def reset(self):
        return np.array([[0.0]], dtype=np.float32), {}

    def step(self, actions):
        self.received_actions.append(np.array(actions, copy=True))
        return (
            np.array([[1.0]], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([True]),
            np.array([False]),
            {},
        )


class OutOfBoundsPolicy:
    def sample_action(self, states):
        return np.full((states.shape[0], 1), 2.5, dtype=np.float32)


class RecordingFloat64Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.observation_dtype = None

    def sample_action(self, states):
        self.observation_dtype = states.dtype
        return np.zeros((states.shape[0], 1), dtype=np.float64)


class ActionClippingTests(unittest.TestCase):
    def test_raw_action_is_stored_and_clipped_action_is_executed(self):
        env = OneStepVectorEnv()
        trajectory = collect_parallel_trajectories(
            env,
            OutOfBoundsPolicy(),
            clip_actions=True,
        )[0]

        np.testing.assert_array_equal(env.received_actions[0], [[1.0]])
        np.testing.assert_array_equal(trajectory.actions[0], [2.5])
        np.testing.assert_array_equal(trajectory.executed_actions[0], [1.0])

    def test_observations_follow_policy_parameter_dtype(self):
        policy = RecordingFloat64Policy()

        collect_parallel_trajectories(OneStepVectorEnv(), policy)

        self.assertEqual(policy.observation_dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()

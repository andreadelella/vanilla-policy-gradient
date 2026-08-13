"""Persistence helpers for training configurations and reward curves."""

import json
import os

import numpy as np


def load_config(path="config.json") -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_training_rewards(
    output_dir,
    rewards,
    seeds,
    filename="training_rewards.npz",
):
    """Save labeled per-seed training curves to a NumPy archive."""

    reward_array = np.asarray(rewards, dtype=np.float32)
    seed_array = np.asarray(seeds, dtype=np.int64)
    if reward_array.shape[0] != seed_array.shape[0]:
        raise ValueError("one seed id is required per reward curve")

    path = os.path.join(output_dir, filename)
    np.savez(path, rewards=reward_array, seeds=seed_array)
    return path

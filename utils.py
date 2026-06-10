# utils.py

import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch


def plot_training_curves(training_rewards, evaluation_rewards, eval_every, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(training_rewards)
    plt.xlabel("Iteration")
    plt.ylabel("Average training return")
    plt.title("Training reward")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_rewards.png"), dpi=300)
    plt.close()

    eval_iterations = np.arange(len(evaluation_rewards)) * eval_every

    plt.figure()
    plt.plot(eval_iterations, evaluation_rewards)
    plt.xlabel("Iteration")
    plt.ylabel("Average evaluation return")
    plt.title("Evaluation reward")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "evaluation_rewards.png"), dpi=300)
    plt.close()


def record_policy_video(
    env_id,
    policy,
    video_dir="videos",
    episode_trigger=lambda episode_id: True,
    seed=23,
):
    os.makedirs(video_dir, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")

    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=episode_trigger,
        name_prefix="policy",
    )

    state, _ = env.reset(seed=seed)
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            action = policy.sample_action(state_tensor)

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
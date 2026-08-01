import os
import numpy as np
import gymnasium as gym
import imageio
import torch

# Comment key: M says what the function does. A says how it works and why.


#M: Runs a policy in an environment and saves rendered episodes as MP4 videos.
#A: Samples actions without gradients, clips continuous actions, and stores every rendered frame.
def record_policy_video(
    env_id,
    policy,
    video_dir="videos",
    seed=23,
    n_episodes=3,
    fps=30,
):
    """Record n_episodes of the policy and save each as an MP4."""
    os.makedirs(video_dir, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")  # no TimeLimit: natural episode length

    for ep in range(n_episodes):
        frames = []
        state, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            frames.append(env.render())

            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action = policy.sample_action(state_tensor)

            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        frames.append(env.render())  # final frame

        video_path = os.path.join(video_dir, f"policy-episode-{ep}.mp4")
        imageio.mimwrite(video_path, frames, fps=fps)
        print(f"Saved video ({len(frames)} frames, {len(frames) / fps:.1f}s): {video_path}")

    env.close()


#M: Loads a saved policy checkpoint and records new videos from it.
#A: Rebuilds the policy from config.json, loads its weights, and calls record_policy_video.
def record_checkpoint_video(
    run_dir,
    checkpoint_name="best.pt",
    n_episodes=3,
    fps=30,
    seed=None,
):
    """Rebuild the policy from a run directory and record videos from a checkpoint.

    `run_dir` must contain `config.json` and `policy/<checkpoint_name>`
    (as produced by run.py with --save_checkpoints 1). The checkpoint stores
    only weights, so config.json is used to reconstruct the matching architecture.
    Videos are written to `<run_dir>/videos/`.
    """
    import json

    # Local import avoids a circular import (policy.py has no dependency on video).
    from vpg.policy import build_policy

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    checkpoint_path = os.path.join(run_dir, "policy", checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")

    # A throwaway env just to read the observation/action spaces for the architecture.
    probe_env = gym.make(cfg["env_id"])
    policy = build_policy(cfg, probe_env)
    probe_env.close()

    # Weights are saved on CPU; replay runs on CPU (see record_policy_video).
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()

    if seed is None:
        seed = cfg.get("seed", 0) + 20_000

    record_policy_video(
        env_id=cfg["env_id"],
        policy=policy,
        video_dir=os.path.join(run_dir, "videos"),
        seed=seed,
        n_episodes=n_episodes,
        fps=fps,
    )

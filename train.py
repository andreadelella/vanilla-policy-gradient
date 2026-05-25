import gymnasium as gym
import numpy as np
import torch

from data_collection import collect_trajectories, collect_parallel_trajectories
from policy import GaussianPolicy
# Future-return version
from gpomdp import compute_gpomdp_loss


ENV_ID = "Pendulum-v1"

SEED = 23
N_ITERATIONS = 1000
N_TRAJECTORIES = 2 #Use for non-parallale computation
N_ENVS = 8

BETA = 0.99

LR = 1e-4

EVAL_EVERY = 5

HIDDEN_STATES=(64,64)
INIT_LOG_STD=-1
LEARN_STD=True


def evaluate_policy(env, policy, n_episodes=5, seed=None):
    episode_rewards = []

    for i in range(n_episodes):
        eval_seed = None if seed is None else seed + i

        state, _ = env.reset(seed=eval_seed)
        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)

            with torch.no_grad():
                action = policy.sample_action(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

    return float(np.mean(episode_rewards))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = gym.make(ENV_ID)
    eval_env = gym.make(ENV_ID)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = GaussianPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=HIDDEN_STATES,
        init_log_std=INIT_LOG_STD,
        learn_std=LEARN_STD
    )

    training_rewards = []
    evaluation_rewards = []

    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    for iteration in range(N_ITERATIONS):
        trajectories = collect_parallel_trajectories(
            env_id=ENV_ID,
            policy=policy,
            n_envs=N_ENVS,
            seed=SEED + iteration * N_ENVS,
        )

        batch_reward = np.mean([
            sum(traj.rewards) for traj in trajectories
        ])

        debug = iteration == 0
        optimizer.zero_grad()

        loss = compute_gpomdp_loss(
            policy=policy,
            trajectories=trajectories,
            beta=BETA,
            debug=debug,
        )

        loss.backward()
        optimizer.step()

        training_rewards.append(batch_reward)

        if iteration % EVAL_EVERY == 0:
            eval_reward = evaluate_policy(
                env=eval_env,
                policy=policy,
                n_episodes=5,
                seed=SEED + 10_000 + iteration,
            )

            evaluation_rewards.append(eval_reward)

            print(
                f"Iteration {iteration:04d} | "
                f"train reward: {batch_reward:.2f} | "
                f"eval reward: {eval_reward:.2f}"
            )

    env.close()
    eval_env.close()

    return policy, training_rewards, evaluation_rewards


if __name__ == "__main__":
    main()
import gymnasium as gym
import numpy as np
import torch

from data_collection import collect_trajectories
from policy import LinearSoftmaxPolicy
# Future-return version
from gpomdp import compute_gpomdp_gradient, apply_gradient_step

# Eligibility traces version
#from gpomdp_elig_traces import compute_gpomdp_textbook_gradient as compute_gpomdp_gradient
#from gpomdp_elig_traces import apply_gradient_step


ENV_ID = "CartPole-v1"

SEED = 23
N_ITERATIONS = 5000
N_TRAJECTORIES = 10

BETA = 0.99
LR = 1e-3

EVAL_EVERY = 100


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
    action_dim = env.action_space.n

    policy = LinearSoftmaxPolicy(state_dim, action_dim)

    training_rewards = []
    evaluation_rewards = []

    for iteration in range(N_ITERATIONS):
        trajectories = collect_trajectories(
            env=env,
            policy=policy,
            n_trajectories=N_TRAJECTORIES,
            seed=SEED + iteration * N_TRAJECTORIES,
        )

        batch_reward = np.mean([
            sum(traj.rewards) for traj in trajectories
        ])

        debug = iteration == 0
        gradients = compute_gpomdp_gradient(
            policy=policy,
            trajectories=trajectories,
            beta=BETA,
            debug=debug
        )

        apply_gradient_step(
            policy=policy,
            gradients=gradients,
            lr=LR,
        )

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
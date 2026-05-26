import numpy as np
import torch


def compute_discounted_returns(rewards, gamma: float, debug: bool = False):
    returns = []
    G = 0.0

    for t, r in reversed(list(enumerate(rewards))):
        G = r + gamma * G
        returns.insert(0, G)

        if debug:
            print(
                f"[Return] t={t}: "
                f"G_t = r_t + gamma * G_next = {r:.3f} + {gamma:.3f} * ... = {G:.3f}"
            )

    return torch.tensor(returns, dtype=torch.float32)


def compute_gpomdp_loss(policy, trajectories, gamma: float, debug: bool = False):
    """
    Batched future-form GPOMDP objective.

    Estimator:

        g = (1/N) sum_i sum_t G_{i,t}^gamma grad_theta log pi_theta(a_{i,t} | s_{i,t})

    We minimize the negative objective:

        L(theta) = -(1/N) sum_i sum_t G_{i,t}^gamma log pi_theta(a_{i,t} | s_{i,t})

    Then PyTorch computes:

        grad L = -g

    Therefore optimizer.step() performs gradient descent on L,
    equivalent to gradient ascent on the GPOMDP objective.
    """

    total_loss = 0.0

    if debug:
        print("\n========== GPOMDP LOSS ==========")
        print("Objective:")
        print("J_hat(theta) = (1/N) * sum_i sum_t G_t^gamma * log pi_theta(a_t | s_t)")
        print("Loss:")
        print("L(theta) = -J_hat(theta)")
        print("=========================================\n")

    for traj_idx, traj in enumerate(trajectories):
        states = torch.tensor(
            np.array(traj.states),
            dtype=torch.float32,
        )

        actions = torch.tensor(
            np.array(traj.actions),
            dtype=torch.float32,
        )

        returns = compute_discounted_returns(
            traj.rewards,
            gamma=gamma,
            debug=debug and traj_idx == 0,
        )

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = policy.log_prob(states, actions)

        trajectory_objective = (returns * log_probs).sum()
        trajectory_loss = -trajectory_objective

        total_loss = total_loss + trajectory_loss

        if debug and traj_idx == 0:
            print(f"\n--- Trajectory {traj_idx} ---")
            print(f"states shape  = {tuple(states.shape)}")
            print(f"actions shape = {tuple(actions.shape)}")
            print(f"returns shape = {tuple(returns.shape)}")
            print(f"log_probs shape = {tuple(log_probs.shape)}")
            print(f"first returns = {returns[:5]}")
            print(f"first log_probs = {log_probs[:5]}")
            print(f"trajectory objective = {trajectory_objective.item():.6f}")
            print(f"trajectory loss = {trajectory_loss.item():.6f}")

    loss = total_loss / len(trajectories)

    if debug:
        print("\n--- Final averaging step ---")
        print(f"Number of trajectories N = {len(trajectories)}")
        print("loss = total_loss / N")
        print(f"Final GPOMDP loss = {loss.item():.6f}")
        print("========== END GPOMDP DEBUG ==========\n")

    return loss
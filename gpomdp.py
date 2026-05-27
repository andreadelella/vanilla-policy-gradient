import numpy as np
import torch


def compute_discounted_returns_matrix(
    rewards: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    rewards: [N, T]

    returns[n, t] = r[n, t] + gamma * r[n, t+1] + ...
    """

    returns = torch.zeros_like(rewards)
    running_return = torch.zeros(
        rewards.shape[0],
        dtype=rewards.dtype,
        device=rewards.device,
    )

    for t in reversed(range(rewards.shape[1])):
        running_return = rewards[:, t] + gamma * running_return
        returns[:, t] = running_return

    return returns


def trajectories_to_tensors(trajectories):
    """
    Converts list[Trajectory] into padded tensors.

    states:  [N, T_max, state_dim]
    actions: [N, T_max, action_dim]
    rewards: [N, T_max]
    mask:    [N, T_max]
    """

    n_trajectories = len(trajectories)
    max_len = max(len(traj.rewards) for traj in trajectories)

    state_dim = np.asarray(trajectories[0].states[0]).shape[-1]
    action_shape = np.asarray(trajectories[0].actions[0]).shape

    states = np.zeros(
        (n_trajectories, max_len, state_dim),
        dtype=np.float32,
    )

    actions = np.zeros(
        (n_trajectories, max_len, *action_shape),
        dtype=np.float32,
    )

    rewards = np.zeros(
        (n_trajectories, max_len),
        dtype=np.float32,
    )

    mask = np.zeros(
        (n_trajectories, max_len),
        dtype=np.float32,
    )

    for i, traj in enumerate(trajectories):
        T = len(traj.rewards)

        states[i, :T] = np.asarray(traj.states, dtype=np.float32)
        actions[i, :T] = np.asarray(traj.actions, dtype=np.float32)
        rewards[i, :T] = np.asarray(traj.rewards, dtype=np.float32)
        mask[i, :T] = 1.0

    return (
        torch.as_tensor(states, dtype=torch.float32),
        torch.as_tensor(actions, dtype=torch.float32),
        torch.as_tensor(rewards, dtype=torch.float32),
        torch.as_tensor(mask, dtype=torch.float32),
    )


def compute_gpomdp_loss(
    policy,
    trajectories,
    gamma: float,
    normalize_returns: bool = True,
    debug: bool = False,
):
    """
        L(theta) = - mean_i sum_t G_{i,t} log pi_theta(a_{i,t}|s_{i,t})
    """

    states, actions, rewards, mask = trajectories_to_tensors(trajectories)

    returns = compute_discounted_returns_matrix(
        rewards=rewards,
        gamma=gamma,
    )

    if normalize_returns:
        valid_returns = returns[mask.bool()]
        returns = (returns - valid_returns.mean()) / (
            valid_returns.std() + 1e-8
        )

    n_trajectories, max_len = rewards.shape

    flat_states = states.reshape(n_trajectories * max_len, -1)

    if actions.ndim == 2:
        # Discrete actions: [N, T] -> [N*T]
        flat_actions = actions.reshape(n_trajectories * max_len).long()
    else:
        # Continuous actions: [N, T, action_dim] -> [N*T, action_dim]
        flat_actions = actions.reshape(n_trajectories * max_len, -1)
    
    log_probs = policy.log_prob(flat_states, flat_actions)
    log_probs = log_probs.reshape(n_trajectories, max_len)

    objective = (returns * log_probs * mask).sum(dim=1).mean()
    loss = -objective

    if debug:
        print("\n========== GPOMDP LOSS ==========")
        print(f"states shape  = {tuple(states.shape)}")
        print(f"actions shape = {tuple(actions.shape)}")
        print(f"rewards shape = {tuple(rewards.shape)}")
        print(f"returns shape = {tuple(returns.shape)}")
        print(f"mask shape    = {tuple(mask.shape)}")
        print(f"log_probs shape = {tuple(log_probs.shape)}")
        print(f"valid steps = {int(mask.sum().item())}")
        print(f"objective = {objective.item():.6f}")
        print(f"loss = {loss.item():.6f}")
        print("========== END GPOMDP DEBUG ==========\n")

    return loss
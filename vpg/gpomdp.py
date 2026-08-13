import math

import numpy as np
import torch
from torch.distributions import Categorical, Normal
from torch.func import functional_call, grad as _fgrad, vmap

# GPOMDP:
#   ∇J(θ) ≈ (1/N) Σ_i Σ_t G_{i,t} · ∇ log π_θ(a_{i,t} | s_{i,t})
# where G_{i,t} = Σ_{k>=t} γ^{k-t} r_{i,k} is the discounted return from step t.


def compute_discounted_returns_matrix(
    rewards: torch.Tensor,
    gamma: float,
    implementation: str = "recursive",
) -> torch.Tensor:
    """Compute reward-to-go for a ``[trajectories, timesteps]`` reward batch."""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")

    if implementation == "recursive":
        returns = torch.empty_like(rewards)
        running = torch.zeros(
            rewards.shape[0], dtype=rewards.dtype, device=rewards.device
        )
        for timestep in range(rewards.shape[1] - 1, -1, -1):
            running = rewards[:, timestep] + gamma * running
            returns[:, timestep] = running
        return returns

    if implementation == "vectorized":
        if gamma == 0.0 or rewards.shape[1] <= 1:
            return rewards.clone()

        last_exponent = (rewards.shape[1] - 1) * math.log(gamma)
        if last_exponent < math.log(torch.finfo(torch.float64).tiny):
            raise ValueError(
                "The vectorized discounted-return formulation is numerically unsafe "
                f"for gamma={gamma} and horizon={rewards.shape[1]}; "
                "use implementation='recursive'."
            )

        # MPS lacks float64 support, so this guarded backend computes on CPU and
        # then returns the result to the caller's original device.
        orig_device = rewards.device
        work = rewards.detach().cpu().to(torch.float64)
        powers = gamma ** torch.arange(rewards.shape[1], dtype=torch.float64)
        scaled = work * powers.unsqueeze(0)
        returns = scaled.flip(1).cumsum(1).flip(1) / powers.unsqueeze(0)
        return returns.to(device=orig_device, dtype=rewards.dtype)

    raise ValueError(
        "implementation must be either 'recursive' or 'vectorized'"
    )


def trajectories_to_tensors(trajectories, device=None):
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
        torch.as_tensor(states, dtype=torch.float32, device=device),
        torch.as_tensor(actions, dtype=torch.float32, device=device),
        torch.as_tensor(rewards, dtype=torch.float32, device=device),
        torch.as_tensor(mask, dtype=torch.float32, device=device),
    )


def compute_gpomdp_loss(
    policy,
    trajectories,
    gamma: float,
    center_returns: bool = True,
    normalize_returns: bool = False,
    entropy_coeff: float = 0.0,
    returns_implementation: str = "recursive",
    device=None,
    debug: bool = False,
):
    """
        L(theta) = - mean_i sum_t G_{i,t} log pi_theta(a_{i,t}|s_{i,t})
                   - entropy_coeff * mean_t H[pi_theta(·|s_{i,t})]

    entropy_coeff > 0 adds an entropy bonus that prevents policy collapse.
    """

    states, actions, rewards, mask = trajectories_to_tensors(trajectories, device=device)

    returns = compute_discounted_returns_matrix(
        rewards=rewards,
        gamma=gamma,
        implementation=returns_implementation,
    )

    valid_returns = returns[mask.bool()]

    if center_returns:
        returns = returns - valid_returns.mean()

    if normalize_returns:
        returns = returns / (
            valid_returns.std() + 1e-8
        )

    n_trajectories, max_len = rewards.shape

    flat_states = states.reshape(n_trajectories * max_len, -1)

    if actions.ndim == 2:
        flat_actions = actions.reshape(n_trajectories * max_len).long()
    else:
        flat_actions = actions.reshape(n_trajectories * max_len, -1)

    log_probs = policy.log_prob(flat_states, flat_actions)
    log_probs = log_probs.reshape(n_trajectories, max_len)

    objective = (returns * log_probs * mask).sum(dim=1).mean()
    loss = -objective

    mean_entropy = None
    if entropy_coeff > 0:
        dist = policy.distribution(flat_states)
        ent = dist.entropy()
        if ent.dim() > 1:
            ent = ent.sum(-1)
        ent = ent.reshape(n_trajectories, max_len)
        mean_entropy = (ent * mask).sum() / mask.sum()
        loss = loss - entropy_coeff * mean_entropy

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
        if mean_entropy is not None:
            print(f"mean entropy = {mean_entropy.item():.4f}")
        print(f"loss = {loss.item():.6f}")
        print("========== END GPOMDP DEBUG ==========\n")

    return loss


def _compute_empirical_fisher(policy, flat_states, flat_actions, flat_mask, damping: float):
    """
    F = (1/M) Σ_i ∇log π(a_i|s_i) · (∇log π(a_i|s_i))ᵀ  +  damping · I

    Returns F of shape [P, P] where P = total number of policy parameters.
    """
    params_dict = dict(policy.named_parameters())
    buffers_dict = dict(policy.named_buffers())
    P = sum(p.numel() for p in params_dict.values())

    valid = flat_mask.bool()
    v_states = flat_states[valid].detach()
    v_actions = flat_actions[valid].detach()
    if v_actions.ndim == 1:
        # Discrete actions must be integer indices.
        v_actions = v_actions.long()
    M = v_states.shape[0]

    def _log_prob_fn(params, state, action):
        out = functional_call(
            policy,
            {**params, **buffers_dict},
            (state.unsqueeze(0),),
        )
        if isinstance(out, tuple):
            # Continuous policy: use the Gaussian action density.
            mean, std = out
            return Normal(mean.squeeze(0), std).log_prob(action.float()).sum()

        # Discrete policy: use the probability of the selected action.
        return Categorical(logits=out.squeeze(0)).log_prob(action.long())

    # vmap produces one parameter-score vector per valid transition.
    per_sample_grads = vmap(
        _fgrad(_log_prob_fn, argnums=0),
        in_dims=(None, 0, 0),
    )(params_dict, v_states, v_actions)

    S = torch.cat([g.reshape(M, -1) for g in per_sample_grads.values()], dim=1)

    F = S.T @ S / M + damping * torch.eye(P, dtype=S.dtype, device=S.device)
    return F


def apply_npg_preconditioning(policy, trajectories, damping: float = 1e-2, device=None, debug: bool = False):
    """
    Preconditions the gradients in .grad with the inverse empirical Fisher:
        F · nat_g = g  →  nat_g  (solved via torch.linalg.solve)

    Modifies .grad in-place; the caller uses SGD to apply the pure natural gradient step.
    """
    states, actions, _, mask = trajectories_to_tensors(trajectories, device=device)
    N, T = mask.shape

    flat_states = states.reshape(N * T, -1)
    flat_actions = actions.reshape(N * T) if actions.ndim == 2 else actions.reshape(N * T, -1)
    flat_mask = mask.reshape(N * T)

    F = _compute_empirical_fisher(policy, flat_states, flat_actions, flat_mask, damping)

    # Join every parameter's current gradient into one vector.
    params = list(policy.parameters())
    g = torch.cat([
        p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1)
        for p in params
    ])

    # Solve directly instead of calculating an explicit matrix inverse.
    try:
        nat_g = torch.linalg.solve(F, g)
    except torch.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"Fisher matrix solve failed (damping={damping}). "
            "Try increasing --npg_damping (e.g. 0.1)."
        ) from exc

    # Split the result back into each parameter's original shape.
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = nat_g[offset: offset + n].reshape(p.shape).clone()
        offset += n

    # Print basic numerical information only when debugging.
    if debug:
        cond = torch.linalg.cond(F.cpu()).item()
        ratio = nat_g.norm().item() / (g.norm().item() + 1e-8)
        print("\n========== NPG PRECONDITIONING ==========")
        print(f"Fisher condition number  : {cond:.3e}")
        print(f"||nat_g|| / ||g||        : {ratio:.3f}")
        print("========== END NPG DEBUG ==========\n")

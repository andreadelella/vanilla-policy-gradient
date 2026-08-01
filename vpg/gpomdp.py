import numpy as np
import math
import torch
from torch.distributions import Categorical, Normal
from torch.func import functional_call, grad as _fgrad, vmap

# Comment key: M says what the function does. A says how it works and why.

# GPOMDP:
#   ∇J(θ) ≈ (1/N) Σ_i Σ_t G_{i,t} · ∇ log π_θ(a_{i,t} | s_{i,t})
# where G_{i,t} = Σ_{k>=t} γ^{k-t} r_{i,k} is the discounted return from step t.


#M: Calculates the discounted future reward from every timestep.
#A: Works backward through rewards by default because this is simple and stable.
def compute_discounted_returns_matrix(
    rewards: torch.Tensor,
    gamma: float,
    implementation: str = "recursive",
) -> torch.Tensor:
    """
    rewards: [N, T]

    returns[n, t] = r[n, t] + gamma * r[n, t+1] + ...
    """
    # Gamma must be a valid discount factor.
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")

    if implementation == "recursive":
        # Start at the final reward and move backward.
        returns = torch.empty_like(rewards)
        running = torch.zeros(
            rewards.shape[0], dtype=rewards.dtype, device=rewards.device
        )
        for timestep in range(rewards.shape[1] - 1, -1, -1):
            running = rewards[:, timestep] + gamma * running
            returns[:, timestep] = running
        return returns

    if implementation == "vectorized":
        # No future reward matters when gamma is zero or there is only one step.
        if gamma == 0.0 or rewards.shape[1] <= 1:
            return rewards.clone()

        # Reject horizons where gamma powers become too small for the computer.
        last_exponent = (rewards.shape[1] - 1) * math.log(gamma)
        if last_exponent < math.log(torch.finfo(torch.float64).tiny):
            raise ValueError(
                "The vectorized discounted-return formulation is numerically unsafe "
                f"for gamma={gamma} and horizon={rewards.shape[1]}; "
                "use implementation='recursive'."
            )

        # Calculate powers in float64, then restore the original type and device.
        orig_device = rewards.device
        work = rewards.detach().cpu().to(torch.float64)
        powers = gamma ** torch.arange(rewards.shape[1], dtype=torch.float64)
        scaled = work * powers.unsqueeze(0)
        returns = scaled.flip(1).cumsum(1).flip(1) / powers.unsqueeze(0)
        return returns.to(device=orig_device, dtype=rewards.dtype)

    raise ValueError(
        "implementation must be either 'recursive' or 'vectorized'"
    )


#M: Converts different-length trajectories into equal-size tensors.
#A: Pads short episodes with zeros and uses a mask to mark real timesteps.
def trajectories_to_tensors(trajectories, device=None):
    """
    Converts list[Trajectory] into padded tensors.

    states:  [N, T_max, state_dim]
    actions: [N, T_max, action_dim]
    rewards: [N, T_max]
    mask:    [N, T_max]
    """
    # Pad all episodes to the length of the longest episode.
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

    # Copy each episode into its row. Mask 1 means real data; 0 means padding.
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


#M: Builds the GPOMDP loss used to update the policy.
#A: Weights each action's log probability by its discounted return and ignores padding.
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

    # Put all trajectories into one padded batch.
    states, actions, rewards, mask = trajectories_to_tensors(trajectories, device=device)

    # Calculate discounted future rewards.
    returns = compute_discounted_returns_matrix(
        rewards=rewards,
        gamma=gamma,
        implementation=returns_implementation,
    )

    valid_returns = returns[mask.bool()]

    if center_returns:
        # Centering often makes the gradient less noisy.
        returns = returns - valid_returns.mean()

    if normalize_returns:
        # Scaling keeps returns at a more consistent size.
        returns = returns / (
            valid_returns.std() + 1e-8
        )

    n_trajectories, max_len = rewards.shape

    # Flatten all timesteps so the policy handles one large batch.
    flat_states = states.reshape(n_trajectories * max_len, -1)

    if actions.ndim == 2:
        # Discrete actions contain one integer per timestep.
        flat_actions = actions.reshape(n_trajectories * max_len).long()
    else:
        # Continuous actions contain one vector per timestep.
        flat_actions = actions.reshape(n_trajectories * max_len, -1)

    # Find the log probability of every sampled action.
    log_probs = policy.log_prob(flat_states, flat_actions)
    log_probs = log_probs.reshape(n_trajectories, max_len)

    # High-return actions receive more weight. The mask removes padding.
    objective = (returns * log_probs * mask).sum(dim=1).mean()
    loss = -objective

    mean_entropy = None
    if entropy_coeff > 0:
        # Entropy encourages the policy to keep exploring.
        dist = policy.distribution(flat_states)
        ent = dist.entropy()
        if ent.dim() > 1:
            ent = ent.sum(-1)
        ent = ent.reshape(n_trajectories, max_len)
        mean_entropy = (ent * mask).sum() / mask.sum()
        loss = loss - entropy_coeff * mean_entropy

    # Print shapes and values only when debugging.
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


#M: Builds the damped Fisher matrix used by NPG.
#A: Computes one score per valid sample, combines them as S^T S / M,
#   and adds damping to make the matrix easier to solve.
def _compute_empirical_fisher(policy, flat_states, flat_actions, flat_mask, damping: float):
    """
    F = (1/M) Σ_i ∇log π(a_i|s_i) · (∇log π(a_i|s_i))ᵀ  +  damping · I

    Returns F of shape [P, P] where P = total number of policy parameters.
    """
    # 1. Collect the policy values.
    # P is the total number of trainable parameters.
    params_dict = dict(policy.named_parameters())
    buffers_dict = dict(policy.named_buffers())
    P = sum(p.numel() for p in params_dict.values())

    # 2. Keep only real timesteps.
    # The mask removes zeros that were added when trajectories were padded.
    valid = flat_mask.bool()
    v_states = flat_states[valid].detach()
    v_actions = flat_actions[valid].detach()
    if v_actions.ndim == 1:
        # Discrete actions must be integer indices.
        v_actions = v_actions.long()
    M = v_states.shape[0]

    #M: Calculates one action's log probability at one state.
    #A: Runs the policy with the supplied parameters and supports both action types.
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

    # 3. Compute one score for every state-action sample.
    # score = gradient of log probability with respect to all parameters.
    # vmap repeats this calculation over the M samples.
    per_sample_grads = vmap(
        _fgrad(_log_prob_fn, argnums=0),
        in_dims=(None, 0, 0),
    )(params_dict, v_states, v_actions)

    # 4. Build score matrix S with shape [M, P].
    # Each row is one sample; each column is one policy parameter.
    S = torch.cat([g.reshape(M, -1) for g in per_sample_grads.values()], dim=1)

    # 5. Build the Fisher by averaging score outer products.
    # Damping adds a small value to the diagonal so F is easier to solve.
    F = S.T @ S / M + damping * torch.eye(P, dtype=S.dtype, device=S.device)
    return F


#M: Changes the normal policy gradient into a natural policy gradient.
#A: Solves F * natural_gradient = gradient and puts the result back into .grad.
def apply_npg_preconditioning(policy, trajectories, damping: float = 1e-2, device=None, debug: bool = False):
    """
    Preconditions the gradients in .grad with the inverse empirical Fisher:
        F · nat_g = g  →  nat_g  (solved via torch.linalg.solve)

    Modifies .grad in-place; the caller uses SGD to apply the pure natural gradient step.
    """
    # Flatten states, actions, and the padding mask.
    states, actions, _, mask = trajectories_to_tensors(trajectories, device=device)
    N, T = mask.shape

    flat_states = states.reshape(N * T, -1)
    flat_actions = actions.reshape(N * T) if actions.ndim == 2 else actions.reshape(N * T, -1)
    flat_mask = mask.reshape(N * T)

    # Build the Fisher from the sampled states and actions.
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

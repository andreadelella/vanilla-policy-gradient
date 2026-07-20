import numpy as np
import math
import torch
from torch.distributions import Categorical, Normal
from torch.func import functional_call, grad as _fgrad, vmap

# GPOMDP:
#   ∇J(θ) ≈ (1/N) Σ_i Σ_t G_{i,t} · ∇ log π_θ(a_{i,t} | s_{i,t})
# where G_{i,t} = Σ_{k>=t} γ^{k-t} r_{i,k} is the discounted return from step t.


#M: Computes the discounted reward-to-go G_{n,t} for every trajectory n and
#   timestep t in one shot, i.e. the full [N, T] matrix of returns used to
#   weight the log-probabilities in the GPOMDP gradient.
#A: The stable backwards recurrence G_t = r_t + gamma*G_{t+1} is used
#   directly. A powers/reverse-cumsum formulation is superficially more
#   vectorized, but gamma**t underflows on long horizons for ordinary
#   discounts and silently corrupts late-timestep returns.
def compute_discounted_returns_matrix(
    rewards: torch.Tensor,
    gamma: float,
    implementation: str = "recursive",
) -> torch.Tensor:
    """
    rewards: [N, T]

    returns[n, t] = r[n, t] + gamma * r[n, t+1] + ...
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")

    if implementation == "recursive":
        # Stable over the full accepted gamma range and all practical horizons.
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

        # Do the powers calculation in float64 even when training uses
        # float32. The original float32 implementation clamped small powers,
        # which changed the mathematical result. Reject a request that would
        # leave float64's normal range instead of silently approximating it.
        last_exponent = (rewards.shape[1] - 1) * math.log(gamma)
        if last_exponent < math.log(torch.finfo(torch.float64).tiny):
            raise ValueError(
                "The vectorized discounted-return formulation is numerically unsafe "
                f"for gamma={gamma} and horizon={rewards.shape[1]}; "
                "use implementation='recursive'."
            )

        orig_device = rewards.device
        work = rewards.detach().cpu().to(torch.float64)
        powers = gamma ** torch.arange(rewards.shape[1], dtype=torch.float64)
        scaled = work * powers.unsqueeze(0)
        returns = scaled.flip(1).cumsum(1).flip(1) / powers.unsqueeze(0)
        return returns.to(device=orig_device, dtype=rewards.dtype)

    raise ValueError(
        "implementation must be either 'recursive' or 'vectorized'"
    )


#M: Converts a Python list of variable-length Trajectory objects into a
#   single batch of fixed-size tensors (states, actions, rewards, mask),
#   so the rest of the pipeline can operate on the whole batch at once
#   instead of per-trajectory.
#A: Episodes have different lengths, so naively you'd need ragged/ per-
#   trajectory tensors and a Python loop over trajectories for every later
#   computation (log_prob, returns, etc.). Instead we pad every trajectory
#   to max_len with zeros and carry along a binary `mask` marking real vs.
#   padded steps. This turns a ragged batch into a single dense [N, T_max]
#   tensor, which lets downstream code (returns, log_probs, loss) use fast
#   batched tensor ops; the mask is then used to zero out the contribution
#   of padded positions instead of branching per-trajectory.
def trajectories_to_tensors(trajectories, device=None):
    """
    Converts list[Trajectory] into padded tensors.

    states:  [N, T_max, state_dim]
    actions: [N, T_max, action_dim]
    rewards: [N, T_max]
    mask:    [N, T_max]
    """
    # Episodes can end at different timesteps; we pad to the longest one and use
    # a binary mask to exclude padding positions from gradient computation.

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


#M: Builds the scalar loss whose gradient (via autograd) equals the negative
#   GPOMDP policy-gradient estimate: -mean_i Σ_t G_{i,t} log π(a_{i,t}|s_{i,t}),
#   optionally with a mean-baseline, return normalization, and an entropy
#   bonus to discourage premature policy collapse.
#A: Two efficiency tricks stack here. (1) Variance reduction is free: sub-
#   tracting the batch-mean return as a baseline (`center_returns`) and/or
#   dividing by its std (`normalize_returns`) does not bias the gradient
#   (Williams 1992) but shrinks its variance, so fewer trajectories are
#   needed per update. (2) All timesteps of all trajectories are flattened
#   to [N*T, ...] and passed through `policy.log_prob` in a single batched
#   call rather than one call per (trajectory, timestep) pair; the returns
#   are reshaped back to [N, T] and multiplied by the padding `mask` so
#   padded steps contribute exactly zero to the objective without any
#   per-trajectory branching.
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
        # Subtract the mean return as a state-independent baseline.
        # Reduces gradient variance without introducing bias (Williams 1992).
        returns = returns - valid_returns.mean()

    if normalize_returns:
        returns = returns / (
            valid_returns.std() + 1e-8
        )

    n_trajectories, max_len = rewards.shape

    flat_states = states.reshape(n_trajectories * max_len, -1)  # [N,T,d_s] → [N*T,d_s]: one batched log_prob call

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

    mean_entropy = None
    if entropy_coeff > 0:
        dist = policy.distribution(flat_states)
        ent = dist.entropy() # [N*T] or [N*T, action_dim]
        if ent.dim() > 1:
            ent = ent.sum(-1) # sum action dims for Gaussian
        ent = ent.reshape(n_trajectories, max_len)
        mean_entropy = (ent * mask).sum() / mask.sum()
        loss = loss - entropy_coeff * mean_entropy  # maximise entropy

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


#M: Estimates the empirical Fisher information matrix F of the policy from
#   a batch of (state, action) samples, F = (1/M) Σ_i g_i g_iᵀ + damping·I,
#   where g_i = ∇log π(a_i|s_i) is the per-sample score vector. This F is
#   what turns the vanilla gradient into a natural gradient in
#   `apply_npg_preconditioning`.
#A: The mathematically direct approach would compute each per-sample
#   gradient g_i with a separate autograd.grad() call in a Python for-loop
#   over the M valid timesteps. Instead the log-prob
#   function is rewritten in a pure-functional form (`functional_call`,
#   parameters passed explicitly) and mapped over the batch with
#   `vmap(grad(...))`: this computes all M per-sample Jacobians as one
#   vectorized operation instead of M sequential backward passes. The
#   resulting per-parameter gradients are flattened and concatenated into
#   an [M, P] score matrix S, so F = SᵀS/M is a single matrix multiply
#   rather than a running sum built up in a loop. Damping (+ damping·I)
#   keeps F invertible when it's rank-deficient (M < P).
def _compute_empirical_fisher(policy, flat_states, flat_actions, flat_mask, damping: float):
    """
    F = (1/M) Σ_i ∇log π(a_i|s_i) · (∇log π(a_i|s_i))ᵀ  +  damping · I

    Returns F of shape [P, P] where P = total number of policy parameters.
    """
    params_dict = dict(policy.named_parameters()) # learnable weights
    buffers_dict = dict(policy.named_buffers()) # non-learnable state
    P = sum(p.numel() for p in params_dict.values()) # total parameters count -> fisher matrix dimension

    # Filter to valid (non-padded) timesteps.
    valid = flat_mask.bool() # 0 -> False, nonzero -> True
    v_states = flat_states[valid].detach() # detached tensor of valid (non-padded) states 
    v_actions = flat_actions[valid].detach() # detached tensor of valid (non-padded) actions
    if v_actions.ndim == 1:
        # 1-D means discrete actions (one action-id per sample); cast to long for
        # Categorical.log_prob. Continuous actions stay 2-D [M, action_dim] float.
        v_actions = v_actions.long()
    M = v_states.shape[0]

    # Pure-functional log_prob for vmap: params are injected, buffers are fixed.
    def _log_prob_fn(params, state, action):
        out = functional_call(policy, {**params, **buffers_dict}, (state.unsqueeze(0),)) # stateless way to run forward pass)
        if isinstance(out, tuple):
            # GaussianPolicy: forward returns (mean, std)
            mean, std = out
            return Normal(mean.squeeze(0), std).log_prob(action.float()).sum()
        # SoftmaxPolicy: forward returns logits
        return Categorical(logits=out.squeeze(0)).log_prob(action.long()) # categorical distribution from unnormalized logits

    # vmap produces all M per-sample Jacobians in one pass instead of M serial autograd.grad calls.
    #   in_dims=(None, 0, 0) — params broadcast, states/actions batched.
    per_sample_grads = vmap(
        _fgrad(_log_prob_fn, argnums=0), # fgrad functional transform that turns scalar-output function into a new function computing its gradient w.r.t argument index argnums
        in_dims=(None, 0, 0),
    )(params_dict, v_states, v_actions) # transforms the function written for a single sample to run it over a batch dimension
    # The combo vmap and fgrad computes, in one call, per_sample_grads: dict {name: [M, *param_shape]}

    S = torch.cat([g.reshape(M, -1) for g in per_sample_grads.values()], dim=1)  # [M, P]
    F = S.T @ S / M + damping * torch.eye(P, dtype=S.dtype, device=S.device)
    return F


#M: Turns the vanilla policy gradient already sitting in `.grad` into the
#   natural policy gradient by preconditioning it with the inverse Fisher
#   information matrix: solves F · nat_g = g for nat_g, then overwrites
#   each parameter's `.grad` in place so a plain SGD step applies NPG.
#A: The naive way to get nat_g = F⁻¹g would explicitly invert the P×P
#   Fisher matrix (expensive and numerically unstable for near-singular F).
#   Instead `torch.linalg.solve` solves the linear system directly via a
#   factorization (no explicit inverse formed), which is both cheaper and
#   more numerically stable. Reusing the existing `.grad` tensors (instead
#   of returning a new gradient object) means the caller's optimizer step
#   is unchanged, NPG becomes a drop-in preprocessing pass before a
#   standard SGD step, and gradient flattening/reassembly reuses the exact
#   parameter ordering from `policy.parameters()` so no name bookkeeping is
#   needed to unflatten nat_g back into per-parameter shapes.
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

    # Assemble the current gradient vector g from .grad of all parameters.
    params = list(policy.parameters())
    g = torch.cat([
        p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1)
        for p in params
    ])

    try:
        nat_g = torch.linalg.solve(F, g)
    except torch.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"Fisher matrix solve failed (damping={damping}). "
            "Try increasing --npg_damping (e.g. 0.1)."
        ) from exc

    offset = 0
    for p in params:
        n = p.numel()
        p.grad = nat_g[offset: offset + n].reshape(p.shape).clone()
        offset += n

    if debug:
        cond = torch.linalg.cond(F.cpu()).item()
        ratio = nat_g.norm().item() / (g.norm().item() + 1e-8)
        print("\n========== NPG PRECONDITIONING ==========")
        print(f"Fisher condition number  : {cond:.3e}")
        print(f"||nat_g|| / ||g||        : {ratio:.3f}")
        print("========== END NPG DEBUG ==========\n")

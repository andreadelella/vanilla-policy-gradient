import torch


def compute_gpomdp_gradient(policy, trajectories, beta: float):
    """
    recursive GPOMDP estimator:

        z_{t+1} = beta z_t + grad log pi(a_t | s_t)

        g_hat = (1/N) sum_i sum_t r_{t+1} z_{t+1}

    where z is the eligibility trace.
    """

    grads = [torch.zeros_like(p) for p in policy.parameters()]

    for traj in trajectories:
        z = [torch.zeros_like(p) for p in policy.parameters()]

        for state, action, reward in zip(traj.states, traj.actions, traj.rewards):
            state_tensor = torch.tensor(state, dtype=torch.float32)

            log_prob = policy.log_prob(state_tensor, action)

            grad_log_prob = torch.autograd.grad(
                log_prob,
                policy.parameters(),
                retain_graph=False,
                create_graph=False,
            )

            for i, g in enumerate(grad_log_prob):
                z[i] = beta * z[i] + g.detach()
                grads[i] += reward * z[i]

    grads = [g / len(trajectories) for g in grads]

    return grads


def apply_gradient_step(policy, gradients, lr: float):
    """
    theta <- theta + lr * g_hat
    """

    with torch.no_grad():
        for param, grad in zip(policy.parameters(), gradients):
            param += lr * grad
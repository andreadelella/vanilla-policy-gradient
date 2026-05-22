import torch


def compute_discounted_returns(rewards, beta: float, debug: bool = False):
    returns = []
    G = 0.0

    for t, r in reversed(list(enumerate(rewards))):
        G = r + beta * G
        returns.insert(0, G)

        if debug:
            print(f"[Return] t={t}: G_t = r_t + beta * G_next = {r:.3f} + {beta:.3f} * ... = {G:.3f}")

    return torch.tensor(returns, dtype=torch.float32)


def compute_gpomdp_gradient(policy, trajectories, beta: float, debug: bool = False):
    """
    Future-form GPOMDP estimator:

        G_t^beta = sum_{k=t}^{T-1} beta^{k-t} r_{k+1}

        g = 1/T sum_t G_t^beta grad log pi(a_t | s_t)
    """

    grads = [torch.zeros_like(p) for p in policy.parameters()]
    total_steps = 0

    if debug:
        print("\n========== GPOMDP GRADIENT ESTIMATION ==========")
        print("Estimator:")
        print("g = (1/T) * sum_t G_t^beta * grad_theta log pi_theta(a_t | s_t)")
        print("================================================\n")

    for traj_idx, traj in enumerate(trajectories):
        if debug:
            print(f"\n--- Trajectory {traj_idx} ---")
            print(f"Length T = {len(traj.rewards)}")
            print(f"Rewards = {traj.rewards}")

        returns = compute_discounted_returns(
            traj.rewards,
            beta=beta,
            debug=debug,
        )

        if debug:
            print(f"Discounted returns G_t^beta = {returns.tolist()}")

        for t, (state, action, G_t) in enumerate(zip(traj.states, traj.actions, returns)):
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # Step 1: log pi_theta(a_t | s_t)
            log_prob = policy.log_prob(state_tensor, action)

            # Step 2: grad_theta log pi_theta(a_t | s_t)
            grad_log_prob = torch.autograd.grad(
                log_prob,
                policy.parameters(),
                retain_graph=False,
                create_graph=False,
            )

            # Step 3: G_t * grad log pi
            for i, g in enumerate(grad_log_prob):
                contribution = G_t * g.detach()
                grads[i] += contribution

                if debug and traj_idx == 0 and t < 5:
                    print(f"\n[t={t}]")
                    print(f"state s_t = {state}")
                    print(f"action a_t = {action}")
                    print(f"log pi(a_t|s_t) = {log_prob.item():.6f}")
                    print(f"G_t^beta = {G_t.item():.6f}")
                    print(f"grad log pi shape = {tuple(g.shape)}")
                    print(f"contribution = G_t * grad log pi")
                    print(contribution)

            total_steps += 1

    grads = [g / total_steps for g in grads]

    if debug:
        print("\n--- Final averaging step ---")
        print(f"Total samples T = {total_steps}")
        print("g_hat = accumulated_gradient / T")

        for i, g in enumerate(grads):
            print(f"\nGradient for parameter tensor {i}:")
            print(g)

        print("\n========== END GPOMDP DEBUG ==========\n")

    return grads


def apply_gradient_step(policy, gradients, lr: float, debug: bool = False):
    """
    Gradient ascent update:

        theta <- theta + lr * g_hat
    """

    if debug:
        print("\n========== PARAMETER UPDATE ==========")
        print("theta <- theta + alpha * g_hat")
        print(f"alpha = {lr}")

    with torch.no_grad():
        for i, (param, grad) in enumerate(zip(policy.parameters(), gradients)):
            if debug:
                print(f"\nParameter tensor {i}")
                print("theta before:")
                print(param.data)
                print("gradient:")
                print(grad)

            param += lr * grad

            if debug:
                print("theta after:")
                print(param.data)

    if debug:
        print("========== END PARAMETER UPDATE ==========\n")
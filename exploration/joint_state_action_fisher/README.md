# Joint state-action Fisher: exact validation stage

This package implements Step 1 of `exploration/fisher_log_barrier/PLAN.md`.
It does not alter or reinterpret historical artifacts. In particular, the old
`pooled_fisher` remains a correct transition-pooled **policy Fisher**:

\[
F_{\pi,\mu}=\mathbb E_{s\sim\mu,a\sim\pi}
[\nabla\log\pi(a\mid s)\nabla\log\pi(a\mid s)^\top].
\]

The new object is the Fisher of the full declared joint distribution
\(\rho_\phi(s,a)=\mu_\phi(s)\pi_\phi(a\mid s)\). Its score includes both
\(\nabla\log\pi(a\mid s)\) and \(\nabla\log\mu(s)\), giving

\[
F_{\rho,\mu}=F_{\pi,\mu}+F_\mu.
\]

For the exact two-state transition-pooled model, \(F_\mu\) is rank one and

\[
\det F_{\rho,\mu}=2\mu_0\det F_{\pi,\mu}.
\]

The factor-one-half normalized joint regularizer is therefore

\[
B_{\mathrm{joint}}
=\tfrac12\log\det F_{\rho,\mu}
=B_{\mathrm{pooled\ policy}}+\tfrac12\log(2\mu_0).
\]

The identity is specific to this two-state model and reduced parameterization;
it is not asserted as a general MDP determinant identity.

Run the full deterministic suite with:

```powershell
python -m exploration.joint_state_action_fisher.verify_identity
```

It uses CPU float64 and checks direct outcome enumeration, closed forms,
legacy equality, expected-score and cross-term identities, PSD/rank, determinant
identities, autograd, and directional finite differences. It writes the
versioned validation artifacts to
`exploration/results/joint_state_action_fisher/step1_identity/`.

No optimization, NPG, sampling, neural policy, damping, clipping, or spectral
floor is part of this stage. Exact training remains gated on all checks passing.

## Step 2: exact two-state objectives

After Gate A passes, run the fixed-objective exact comparison with:

```powershell
python -m exploration.joint_state_action_fisher.run_exact_two_state
```

This compares reward only, the detached statewise conditional barrier, the
historical pooled-policy logdet under its corrected name, the new joint
state-action logdet, and two diagnostic decompositions. It uses exact Euclidean
gradients, the declared same-beta grid, and an adverse-start magnitude-matched
protocol. There is no sampling, NPG, damping, or statistical inference.

`state_distribution_only` is explicitly the visitation component derived from
the joint determinant identity. It is not `logdet(F_mu)`: the four-dimensional
state-distribution Fisher has rank one, so that determinant is zero.

Step 2 artifacts are isolated under
`exploration/results/joint_state_action_fisher/step2_two_state/`. Handoff runs
remain deferred until the fixed objectives and vector fields have been reviewed,
as required by Gate B in the plan.

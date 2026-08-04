# Step 4: Finite-Batch Sampled Two-State MDP

This package is the stochastic bridge between the exact tabular geometry in
`exploration.tabular_mdp` and a later neural discrete-control experiment. It
uses the same two-state, three-action MDP and reduced logits, but replaces
population expectations with finite batches of sampled trajectories.

It does not modify `vpg/`, use Gym, introduce a neural network, damp a Fisher
matrix, clip an update, or hide missing-state batches with a count floor.

## Why this step exists

The primary paper proves that, for a reduced categorical softmax
bandit, `log det(F_red) = sum_a log pi(a)`. Step 3 showed exactly what changes
in a tabular MDP when the Fisher also contains policy-dependent state
frequencies. Step 4 asks what an ordinary finite rollout batch estimates.

For `N` trajectories, let `K1` be the number selecting the transition action
at `s0`. The pooled dataset has `M=N+K1` valid transitions, so
`mu1_hat=K1/(N+K1)`, with `K1` binomially distributed. The ratio is random and
its finite-batch expectation is not `q/(1+q)`. The probability that a batch
offers no direct barrier protection to `s1` is exactly `(1-q)^N`.

## Estimators

The primary reward estimator is raw reward-to-go REINFORCE, averaged over
trajectories. The practical barrier estimator averages the statewise
categorical barrier gradient over all valid pooled transitions. The empirical
Fisher follows the project convention exactly: `F_hat = S.T @ S / M`, using
sampled actions and no damping.

Rank-deficient batches keep an undefined log-determinant; the implementation
does not add a spectral floor. `exact_finite_batch_moments` computes exact
binomial moments for the pooled ratio and conditional barrier, the exact
expected sampled Fisher conditional on `K1`, and the exact mean/covariance of
the REINFORCE estimator.

## Training comparisons

Every method uses a sampled reward gradient. Six methods retain the Step 3
population regularizer as an explicitly labelled oracle control:

- `reward_only`;
- `detached_conditional_oracle`;
- `complete_weighted_oracle`;
- `uniform_action_oracle`;
- `visitation_only_oracle`;
- `full_pooled_fisher_oracle`.

The seventh method, `detached_conditional_sampled`, is the implementable
rollout-based candidate. The visitation and full oracle methods are not
presented as algorithms available from ordinary neural rollout states.

The primary grid uses `alpha=0.05`, 2000 updates,
`beta in {0.01, 0.1, 0.2}`, batch sizes `{4, 32, 128}`, and the uniform and
adverse Step 3 initializations. Complete seed runs are the independent units.
Two-sided 95% Student-t intervals are computed across seeds, never across
checkpoints.

Secondary experiments add terminal Gaussian reward noise with standard
deviation one and reproduce the project's centered-and-normalized return
convention. These do not replace the deterministic raw-return primary result.

## Commands

```powershell
python -m exploration.sampled_tabular_mdp.verify
python -m exploration.sampled_tabular_mdp.run_experiment --preset smoke
python -m exploration.sampled_tabular_mdp.run_experiment --preset pilot
python -m exploration.sampled_tabular_mdp.run_experiment --preset full
python -m exploration.sampled_tabular_mdp.run_handoff --preset full
python -m exploration.sampled_tabular_mdp.run_switch_sweep
```

Use `--resume` only with an output directory containing the identical saved
configuration. Incompatible nonempty directories are rejected. Outputs are
written beneath `exploration/results/tabular_mdp/two_step_trap_sampled/`.

## Temporary-barrier handoff experiment

The dedicated handoff command runs a separate, paired experiment with
`N=32`, deterministic raw returns, `alpha=0.05`, and both prescribed
initializations. It compares reward-only, a fixed sampled conditional barrier
at `beta=0.2`, the practical sampled conditional barrier at `beta=0.2` for the
first 2000 of 4000 updates followed by reward-only learning, and a full-oracle
`beta=0.1` handoff used only as a diagnostic upper reference.

The schedule uses zero-based update index `t`: `t < 2000` is regularized and
`t >= 2000` is unregularized. All units reset the same generator seed and use
the same batch tensor shapes, preserving paired base-uniform streams. Results
are isolated under `two_step_trap_sampled/handoff/<preset>/`.

## Interpretation boundary

This stage can establish finite-batch bias, variance, missing-state frequency,
and their association with training behavior. It cannot show that larger
Fisher volume causes higher return, that an oracle visitation gradient is
available to neural RL, or that a tabular result transfers through a shared
neural Jacobian.

If the practical estimator is sufficiently characterized, the later neural
method must be called an **on-policy sampled-state conditional categorical log
barrier**, not a global neural Fisher log-determinant.

Primary references are [How Log-Barrier Helps Exploration in
Policy Optimization](https://arxiv.org/html/2603.15001v2), Williams's
[REINFORCE paper](https://doi.org/10.1007/BF00992696), and Kunstner et al. on
[limitations of the empirical Fisher](https://proceedings.neurips.cc/paper/2019/hash/46a558d97954d0692411c861cf78ef79-Abstract.html).

# Step 3: Exact Two-State Tabular MDP Geometry

This package isolates policy-dependent state weighting from conditional categorical action geometry. It contains no Gym environment, neural network, Monte Carlo sampling, damping, clipping, spectral floor, or stochastic noise. Every expectation and gradient is exact and uses CPU `torch.float64`.

## Model

At `s0`, action `a0` terminates with reward `0.5`, `a1` moves to `s1`, and `a2` terminates with reward zero. At `s1`, the terminal rewards are `(1, 0.2, 0)`. Each state has a three-action categorical policy represented by two reduced logits; `a2` is the fixed reference action.

With `q = pi0(a1)`,

```math
J=0.5\pi_0(a_0)+q[\pi_1(a_0)+0.2\pi_1(a_1)].
```

## Transition-pooled weighting convention

```math
\mu_0=\frac1{1+q},\qquad \mu_1=\frac q{1+q}.
```

`mu_pool` is **the population state distribution obtained by uniformly selecting one valid action transition from an infinitely large transition-pooled rollout dataset**. It is a global average over valid transitions: all trajectories contain one `s0` transition, while only the fraction `q` contain an `s1` transition. Shorter terminated trajectories contribute fewer transitions.

This is not the standard fixed-horizon occupancy

```math
d_T(s)=\frac1T\sum_t \Pr(S_t=s).
```

In a finite sampled batch, the empirical transition-pooled state weight is a ratio with a random denominator. Its expectation need not equal the ratio of population expected counts above. Measuring that discrepancy belongs to the next stochastic stage.

## Fisher and barrier decomposition

For the two reduced categorical Fishers,

```math
F_{pool}=\operatorname{blockdiag}(\mu_0F_0,\mu_1F_1),
```

```math
\log\det F_{pool}=B_0+B_1+2\log\mu_0+2\log\mu_1,
\qquad B_s=\sum_a\log\pi_s(a).
```

The normalized full barrier is

```math
B_{full}=\tfrac12\log\det F_{pool}
=\underbrace{\tfrac12(B_0+B_1)}_{B_{uniform}}
+\underbrace{\log\mu_0+\log\mu_1}_{B_{visit}}.
```

The factor `1/2` is not canonical. It is an action-matched normalization making the action component equal to the uniform-state action barrier, so visitation is the isolated difference.

The six methods are `reward_only`, `detached_conditional`, `complete_weighted`, `uniform_action`, `visitation_only`, and `full_pooled_fisher`. In particular,

```math
\nabla(\mu_0B_0+\mu_1B_1)
=\mu_0\nabla B_0+\mu_1\nabla B_1+B_0\nabla\mu_0+B_1\nabla\mu_1.
```

Thus `E_mu[grad B_s]`, `grad E_mu[B_s]`, and `grad logdet(F_pool)` are generally different.

## Commands and outputs

```powershell
python -m exploration.tabular_mdp.verify
python -m exploration.tabular_mdp.run_experiment
```

The experiment writes deterministic, versioned results to `exploration/results/tabular_mdp/two_step_trap/`. Existing compatible units can be reused with `--resume`; incompatible nonempty result directories are rejected.

The focused report is `exploration/tabular_mdp/two_state_geometry.tex`. It deliberately remains separate from the longer Step 1--2 report.

## Interpretation boundary

The detached transition-pooled conditional surrogate protects each state's categorical geometry in proportion to that state's dataset frequency, so its direct protection of rare states is weaker. The full pooled-Fisher barrier adds explicit state-visitation pressure, which may aid recovery but may also bias the policy toward visiting a state independently of reward.

This experiment does not show that a larger determinant guarantees good conditioning or return, establish a causal geometric mechanism from return alone, or transfer directly to shared neural parameters. The later neural candidate must be called an **on-policy sampled-state conditional categorical log barrier**, not a global neural Fisher log-determinant.

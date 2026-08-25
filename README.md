# Vanilla Policy Gradient and Fisher Analysis

PyTorch implementation of GPOMDP and Natural Policy Gradient (NPG), together
with a fixed-policy empirical Fisher eigenspectrum analysis for continuous-control
Gymnasium environments and a focused categorical log-barrier study.

## Project structure

```text
vpg/                 Policy models, rollout collection, GPOMDP/NPG, training,
                     diagnostics, persistence, plotting, tuning, and video utilities.
fisher_analysis/     Fixed-policy rollouts, streaming Fisher construction,
                     spectral metrics, plotting, and experiment orchestration.
log_barrier/         Exact finite-MDP policy/joint Fisher experiment and the
                     sampled categorical Acrobot validation.
fisher_log_barrier/  Strict trajectory-score Fisher logdet Loss-Function (1)
                     surrogate and Acrobot feasibility preflight.
results/figures/     Fixed-policy Fisher eigenspectrum plots.
results/csv/         Fixed-policy Fisher summaries and iteration statistics.
results/runs_new/    Current NPG width-comparison figures, videos, and metrics.
results/log_barrier/ Compact retained outputs from the log-barrier experiments.
run.py               Training command.
analysis.py          Reward-curve analysis command.
tune.py              Single environment/algorithm hyperparameter search.
tune_all.py          Multi-environment hyperparameter search.
config.json          Default experiment configuration.
requirements.txt     Exact Python dependencies.
.python-version      Required Python version.
```

## Installation

The project uses Python 3.13.

```bash
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Policy-gradient training

Run the defaults from `config.json`:

```bash
python run.py
```

Examples:

```bash
python run.py --env_id CartPole-v1 --algorithm gpomdp
python run.py --env_id Hopper-v5 --algorithm npg
python run.py --run_mode multiseed --seeds 23 24 25 26 27
```

GPOMDP estimates the policy gradient from sampled trajectories. NPG uses the
same gradient and preconditions it with the damped empirical Fisher matrix.
Training outputs are written to `runs/`, which is intentionally ignored.

The checked-in `config.json` supplies CLI defaults. Explicit command-line
arguments take precedence. The default run is GPOMDP on CartPole; select NPG
with `--algorithm npg`.

## Fisher eigenspectrum analysis

The analysis estimates the undamped empirical Fisher matrix

```text
F = (1 / M) S^T S,
```

where each row of `S` is the score of one sampled state-action transition and
`M` is the total number of valid transitions. It then diagonalizes `F` and
reports how many eigenvalues are required to explain 90% of its trace.

Run the default Hopper experiment:

```bash
python -m fisher_analysis.run_fisher_analysis
```

Run another environment or choose an output directory:

```bash
python -m fisher_analysis.run_fisher_analysis \
  --env-id HalfCheetah-v5 \
  --output-dir output/halfcheetah_width_sweep
```

The retained summary is:

| Environment | Width 4 | Width 8 | Width 16 |
|---|---:|---:|---:|
| Hopper | 6 / 86 | 7 / 198 | 10 / 518 |
| HalfCheetah | 32 / 128 | 45 / 276 | 70 / 668 |
| Swimmer | 6 / 68 | 6 / 164 | 7 / 452 |

Each entry is `directions needed for 90% of Fisher trace / policy parameters`.
These are local sensitivity dimensions of the sampled policy, not reduced
network architectures.

The final plots are in `results/figures/`. The `*_summary.csv` files contain
one row per network width, while `*_iteration_stats.csv` contains the rollout
and Fisher statistics for each analysis iteration.

## Training results

The selected NPG width-sweep outputs are organized under `results/runs_new/`:

```text
results/runs_new/
├── figures/             Comparison, confidence-interval, seed, and KL plots.
├── videos/              Recorded policies, preserving environment and width.
└── metrics_summary.csv  One row per environment, width, and seed.
```

For HalfCheetah, Hopper, and Swimmer, the
`kl_by_iteration_and_cumulative_kl.png` figure contains both comparisons:

- return at equal training iterations;
- return at equal cumulative KL, which compares policies after matched
  policy-space movement.

`metrics_summary.csv` records the completed iteration count, final reward,
mean reward over the final 100 iterations, best observed reward, final-step KL,
and cumulative KL for each retained run.

## Categorical log-barrier experiments

The exact finite MDP is a three-state, three-action chain with two continuation
bottlenecks. It uses normalized discounted occupancy (including the absorbing
terminal state) and computes two exact reduced-coordinate matrices:

```text
F_policy = E_d E_pi[score_pi score_pi^T]
F_joint  = F_policy + F_state
```

The experiment applies ordinary Euclidean gradient ascent to the exact return
with no barrier, `0.5 * logdet(F_policy)`, or `0.5 * logdet(F_joint)`. At the
eleven training deciles it records all six eigenvalues, minimum and maximum
eigenvalues, condition number, trace, and log determinant for both matrices.

```bash
python -m log_barrier.exact_mdp.verify
python -m log_barrier.exact_mdp.run
```

The Acrobot experiment does not estimate a state-density Fisher.
It compares reward-only GPOMDP with an on-policy sampled-state conditional
categorical barrier and records the undamped, action-enumerated empirical policy
Fisher spectrum at the same eleven deciles. The barrier remains active for the
complete run. The configuration uses Adam, learning rate 0.003, eight complete
episodes per update, and centered but unnormalized returns.

For sampled rollout states `s_1,...,s_M` and `K` discrete actions, the fixed
regularizer is

```text
B(theta) = (1 / MK) sum_m sum_a log pi_theta(a | s_m).
```

The action-enumerated empirical policy Fisher used for analysis is

```text
F_hat = (1 / M) sum_m sum_a pi_theta(a | s_m)
        grad log pi_theta(a | s_m) grad log pi_theta(a | s_m)^T.
```

`k90` is the smallest number of descending Fisher eigenvalues whose sum reaches
90% of the positive Fisher trace. It measures how concentrated local policy
sensitivity is; it is not a proposed smaller network size.

```bash
python -m log_barrier.acrobot.run --seeds 1,2,3
python -m log_barrier.acrobot.analyze_checkpoints
python -m unittest discover -s log_barrier/tests -p "test_*.py"
```

The empirical neural Fisher is a sampled-state diagnostic and can be rank
deficient. Accordingly, its CSV reports the positive spectrum, numerical rank,
trace, condition number, log pseudodeterminant, and the numbers of principal
directions needed for 90%, 95%, and 99% of its positive trace. The fixed-barrier
checkpoint analysis also generates raw, trace-normalized, cumulative-trace, and
`k90`-through-training plots matching `fisher_analysis`. No damping or spectral
floor is used. This is not presented as a global joint state-action Fisher
determinant.

The fixed-barrier coefficient and training budget were selected after preliminary
experiments on exploration loss and catastrophic failures. Those preliminary
runs are not part of the final command interface; the submitted comparison uses
only reward-only GPOMDP and the barrier kept active for the complete training
run. Retained result files record the exact seeds, coefficient, checkpoints, and
state-bank construction used to produce each figure.

## Trajectory-Fisher logdet barrier

The separate `fisher_log_barrier` package implements Loss-Function (1) for
the whole-trajectory score Fisher

```text
z_k = sum_t grad log pi(a_kt | s_kt)
F_hat = (1 / N) sum_k z_k z_k^T
```

and the strict domain `F_hat - mu I > 0`. It is distinct from the categorical
action barrier above. The Acrobot implementation learns two logits and fixes the
third to zero, removing the categorical common-logit null direction while
preserving the complete policy family. The `[8, 8]` reference-logit policy has
146 trainable parameters.

Each update uses a large, independent batch to estimate `F_hat` and the normal
eight-trajectory batch for GPOMDP and the outer Loss-Function (1) expectation.
Fisher construction and factorization use float64. Only the small outer batch
retains second-order autograd graphs. Environment-step reporting includes both
batches. The implementation does not use damping, eigenvalue clipping, a
pseudoinverse, or a pseudodeterminant.

Run the mandatory Acrobot feasibility preflight before attempting training:

```bash
python -m fisher_log_barrier.preflight \
  --episodes-per-update 256 \
  --parallel-envs 16 \
  --mu 1e-10 \
  --output results/fisher_log_barrier/acrobot/preflight.json
```

Only if that preflight reports `training_can_proceed: true`, run a short pilot:

```bash
python -m log_barrier.acrobot.run \
  --methods fisher_logdet \
  --seeds 1 \
  --updates 5 \
  --fisher-episodes-per-update 256 \
  --fisher-parallel-envs 16 \
  --fisher-mu 1e-10 \
  --fisher-beta 1.0 \
  --output results/fisher_log_barrier/acrobot/pilot
```

The empirical estimator has rank at most the number of Fisher trajectories. The
fixed default `mu=1e-10` was chosen below the smallest initial `lambda_min`
observed in a five-seed, 256-trajectory preflight recorded in
`results/fisher_log_barrier/acrobot/mu_selection.json`. If a new preflight fails,
increase the Fisher batch rather than adapting `mu` during training or silently
damping or clipping the matrix. To compare against a reward-only baseline in
exactly the same identifiable coordinates, run both methods with
`--policy-parameterization reference`.

### Lunar-Barrier paired reliability study

The Acrobot categorical barrier is also available for discrete
`LunarLander-v3`. Run hyperparameter screening on a small, reserved seed set:

```bash
python -m log_barrier.lunar_barrier.run smoke \
  --updates 200 \
  --workers 4
```

The smoke phase is staged to avoid a full Cartesian explosion. Stage 1 tunes
learning rate (`0.001`, `0.003`), gamma (`0.97`, `0.99`), episodes per update
(`4`, `8`), and centered versus centered-normalized returns. Stage 2 tunes beta
(`2`, `5`, `10`, `20`, `40`) and barrier handoff fraction (`0.05`, `0.10`,
`0.25`) on the selected Stage 1 configuration. Only the Acrobot-comparable
`[8, 8]` network and 32 evaluation episodes are fixed. Candidates are ranked by
the 25th percentile of held-out stochastic evaluation return, with mean return
and paired improvement as tie breakers. The beta grid brackets LunarLander's
selected normalized-return gradient scale and is not copied from Acrobot. Run
the confirmatory experiment on 200 new, paired seeds:

```bash
python -m log_barrier.lunar_barrier.run reliability \
  --selection results/log_barrier/lunar_barrier/smoke/selection.json \
  --updates 1000 \
  --n-seeds 200 \
  --workers 4
```

Each seed uses the same initialization and environment seed schedule for
reward-only and barrier training. Evaluation also uses common environment seeds
and common action uniforms across 32 episodes per trained policy. Runs are saved
independently and resumed automatically. `reliability_summary.json` reports the
paired mean difference and bootstrap confidence interval, win rate, solved rate,
catastrophic-failure rate, and exact paired McNemar test. Smoke and confirmatory
seed sets are required to be disjoint.

## Tests

Run the complete repository suite with:

```bash
python -m unittest discover -v
```

The core tests cover configuration parsing, policies, discounted returns,
trajectory masking, action clipping, statistics, artifact persistence, and
streaming Fisher construction in addition to the log-barrier checks.

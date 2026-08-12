# Vanilla Policy Gradient and Fisher Analysis

PyTorch implementation of GPOMDP and Natural Policy Gradient (NPG), together
with a fixed-policy empirical Fisher eigenspectrum analysis for continuous-control
Gymnasium environments.

## Project structure

```text
vpg/                 Policy models, rollout collection, GPOMDP/NPG, training,
                     plotting, tuning, statistics, and video utilities.
fisher_analysis/     Fixed-policy empirical Fisher analysis.
results/figures/     Final plots used to present the experiments.
results/csv/         Compact Fisher summaries and per-iteration statistics.
results/runs_new/    NPG width-comparison figures, videos, and compact metrics.
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
python run.py --env_id CartPole-v1 --algorithms gpomdp
python run.py --env_id Hopper-v5 --algorithms npg
python run.py --run_mode multiseed --seeds 23 24 25 26 27
```

GPOMDP estimates the policy gradient from sampled trajectories. NPG uses the
same gradient and preconditions it with the damped empirical Fisher matrix.
Training outputs are written to `runs/`, which is intentionally ignored.

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

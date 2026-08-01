# Vanilla Policy Gradient

A from-scratch PyTorch implementation of **GPOMDP** and **Natural Policy Gradient (NPG)**, runnable on standard Gymnasium environments.

## Overview

The core algorithm is the **GPOMDP gradient estimator**:

```
∇J(θ) ≈ (1/N) Σ_i Σ_t G_{i,t} · ∇ log π_θ(a_{i,t} | s_{i,t})
```

where `G_{i,t}` is the discounted return from step `t` in trajectory `i`.
An optional baseline (mean centering) and return normalization reduce gradient variance.
An entropy bonus (`--entropy_coeff`) prevents policy collapse on Adam.

**Natural Policy Gradient** (`--algorithms npg`) preconditions the GPOMDP gradient with the inverse empirical Fisher Information Matrix:

```
θ ← θ + α · F⁻¹ · g,    F = (1/M) Σ ∇log π · (∇log π)ᵀ  +  λI
```

This removes the dependence on the parameter scale, making NPG significantly more sample-efficient than vanilla GPOMDP in practice.

The repository also includes:
- a standalone environment wrapper ([vpg/env_wrappers.py](vpg/env_wrappers.py)) for single-env experimentation outside the vectorised loop.
- a fixed-policy empirical Fisher eigenspectrum analysis for studying local
  policy compression on Hopper, HalfCheetah, and Swimmer
  ([analysis documentation](fisher_analysis/fisher_analysis.md)).

## Project Structure

```
vpg/                  # core library: policy, GPOMDP/NPG math, training loop, plotting, video
run.py                # CLI: launch training runs (thin wrapper around vpg.run)
analysis.py           # CLI: post-hoc plotting of saved reward files (thin wrapper around vpg.analysis)
tune.py               # CLI: Optuna hyperparameter search for one (env, algorithm) pair
tune_all.py           # CLI: sweeps tune.py across every environment x algorithm
notebooks/            # interactive equivalents of analysis.py's ci/compare workflows
fisher_analysis/      # self-contained fixed-policy Fisher eigenspectrum experiment
tests/                # unit tests for analysis.py
runs/                 # experiment outputs: runs/<env_id>/<algorithm>/[<width>/[<seed>/]]
report/               # curated figures for the write-up (see report/README.md)
```

`run.py`, `analysis.py`, `tune.py`, and `tune_all.py` stay runnable exactly as
documented below (`python3 run.py`, `python3 analysis.py ...`); each just
delegates to the matching module inside `vpg/`.

## Installation

Requires Python 3.10+. Install dependencies with pip:

```bash
pip install -r requirements.txt
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -r requirements.txt
```

> **MuJoCo note:** `gymnasium[mujoco]` pulls in the MuJoCo bindings automatically. No separate license is needed for MuJoCo 2.1.0+.

## Usage

Training experiments are launched via `run.py`. Defaults are read from
`config.json`; any CLI argument overrides the file.

### Single-seed run

```bash
python3 run.py --env_id CartPole-v1
```

### Multi-seed run (with 95% CI plots)

```bash
python3 run.py \
    --run_mode multiseed \
    --env_id CartPole-v1 \
    --seeds 23 24 25 26 27
```

### Natural Policy Gradient

```bash
python3 run.py --algorithms npg --env_id CartPole-v1
```

### Algorithm comparison (GPOMDP vs NPG on the same plot)

```bash
python3 run.py --algorithms gpomdp npg --run_mode multiseed --env_id CartPole-v1
```

This trains both algorithms independently (each in its own subdirectory) and saves a joint `comparison.png` with overlapping CI bands.

### Record a video of the trained policy

```bash
python3 run.py --env_id CartPole-v1 --record_video 1
```

To re-record from an existing run without retraining, use `record_checkpoint_video`, which
rebuilds the policy from that run's `config.json` and loads a saved checkpoint:

```python
from vpg.video import record_checkpoint_video

# Uses runs/CartPole-v1/policy/best.pt and writes to runs/CartPole-v1/videos/
record_checkpoint_video("runs/CartPole-v1", checkpoint_name="best.pt", n_episodes=3)
```

## Fixed-Policy Fisher Analysis

The standalone [`fisher_analysis`](fisher_analysis/) package estimates the
undamped empirical Fisher matrix of fixed random policies:

```
F = (1/M) sum_i grad log pi(a_i|s_i) grad log pi(a_i|s_i)^T
```

No learning, loss, optimizer, or parameter update occurs. The eigenvalues
measure how strongly independent parameter-space directions change the local
action distribution. A rapidly decaying spectrum indicates that most local
policy sensitivity lies in a subspace much smaller than the full parameter
space.

The standard analysis uses widths 4, 8, and 16, two hidden layers, 10
iterations, 32 asynchronous environments, 4 trajectories per environment, a
200-step horizon, and seed 23.

Run all three environments from the repository root:

```bash
# Hopper is the default.
.venv/bin/python -m fisher_analysis.run_fisher_analysis

.venv/bin/python -m fisher_analysis.run_fisher_analysis \
  --env-id HalfCheetah-v5 \
  --output-dir fisher_analysis/results/halfcheetah_width_sweep

.venv/bin/python -m fisher_analysis.run_fisher_analysis \
  --env-id Swimmer-v5 \
  --output-dir fisher_analysis/results/swimmer_width_sweep
```

The measured number of principal directions needed to explain 90% of Fisher
trace is:

| Environment | Width 4 | Width 8 | Width 16 |
|---|---:|---:|---:|
| Hopper | 6 / 86 | 7 / 198 | 10 / 518 |
| HalfCheetah | 32 / 128 | 45 / 276 | 70 / 668 |
| Swimmer | 6 / 68 | 6 / 164 | 7 / 452 |

Each cell is `90%-trace directions / policy parameters`. These are local
policy-sensitivity dimensions, not already-compressed network sizes.

Artifacts are stored in:

- [`fisher_analysis/results/hopper_width_sweep/`](fisher_analysis/results/hopper_width_sweep/)
- [`fisher_analysis/results/halfcheetah_width_sweep/`](fisher_analysis/results/halfcheetah_width_sweep/)
- [`fisher_analysis/results/swimmer_width_sweep/`](fisher_analysis/results/swimmer_width_sweep/)

Each directory contains the Fisher matrices, descending eigenvalues, parameter
layouts, rollout statistics, policy checkpoints, summary metrics, and spectrum
plots. See [`fisher_analysis/fisher_analysis.md`](fisher_analysis/fisher_analysis.md)
for the derivation, implementation details, full results, compression
interpretation, limitations, and verification record. The
[`eigenvalue_analysis.ipynb`](fisher_analysis/eigenvalue_analysis.ipynb)
notebook reproduces the plots from a single configuration cell. Select a
result directory, all or some saved widths, an output directory, plot titles,
and the optional NPG damping view:

```python
RESULTS_DIR = "fisher_analysis/results/hopper_width_sweep"
WIDTHS = None  # Uses every width in RESULTS_DIR/config.json; or e.g. [4, 16].
OUTPUT_DIR = "fisher_analysis/results/hopper_width_sweep"

RAW_TITLE = "Undamped empirical Fisher eigenspectrum"
NORMALIZED_TITLE = "Trace-normalized Fisher eigenspectrum"
CUMULATIVE_TITLE = "Cumulative Fisher trace"
DAMPING_TITLE = "Fisher spectrum after NPG diagonal damping"

SHOW_NPG_DAMPING = True
NPG_DAMPING = 0.01
DPI = 180
```

## Hyperparameters

Defaults below are `run.py`'s hardcoded argparse defaults (what you get with no `config.json`
present). The `config.json` checked into this repo overrides several of them — see
[Usage](#usage) for the priority order (hardcoded < `config.json` < CLI flags).

| Argument | Default | Description |
|---|---|---|
| `--output_dir` | `runs/<env_id>/` | Directory where all outputs are saved. Auto-named from env if not set. |
| `--run_mode` | `single` | `single` trains one seed; `multiseed` loops over `--seeds` and produces CI plots. |
| `--env_id` | `CartPole-v1` | Gymnasium environment ID. Continuous action spaces use a Gaussian policy; discrete use a softmax MLP. |
| `--seed` | `23` | Random seed for single-seed runs. |
| `--seeds` | `23 24 25 26 27` | List of seeds for multiseed runs. |
| `--n_iterations` | `2000` | Number of policy gradient update steps. |
| `--n_envs` | `16` | Number of parallel environments. Total trajectories per iteration = `n_envs × n_trajectories`. |
| `--n_trajectories` | `1` | Number of episodes collected per environment per iteration. |
| `--horizon` | `200` | Maximum episode length (truncates via `TimeLimit` wrapper). Set to `0` to use the environment default. |
| `--gamma` | `0.99` | Discount factor γ ∈ (0, 1]. |
| `--returns_implementation` | `recursive` | Discounted-return backend: numerically safe `recursive`, or guarded float64 `vectorized`. |
| `--algorithms` | `gpomdp` | Algorithm(s) to run: `gpomdp` (Adam) or `npg` (SGD + Fisher preconditioning). Pass both to trigger comparison mode. |
| `--lr` | `1e-4` | Learning rate for GPOMDP (Adam optimizer). |
| `--lr_npg` | `None` | Learning rate for NPG (SGD optimizer). Defaults to `--lr` if not set. |
| `--npg_damping` | `0.01` | Tikhonov damping λ added to the Fisher diagonal: `(F + λI)⁻¹`. Increase if the linear solve fails. |
| `--entropy_coeff` | `0.01` | Entropy bonus coefficient. Adds `entropy_coeff · H[π]` to the objective to prevent policy collapse. |
| `--center_returns` | `1` | Subtract the mean return from all returns (baseline trick). Reduces gradient variance without bias. |
| `--normalize_returns` | `0` | Divide returns by their standard deviation. Further reduces variance. |
| `--clip_actions` | `1` | Clip continuous actions to the environment's action bounds before stepping. |
| `--hidden_sizes` | `8,8` | Hidden layer sizes for the Gaussian policy (continuous envs). Comma-separated, e.g. `64,64`. |
| `--hidden_dim` | `32` | Hidden layer size for the softmax MLP policy (discrete envs). |
| `--init_log_std` | `-0.5` | Initial log standard deviation of the Gaussian policy (σ ≈ 0.6). |
| `--learn_std` | `1` | If `1`, log std is a learnable parameter. If `0`, it is fixed at `init_log_std`. |
| `--save_plots` | `1` | Save training reward plots to `output_dir`. |
| `--save_checkpoints` | `1` | Save `best.pt`/`final.pt` to `output_dir/policy/` and periodic snapshots to `output_dir/checkpoints/`. |
| `--checkpoint_interval` | `500` | Save a policy snapshot to `output_dir/checkpoints/` every N iterations. `0` disables periodic snapshots. |
| `--record_video` | `0` | Record a video of the best policy and save it to `output_dir/videos/`. |

## Outputs

### Single algorithm

All outputs are written to `--output_dir` (default: `runs/<env_id>/`).

| File | Produced by | Description |
|---|---|---|
| `config.json` | always | Full hyperparameter config for the run. |
| `training_rewards.npz` | always | Training curves as a NumPy archive: `rewards` of shape `[1, n_iterations]` (single) or `[n_seeds, n_iterations]` (multiseed), plus a `seeds` array labeling each row with its seed id. |
| `training_rewards.npy` | legacy runs | Raw one- or two-dimensional reward arrays. Supported by `notebooks/ci_plots.ipynb`, but does not store seed IDs. |
| `training_rewards_seed<seed>.npz` | multiseed mode | One archive per seed (`rewards` shape `[1, n_iterations]`, `seeds` `[<seed>]`) so each seed's curve is identifiable on disk. |
| `training_rewards.png` | single mode | Per-iteration mean return over the training batch. |
| `training_rewards_ci.png` | analysis tool/notebook | Mean training return ± 95% CI across selected `.npz` or legacy `.npy` rewards. |
| `best_seed_comparison.png` | analysis tool/notebook | Best-performing seed of each run overlaid (`analysis.py compare --mode best`). |
| `policy/best.pt` | `--save_checkpoints 1` | Policy weights with the highest training return during learning. |
| `policy/final.pt` | `--save_checkpoints 1` | Policy weights at the end of training. |
| `checkpoints/snapshot_iter_<N>.pt` | `--save_checkpoints 1` | Periodic policy snapshot saved every `--checkpoint_interval` iterations (default 500). |
| `videos/` | `--record_video 1` or `record_checkpoint_video(...)` | MP4 recordings of the policy. |

### Reward analysis

`analysis.py` accepts `.npz`/legacy `.npy` reward files or run directories. A
run directory automatically resolves to `training_rewards.npz`, falling back
to `training_rewards.npy`. Every command supports `--title`, `--xlabel`,
`--ylabel`, and `--output`.

Plot every curve from one file or run without aggregation:

```bash
python3 analysis.py single runs/Swimmer/gpomdp \
  --title "Swimmer GPOMDP"
```

Build a 95% confidence interval from an exact selection of seed files:

```bash
python3 analysis.py ci \
  runs/Hopper/npg/4x4/training_rewards_seed230.npz \
  runs/Hopper/npg/4x4/training_rewards_seed24.npz \
  runs/Hopper/npg/4x4/training_rewards_seed25.npz \
  --title "Hopper NPG 4x4" \
  --output runs/Hopper/npg/4x4/selected_seeds_ci.png
```

Compare different runs. Repeat `--run LABEL INPUT...`; each group may contain
one aggregate run or several selected files:

```bash
python3 analysis.py compare \
  --run "NPG 4x4" runs/Hopper/npg/4x4 \
  --run "NPG 8x8" runs/Hopper/npg/8x8 \
  --run "NPG 16x16" runs/Hopper/npg/16x16 \
  --title "Hopper network-width comparison" \
  --output runs/Hopper/comparison.png
```

Add `--mode best --final-window 100` to compare each group's best curve
instead of its mean and CI. Run `python3 analysis.py <mode> --help` for all
options.

The notebooks in `notebooks/` provide the same workflows interactively. Edit
the single configuration cell in `ci_plots.ipynb` to choose an aggregate run
or an exact set of reward files:

```python
INPUTS = ["runs/Swimmer/gpomdp"]
TITLE = "Swimmer GPOMDP across seeds"
XLABEL = "Iteration"
YLABEL = "Average training return"
LABEL = "Mean"
OUTPUT = "runs/Swimmer/gpomdp/training_rewards_ci.png"
```

In `comparison.ipynb`, each label can likewise contain one aggregate run or
several exact reward files. Both mean/CI and best-seed plots are generated:

```python
RUNS = {
    "NPG 4x4": ["runs/HalfCheetah/npg/4x4"],
    "NPG 8x8": ["runs/HalfCheetah/npg/8x8"],
    "NPG 16x16": ["runs/HalfCheetah/npg/16x16"],
}

MEAN_TITLE = "HalfCheetah network-width comparison"
BEST_TITLE = "HalfCheetah best-seed comparison"
XLABEL = "Iteration"
YLABEL = "Average training return"
MEAN_OUTPUT = "runs/HalfCheetah/comparison.png"
BEST_OUTPUT = "runs/HalfCheetah/best_seed_comparison.png"
FINAL_WINDOW = 100
```

Train each variant into its own directory, then analyze them together:

```
runs/<env_id>/
    gpomdp/          # full outputs for GPOMDP
    npg/             # full outputs for NPG
    comparison.png   # overlapping mean ± 95% CI curves
```

### Training reward

The training reward at each iteration is the mean episode return over the `n_envs × n_trajectories` trajectories collected by the parallel environments. It reflects the current policy's performance under stochastic exploration and is the primary learning signal to monitor.

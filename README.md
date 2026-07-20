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
- a standalone environment wrapper ([env_wrappers.py](env_wrappers.py)) for single-env experimentation outside the vectorised loop.

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

All experiments are launched via `run.py`. Defaults are read from `config.json`; any CLI argument overrides the file.

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
| `--save_checkpoints` | `1` | Save `best.pt` and `final.pt` to `output_dir/checkpoints/`. |
| `--record_video` | `0` | Record a video of the best policy and save it to `output_dir/videos/`. |

## Outputs

### Single algorithm

All outputs are written to `--output_dir` (default: `runs/<env_id>/`).

| File | Produced by | Description |
|---|---|---|
| `config.json` | always | Full hyperparameter config for the run. |
| `training_rewards.npz` | always | Training curves as a NumPy archive: `rewards` of shape `[1, n_iterations]` (single) or `[n_seeds, n_iterations]` (multiseed), plus a `seeds` array labeling each row with its seed id. |
| `training_rewards_seed<seed>.npz` | multiseed mode | One archive per seed (`rewards` shape `[1, n_iterations]`, `seeds` `[<seed>]`) so each seed's curve is identifiable on disk. |
| `training_rewards.png` | single mode | Per-iteration mean return over the training batch. |
| `training_rewards_ci.png` | multiseed mode | Mean training return ± 95% CI across seeds. |
| `checkpoints/best.pt` | `--save_checkpoints 1` | Policy weights with the highest training return during learning. |
| `checkpoints/final.pt` | `--save_checkpoints 1` | Policy weights at the end of training. |
| `videos/` | `--record_video 1` | MP4 recordings of the best policy. |

### Comparison mode (`--algorithms gpomdp npg`)

Each algorithm is written to its own subdirectory; the comparison plot is saved alongside.

```
runs/<env_id>/
    gpomdp/          # full outputs for GPOMDP
    npg/             # full outputs for NPG
    comparison.png   # overlapping mean ± 95% CI curves for both algorithms
```

### Training reward

The training reward at each iteration is the mean episode return over the `n_envs × n_trajectories` trajectories collected by the parallel environments. It reflects the current policy's performance under stochastic exploration and is the primary learning signal to monitor.

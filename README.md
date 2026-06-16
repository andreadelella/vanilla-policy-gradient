# Vanilla Policy Gradient

A from-scratch PyTorch implementation of **GPOMDP** (Gradient of the Performance of Markov Decision Processes, Baxter & Bartlett 2001) — also known as the REINFORCE / vanilla policy gradient estimator.

## Overview

The goal of this repository is to implement and experiment with policy gradient algorithms on standard Gymnasium environments (discrete: CartPole, continuous: MuJoCo locomotion tasks).

The core algorithm is the **GPOMDP gradient estimator**:

```
∇J(θ) ≈ (1/N) Σ_i Σ_t G_{i,t} · ∇ log π_θ(a_{i,t} | s_{i,t})
```

where `G_{i,t}` is the discounted return from step `t` in trajectory `i`.
An optional baseline (mean centering) and return normalization are supported to reduce gradient variance.

The repository also includes:
- an alternative implementation of the same estimator using **eligibility traces** ([gpomdp_elig_traces.py](gpomdp_elig_traces.py)) — the recursive form `z_{t+1} = β z_t + ∇log π`, equivalent to the batched version for β = γ.
- a standalone environment wrapper ([env_wrappers.py](env_wrappers.py)) that adds horizon truncation and action clipping, useful for single-env experimentation outside the vectorised training loop.

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

All experiments are launched via `run.py`.

### Single-seed run

```bash
python3 run.py --env_id CartPole-v1 --output_dir results/cartpole
```

### Multi-seed run (with confidence intervals)

```bash
python3 run.py \
    --run_mode multiseed \
    --env_id Hopper-v5 \
    --seeds 0 1 2 3 4 \
    --output_dir results/hopper_multiseed
```

### Record a video of the trained policy

```bash
python3 run.py --env_id CartPole-v1 --record_video 1 --output_dir results/cartpole
```

## Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--output_dir` | `runs` | Directory where all outputs are saved (config, plots, videos). |
| `--run_mode` | `single` | `single` trains one agent with `--seed`; `multiseed` loops over `--seeds` and produces confidence-interval plots. |
| `--env_id` | `CartPole-v1` | Gymnasium environment ID. Continuous action spaces use a Gaussian policy; discrete spaces use a softmax MLP. |
| `--seed` | `23` | Random seed for single-seed runs. |
| `--seeds` | `23 24 25 26 27` | List of seeds for multiseed runs. |
| `--n_iterations` | `2000` | Number of policy gradient update steps. |
| `--n_envs` | `16` | Number of parallel environments. Total trajectories per iteration = `n_envs × n_trajectories`. |
| `--n_trajectories` | `1` | Number of episodes collected per environment per iteration. |
| `--horizon` | `200` | Maximum episode length (truncates via `TimeLimit` wrapper). Set to `0` to use the environment default. |
| `--gamma` | `0.99` | Discount factor γ ∈ (0, 1]. Controls how much future rewards are down-weighted. |
| `--lr` | `1e-4` | Adam learning rate. |
| `--center_returns` | `1` | Subtract the mean return from all returns (baseline trick). Reduces gradient variance without introducing bias. |
| `--normalize_returns` | `0` | Divide returns by their standard deviation. Further reduces variance but can destabilize early training. |
| `--clip_actions` | `1` | Clip continuous actions to the environment's action bounds before stepping. |
| `--hidden_sizes` | `8,8` | Hidden layer sizes for the Gaussian policy (continuous envs), e.g. `64,64`. Comma-separated. |
| `--hidden_dim` | `32` | Hidden layer size for the softmax MLP policy (discrete envs). |
| `--init_log_std` | `-0.5` | Initial log standard deviation of the Gaussian policy. Corresponds to σ ≈ 0.6 at start. |
| `--learn_std` | `1` | If `1`, the log standard deviation is a learnable parameter. If `0`, it stays fixed at `init_log_std`. |
| `--save_plots` | `1` | Save training reward plots to `output_dir`. |
| `--save_checkpoints` | `1` | Save `best.pt` and `final.pt` to `output_dir/checkpoints/`. |
| `--record_video` | `0` | Record a video of the best policy and save it to `output_dir/videos/`. |

## Outputs

All outputs are written to `--output_dir` (default: `runs/`).

| File | Produced by | Description |
|---|---|---|
| `config.json` | always | Full hyperparameter config for the run. |
| `training_rewards.png` | single mode | Per-iteration mean return over the training batch. |
| `training_rewards_ci.png` | multiseed mode | Mean training return ± 95 % CI across seeds. |
| `training_rewards.npy` | multiseed mode | Raw per-seed training curves, shape `[n_seeds, n_iterations]`. |
| `checkpoints/best.pt` | `--save_checkpoints 1` | Weights of the policy with the highest training return seen during learning. |
| `checkpoints/final.pt` | `--save_checkpoints 1` | Weights of the policy at the end of training. |
| `videos/` | `--record_video 1` | MP4 of the best policy found during training, capped at `--horizon` steps. |
| `config_seed_<n>.json` | multiseed mode | Per-seed config snapshot. |

### Training rewards

The training reward at each iteration is the mean episode return over the `n_envs × n_trajectories` trajectories collected by the parallel environments. It reflects the current policy's performance under stochastic exploration and is the primary learning signal to monitor.

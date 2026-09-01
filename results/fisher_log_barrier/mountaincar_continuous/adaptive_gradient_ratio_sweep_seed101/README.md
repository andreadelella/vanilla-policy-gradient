# Adaptive Fisher-gradient ratio sweep

Single-seed MountainCarContinuous-v0 exploration with seed 101.

The applied Fisher gradient is rescaled on every update to have exactly the
requested norm relative to the reward gradient. This is an adaptive target,
not a maximum-norm cap. The 0% run still estimates the Fisher but applies no
Fisher gradient.

## Training protocol

- Ratios: 0%, 5%, 10%, 20%, 50%
- Updates: 250
- Learning rate: 0.02
- Reward trajectories: 32
- Fisher trajectories: 256, collected separately
- Hidden layers: 2, 2
- Horizon: 500 primitive steps
- Action repeat: 5
- Native reward only
- Fisher boundary: mu = 0

All five runs completed without a strict-domain failure. Each run was
reanalyzed at seven checkpoints using 4,096 common-random-number trajectories
in float64.

## Endpoint summary

| Target | Last-10 return | Minimum eigenvalue | Condition number |
| ---: | ---: | ---: | ---: |
| 0% | 88.76 | 8.06e-7 | 1.18e9 |
| 5% | 88.67 | 5.87e-8 | 1.55e10 |
| 10% | 90.97 | 1.80e-7 | 5.67e9 |
| 20% | 90.84 | 2.23e-8 | 6.25e10 |
| 50% | 86.04 | 5.79e-11 | 4.78e12 |

For this seed, 10% and 20% solve the task by the last-10-return criterion.
The 0% run has the best endpoint conditioning. Increasing the target does not
monotonically improve Fisher non-degeneracy, and 50% is worse on both task
return and endpoint conditioning. Multi-seed confirmation is required.

Aggregate tables and plots are under `aggregate/`.

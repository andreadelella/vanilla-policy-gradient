# MountainCar beta geometry sweep

All runs use native `MountainCar-v0` reward, paired seeds 24-29, a 4x4
reference-logit policy, 400 updates, `mu=0`, 32 reward trajectories, 128
Fisher trajectories, and the strict undamped Fisher domain.

| Barrier ratio | Complete | Return mean | Paired vs reward | Strict / near solved | Rank | Effective rank | Median min eigenvalue | Mean log10 positive condition | 90% trace components |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reward only | 6/6 | -129.55 | 0.00 | 1 / 2 | 24.50 | 1.120 | -6.77e-16 | 13.742 | 1.00 |
| 5% | 6/6 | -127.55 | +2.00 | 0 / 2 | 41.17 | 1.873 | 6.65e-12 | 12.746 | 2.17 |
| 10% | 4/6 | -130.87 | +6.26 | 0 / 2 | 40.75 | 1.596 | 1.29e-12 | 13.296 | 1.50 |
| 20% | 4/6 | -113.79 | +11.12 | 0 / 3 | 42.00 | 1.775 | 3.26e-11 | 11.807 | 1.75 |
| 35% | 6/6 | -115.33 | +14.22 | 1 / 4 | 41.67 | 1.950 | 5.21e-12 | 12.638 | 2.50 |
| 50% | 6/6 | -122.15 | +7.40 | 0 / 2 | 41.17 | 1.776 | 3.67e-11 | 12.057 | 1.67 |

The 35% setting gives the best overall balance: all seeds complete, task
return improves, and the Fisher has the highest mean effective rank and the
widest 90%-trace support. The 20% setting has the best conditioning among its
completed runs, but two seeds leave the strict domain. At 50%, completion
remains reliable and the spectral floor is stronger, but return and effective
rank regress relative to 35%.

Statistics for incomplete levels use completed seeds only. The 10% failures
occur at updates 384 (seed 24) and 207 (seed 28); the 20% failures occur at
updates 272 (seed 25) and 261 (seed 27). Final Fisher diagnostics use a fresh
sample, so a completed training run can still have a rank-deficient final
empirical Fisher.

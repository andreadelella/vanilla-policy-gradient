# Log-barrier versus reward-only trajectory Fisher

This comparison uses MountainCarContinuous seed 101 at checkpoints 50, 100,
150, 200, and 250.

Both policies are analyzed on-policy using:

- 4,096 fresh trajectories per checkpoint;
- diagnostic seed 4101 and matched environment reset seeds;
- the same Gaussian random-number stream;
- native horizon 500 and action repeat 5;
- raw Gaussian samples for policy scores and clipped environment actions;
- float64 trajectory scores, Fisher matrices, and eigendecomposition;
- `fisher_analysis.analyze_fisher` for all spectral metrics.

The state-action paths cannot be literally shared because the policies select
different actions and therefore induce different on-policy trajectory
distributions. Evaluating both policies on one policy's paths would estimate an
off-policy matrix. The matched seeds and random stream provide the valid paired
comparison.

## Results

| Update | Barrier min eig | GPOMDP min eig | Min-eig ratio | Condition improvement |
|---:|---:|---:|---:|---:|
| 50  | 2.992e-13 | 5.472e-14 | 5.47x | 6.87x |
| 100 | 1.695e-9  | 2.648e-10 | 6.40x | 14.14x |
| 150 | 6.890e-8  | 1.360e-8  | 5.06x | 36.48x |
| 200 | 8.586e-7  | 2.036e-8  | 42.18x | 126.42x |
| 250 | 1.743e-6  | 4.984e-8  | 34.96x | 98.05x |

Condition improvement is GPOMDP's full condition number divided by the
barrier's. The barrier has the better value when this ratio is above one.

The barrier consistently improves the smallest eigenvalue and full condition
number. Reward-only GPOMDP learns the task faster and finishes with the higher
diagnostic return. Both remain spectrally concentrated: two components explain
90% of the trace at every checkpoint.

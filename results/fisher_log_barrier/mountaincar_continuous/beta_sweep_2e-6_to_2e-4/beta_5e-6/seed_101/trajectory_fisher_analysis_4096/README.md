# Trajectory Fisher spectral analysis

This analysis uses seed 101 and beta `5e-6`.

For each saved policy checkpoint, it collects 4,096 fresh trajectories and
estimates the undamped whole-trajectory score Fisher

```
z_k = sum_t grad_theta log pi_theta(a_kt | s_kt)
F_hat = (1 / N) sum_k z_k z_k^T
```

This is the Fisher used by the log-barrier. It is not the transition-level
Fisher. Raw Gaussian samples are used in each score while clipped actions are
sent to the environment. The native 500-step horizon and action repeat 5 are
preserved.

Policy scores, Fisher accumulation, and eigendecomposition use float64.
`fisher_analysis.analyze_fisher` performs the symmetry, PSD, trace, numerical
rank, conditioning, effective-rank, stable-rank, and trace-coverage analysis.

## Results

| Update | Rank | Minimum eigenvalue | Condition number | Effective rank | 90% trace components |
|---:|---:|---:|---:|---:|---:|
| 50  | 15/16 | 2.992e-13 | 9.364e13 | 2.107 | 2 |
| 100 | 16/16 | 1.695e-9  | 9.388e10 | 2.062 | 2 |
| 150 | 16/16 | 6.890e-8  | 2.040e9  | 2.476 | 2 |
| 200 | 16/16 | 8.586e-7  | 5.647e8  | 2.020 | 2 |
| 250 | 16/16 | 1.743e-6  | 3.668e8  | 1.869 | 2 |

The low end of the spectrum improves strongly during training. The Fisher is
nevertheless concentrated: two directions account for 90% of its trace at
every checkpoint.

## Batch-size check

The saved 4,096 trajectory-score rows were also analyzed at prefixes of 256,
512, 1,024, and 2,048. At update 250:

- the minimum eigenvalue stays between `1.713e-6` and `1.801e-6`;
- the condition number stays between `3.47e8` and `3.83e8`;
- the effective rank stays between `1.857` and `1.891`;
- the 2,048-sample Fisher is 2.16% from the 4,096-sample Fisher in relative
  Frobenius norm.

The endpoint spectrum is therefore stable at this diagnostic batch scale.

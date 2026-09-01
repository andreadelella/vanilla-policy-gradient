# True 20% Fisher-gradient clipping

Single-seed MountainCarContinuous sweep using native reward, `mu=0`, and
learning rate `0.015`.

The nominal Fisher log-barrier gradient is capped by

```text
scale = min(1, 0.2 * ||reward gradient|| / ||nominal Fisher gradient||)
```

This is an upper cap only: a weak Fisher gradient is never amplified.

## Main result

The cap was active on 15.6% to 50.0% of updates, mostly early in training.
No run had a last-10 mean return above the solve threshold of 90. The best
last-10 mean was 89.35 at nominal beta `5e-5`.

For the largest nominal betas, clipping avoided the fixed-beta collapse:

| Nominal beta | Fixed return | Capped return | Fixed condition | Capped condition |
| ---: | ---: | ---: | ---: | ---: |
| `1e-4` | 31.70 | 88.53 | `1.13e14` | `6.24e9` |
| `2e-4` | 27.60 | 88.65 | `1.29e14` | `4.35e9` |

The capped endpoint minimum eigenvalues were `1.17e-7` and `1.69e-7`,
respectively, from matched 4,096-trajectory Fisher estimates. Nominal beta
`2e-5` was the main outlier, with last-10 return 81.13.

These are seed-101 results and should be treated as exploratory.

## Contents

- `runs/`: checkpoints, training logs, and per-run trajectory Fisher analysis
- `aggregate/summary.csv`: endpoint metrics and fixed-beta comparison
- `aggregate/training_long.csv`: per-update clipping and gradient metrics
- `aggregate/spectral_long.csv`: matched checkpoint spectra
- `aggregate/figures/`: learning, clipping, and geometry plots

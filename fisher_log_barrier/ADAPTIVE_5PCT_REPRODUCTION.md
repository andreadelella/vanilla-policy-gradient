# Adaptive Fisher gradient experiment

This experiment keeps the applied Fisher regularizer gradient at 5% of the
reward gradient norm on every policy update. The scalar beta is recomputed from
the two component gradients and the optimizer receives their combined gradient.

Install the dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Run the reproduced seed-101 experiment:

```powershell
python -m fisher_log_barrier.continuous_mountain_car_experiment `
  --seed 101 `
  --target-gradient-ratio 0.05 `
  --updates 250 `
  --output results/fisher_log_barrier/mountaincar_continuous/adaptive_5pct_seed101
```

The run writes its configuration, per-update diagnostics, checkpoints, and
summary to the output directory. The diagnostics include the effective beta,
component gradient norms, achieved ratio, and gradient cosine.

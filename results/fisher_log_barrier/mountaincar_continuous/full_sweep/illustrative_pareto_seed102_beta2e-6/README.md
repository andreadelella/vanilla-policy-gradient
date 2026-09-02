# Tuned illustrative case

This single-seed comparison was selected to illustrate the desired outcome:
the log-barrier policy solves MountainCarContinuous while the paired GPOMDP
policy does not, and its endpoint Fisher has better non-degeneracy metrics.

- Seed: 102
- GPOMDP learning rate: 0.015
- Log-barrier learning rate: 0.02
- Barrier coefficient: 2e-6
- Fisher evaluation: 4,096 matched trajectories per checkpoint

At update 250, the log-barrier run has:

- last-10 training return 90.77 versus 88.24;
- diagnostic return 90.78 versus 88.09;
- minimum eigenvalue 1.95x larger;
- condition number 1.29x smaller;
- normalized positive-spectrum log-determinant 0.60 larger.

This is an intentionally tuned, selected single-seed example. It is not
evidence that the barrier dominates GPOMDP across seeds or under equal
hyperparameters.

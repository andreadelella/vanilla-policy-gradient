# Neural discrete categorical log barrier

This isolated package implements the **on-policy sampled-state conditional
categorical log barrier** for discrete policies. It does not modify `vpg/` and
does not identify the barrier with a global neural Fisher log determinant.

Commands:

```powershell
python -m exploration.neural_discrete_log_barrier.verify
python -m exploration.neural_discrete_log_barrier.run_experiment --stage cartpole-smoke
python -m exploration.neural_discrete_log_barrier.run_experiment --stage acrobot-lr-screen
python -m exploration.neural_discrete_log_barrier.run_experiment --stage acrobot-lr-continuation
python -m exploration.neural_discrete_log_barrier.run_experiment --stage acrobot-gpomdp-confirmation
python -m exploration.neural_discrete_log_barrier.run_experiment --stage acrobot-regularizer-calibration
python -m exploration.neural_discrete_log_barrier.run_experiment --stage acrobot-gpomdp-ablation
python -m exploration.neural_discrete_log_barrier.run_experiment --stage acrobot-pilot
python -m exploration.neural_discrete_log_barrier.run_experiment --stage acrobot-confirmatory
python -m exploration.neural_discrete_log_barrier.run_experiment --stage fisher
python -m exploration.neural_discrete_log_barrier.run_experiment --stage report
```

The stages are configuration-validated and resumable at complete run-unit
boundaries. Outputs are under
`exploration/results/neural_discrete_log_barrier/`. The report and manifest
record the negative Acrobot decision gate, the fixed reference-state-bank hash,
and the preserved but scientifically excluded fixed-segment pilot quarantine.

## Baseline gate before barrier ablations

The earlier Acrobot comparison is not evidence against GPOMDP or the barrier:
the nominal 120-update step budget produced only 31 complete-episode optimizer
updates.  The replacement workflow therefore uses
`complete_episodes_by_update`.  One update contains exactly eight complete
episodes, and requested optimizer-update and episode counts are both checked in
every run archive.

The focused screen holds the architecture and estimator fixed and tests
learning rates `1e-4`, `3e-4`, `1e-3`, `3e-3`, and `1e-2`.  It runs two paired
seeds for 300 updates (2,400 episodes), continues the best two configurations
from scratch for 1,000 updates (8,000 episodes), then confirms the provisional
choice on five new seeds.  Stochastic complete-episode evaluation is primary;
deterministic evaluation, termination rate, episode length, actual environment
steps, and optimizer-update counts are retained alongside it.

The barrier ablation remains closed unless the five-seed GPOMDP confirmation is
finite, at least four seeds improve by 100 reward points, the median final
stochastic return is at least -300, the median goal-termination rate is at
least 0.8, and the two-sided 95% Student-t interval for paired seed improvement
has a lower endpoint above zero.  Gymnasium's official solved threshold of -100
is reported separately and is not silently substituted for this baseline gate.

After the gate passes, the episode-based ablation calibrates the entropy and
conditional-barrier coefficients on five disjoint early-policy seeds.  It
matches the median regularizer-gradient norm to 0.3 times the reward-gradient
norm and does not inspect reward outcomes when selecting either coefficient.
Five further paired seeds then compare reward-only GPOMDP, fixed entropy, fixed
sampled conditional barrier, and a barrier disabled after 25% of the 1,000
updates.  NPG is excluded until its own learning-rate and damping baseline gate
is run; the old hard-coded NPG rate is not reused.

## macOS: 60-pair reliability extension

The frozen reliability extension adds seeds 521--560 to the existing paired
seeds 501--520. Run one method at a time; each command parallelizes across that
method's 40 seeds and resumes completed seed directories:

```bash
.venv/bin/python -m exploration.neural_discrete_log_barrier.run_experiment \
  --stage acrobot-reliability-extension \
  --method reward_only \
  --parallel-workers 4

.venv/bin/python -m exploration.neural_discrete_log_barrier.run_experiment \
  --stage acrobot-reliability-extension \
  --method logbarrier_handoff_h25 \
  --parallel-workers 4
```

Choose the worker count conservatively for the Mac's available performance
cores and memory. Each worker sets PyTorch to one thread. After both commands
finish, combine the original and extension cohorts and run the mechanism audit:

```bash
.venv/bin/python -m exploration.neural_discrete_log_barrier.run_experiment \
  --stage acrobot-reliability-extension-summary
```

The summary command requires the existing seeds 501--520 and the frozen
reference-state bank to be present under the results root. It reports the
predeclared catastrophic-failure endpoint, paired return and true
environment-step AUC, and an exact paired McNemar test. At update 250 it saves
per-state action probabilities, entropy distributions, greedy-action
agreement, top-two action margins, disagreement-state masks, and the frequency
of the disagreement region under each policy's independently sampled on-policy
state distribution. It does not launch a hyperparameter search.

# NPG × temporary categorical log-barrier factorial

This isolated package asks whether temporary categorical support protection is
still useful after natural preconditioning. It does not modify `vpg/`, add an
environment, implement TRPO, or identify the sampled-state barrier with a
global neural Fisher log determinant.

Run stages in order:

```powershell
python -m exploration.npg_logbarrier_factorial.run_experiment fisher-validation
python -m exploration.npg_logbarrier_factorial.run_experiment exact-two-state
python -m exploration.npg_logbarrier_factorial.run_experiment sampled-two-state --preset smoke
python -m exploration.npg_logbarrier_factorial.run_experiment sampled-two-state --preset pilot
python -m exploration.npg_logbarrier_factorial.run_experiment sampled-two-state --preset full
python -m exploration.npg_logbarrier_factorial.run_experiment acrobot-pilot --parallel-workers 4
python -m exploration.npg_logbarrier_factorial.run_experiment acrobot-confirmatory --parallel-workers 4
python -m exploration.npg_logbarrier_factorial.run_experiment fisher-diagnostics
python -m exploration.npg_logbarrier_factorial.run_experiment report
```

The validation stage is a hard gate. Main runners refuse to execute unless the
categorical enumerated Fisher and analytic Gaussian Fisher match their
forward-KL Hessians in float64.

Primary natural steps solve

\[
(F+\lambda I)x=g_J+\beta_t g_B,
\qquad
\Delta\theta=x\sqrt{\frac{2\delta}{x^T F x}}.
\]

The damped matrix is used for the solve and the undamped Fisher for predicted
KL. A nonpositive quadratic form is an explicit invalid update. There is no
absolute-value repair, silent fallback, or line search.

Acrobot preserves the frozen `(8,8)` categorical MLP, eight complete episodes
per update, 1000 updates, centered but unnormalized reward-to-go, and the
existing barrier coefficient/handoff schedule. The NPG pilot uses disjoint
seeds and selects damping/target KL before the fresh 30-pair confirmatory
cohort. Historical 60-pair GP archives are never overwritten.


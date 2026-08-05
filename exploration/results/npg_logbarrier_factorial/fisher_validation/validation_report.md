# Policy-Fisher validation

Gate status: **PASS**.

The categorical action-enumerated Fisher and the analytic diagonal-Gaussian Fisher are compared with the Hessian of mean forward KL at the reference parameters. Sampled score Fishers use raw policy actions and are reported over increasing action samples per state. Main factorial runners must refuse to execute unless `validation_result.json` has `passed=true`.

param(
    [string]$OutputPath = "exploration/experiment_results_catalog.csv"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$rows = [System.Collections.Generic.List[object]]::new()
$script:recordNumber = 0

function Relative-Path([string]$Path) {
    $absolute = (Resolve-Path (Join-Path $root $Path)).Path
    return $absolute.Substring($root.Length + 1).Replace('\', '/')
}

function Compact-Json([hashtable]$Table) {
    if ($Table.Count -eq 0) { return "{}" }
    return ($Table | ConvertTo-Json -Compress -Depth 8)
}

function Add-Result {
    param(
        [int]$StageOrder,
        [string]$Stage,
        [string]$ExperimentId,
        [string]$ExperimentTitle,
        [string]$ScientificRole,
        [string]$Status,
        [string]$Environment,
        [string]$Method = "",
        [string]$Comparison = "",
        [string]$Initialization = "",
        [string]$SampleSize = "",
        [string]$UnitOfAnalysis = "",
        [hashtable]$Configuration = @{},
        [string]$Metric,
        [string]$Estimate,
        [string]$CiLow = "",
        [string]$CiHigh = "",
        [string]$Uncertainty = "none",
        [string]$ResultScope = "aggregate",
        [string]$SourceTex = "",
        [string]$SourceArtifact,
        [string]$Interpretation = "",
        [string]$Limitations = ""
    )
    $script:recordNumber++
    $rows.Add([pscustomobject][ordered]@{
        catalog_schema_version = 1
        record_id = ('R{0:D6}' -f $script:recordNumber)
        stage_order = $StageOrder
        stage = $Stage
        experiment_id = $ExperimentId
        experiment_title = $ExperimentTitle
        scientific_role = $ScientificRole
        status = $Status
        environment = $Environment
        method = $Method
        comparison = $Comparison
        initialization = $Initialization
        sample_size = $SampleSize
        unit_of_analysis = $UnitOfAnalysis
        configuration_json = (Compact-Json $Configuration)
        metric = $Metric
        estimate = $Estimate
        ci95_low = $CiLow
        ci95_high = $CiHigh
        uncertainty_method = $Uncertainty
        result_scope = $ResultScope
        source_tex = $SourceTex
        source_artifact = (Relative-Path $SourceArtifact)
        interpretation = $Interpretation
        limitations = $Limitations
    }) | Out-Null
}

function Row-Config($Row, [string[]]$Exclude) {
    $h = @{}
    foreach ($property in $Row.PSObject.Properties) {
        if ($Exclude -notcontains $property.Name) { $h[$property.Name] = [string]$property.Value }
    }
    return $h
}

function Add-MeanCsv {
    param(
        [string]$Path, [int]$StageOrder, [string]$Stage, [string]$ExperimentId,
        [string]$Title, [string]$Role, [string]$Status, [string]$Environment,
        [string]$Tex, [string]$MetricColumn, [string]$MeanColumn,
        [string]$LowColumn, [string]$HighColumn, [string]$MethodColumn,
        [string]$SampleColumn, [string]$InitializationColumn = "",
        [string]$ComparisonColumn = "", [string]$Interpretation = "",
        [string]$Limitations = "", [string]$Uncertainty = "two-sided 95% Student-t interval across independent runs/seeds"
    )
    $data = Import-Csv (Join-Path $root $Path)
    foreach ($r in $data) {
        $exclude = @($MetricColumn, $MeanColumn, $LowColumn, $HighColumn, $MethodColumn, $SampleColumn)
        if ($InitializationColumn) { $exclude += $InitializationColumn }
        if ($ComparisonColumn) { $exclude += $ComparisonColumn }
        Add-Result -StageOrder $StageOrder -Stage $Stage -ExperimentId $ExperimentId `
            -ExperimentTitle $Title -ScientificRole $Role -Status $Status -Environment $Environment `
            -Method ([string]$r.$MethodColumn) -Comparison $(if ($ComparisonColumn) {[string]$r.$ComparisonColumn} else {""}) `
            -Initialization $(if ($InitializationColumn) {[string]$r.$InitializationColumn} else {""}) `
            -SampleSize ([string]$r.$SampleColumn) -UnitOfAnalysis "independent run/seed" `
            -Configuration (Row-Config $r $exclude) -Metric ([string]$r.$MetricColumn) `
            -Estimate ([string]$r.$MeanColumn) -CiLow ([string]$r.$LowColumn) -CiHigh ([string]$r.$HighColumn) `
            -Uncertainty $Uncertainty -SourceTex $Tex -SourceArtifact $Path `
            -Interpretation $Interpretation -Limitations $Limitations
    }
}

function Add-WideCsv {
    param(
        [string]$Path, [int]$StageOrder, [string]$Stage, [string]$ExperimentId,
        [string]$Title, [string]$Role, [string]$Status, [string]$Environment,
        [string]$Tex, [string[]]$IdColumns, [string[]]$MetricColumns,
        [string]$MethodColumn = "", [string]$InitializationColumn = "",
        [string]$SampleColumn = "", [string]$Interpretation = "", [string]$Limitations = "",
        [string]$ResultScope = "aggregate"
    )
    foreach ($r in (Import-Csv (Join-Path $root $Path))) {
        $config = @{}
        foreach ($id in $IdColumns) { if ($r.PSObject.Properties.Name -contains $id) { $config[$id] = [string]$r.$id } }
        foreach ($metric in $MetricColumns) {
            if (($r.PSObject.Properties.Name -contains $metric) -and ([string]$r.$metric -ne "")) {
                Add-Result -StageOrder $StageOrder -Stage $Stage -ExperimentId $ExperimentId `
                    -ExperimentTitle $Title -ScientificRole $Role -Status $Status -Environment $Environment `
                    -Method $(if ($MethodColumn) {[string]$r.$MethodColumn} else {""}) `
                    -Initialization $(if ($InitializationColumn) {[string]$r.$InitializationColumn} else {""}) `
                    -SampleSize $(if ($SampleColumn) {[string]$r.$SampleColumn} else {""}) `
                    -UnitOfAnalysis $(if ($ResultScope -eq 'seed') {'one seed'} else {'deterministic run/configuration'}) `
                    -Configuration $config -Metric $metric -Estimate ([string]$r.$metric) `
                    -ResultScope $ResultScope -SourceTex $Tex -SourceArtifact $Path `
                    -Interpretation $Interpretation -Limitations $Limitations
            }
        }
    }
}

# Step 1: values documented in the exact-identity TeX table.
$identityCases = @(
    @{name='k2_uniform'; k='2'; minp='0.5'; logdet='0'; fd='0'},
    @{name='k3_asymmetric'; k='3'; minp='0.1052'; logdet='0'; fd='5.94e-11'},
    @{name='k10_random_seed23'; k='10'; minp='0.02278'; logdet='3.55e-15'; fd='6.03e-11'},
    @{name='k100_random_seed23'; k='100'; minp='0.0006007'; logdet='5.68e-14'; fd='2.80e-9'},
    @{name='k10_near_boundary'; k='10'; minp='9.352e-8'; logdet='1.42e-14'; fd='7.38e-11'}
)
foreach ($c in $identityCases) {
    foreach ($pair in @(@('minimum_probability',$c.minp),@('logdet_identity_error',$c.logdet),@('directional_finite_difference_error',$c.fd),@('all_required_checks_passed','1'))) {
        Add-Result -StageOrder 1 -Stage 'categorical_identity' -ExperimentId 'step1_exact_identity' `
            -ExperimentTitle 'Exact categorical Fisher and reduced log-determinant identity' `
            -ScientificRole 'deterministic algebraic verification' -Status 'complete' -Environment 'categorical softmax, no rewards' `
            -Method 'exact_action_enumeration' -Configuration @{case=$c.name; action_count=$c.k; dtype='float64'; seed='23 where applicable'} `
            -Metric $pair[0] -Estimate $pair[1] -SourceTex 'exploration/categorical_bandit_identity.tex' `
            -SourceArtifact 'exploration/categorical_bandit_exploration.tex' `
            -Interpretation 'The reduced categorical Fisher identity, null direction, Bartlett identity, and barrier derivatives passed the declared checks.' `
            -Limitations 'This is an exact categorical identity at a fixed input, not a global neural Fisher determinant.'
    }
}

# Step 2: all categorical training presets, including smoke (clearly labelled).
foreach ($preset in @('smoke','pilot','eta','paper')) {
    $role = if ($preset -eq 'smoke') {'software smoke test'} elseif ($preset -eq 'paper') {'primary paper-grid reproduction'} elseif ($preset -eq 'eta') {'barrier-strength ablation'} else {'pilot'}
    $status = if ($preset -eq 'smoke') {'complete_non_scientific'} else {'complete'}
    $base = "exploration/results/categorical_bandit/$preset"
    Add-MeanCsv -Path "$base/summary.csv" -StageOrder 2 -Stage 'categorical_bandit_training' `
        -ExperimentId "step2_$preset" -Title "Categorical Gaussian bandit training: $preset preset" `
        -Role $role -Status $status -Environment 'stationary Gaussian K-armed bandit' `
        -Tex 'exploration/categorical_bandit_exploration.tex' -MetricColumn 'metric' -MeanColumn 'mean' `
        -LowColumn 'ci95_lower' -HighColumn 'ci95_upper' -MethodColumn 'algorithm' -SampleColumn 'num_runs' `
        -Interpretation 'Performance and categorical geometry are reported together; the paper grid reproduces support preservation and the barrier-strength trade-off.' `
        -Limitations 'A higher determinant or probability floor does not by itself prove better conditioning or better reward.'
    Add-MeanCsv -Path "$base/paired_final_differences.csv" -StageOrder 2 -Stage 'categorical_bandit_training' `
        -ExperimentId "step2_${preset}_paired" -Title "Paired LB-SGB differences: $preset preset" `
        -Role $role -Status $status -Environment 'stationary Gaussian K-armed bandit' `
        -Tex 'exploration/categorical_bandit_exploration.tex' -MetricColumn 'metric' -MeanColumn 'mean_lb_minus_baseline' `
        -LowColumn 'ci95_lower' -HighColumn 'ci95_upper' -MethodColumn 'lb_algorithm' -ComparisonColumn 'baseline' `
        -SampleColumn 'paired_runs' -Interpretation 'Positive values mean LB-SGB exceeds the named baseline for the recorded metric; metric direction still matters.' `
        -Limitations 'Paired intervals quantify seed-level uncertainty and do not prove the proposed geometric mechanism is causal.'
}

# Step 3 exact two-state MDP: one deterministic row per configuration and endpoint metric.
Add-WideCsv -Path 'exploration/results/tabular_mdp/two_step_trap/summary.csv' -StageOrder 3 `
    -Stage 'exact_two_state_mdp' -ExperimentId 'step3_exact_geometry' `
    -Title 'Exact two-state tabular MDP geometry and six-objective comparison' `
    -Role 'primary exact tabular experiment' -Status 'complete' -Environment 'deterministic two-step three-action MDP' `
    -Tex 'exploration/tabular_mdp/two_state_geometry.tex' `
    -IdColumns @('experiment','label','alpha','beta','updates','run','finite') `
    -MetricColumns @('final_return','final_q','final_p_good','final_min_pi0','final_min_pi1','final_lambda_min_f_pool','final_logdet_f_pool') `
    -MethodColumn 'method' -Interpretation 'Separates conditional action protection, policy-dependent state weighting, and explicit visitation pressure without sampling noise.' `
    -Limitations 'Exact tabular gradients and disjoint state parameters do not directly transfer to shared neural policies.'

Add-WideCsv -Path 'exploration/results/tabular_mdp/smoke_check/summary.csv' -StageOrder 3 `
    -Stage 'exact_two_state_mdp' -ExperimentId 'step3_smoke_check' `
    -Title 'Exact two-state implementation smoke check' -Role 'software smoke test' `
    -Status 'complete_non_scientific' -Environment 'deterministic two-step three-action MDP' `
    -Tex 'exploration/tabular_mdp/two_state_geometry.tex' `
    -IdColumns @('experiment','label','alpha','beta','updates','run','finite') `
    -MetricColumns @('final_return','final_q','final_p_good','final_min_pi0','final_min_pi1','final_lambda_min_f_pool','final_logdet_f_pool') `
    -MethodColumn 'method' -Interpretation 'Short deterministic execution check retained for reproducibility.' `
    -Limitations 'Non-scientific smoke preset; use the full Step 3 experiment for conclusions.'

$step3Verification = Get-Content (Join-Path $root 'exploration/results/tabular_mdp/two_step_trap/verification.json') -Raw | ConvertFrom-Json
foreach ($p in $step3Verification.metrics.PSObject.Properties) {
    Add-Result -StageOrder 3 -Stage 'exact_two_state_mdp' -ExperimentId 'step3_verification' `
        -ExperimentTitle 'Exact Step 3 implementation verification' -ScientificRole 'verification' -Status 'complete' `
        -Environment 'deterministic two-step three-action MDP' -Method 'exact_enumeration_and_autograd' `
        -Metric $p.Name -Estimate ([string]$p.Value) -SourceTex 'exploration/tabular_mdp/two_state_geometry.tex' `
        -SourceArtifact 'exploration/results/tabular_mdp/two_step_trap/verification.json' `
        -Interpretation 'All declared exact identities and finite-run checks passed.'
}

# Step 4 finite-batch audit and sampled training (pilot and full remain distinct).
foreach ($preset in @('pilot','full')) {
    $base = "exploration/results/tabular_mdp/two_step_trap_sampled/$preset"
    $role = if ($preset -eq 'full') {'primary finite-batch experiment'} else {'pilot'}
    Add-MeanCsv -Path "$base/summary.csv" -StageOrder 4 -Stage 'sampled_two_state_mdp' `
        -ExperimentId "step4_training_$preset" -Title "Finite-batch sampled two-state training: $preset" `
        -Role $role -Status 'complete' -Environment 'sampled two-step three-action MDP' `
        -Tex 'exploration/sampled_tabular_mdp/sampled_two_state.tex' -MetricColumn 'metric' -MeanColumn 'mean' `
        -LowColumn 'ci_lower' -HighColumn 'ci_upper' -MethodColumn 'method' -SampleColumn 'n_seeds' `
        -InitializationColumn 'initialization' -Interpretation 'Compares sampled REINFORCE training with exact regularizer controls and the practical sampled conditional barrier.' `
        -Limitations 'The practical estimator is a sampled-state conditional barrier, not a global neural Fisher log determinant.'
    Add-MeanCsv -Path "$base/paired_differences.csv" -StageOrder 4 -Stage 'sampled_two_state_mdp' `
        -ExperimentId "step4_training_${preset}_paired" -Title "Finite-batch paired differences: $preset" `
        -Role $role -Status 'complete' -Environment 'sampled two-step three-action MDP' `
        -Tex 'exploration/sampled_tabular_mdp/sampled_two_state.tex' -MetricColumn 'metric' -MeanColumn 'mean_paired_difference' `
        -LowColumn 'ci_lower' -HighColumn 'ci_upper' -MethodColumn 'method' -SampleColumn 'n_pairs' `
        -InitializationColumn 'initialization' -ComparisonColumn 'experiment' `
        -Interpretation 'Each value is a paired seed-level difference against the experiment reference encoded in the source table.' `
        -Limitations 'Repeated updates within a seed are not independent observations.'
    $auditMetrics = @('conditional_exact_bias_norm','conditional_exact_sd_norm','conditional_mc_bias_norm','conditional_mc_mean_cosine','conditional_mc_rmse','exact_mu1_bias','exact_mu1_mean','exact_mu1_variance','fisher_exact_bias_fro','fisher_full_rank_fraction','fisher_logdet_defined_fraction','fisher_logdet_mean_when_defined','fisher_mc_bias_fro','fisher_mc_rmse_fro','fisher_min_eigenvalue_mean','fisher_rank_mean','mc_mu1_mean','mc_mu1_variance','mc_zero_s1_fraction','population_mu1','reward_exact_bias_norm','reward_mc_bias_norm','reward_mc_rmse','zero_s1_probability')
    Add-WideCsv -Path "$base/audit.csv" -StageOrder 4 -Stage 'sampled_two_state_mdp' `
        -ExperimentId "step4_estimator_audit_$preset" -Title "Finite-batch ratio, gradient, and empirical-Fisher audit: $preset" `
        -Role $role -Status 'complete' -Environment 'fixed policies in sampled two-step MDP' `
        -Tex 'exploration/sampled_tabular_mdp/sampled_two_state.tex' -IdColumns @('policy','n','q','p_good','repetitions') `
        -MetricColumns $auditMetrics -MethodColumn 'policy' -SampleColumn 'repetitions' `
        -Interpretation 'Quantifies finite-batch ratio bias, missing downstream-state batches, conditional-gradient error, and empirical-Fisher rank/log-determinant availability.' `
        -Limitations 'Monte Carlo nonlinear summaries coexist with exact binomial moments; consult the metric name and source table.'
}

foreach ($verificationPath in @(
    'exploration/results/tabular_mdp/two_step_trap_sampled/full/verification.json',
    'exploration/results/tabular_mdp/two_step_trap_sampled/handoff/full/verification.json',
    'exploration/results/tabular_mdp/two_step_trap_sampled/handoff/robustness/verification.json'
)) {
    $verification=Get-Content (Join-Path $root $verificationPath) -Raw | ConvertFrom-Json
    if ($verification.PSObject.Properties.Name -contains 'residuals') {
        foreach ($property in $verification.residuals.PSObject.Properties) {
            Add-Result -StageOrder 4 -Stage 'sampled_two_state_mdp' -ExperimentId 'step4_verification' `
                -ExperimentTitle 'Sampled two-state estimator and experiment verification' `
                -ScientificRole 'verification' -Status 'complete' -Environment 'sampled two-step three-action MDP' `
                -Method 'exact_enumeration_and_deterministic_replay' -Configuration @{verification_file=$verificationPath; all_checks_passed=[string]$verification.passed} `
                -Metric $property.Name -Estimate ([string]$property.Value) `
                -SourceTex 'exploration/sampled_tabular_mdp/sampled_two_state.tex' -SourceArtifact $verificationPath `
                -Interpretation 'Declared exact-moment, replay, rank-deficiency, and artifact checks passed.'
        }
    } else {
        Add-Result -StageOrder 4 -Stage 'sampled_two_state_mdp' -ExperimentId 'step4_verification' `
            -ExperimentTitle 'Sampled two-state estimator and experiment verification' `
            -ScientificRole 'verification' -Status 'complete' -Environment 'sampled two-step three-action MDP' `
            -Method 'deterministic_replay_and_artifact_checks' -Configuration @{verification_file=$verificationPath} `
            -Metric 'all_checks_passed' -Estimate ([string]$verification.passed) `
            -SourceTex 'exploration/sampled_tabular_mdp/sampled_two_state.tex' -SourceArtifact $verificationPath `
            -Interpretation 'Declared estimator and experiment checks passed.'
    }
}

# Step 4 handoff, switch robustness, and focused post-hoc analyses.
$handoff='exploration/results/tabular_mdp/two_step_trap_sampled/handoff/full'
Add-MeanCsv -Path "$handoff/summary.csv" -StageOrder 4 -Stage 'sampled_two_state_handoff' `
    -ExperimentId 'step4_handoff_main' -Title 'Temporary sampled conditional barrier handoff' `
    -Role 'focused handoff experiment' -Status 'complete' -Environment 'sampled two-step three-action MDP' `
    -Tex 'exploration/sampled_tabular_mdp/sampled_two_state.tex' -MetricColumn 'metric' -MeanColumn 'mean' `
    -LowColumn 'ci_lower' -HighColumn 'ci_upper' -MethodColumn 'label' -SampleColumn 'n_seeds' `
    -InitializationColumn 'initialization' -Interpretation 'Tests whether reward-only learning can continue after temporary sampled conditional support protection.' `
    -Limitations 'The full oracle handoff is an upper diagnostic reference, not the candidate neural algorithm.'
Add-MeanCsv -Path "$handoff/paired_differences.csv" -StageOrder 4 -Stage 'sampled_two_state_handoff' `
    -ExperimentId 'step4_handoff_paired' -Title 'Paired temporary-barrier handoff differences' `
    -Role 'focused handoff experiment' -Status 'complete' -Environment 'sampled two-step three-action MDP' `
    -Tex 'exploration/sampled_tabular_mdp/sampled_two_state.tex' -MetricColumn 'metric' -MeanColumn 'mean_paired_difference' `
    -LowColumn 'ci_lower' -HighColumn 'ci_upper' -MethodColumn 'label' -SampleColumn 'n_pairs' `
    -InitializationColumn 'initialization' -ComparisonColumn 'method' `
    -Interpretation 'Paired differences isolate schedule effects under common seed streams.'
Add-MeanCsv -Path "$handoff/post_handoff_changes.csv" -StageOrder 4 -Stage 'sampled_two_state_handoff' `
    -ExperimentId 'step4_handoff_change' -Title 'Within-run change after handoff' `
    -Role 'mechanism diagnostic' -Status 'complete' -Environment 'sampled two-step three-action MDP' `
    -Tex 'exploration/sampled_tabular_mdp/sampled_two_state.tex' -MetricColumn 'metric' -MeanColumn 'mean_change' `
    -LowColumn 'ci_lower' -HighColumn 'ci_upper' -MethodColumn 'label' -SampleColumn 'n_seeds' `
    -InitializationColumn 'initialization' -Interpretation 'Measures whether the policy continues improving after the regularizer is disabled.'

$rob='exploration/results/tabular_mdp/two_step_trap_sampled/handoff/robustness'
Add-MeanCsv -Path "$rob/final_endpoints.csv" -StageOrder 4 -Stage 'sampled_two_state_handoff' `
    -ExperimentId 'step4_switch_time_sweep' -Title 'Temporary-barrier switch-time robustness sweep' `
    -Role 'predeclared robustness check' -Status 'complete' -Environment 'sampled two-step three-action MDP' `
    -Tex 'exploration/sampled_tabular_mdp/sampled_two_state.tex' -MetricColumn 'metric' -MeanColumn 'mean' `
    -LowColumn 'ci_lower' -HighColumn 'ci_upper' -MethodColumn 'switch_time' -SampleColumn 'n_seeds' `
    -InitializationColumn 'initialization' -Interpretation 'Checks whether the handoff conclusion depends accidentally on switching at update 2000.' `
    -Limitations 'This is a narrow schedule robustness sweep, not a broad hyperparameter search.'
Add-MeanCsv -Path "$rob/paired_vs_switch_2000.csv" -StageOrder 4 -Stage 'sampled_two_state_handoff' `
    -ExperimentId 'step4_switch_time_paired' -Title 'Switch-time paired differences versus update 2000' `
    -Role 'predeclared robustness check' -Status 'complete' -Environment 'sampled two-step three-action MDP' `
    -Tex 'exploration/sampled_tabular_mdp/sampled_two_state.tex' -MetricColumn 'metric' -MeanColumn 'mean_paired_difference' `
    -LowColumn 'ci_lower' -HighColumn 'ci_upper' -MethodColumn 'switch_time' -SampleColumn 'n_pairs' `
    -InitializationColumn 'initialization' -ComparisonColumn 'reference_switch_time' `
    -Interpretation 'Paired endpoint change relative to switch time 2000.'

foreach ($name in @('posthoc_switch_summary','posthoc_endpoint_summary','posthoc_counterfactual_summary')) {
    $path="exploration/results/tabular_mdp/two_step_trap_sampled/handoff_posthoc/$name.csv"
    Add-MeanCsv -Path $path -StageOrder 4 -Stage 'sampled_two_state_handoff_posthoc' `
        -ExperimentId "step4_$name" -Title 'Focused handoff threshold post-hoc analysis' `
        -Role 'post-hoc mechanism analysis' -Status 'complete' -Environment 'sampled two-step three-action MDP' `
        -Tex 'exploration/sampled_tabular_mdp/sampled_two_state.tex' -MetricColumn 'metric' -MeanColumn 'mean' `
        -LowColumn 'ci_lower' -HighColumn 'ci_upper' -MethodColumn 'switch_time' -SampleColumn 'n' `
        -InitializationColumn 'initialization' -Interpretation 'Tests behavioral, exact reward-vector-field, and Fisher-geometry explanations at failed and successful handoffs.' `
        -Limitations 'Post-hoc associations do not identify Fisher geometry as the causal explanation.'
}

# Neural stage: the initial confirmatory result is retained as superseded evidence.
Add-Result -StageOrder 5 -Stage 'neural_discrete_cartpole' -ExperimentId 'neural_cartpole_smoke' `
    -ExperimentTitle 'CartPole neural conditional-barrier smoke experiment' -ScientificRole 'software and coefficient smoke test' `
    -Status 'complete_non_scientific' -Environment 'CartPole-v1' -Method 'ten short neural runs' `
    -SampleSize '10' -UnitOfAnalysis 'short run' `
    -Configuration @{selected_beta='16.069172515736582'; selected_entropy_coefficient='16.849303154725007'; pilot_seeds='91,92'; target_gradient_ratio='0.3'} `
    -Metric 'all_runs_finite' -Estimate '1' -SourceArtifact 'exploration/results/neural_discrete_log_barrier/cartpole_smoke/smoke_result.json' `
    -Interpretation 'Validated the neural categorical barrier and artifact pipeline before Acrobot.' `
    -Limitations 'Smoke test only; no scientific performance claim.'

Add-WideCsv -Path 'exploration/results/neural_discrete_log_barrier/acrobot_pilot/learning_rate_candidates.csv' `
    -StageOrder 5 -Stage 'neural_discrete_acrobot' -ExperimentId 'neural_original_acrobot_pilot' `
    -Title 'Original Acrobot learning-rate pilot' -Role 'superseded pilot' -Status 'complete_superseded' `
    -Environment 'Acrobot-v1' -Tex '' -IdColumns @('learning_rate','finite') `
    -MetricColumns @('mean_final_deterministic_return','finite') -MethodColumn 'learning_rate' `
    -Interpretation 'All candidates returned -500, motivating the later complete-episode baseline gate.' `
    -Limitations 'The short pilot did not establish a learning-capable baseline and is not a method comparison.'

Add-MeanCsv -Path 'exploration/results/neural_discrete_log_barrier/paired_method_differences.csv' -StageOrder 5 `
    -Stage 'neural_discrete_acrobot' -ExperimentId 'neural_initial_confirmatory_paired' `
    -Title 'Initial fixed-budget Acrobot confirmatory comparison' -Role 'superseded negative gate' `
    -Status 'complete_superseded' -Environment 'Acrobot-v1' -Tex '' -MetricColumn 'metric' `
    -MeanColumn 'mean_paired_difference' -LowColumn 'ci95_lower' -HighColumn 'ci95_upper' `
    -MethodColumn 'method' -ComparisonColumn 'reference' -SampleColumn 'seed_count' `
    -Interpretation 'All methods failed under a configuration that did not establish a learning-capable GPOMDP baseline.' `
    -Limitations 'Do not use this stage as evidence against GPOMDP or the barrier; it motivated the baseline-learning-rate gate.'
Add-WideCsv -Path 'exploration/results/neural_discrete_log_barrier/failure_rates.csv' -StageOrder 5 `
    -Stage 'neural_discrete_acrobot' -ExperimentId 'neural_initial_confirmatory_failures' `
    -Title 'Initial fixed-budget Acrobot failure rates' -Role 'superseded negative gate' `
    -Status 'complete_superseded' -Environment 'Acrobot-v1' -Tex '' `
    -IdColumns @('run_label','seed_count','failure_definition') -MetricColumns @('failure_count','failure_rate') `
    -MethodColumn 'run_label' -SampleColumn 'seed_count' `
    -Interpretation 'Documents the failed baseline gate honestly.' `
    -Limitations 'The protocol was not learning-capable and is superseded by the complete-episode GPOMDP baseline experiments.'

# Learning-rate screen, continuation, and five-seed baseline confirmation.
foreach ($spec in @(
    @{path='exploration/results/neural_discrete_log_barrier/acrobot_gpomdp_baseline/lr_screen_300_updates/candidate_results.csv'; id='neural_lr_screen'; title='Acrobot GPOMDP learning-rate screen'; role='baseline selection pilot'},
    @{path='exploration/results/neural_discrete_log_barrier/acrobot_gpomdp_baseline/lr_continuation_1000_updates/candidate_results.csv'; id='neural_lr_continuation'; title='Acrobot GPOMDP learning-rate continuation'; role='baseline selection continuation'}
)) {
    Add-WideCsv -Path $spec.path -StageOrder 5 -Stage 'neural_discrete_acrobot' -ExperimentId $spec.id `
        -Title $spec.title -Role $spec.role -Status 'complete' -Environment 'Acrobot-v1' -Tex '' `
        -IdColumns @('learning_rate','seed_count','all_finite','rank') `
        -MetricColumns @('median_stochastic_return_auc','mean_final_stochastic_return','median_final_stochastic_return','minimum_final_stochastic_return','mean_stochastic_improvement','minimum_stochastic_improvement','mean_final_termination_rate','rank') `
        -MethodColumn 'learning_rate' -SampleColumn 'seed_count' `
        -Interpretation 'Selects a learning-capable reward-only GPOMDP setting before testing any regularizer.' `
        -Limitations 'Small pilot cohorts select a baseline; they are not confirmatory method comparisons.'
}
Add-WideCsv -Path 'exploration/results/neural_discrete_log_barrier/acrobot_gpomdp_baseline/gpomdp_confirmation_1000_updates/seed_results.csv' `
    -StageOrder 5 -Stage 'neural_discrete_acrobot' -ExperimentId 'neural_gpomdp_baseline_confirmation' `
    -Title 'Five-seed Acrobot GPOMDP baseline gate' -Role 'baseline gate' -Status 'complete_passed' `
    -Environment 'Acrobot-v1' -Tex '' -IdColumns @('seed','learning_rate','finite','actual_optimizer_updates','actual_training_episodes') `
    -MetricColumns @('initial_stochastic_return','final_stochastic_return','stochastic_improvement','final_deterministic_return','final_stochastic_termination_rate','stochastic_return_auc','environment_steps') `
    -MethodColumn 'learning_rate' -ResultScope 'seed' `
    -Interpretation 'Learning rate 0.003 passed the predeclared reward-only baseline gate.' `
    -Limitations 'Five seeds establish viability, not a precise failure-rate estimate.'

# Coefficient calibration and five-seed mechanism ablation.
Add-WideCsv -Path 'exploration/results/neural_discrete_log_barrier/acrobot_gpomdp_regularizer_ablation/calibration/early_gradient_audit.csv' `
    -StageOrder 5 -Stage 'neural_discrete_acrobot' -ExperimentId 'neural_regularizer_calibration' `
    -Title 'Independent early-gradient regularizer calibration' -Role 'coefficient calibration' -Status 'complete' `
    -Environment 'Acrobot-v1' -Tex '' -IdColumns @('seed','update','training_episodes','environment_steps') `
    -MetricColumns @('reward_gradient_norm','unscaled_barrier_gradient_norm','unscaled_entropy_gradient_norm','unscaled_barrier_to_reward_ratio','unscaled_entropy_to_reward_ratio') `
    -ResultScope 'seed' -Interpretation 'Coefficients were selected by matching the median regularizer-gradient norm to 0.3 times the reward-gradient norm without inspecting final returns.' `
    -Limitations 'Gradient-norm matching equalizes initial scale, not objective semantics or later dynamics.'
Add-WideCsv -Path 'exploration/results/neural_discrete_log_barrier/acrobot_gpomdp_regularizer_ablation/five_seed_1000_updates/seed_endpoints.csv' `
    -StageOrder 5 -Stage 'neural_discrete_acrobot' -ExperimentId 'neural_five_seed_ablation' `
    -Title 'Five-seed Acrobot regularizer ablation' -Role 'mechanism pilot' -Status 'complete' `
    -Environment 'Acrobot-v1' -Tex '' -IdColumns @('run_label','method','seed','finite','learning_rate','optimizer_updates','training_episodes') `
    -MetricColumns @('final_stochastic_return','final_deterministic_return','final_stochastic_termination_rate','stochastic_return_auc','environment_steps') `
    -MethodColumn 'run_label' -ResultScope 'seed' `
    -Interpretation 'Compared reward-only, fixed entropy, fixed barrier, and a 25-percent barrier handoff after baseline validation.' `
    -Limitations 'Five seeds are exploratory; fixed regularizers can retain stochasticity and hurt stochastic evaluation despite good greedy behavior.'
Add-MeanCsv -Path 'exploration/results/neural_discrete_log_barrier/acrobot_gpomdp_regularizer_ablation/five_seed_1000_updates/paired_differences.csv' `
    -StageOrder 5 -Stage 'neural_discrete_acrobot' -ExperimentId 'neural_five_seed_ablation_paired' `
    -Title 'Five-seed Acrobot paired method differences' -Role 'mechanism pilot' -Status 'complete' `
    -Environment 'Acrobot-v1' -Tex '' -MetricColumn 'metric' -MeanColumn 'mean_difference_method_minus_reward_only' `
    -LowColumn 'ci95_low' -HighColumn 'ci95_high' -MethodColumn 'run_label' -SampleColumn 'paired_seed_count' `
    -Interpretation 'Exploratory paired differences relative to reward-only.' `
    -Limitations 'Wide intervals reflect the five-seed pilot size.'

$seed402Path='exploration/results/neural_discrete_log_barrier/acrobot_gpomdp_regularizer_ablation/seed_402_divergence_audit/checkpoint_audit.csv'
$seed402Rows=Import-Csv (Join-Path $root $seed402Path)
$seed402Ids=@('seed','run_label','update','true_environment_steps','barrier_active')
$seed402Metrics=($seed402Rows[0].PSObject.Properties.Name | Where-Object { $seed402Ids -notcontains $_ })
Add-WideCsv -Path $seed402Path -StageOrder 5 -Stage 'neural_discrete_acrobot' `
    -ExperimentId 'neural_seed402_divergence_audit' -Title 'Seed 402 paired divergence audit with seed 403 control' `
    -Role 'focused post-hoc mechanism analysis' -Status 'complete' -Environment 'Acrobot-v1' `
    -Tex 'exploration/neural_discrete_log_barrier/acrobot_reliability_extension.tex' -IdColumns $seed402Ids `
    -MetricColumns $seed402Metrics -MethodColumn 'run_label' -ResultScope 'seed' `
    -Interpretation 'Compares return, action support, gradient norms, and on-policy/fixed-reference Fisher spectra at every checkpoint for a diverged pair and an ordinary successful control seed.' `
    -Limitations 'Outcome-selected seeds are mechanistic case studies, not population effect estimates.'

$fisherMeta=Get-Content (Join-Path $root 'exploration/results/neural_discrete_log_barrier/fisher_analysis/analysis_result.json') -Raw | ConvertFrom-Json
foreach ($property in $fisherMeta.PSObject.Properties) {
    Add-Result -StageOrder 5 -Stage 'neural_discrete_acrobot' -ExperimentId 'neural_initial_fisher_analysis' `
        -ExperimentTitle 'Initial Acrobot on-policy and fixed-reference Fisher checkpoint analysis' `
        -ScientificRole 'geometric diagnostic' -Status 'complete_superseded' -Environment 'Acrobot-v1' `
        -Method 'all initial confirmatory methods' -Metric $property.Name -Estimate ([string]$property.Value) `
        -SourceArtifact 'exploration/results/neural_discrete_log_barrier/fisher_analysis/analysis_result.json' `
        -Interpretation 'Retains the checkpoint-analysis dimensions and frozen reference-bank identity for auditability.' `
        -Limitations 'The associated initial Acrobot training protocol failed for every method; spectra cannot explain successful learning in that cohort.'
}

# Twenty-seed confirmation and the final sixty-pair extension.
foreach ($cohort in @(
    @{dir='reliability_confirmation_20_seeds'; id='neural_reliability_20'; title='Twenty-seed Acrobot reliability confirmation'; summary='method_summaries.csv'; paired='paired_differences.csv'; role='intermediate confirmation'},
    @{dir='reliability_extension_60_total'; id='neural_reliability_60'; title='Sixty-pair Acrobot reliability extension'; summary='combined_method_summaries.csv'; paired='combined_paired_differences.csv'; role='primary neural comparison'}
)) {
    $base="exploration/results/neural_discrete_log_barrier/acrobot_gpomdp_regularizer_ablation/$($cohort.dir)"
    foreach ($r in (Import-Csv (Join-Path $root "$base/$($cohort.summary)"))) {
        foreach ($m in @(
            @{metric='failure_rate'; value=$r.failure_rate; low=$r.failure_rate_wilson95_low; high=$r.failure_rate_wilson95_high; unc='Wilson 95% binomial interval'},
            @{metric='failure_count'; value=$r.failures; low=''; high=''; unc='none'},
            @{metric='mean_final_stochastic_return'; value=$r.mean_final_stochastic_return; low=''; high=''; unc='point estimate; paired uncertainty is in the paired table'},
            @{metric='mean_environment_step_return_auc'; value=$r.mean_environment_step_return_auc; low=''; high=''; unc='point estimate; paired uncertainty is in the paired table'}
        )) {
            Add-Result -StageOrder 5 -Stage 'neural_discrete_acrobot' -ExperimentId $cohort.id `
                -ExperimentTitle $cohort.title -ScientificRole $cohort.role -Status 'complete' -Environment 'Acrobot-v1' `
                -Method $r.run_label -SampleSize $r.seed_count -UnitOfAnalysis 'paired training seed' `
                -Configuration @{learning_rate='0.003'; updates='1000'; episodes_per_update='8'; horizon='500'; centered_returns='true'; normalized_returns='false'; handoff_fraction='0.25'; barrier_beta='546.4135158976487'; failure_definition='final stochastic return < -300 OR termination rate < 0.8'} `
                -Metric $m.metric -Estimate ([string]$m.value) -CiLow ([string]$m.low) -CiHigh ([string]$m.high) `
                -Uncertainty $m.unc -SourceTex 'exploration/neural_discrete_log_barrier/acrobot_reliability_extension.tex' `
                -SourceArtifact "$base/$($cohort.summary)" `
                -Interpretation 'The primary endpoint is catastrophic failure; higher Acrobot return is better (less negative).' `
                -Limitations 'Failure events are rare; paired uncertainty and the exact McNemar result must accompany descriptive rates.'
        }
    }
    Add-MeanCsv -Path "$base/$($cohort.paired)" -StageOrder 5 -Stage 'neural_discrete_acrobot' `
        -ExperimentId "$($cohort.id)_paired" -Title "$($cohort.title): paired differences" `
        -Role $cohort.role -Status 'complete' -Environment 'Acrobot-v1' `
        -Tex 'exploration/neural_discrete_log_barrier/acrobot_reliability_extension.tex' -MetricColumn 'metric' `
        -MeanColumn 'mean_difference_method_minus_reward_only' -LowColumn 'ci95_low' -HighColumn 'ci95_high' `
        -MethodColumn 'run_label' -SampleColumn 'paired_seed_count' `
        -Interpretation 'Difference is handoff/control minus reward-only on the same seed; positive return differences and negative failure differences favor handoff.' `
        -Limitations 'The 95% intervals include zero for the final 60-pair endpoints.'
}

$combined = Get-Content (Join-Path $root 'exploration/results/neural_discrete_log_barrier/acrobot_gpomdp_regularizer_ablation/reliability_extension_60_total/combined_result.json') -Raw | ConvertFrom-Json
Add-Result -StageOrder 5 -Stage 'neural_discrete_acrobot' -ExperimentId 'neural_reliability_60_mcnemar' `
    -ExperimentTitle 'Sixty-pair exact failure discordance test' -ScientificRole 'primary paired inference' `
    -Status 'complete' -Environment 'Acrobot-v1' -Method 'logbarrier_handoff_h25' -Comparison 'reward_only' `
    -SampleSize '60' -UnitOfAnalysis 'paired training seed' -Configuration @{reward_failed_handoff_succeeded=[string]$combined.paired_failure_discordance.reward_failed_handoff_succeeded; reward_succeeded_handoff_failed=[string]$combined.paired_failure_discordance.reward_succeeded_handoff_failed} `
    -Metric 'exact_two_sided_mcnemar_p' -Estimate ([string]$combined.paired_failure_discordance.exact_two_sided_mcnemar_p) `
    -Uncertainty 'exact paired McNemar test' -SourceTex 'exploration/neural_discrete_log_barrier/acrobot_reliability_extension.tex' `
    -SourceArtifact 'exploration/results/neural_discrete_log_barrier/acrobot_gpomdp_regularizer_ablation/reliability_extension_60_total/combined_result.json' `
    -Interpretation 'Eight reward-only failures were rescued and two successes were harmed; the two-sided exact p-value is 0.109375.' `
    -Limitations 'The descriptive 75% relative failure reduction is not conventionally statistically significant.'

# Preserve every final 60-pair seed endpoint and every discordant-seed mechanism datum.
Add-WideCsv -Path 'exploration/results/neural_discrete_log_barrier/acrobot_gpomdp_regularizer_ablation/reliability_extension_60_total/combined_seed_endpoints.csv' `
    -StageOrder 5 -Stage 'neural_discrete_acrobot' -ExperimentId 'neural_reliability_60_seed_endpoints' `
    -Title 'Sixty-pair Acrobot seed-level endpoints' -Role 'primary neural comparison data' -Status 'complete' `
    -Environment 'Acrobot-v1' -Tex 'exploration/neural_discrete_log_barrier/acrobot_reliability_extension.tex' `
    -IdColumns @('run_label','method','seed','finite','learning_rate','optimizer_updates','training_episodes') `
    -MetricColumns @('final_stochastic_return','final_deterministic_return','final_stochastic_termination_rate','stochastic_return_auc','environment_step_return_auc','environment_steps','failure') `
    -MethodColumn 'run_label' -ResultScope 'seed' `
    -Interpretation 'Complete seed-level primary and secondary endpoints for independent re-analysis.' `
    -Limitations 'Seeds are paired across methods; rows from different methods with the same seed are not independent.'

$diagPath='exploration/results/neural_discrete_log_barrier/acrobot_gpomdp_regularizer_ablation/reliability_extension_60_total/discordant_seed_diagnostics.csv'
$diagRows=Import-Csv (Join-Path $root $diagPath)
$diagIds=@('seed','outcome_group','run_label')
$diagMetrics=($diagRows[0].PSObject.Properties.Name | Where-Object { $diagIds -notcontains $_ })
Add-WideCsv -Path $diagPath -StageOrder 5 -Stage 'neural_discrete_acrobot' `
    -ExperimentId 'neural_reliability_60_discordant_diagnostics' -Title 'Discordant-seed action-support and trajectory diagnostics' `
    -Role 'post-hoc mechanism analysis' -Status 'complete' -Environment 'Acrobot-v1' `
    -Tex 'exploration/neural_discrete_log_barrier/acrobot_reliability_extension.tex' -IdColumns $diagIds `
    -MetricColumns $diagMetrics -MethodColumn 'run_label' -ResultScope 'seed' `
    -Interpretation 'Compares every seed where the two methods disagree on catastrophic failure, including handoff behavior, statewise entropy/margins, disagreement regions, and return milestones.' `
    -Limitations 'Selected on outcome discordance and therefore descriptive, not an unbiased causal estimate.'

# Quarantined legacy collector is retained as one explicit catalog record.
Add-Result -StageOrder 5 -Stage 'neural_discrete_acrobot' -ExperimentId 'neural_legacy_fixed_segment_pilot' `
    -ExperimentTitle 'Legacy fixed-segment Acrobot pilot' -ScientificRole 'implementation diagnostic only' `
    -Status 'quarantined_invalid_for_science' -Environment 'Acrobot-v1' -Method 'multiple pilot methods' `
    -Metric 'included_in_scientific_conclusions' -Estimate '0' `
    -SourceArtifact 'exploration/results/neural_discrete_log_barrier/acrobot_pilot_fixed_segment_quarantine/run_summaries.json' `
    -Interpretation 'Preserved for auditability after discovering that fixed segments truncated episodes too early.' `
    -Limitations 'Exclude all outcomes from scientific synthesis.'

$destination = Join-Path $root $OutputPath
$destinationDirectory = Split-Path -Parent $destination
if (-not (Test-Path $destinationDirectory)) { New-Item -ItemType Directory -Path $destinationDirectory | Out-Null }
$rows | Export-Csv -LiteralPath $destination -NoTypeInformation -Encoding utf8
Write-Output "Wrote $($rows.Count) catalog rows to $OutputPath"

# Synthetic reject-inference experiment

```powershell
.venv\Scripts\python.exe -m credit_risk.cli reject-inference --config configs/reject_inference.yaml --output-dir artifacts/phase9-reject-002
```

Choose a fresh output directory for every run. The command has no source CSV or
Phase 4/5/8 artifact argument: this is a separate simulation, not an inferred
funding history from Give Me Some Credit. The original final test is untouched.
`RejectInferenceConfig` is strict, rejects unknown fields and records all defaults
in the run manifest. Seeds, sample size, selection assumptions, latent/outcome
strengths, overlap, weight clipping and sensitivity multipliers are configurable.

For each seed, generate X=(risk, debt, income) and latent U as independent standard
normal variables. The same outcome and features are reused across the scenarios.
The generator uses:

```
P(Y=1 | X,U) = sigmoid(outcome_intercept + risk + .5*debt - .5*income
                      + nonlinear_risk_effect*risk^2 + hidden_outcome_effect*U)
selection_logit = selection_intercept + risk_coefficient*risk
                  + debt_coefficient*debt + income_coefficient*income
```

MAR selection uses only X and clips true financing probabilities to [.05,.95]
by default. MNAR additionally uses `hidden_selection_effect * U`; the recorded X
cannot account for this common cause of financing and outcomes. Its recorded
true propensity diagnostic is conditional on X and latent U, not an estimated
marginal propensity given X. Deterministic selection accepts only when the base
selection probability reaches the configured cutoff, creating exact zero support.

A = financing/observed-outcome indicator. `observed_outcome` is Y for A=1 and NaN
for A=0. `fit_observed_models` rejects any supplied rejected outcome label. The
interface receives only recorded X, A and masked Y. Hidden truth is reserved for
simulation evaluation, plus the explicitly unattainable all-label benchmark.

Training/holdout positions are a seeded random partition. Propensity estimation
uses stratified cross-fitting on training rows only. Each fold fits a scaler and
logistic P(A=1|X) on the other folds and scores its held-out fold. No Y is supplied.
Actual known simulation probabilities are diagnostic, not training weights.

For accepted training rows:

```
raw_weight = 1 / estimated_propensity
clipped_weight = min(1 / max(estimated_propensity, configured_floor), maximum_weight)
fit_weight = clipped_weight / mean(clipped_weight)
ESS = sum(clipped_weight)^2 / sum(clipped_weight^2)
```

Accepted-only and IPW outcome models use the same scaler/logistic family and
regularization; scaling is fitted on accepted training X. IPW weights enter the
logistic objective. The outcome family deliberately omits nonlinear/latent risk
so the experiment can show changes in population fit under selection. An oracle
benchmark fits the same family on all synthetic training labels. It is neither a
true conditional-PD oracle nor an attainable production model.

Interpretation requires outcome observation independent of Y given recorded X,
positive financing probability throughout the target population, and an adequate
propensity model. These assumptions cannot be verified from accepted outcomes
alone. Weight flooring/capping changes the weighted objective and trades variance
against bias. Logistic scores may be positive even where real policies have zero
support; the deterministic scenario therefore disables IPW explicitly. MNAR may
still show better measured fit while failing to identify the target risk.

The assumption focus follows the [Ehrhardt et al. reject-inference study](https://adimajo.github.io/assets/publications/rejectInference.pdf),
which finds no universally superior method, and the [Cole and Hernan weighting paper](https://epiresearch.org/wp-content/uploads/2014/07/Cole_AJE_2008_168_656.pdf),
which emphasizes exchangeability, positivity and weight-model specification.

Holdout evaluation reports discrimination and probability metrics for all,
accepted and rejected synthetic applicants. Using true synthetic reject outcomes
for evaluation is possible only because the generator supplies them. Those labels
never tune hyperparameters or thresholds. Three seeds give descriptive Monte Carlo
variation, not confidence intervals or evidence about a real bank portfolio.

Sensitivity multiplies model-predicted reject odds by configured factors:
`p_adjusted = multiplier*p / (1-p+multiplier*p)`. Aggregate assumptions use observed
accepted bad counts plus adjusted rejected probabilities. No parceling labels are
created or trained upon. Worst-case aggregate bounds are observed bad count / N
through (observed bad count + missing outcome count) / N; these do not require MAR
and do not include sampling uncertainty. Neither output recovers unknown outcomes.

A real application would need through-the-door features, actual financing and
take-up indicators, comparable terms, mature observed outcomes, consistent target
construction, selection-history provenance and external outcome evidence or an
explicit identification strategy. Propensity fit, overlap and missingness should
be assessed before any model promotion. This experiment does not recommend
financing rejected applicants or establish regulatory/fairness compliance.

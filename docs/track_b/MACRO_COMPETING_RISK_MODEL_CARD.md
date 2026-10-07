# Track B Task 10 — Research Model Card

## Purpose and boundary

Test incremental predictive association of point-in-time macro information
with monthly research-default and payoff/maturity hazards in the frozen
Freddie Mac mortgage sample. This is a research mortgage model, not
regulatory PD, Basel IRB PD, IFRS 9 PD, an underwriting approval model,
borrower-level risk or a causal macro model. No EAD, LGD or ECL is calculated.
Protected-class information is not established; fairness has not been assessed.

## Population and information

Task 9A eligibility remains frozen: all eight engineered features must be
available for primary interval membership over September 2010–February 2026.
The fitted primary macro vector has seven terms: the mortgage–Treasury spread
is excluded from coefficients for algebraic identification, not selection.
Each interval uses macro information known at its preceding month-end.
Mortgage records are retrospective disclosure; their historical operational
knowledge time remains unverified even though macro vintage provenance is audited.

Static origination characteristics, prespecified duration bands and vintage
indicators enter the model ladder. Future delinquency or exposure states are
not supplied as predictors. Preprocessing fits development data only. The
model is a weak-L2 multinomial monthly hazard for no event, default and payoff.

## Evaluation governance

Development ends December 2017; 2018 is excluded; temporal evaluation spans
January 2019–February 2026 with facility-disjoint hash roles. Primary temporal
results cover vintages represented in development. Unseen-vintage effects
are explicitly fixed to the reference contribution and reported separately.
Duration bands with no development exposure have unestimated zero effects;
they must not be described as learned seasoning effects.

Prior aggregate outcomes were inspected. The correct label is **LOCKED
TEMPORAL EVALUATION AFTER SUPPORT DESIGN**, not a virgin holdout. A new
Task 10 ledger binds facilities, risk arrays, eligibility, model specification
and implementation hashes before fitting. All prespecified final temporal
predictions are generated in one consumption session. Subsequent deterministic
verification replays frozen artifacts without refitting, selection or changing
the scientific result.

## Cumulative incidence

There is one first eligible evaluation landmark per facility. Observed default,
payoff and event-free survival use Aalen–Johansen; payoff is not censoring.
IPCW uses the prior Task 6 pooled censoring method with an unverified
conditional-independent-censoring assumption. Unsupported tails are suppressed.

Modeled CIF paths use historical macro values available at each future
historical interval. These are rolling PIT-path evaluations, not forecasts
made with information available at the original landmark. Future economic
paths beyond the frozen reporting cutoff are neither imputed nor generated.
Prospective scenario-conditioned forecasting requires a separate study.

## Interpretation and limitations

Coefficients describe conditional cause-versus-no-event odds, not proportional
hazard ratios or causal effects. Numerical convergence does not establish
economic stability or external validity. Age–period–cohort identification
depends on constrained bands, vintage effects and absence of unrestricted
calendar fixed effects. Unseen cohorts and seasoning bands are extrapolations.

Facility resampling conditions on the realized calendar/macro path. A separate
calendar-year block sensitivity reflects common-period uncertainty but has
few blocks. Neither establishes borrower clustering or population representativeness.
Default is a monthly research proxy; payoff includes maturity. Delayed entry
conditions on surviving into support. Unknown, ambiguous and administrative
states censor according to the unchanged source protocol.

No model is automatically promoted. A negative temporal macro increment is
a valid research result and must be retained without retuning.

Results: [validation report](../../reports/track_b/MACRO_COMPETING_RISK_VALIDATION.md).
Specification: [frozen protocol](macro_competing_risk_protocol.json).

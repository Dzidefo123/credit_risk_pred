# Task 3: prespecified twelve-month mortgage baseline

Design version: track-b-pd-v1. Fixed before model fitting; feasibility counts were inspected, no predictions or metrics were available. This is a retrospective nominal-time research experiment. Historical operational knowledge time is UNVERIFIED.

## Estimand and cohort

Facility-level probability of protocol research default before payoff/maturity in months 1 through 12 after the reporting-month endpoint t0. This is a default-before-payoff cumulative-incidence endpoint, not default conditional on remaining open and not borrower default. Documented payoff/maturity ends this facility risk and is coded zero; it is retained as its own outcome category.

Use Task 2 eligibility without amendment: six consecutive known months in the clean history prefix, no prevalent/prior default, gap, unknown state or terminal event; no cure/reentry. Research default is numeric delinquency 03-99, RA, or credit termination 02/03/09. Payoff/maturity is 01; administrative exit and unknown follow-up do not imply zero. Ambiguous ordering is excluded. Do not require twelve physically observed future records. Include positive_default=1, negative_survived_horizon=0, competing_payoff=0; exclude right_censored, ambiguous_event_order and any other unknown status. Identity and source hashes are fixed to Task 2. Neither sample nor outcome contract may be changed.

## Frequency and evaluation unit

Monthly is primary because the intended research use is monthly risk refresh. Each eligible facility-month has equal weight; longer surviving loans contribute more observations. This is a landmark-weighted, not equal-loan, estimand. Quarterly quarter-end months (March/June/September/December) are a prespecified sensitivity, never selected by AUC. Loans are the resampling unit; borrower identity is unavailable, so cross-facility borrower dependence remains unknown.

## Fixed calendar and group split

SHA256(UTF8('track-b-pd-v1:' + loan_id)) interpreted as an integer modulo 10: residues 0-6 development, 7-9 evaluation. No labels enter assignment; no seed/date search or reshuffle permitted. Development t0 ends 2014-12. Evaluation t0 begins 2016-01. All 2015 landmarks are purged; the latest development outcome window ends 2015-12, strictly before evaluation. Evaluation includes known windows through the panel end, not only loans surviving to a fixed end date. Earlier evaluation-group and later development-group rows are unused. Thus calendar drift is inseparable from differences in the fixed loan groups and ageing/survival of the 2010 origination cohort; this is not validation on new origination cohorts. The split sacrifices rows and event support to eliminate loan leakage.

## Pre-fit feasibility gate

Before fitting, emit annual eligibility/status/effective-sample counts and partition support. Require both classes, at least 10 distinct defaulting development loans, and at least one distinct defaulting evaluation loan. Ten is a minimal exploratory engineering gate, not a statistical adequacy claim. Use only four fixed numeric predictors and ridge regularization. Fewer than 20 evaluation event loans implies EXPLORATORY ONLY; passing 20 alone would not establish production adequacy. This frozen experiment has 13 development and 5 evaluation event loans and is necessarily exploratory. If a sensitivity fails its gate, report unestimable; do not relax the design.

## Feature and preprocessing specification

Machine registry: pd_feature_registry.json, covering every panel column plus explicit prohibited outcome/recovery/future fields. Predictors: orig_credit_score and orig_ltv (STATIC_AT_ORIGINATION); loan_age and current delinquency_state (TIME_VARYING_KNOWN_AT_T0 in nominal reporting time, exact operational vintage UNVERIFIED). Eligible delinquency is numerically 0, 1 or 2. Origination score/LTV are retrospective released origination attributes, with no proof of original publication vintage. These four are chosen on credit mechanism and low dimensionality, not outcome associations. Other panel candidates are excluded; all-missing VantageScore is unusable. No future/macro/termination/recovery/label/identity/window metadata can enter preprocessing.

Development-only median imputation and standard scaling, then LogisticRegression C=1, lbfgs, max_iter=2000, tol=1e-8, seed=31003, class_weight=None. No feature selection, nonlinear transforms, resampling, tuning, calibrator or lending threshold. Natural landmark prevalence is preserved. Null predicts development prevalence. Fit convergence failure or all-missing development predictors causes failure.

## Evaluation and uncertainty

Report ROC-AUC, Gini, trapezoidal PR-AUC separately from average precision, Brier and log loss. PR curves' trapezoidal interpolation can give a constant null an unusually high area; average precision is also required. Calibration-in-the-large fits an offset intercept with slope fixed at one; joint intercept/slope is diagnostic only and never changes predictions. Five event loans imply UNSTABLE / INSUFFICIENT EVENT SUPPORT even if optimization returns numbers. Constant null logits have no identifiable slope. Three fixed probability reliability bins [0,.005], (.005,.02], (.02,1] include row/loan/event support; no deciles or calibration claim. All probability estimates and coefficients are exploratory.

Use 500 fixed-seed evaluation loan-cluster bootstrap draws, sampling loans with replacement and retaining all their landmarks. Percentile 95% intervals; report valid draws and single-class draws. Single-class draws contribute Brier/log loss but not discrimination. These intervals condition on the development fit; they omit training/model uncertainty and unknown borrower clustering. With five event loans they are unstable and not confirmatory.

## Fixed sensitivities

S1 exclude competing payoff and refit the same pipeline using the same groups/dates. This conditions on ascertainment without payoff and changes selection/estimand; it is NOT a proper competing-risk estimator. S2 quarterly landmarks and refit the same pipeline, same groups/dates. S3 descriptive removal of uncertain follow-up by status/calendar/effective loans; additionally compare a known-label twelve-physical-record subset with the primary cohort to quantify the naive completeness selection. Refit only on known labels for that selection sensitivity; never unknown-to-zero. No sensitivity is a model-shopping exercise.

## Interpretation boundary

Only research default in the frozen sampled Freddie mortgage facilities is addressed. No regulatory PD, IFRS 9, production readiness, profit, fairness, borrower independence, external validity or future stability claim. No protected-characteristic coverage adequate for fairness is available. Track A remains frozen and historical metrics are not re-evaluated. Survival/competing-risk estimation is deferred.

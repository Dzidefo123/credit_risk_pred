# Track B twelve-month PD baseline validation

## Executive Summary

BASELINE ESTABLISHED — EXPLORATORY ONLY

Historical operational knowledge time is **UNVERIFIED**. This is retrospective nominal-time mortgage research. Only five evaluation default loans support the results; point estimates are exploratory, not validation of lending performance.

## Cohort Definition

See docs/track_b/PD_COHORT_DESIGN.md for the fixed design. Six known clean months, no prevalent/prior event or reentry. Horizon months 1-12. Known facility outcomes only; no twelve-physical-row requirement.

| partition | landmarks | loans | positive_landmarks | default_loans |
| --- | --- | --- | --- | --- |
| eligible | 65064 | 983 | 368 | 31 |
| primary | 64304 | 982 | 368 | 31 |


| status | landmarks | loans | positive_landmarks | default_loans |
| --- | --- | --- | --- | --- |
| ambiguous_event_order | 52 | 5 | 0 | 0 |
| competing_payoff | 10288 | 888 | 0 | 0 |
| negative_survived_horizon | 53648 | 891 | 0 | 0 |
| positive_default | 368 | 31 | 368 | 31 |
| right_censored | 708 | 59 | 0 | 0 |


## Temporal Structure

Fixed identifier hash groups (70/30 expected, no outcome-based redraw). Development ends December 2014; evaluation starts January 2016; 2015 purges all overlapping development outcome windows. No shared loans. Earlier evaluation-group and later development-group observations are unused. Calendar, ageing and group-composition effects remain confounded.

| year | landmarks | loans | positive_landmarks | default_loans | payoff | censored | ambiguous |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2010 | 1254 | 363 | 6 | 1 | 126 | 0 | 0 |
| 2011 | 9631 | 966 | 46 | 7 | 1459 | 8 | 4 |
| 2012 | 9731 | 889 | 65 | 10 | 2072 | 4 | 0 |
| 2013 | 7595 | 709 | 50 | 8 | 873 | 0 | 0 |
| 2014 | 6679 | 582 | 28 | 5 | 875 | 0 | 0 |
| 2015 | 5783 | 529 | 25 | 4 | 924 | 0 | 0 |
| 2016 | 4851 | 439 | 16 | 2 | 764 | 0 | 0 |
| 2017 | 4071 | 369 | 5 | 1 | 612 | 0 | 5 |
| 2018 | 3449 | 307 | 19 | 2 | 414 | 0 | 7 |
| 2019 | 3009 | 267 | 50 | 8 | 521 | 0 | 0 |
| 2020 | 2438 | 227 | 46 | 8 | 612 | 0 | 0 |
| 2021 | 1780 | 171 | 0 | 0 | 364 | 0 | 0 |
| 2022 | 1416 | 128 | 4 | 1 | 192 | 0 | 0 |
| 2023 | 1220 | 105 | 8 | 1 | 125 | 0 | 0 |
| 2024 | 1087 | 96 | 0 | 0 | 184 | 0 | 7 |
| 2025 | 896 | 86 | 0 | 0 | 171 | 522 | 29 |
| 2026 | 174 | 58 | 0 | 0 | 0 | 174 | 0 |


## Competing Events

Primary zeros after documented payoff refer to default before payoff on this facility, not ordinary ongoing event-free exposure. S1 excludes payoff and changes selection/estimand; it is not a competing-risk estimator.

## Feature-Time Audit

Four fixed numerical features: released origination FICO/LTV; reporting-month loan age/delinquency 0-2. Exact historical availability/version is unverified. Every panel field is classified in the machine registry; unselected/unknown/future/outcome fields fail the predictor firewall.

## Missingness

Median imputation is prespecified and fitted only on development. Outcome stratification is descriptive only; no imputation choices use outcomes. Annual/outcome breakdowns are in JSON.

| feature | landmarks | loans | positive_landmarks | default_loans | missing | rate |
| --- | --- | --- | --- | --- | --- | --- |
| orig_credit_score | 65064 | 983 | 368 | 31 | 0 | 0 |
| orig_ltv | 65064 | 983 | 368 | 31 | 0 | 0 |
| loan_age | 65064 | 983 | 368 | 31 | 0 | 0 |
| delinquency_state | 65064 | 983 | 368 | 31 | 0 | 0 |


## Baseline Models

Null: development landmark prevalence. Logistic: four standardized predictors, development medians/scales, L2 C=1, lbfgs, 2000 iterations, tolerance 1e-8, seed 31003. Natural prevalence; no class weighting/resampling/search/calibrator/threshold. Coefficients and preprocessing parameters are recorded in JSON.

## Temporal Validation

| partition | landmarks | loans | positive_landmarks | default_loans |
| --- | --- | --- | --- | --- |
| development | 23635 | 665 | 137 | 13 |
| evaluation | 7848 | 130 | 60 | 5 |


## Discrimination

| model | metric | landmarks | loans | positive_landmarks | default_loans | estimate | lower | upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null | roc_auc | 7848 | 130 | 60 | 5 | 0.5 | 0.5 | 0.5 |
| null | gini | 7848 | 130 | 60 | 5 | 0 | 0 | 0 |
| null | pr_auc_trapezoid | 7848 | 130 | 60 | 5 | 0.5038 | 0.5008 | 0.5073 |
| null | average_precision | 7848 | 130 | 60 | 5 | 0.007645 | 0.001619 | 0.01467 |
| logistic | roc_auc | 7848 | 130 | 60 | 5 | 0.8365 | 0.6782 | 0.9597 |
| logistic | gini | 7848 | 130 | 60 | 5 | 0.6729 | 0.3565 | 0.9193 |
| logistic | pr_auc_trapezoid | 7848 | 130 | 60 | 5 | 0.1619 | 0.02988 | 0.2766 |
| logistic | average_precision | 7848 | 130 | 60 | 5 | 0.1699 | 0.04845 | 0.2936 |


## Probability Quality

| model | metric | landmarks | loans | positive_landmarks | default_loans | estimate | lower | upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null | brier | 7848 | 130 | 60 | 5 | 0.00759 | 0.001605 | 0.01452 |
| null | log_loss | 7848 | 130 | 60 | 5 | 0.04515 | 0.01399 | 0.08122 |
| null | observed_rate | 7848 | 130 | 60 | 5 | 0.007645 | 0.00159 | 0.01466 |
| null | mean_probability | 7848 | 130 | 60 | 5 | 0.005796 | 0.005796 | 0.005796 |
| logistic | brier | 7848 | 130 | 60 | 5 | 0.006837 | 0.001607 | 0.01286 |
| logistic | log_loss | 7848 | 130 | 60 | 5 | 0.03849 | 0.01029 | 0.07529 |
| logistic | observed_rate | 7848 | 130 | 60 | 5 | 0.007645 | 0.00159 | 0.01466 |
| logistic | mean_probability | 7848 | 130 | 60 | 5 | 0.0034 | 0.001463 | 0.006151 |


PR-AUC uses trapezoidal interpolation; average precision is the step-weighted measure. A constant score's trapezoidal PR area can be misleading; do not interpret it as strong discrimination.

## Calibration

UNSTABLE / INSUFFICIENT EVENT SUPPORT. Numeric diagnostic fits below are exploratory and do not recalibrate predictions. Null slope is unidentifiable.

null: {"calibration_intercept": 0.27869486002937904, "joint_intercept": null, "calibration_slope": null, "slope_status": "constant logits: joint slope unidentifiable", "support_status": "UNSTABLE / INSUFFICIENT EVENT SUPPORT"}

| bin | landmarks | loans | positive_landmarks | default_loans | observed_rate | mean_probability | sparse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (0.005, 0.02] | 7848 | 130 | 60 | 5 | 0.007645 | 0.005796 | True |


logistic: {"calibration_intercept": 1.1384786076641769, "joint_intercept": -0.00042852000982144086, "calibration_slope": 0.764554692505559, "slope_status": "estimated", "support_status": "UNSTABLE / INSUFFICIENT EVENT SUPPORT"}

| bin | landmarks | loans | positive_landmarks | default_loans | observed_rate | mean_probability | sparse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (-0.001000000001, 0.005] | 7665 | 129 | 33 | 4 | 0.004305 | 0.0009373 | True |
| (0.005, 0.02] | 54 | 12 | 0 | 0 | 0 | 0.009932 | True |
| (0.02, 1.0] | 129 | 14 | 27 | 5 | 0.2093 | 0.147 | True |


## Clustered Uncertainty

500 loan-cluster draws with all landmarks retained. 95% percentile intervals condition on the fixed fitted model; no training uncertainty or borrower clustering. Single-class discrimination draws are omitted; probability-quality draws are retained. Sparse-event intervals are not confirmatory.

Bootstrap diagnostics: {"null": {"draws": 500, "seed": 31003, "single_class_draws": 2}, "logistic": {"draws": 500, "seed": 31003, "single_class_draws": 2}}

## Sensitivity Analyses

Monthly is primary for monthly refresh, not chosen by AUC. Quarterly reduces repeated observations. S3 unknown statuses are excluded without labeling; the physical-presence sensitivity shows additional selection among already-known labels.

### S1_payoff_exclusion

BASELINE ESTABLISHED — EXPLORATORY ONLY

| partition | landmarks | loans | positive_landmarks | default_loans |
| --- | --- | --- | --- | --- |
| cohort | 54016 | 893 | 368 | 31 |
| development | 20040 | 605 | 137 | 13 |
| evaluation | 6669 | 119 | 60 | 5 |


| model | landmarks | loans | positive_landmarks | default_loans | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null | 6669 | 119 | 60 | 5 | 0.5 | 0.008997 | 0.008921 | 0.05165 |
| logistic | 6669 | 119 | 60 | 5 | 0.8286 | 0.1743 | 0.007992 | 0.04423 |


### S2_quarterly

BASELINE ESTABLISHED — EXPLORATORY ONLY

| partition | landmarks | loans | positive_landmarks | default_loans |
| --- | --- | --- | --- | --- |
| cohort | 21465 | 977 | 122 | 31 |
| development | 8001 | 661 | 46 | 13 |
| evaluation | 2579 | 129 | 20 | 5 |


| model | landmarks | loans | positive_landmarks | default_loans | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null | 2579 | 129 | 20 | 5 | 0.5 | 0.007755 | 0.007699 | 0.04573 |
| logistic | 2579 | 129 | 20 | 5 | 0.8113 | 0.186 | 0.007278 | 0.04273 |


### S3_physical_future_presence_known_labels

BASELINE ESTABLISHED — EXPLORATORY ONLY

| partition | landmarks | loans | positive_landmarks | default_loans |
| --- | --- | --- | --- | --- |
| cohort | 54781 | 902 | 325 | 31 |
| development | 20303 | 611 | 114 | 13 |
| evaluation | 6764 | 122 | 60 | 5 |


| model | landmarks | loans | positive_landmarks | default_loans | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null | 6764 | 122 | 60 | 5 | 0.5 | 0.00887 | 0.008802 | 0.05155 |
| logistic | 6764 | 122 | 60 | 5 | 0.8354 | 0.1813 | 0.008056 | 0.04561 |


### S3_unknown_exclusion

| status | landmarks | loans | positive_landmarks | default_loans |
| --- | --- | --- | --- | --- |
| ambiguous_event_order | 52 | 5 | 0 | 0 |
| right_censored | 708 | 59 | 0 | 0 |


## Limitations

- Historical operational knowledge time UNVERIFIED: nominal-time retrospective research only
- Only 13 development and 5 evaluation default loans; coefficients and metrics exploratory
- Repeated overlapping windows; loan clustering cannot account for unknown borrower links
- Fixed-fit bootstrap omits development uncertainty; percentile intervals unstable with five event loans
- 2010 mortgage cohort ageing/survival selection confounded with calendar drift; no new-vintage validation
- No regulatory/IFRS9/production/profit/fairness/external-validity claim
- No proper competing-risk or censoring-adjusted estimator fitted

## Decision

BASELINE ESTABLISHED — EXPLORATORY ONLY

Next task: Prespecified identifier-only sample-expansion feasibility study under a protocol amendment; no expansion in Task 3

## Reproducibility and preservation

Source/sample/panel/protocol/design/registry/cohort/code hashes, package versions, configuration and all sensitivity uncertainty details are recorded in pd_baseline_validation.json. Frozen Task 2 inputs were verified before/after fitting. Track A is untouched and its locked holdout is not evaluated. No models or raw licensed records are published.

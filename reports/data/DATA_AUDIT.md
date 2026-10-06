# Credit-risk data audit

Audit date: 2026-10-05. Source inspected locally: `cs-training.csv`; no rows were deleted, repaired or rewritten. Aggregate computations used the current adapter plus pandas descriptive statistics. The machine-readable working evidence is local ignored `artifacts/audit-pd-20261005/facts.json`; this report embeds the relevant observed results. Reusable audit implementation is deferred until approval, as required by the supplied first-execution gate.

## Observed facts

Source SHA-256: `9d863f88e2ef2190ed608c15c3dd362d1833d940e35f4f2bf22660e04b5f05cb`. There are **150,000 rows and 12 columns**: ten predictors, `SeriousDlqin2yrs`, and source index `source_row_id` (renamed by the adapter). Target counts: 10,026 positive and 139,974 negative, a **6.684%** positive rate. A majority-only classifier would have 93.316% accuracy; accuracy alone is unsuitable evidence of minority-outcome performance.

| Predictor | dtype | Missing count (%) | Minimum | Median | P99 | Maximum |
| --- | --- | --- | --- | --- | --- | --- |
| RevolvingUtilizationOfUnsecuredLines | float64 | 0 (0.000%) | 0 | 0.154181 | 1.09296 | 50708 |
| age | int64 | 0 (0.000%) | 0 | 52 | 87 | 109 |
| NumberOfTime30_59DaysPastDueNotWorse | int64 | 0 (0.000%) | 0 | 0 | 4 | 98 |
| DebtRatio | float64 | 0 (0.000%) | 0 | 0.366508 | 4979.04 | 329664 |
| MonthlyIncome | float64 | 29,731 (19.821%) | 0 | 5400 | 25000 | 3.00875e+06 |
| NumberOfOpenCreditLinesAndLoans | int64 | 0 (0.000%) | 0 | 8 | 24 | 58 |
| NumberOfTimes90DaysLate | int64 | 0 (0.000%) | 0 | 0 | 3 | 98 |
| NumberRealEstateLoansOrLines | int64 | 0 (0.000%) | 0 | 1 | 4 | 54 |
| NumberOfTime60_89DaysPastDueNotWorse | int64 | 0 (0.000%) | 0 | 0 | 2 | 98 |
| NumberOfDependents | float64 | 3,924 (2.616%) | 0 | 0 | 4 | 20 |

Quantiles exclude missing values. The target and source ID are int64 and have no missing values. No negative predictor values were observed. No reliable borrower, account, observation-date, application-date or outcome-date field is present. The unique source row index does not establish unique borrowers.

## Duplicate records and contamination

There are **609 duplicate predictor-plus-target rows beyond their first occurrence**, **646 duplicate predictor rows**, and **37 exact-predictor groups containing both outcome labels**. These differences were computed before imputation or clipping. Conflicting labels can reflect distinct people sharing the same coarse feature profile, inconsistent labels, or other source issues; this audit cannot distinguish them. They are not silently collapsed or relabeled. Source IDs have zero duplicates.

| Existing partition | Rows | Positive outcomes |
| --- | --- | --- |
| Training | 67,562 | 4,514 |
| Development | 22,483 | 1,504 |
| Calibration | 29,964 | 2,005 |
| Final test, already consumed | 29,991 | 2,003 |

**Zero exact-predictor groups span multiple saved partitions.** Saved group hashes and assignments also matched the current frozen experiment verifier. This controls observed exact-profile overlap, not all borrower-level dependence: different snapshots of one unidentified borrower could have different predictors. Missingness rates by partition are retained in the original validation manifest. No inference of temporal validation follows from these random partitions.

## Impossible and suspicious values

| Observation | Count or example | Interpretation |
| --- | --- | --- |
| Age zero or over 110 | 1; observed minimum 0, maximum 109 | Age zero is implausible for an adult lending applicant; source explanation absent |
| Utilization over one | 3,321; maximum 50,708 | Over-limit balances may exceed one; extreme values need source definitions, not automatic deletion |
| Any delinquency count at least 90 | 269; maxima 98 | Possible coded/sentinel values; not proven to be literal counts or missing codes |
| Zero monthly income | 1,634 | Could be real zero, missing encoding, or definition issue; keep separately from NA |
| DebtRatio | Median 0.366508; P99 4,979.04; maximum 329,664 | Units/denominator/zero-income handling unknown; heavy tail is observed |
| MonthlyIncome | Median 5,400; maximum 3,008,750 | Currency, period and extreme-value legitimacy are unverified |
| Real-estate lines and dependents | Maxima 54 and 20 | Unusual values; no demonstrated source error |

The source contract flags suspicious values rather than silently dropping rows. The current model caps features using training-fitted 0.001/0.999 quantiles and imputes training medians with indicators. That is an explicit modeling transform, not a correction of the source truth. It can reduce information in extreme profiles, and should be assessed using training-only sensitivity experiments.

## Correlations and distributions

Full-source Spearman correlations are descriptive audit statistics, not a feature-selection experiment. No predictor-predictor pair has absolute Spearman correlation at least 0.5. This does not rule out nonlinear dependence, tail/sentinel effects, or collinearity after transformations. Predictor-target associations are:

| Predictor | Spearman correlation with inherited target |
| --- | --- |
| RevolvingUtilizationOfUnsecuredLines | 0.240378 |
| age | -0.117034 |
| NumberOfTime30_59DaysPastDueNotWorse | 0.257411 |
| DebtRatio | 0.020597 |
| MonthlyIncome | -0.066980 |
| NumberOfOpenCreditLinesAndLoans | -0.038587 |
| NumberOfTimes90DaysLate | 0.342349 |
| NumberRealEstateLoansOrLines | -0.034118 |
| NumberOfTime60_89DaysPastDueNotWorse | 0.277111 |
| NumberOfDependents | 0.045972 |

Past-due counts and utilization have stronger positive monotone associations than other fields. This is consistent with predictive usefulness, but cannot establish availability before the target window or causality. Income has 19.821% missingness and dependents 2.616%; mechanisms are unknown and may depend on customer/source selection. Several distributions are highly skewed, motivating the declared transforms without proving they are optimal.

## Modeling problem: assumptions and unknowns

- **Target fact:** the inherited field is named `SeriousDlqin2yrs`; the current contract preserves its source two-year delinquency meaning. The audit did not reconstruct events from dated histories.
- **Observation-point assumption:** predictors are treated as an origination-like snapshot. Their actual measurement dates are absent.
- **Unit:** a source record; borrower/account/loan identity is not verifiable.
- **Probability:** a model-estimated probability of the inherited label conditional on this source and pipeline. It is not a verified twelve-month, regulatory or contractual-default PD.
- **Source population:** original collection dates, currency/units, acceptance/rejection mechanism, representativeness, censoring and data rights are not established by the local file.
- **Duplicate interpretation:** predictor hashes are grouping proxies and can merge different borrowers; they do not recover genuine identities.

## Potential problems versus confirmed problems

**Confirmed current data issues:** substantial missing income, duplicate profiles, 37 conflicting-label groups, implausible age zero and extreme numeric tails. They are confirmed observations; most underlying causes remain unknown. Zero exact-profile partition overlap is a confirmed check, not a declaration that all leakage is absent.

**Confirmed historical methodological problems:** V1 performs outcome-dependent exclusion, learned imputation before splitting and initial in-sample evaluation; the notebook population does not match the retained CSV. See the [baseline report](../baseline/Baseline_Model_Report.md). The current train-only package addresses those particular methodological failures.

**Potential unresolved risks:** post-observation information in delinquency histories, repeated unidentified borrowers, dataset selection/reject bias, missingness mechanisms, changing economic conditions and unverified target/window definitions. No direct target or row ID is used as a predictor, but that is insufficient to prove prediction-time integrity.

## Reproduction and boundaries

The aggregate audit read the retained source and saved partition assignments. `verify_experiment` confirmed the frozen source, code, versions, artifacts and assignments without model deserialization. No new original-holdout prediction, recalibration, tuning or champion selection occurred. Whole-source descriptive auditing is not a new independent holdout evaluation and must not become a route for choosing features against the consumed test.

Before a lender-specific model is developed, obtain a dated data dictionary, borrower/account keys, target event definitions, observation/performance windows, units, censoring rules and sampling/approval lineage. Keep the current artifact and consumed holdout intact. Any new feature/capping/CV experiment should be approved and confined to training records; genuinely independent final validation requires new unseen data.

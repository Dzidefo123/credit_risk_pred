# Leakage review

Task 2, 2026-10-05. Documentary/static review plus training-only descriptive audit. No locked holdout predictions were inspected, scored or evaluated. Full-source overlap facts below are retained from the preceding repository audit, not new model-selection analysis.

## Classification meaning

**LOW CONCERN**: no evident direct outcome encoding in the feature meaning; not a guarantee of clean lineage. **REVIEW**: plausible pre-outcome use, but timing/definition needs evidence. **HIGH CONCERN**: demonstrated leakage or strong direct outcome-derived behavior in an implementation. **UNKNOWN**: semantics too weak to assess even conceptually. These are investigation labels, not thresholds for removing predictors. No reviewed feature was removed.

Source definitions and reproduced-definition provenance are in the [feature dictionary](../../docs/DATA_DICTIONARY.md). The [target document](../../docs/TARGET_DEFINITION.md) explains the intended relative window and missing dates.

## Predictor-by-predictor review

| Predictor | Classification | Reason and required evidence |
| --- | --- | --- |
| RevolvingUtilizationOfUnsecuredLines | REVIEW | Financial snapshot could be updated after t0; confirm balance/limit timestamps and reporting lags. Extreme ratios are not proof of leakage |
| age | LOW CONCERN | Demographic age does not directly encode the outcome; measurement date remains unverified and age zero needs explanation |
| NumberOfTime30_59DaysPastDueNotWorse | REVIEW | Backward-looking count is legitimate only if its lookback ends at t0; future-window overlap would be leakage |
| DebtRatio | REVIEW | Need snapshot and numerator/denominator lineage; later income or arrears updates could contaminate it |
| MonthlyIncome | REVIEW | Need measurement/verification date and reporting lag; outcome-related refreshes could change it |
| NumberOfOpenCreditLinesAndLoans | REVIEW | Need dated facility status; a later closure or draw could reflect the outcome window |
| NumberOfTimes90DaysLate | REVIEW | Same delinquency threshold as outcome does not prove leakage; need its unspecified lookback and ensure no prospective target events entered it |
| NumberRealEstateLoansOrLines | REVIEW | Need snapshot timing for openings/closures; aggregate count is not dated account-level history |
| NumberOfTime60_89DaysPastDueNotWorse | REVIEW | Past-count semantics require a documented t0 boundary; overlap with prospective events is not presently testable |
| NumberOfDependents | REVIEW | Reported household information could be refreshed after t0; no timestamp demonstrates alignment |

No individual predictor is labeled HIGH CONCERN on unsupported inference. None has verified record-level timestamp lineage. UNKNOWN remains appropriate for unprovided supplier/sampling/measurement processes. REVIEW for the behavioral counts should be treated seriously: conceptual backward/forward windows are distinct, but the local file cannot establish the boundary. Preexisting-default eligibility also cannot be reconstructed from historical occurrence counts alone.

## Preprocessing and outcome-derived information

**Confirmed historical HIGH CONCERN:** V1 notebook learns imputations before the random split, uses outcome-dependent row exclusion, and initially fits/evaluates XGBoost on the same records. Those mechanisms bias the historical evaluation and population. The historical serialized model lacks the claimed ensemble/preprocessing. These findings are in the [baseline report](../baseline/Baseline_Model_Report.md); V1 scores remain unverified.

**Current pipeline:** static code tracing shows caps, imputation and logistic scaling fit only training rows. The canonical feature allowlist excludes target and row ID; no direct target-derived feature enters the current model. This resolves the specific historical preprocessing defects, but cannot authenticate upstream data construction. Task 2 neither changes nor refits any transformer.

## Duplicates and borrower overlap

The preceding full-source audit recorded 609 extra predictor-plus-target duplicate rows, 646 extra predictor profiles and 37 profiles with conflicting labels. Saved exact-predictor hashes do not cross partitions. Exact-profile grouping therefore addresses the observed exact duplication mechanism. Fresh training-only audit records 332 extra predictor-plus-target profiles; this is a different population/quantity and must not replace the full-source count.

Source IDs are row indices, not stable borrower identifiers. Repeated borrowers with different snapshots could still cross partitions; different borrowers can share a profile. No test of true borrower independence or account overlap is possible. Conflicting profile labels can have several causes and do not establish erroneous outcomes. No deduplication, repair or relabeling was performed.

## Calibration, test separation and experiment reuse

Static tracing and retained manifests show separate training, development, calibration and final-test partitions. Calibration fits its designated partition; candidate/method selection uses development; the locked final evaluation follows selection. The final test has already been consumed. Its historical metrics are evidence, not a tuning resource or fresh validation permission.

A current engineering limitation remains: consumed-test protection is scoped to a run directory. A different directory/seed can expose previously viewed records again. Documentation prohibits that, but a repository-wide source/sample ledger is not implemented. Repeated experiments cannot manufacture new independent data.

This task used saved training positions and skipped other CSV rows at parsing for its fresh aggregate audit. Previously recorded whole-source descriptions are reused for the dictionary. Reading retained evidence/checksums for preservation is not a new test-performance evaluation. No test prediction file was parsed, no model bundle was loaded and no holdout probability was computed.

## Post-outcome and temporal risks

The file has no observation/event dates and no trusted temporal order. Competition year, row ID, label name and two-year feature lookbacks cannot serve as substitutes. True out-of-time validation, exact feature-window separation, follow-up maturity and reporting-lag checks remain unavailable. The intended source horizon is externally documented, but the pipeline cannot reconstruct event-window labels from these rows.

Before lender use, require an observation-time feature contract, stable borrower/account keys, dated raw events, independent future performance, source sampling/acceptance lineage, default adjudication and eligibility rules. Confirm historical delinquency boundaries without dropping predictive history merely because it is associated with the label.

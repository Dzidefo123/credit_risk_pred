# Feature data dictionary

Task 2: 2026-10-05. Ten model predictors; target and source index are excluded from the feature allowlist.

## Source definitions versus our interpretation

The **Source definition** entries are concise paraphrases of the published [Empulse data-description table](https://empulse.readthedocs.io/en/stable/guide/datasets/give_me_some_credit.html#data-description), corroborated where available by a [published dataset study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9041569/). They are externally reproduced definitions: the original competition dictionary XLS was listed but not retrieved. Empulse uses normalized feature names and a processed population; neither its population nor its assumed loss/cost parameters is substituted for this lab source. The [target/provenance document](TARGET_DEFINITION.md) records the evidence chain and remaining gaps.

The **Our interpretation** entries are project judgments, not additional facts supplied by the publisher. Availability categories describe evidence, not a claim that timestamp lineage was audited.

## Observed ranges and missingness

Full-source numbers below are retained descriptive evidence from the [earlier source audit](../reports/data/DATA_AUDIT.md), not new holdout analysis. Fresh Task 2 calculations use only the saved training partition in [TRAINING_DATA_AUDIT.json](../reports/data/TRAINING_DATA_AUDIT.json). Source provenance is not inferred from numerical resemblance alone.

| Canonical predictor | Current pandas dtype | Retained full-source min–max | Missing count (%) |
| --- | --- | --- | --- |
| RevolvingUtilizationOfUnsecuredLines | float64 | 0–50708 | 0 (0.000%) |
| age | int64 | 0–109 | 0 (0.000%) |
| NumberOfTime30_59DaysPastDueNotWorse | int64 | 0–98 | 0 (0.000%) |
| DebtRatio | float64 | 0–329664 | 0 (0.000%) |
| MonthlyIncome | float64 | 0–3008750 | 29,731 (19.821%) |
| NumberOfOpenCreditLinesAndLoans | int64 | 0–58 | 0 (0.000%) |
| NumberOfTimes90DaysLate | int64 | 0–98 | 0 (0.000%) |
| NumberRealEstateLoansOrLines | int64 | 0–54 | 0 (0.000%) |
| NumberOfTime60_89DaysPastDueNotWorse | int64 | 0–98 | 0 (0.000%) |
| NumberOfDependents | float64 | 0–20 | 3,924 (2.616%) |

Count fields are conceptually integral. NumberOfDependents is represented as float64 because of missing values; that dtype does not imply fractional dependents are meaningful. Other numeric dtype/range entries describe parsing, not a verified economic definition.

## Shared fitted transformations

All predictors use training-fitted 0.001/0.999 quantile capping and median imputation with missing indicators. Logistic applies log1p to all predictors except age, then scaling. XGBoost does not use the logistic log transform/scaling. Bounds, medians and scaling are learned only from training; no transformation was changed in Task 2. These fitted transformations do not authenticate or repair original values. Raw-unit coefficient/odds-ratio interpretation would need to account for them.

## RevolvingUtilizationOfUnsecuredLines

- **Source definition:** Unsecured revolving balances relative to combined credit limits, excluding real estate and installment debt.
- **Our interpretation:** Credit-card/line utilization ratio, not a cash balance.
- **Availability at prediction time:** Ambiguous in the local evidence; usable before the outcome only if its snapshot/lookback is verified at t0.
- **Leakage assessment:** **REVIEW**. Financial/household state could have been refreshed after the scoring point; no timestamps demonstrate alignment.
- **Known transformation:** shared capping/imputation; log1p and scaling for logistic; neither for XGBoost.
- **Important caveats:** Values above one may reflect over-limit balances; extreme values and denominator definitions need investigation.

## age

- **Source definition:** Borrower age, measured in years.
- **Our interpretation:** Age is a demographic characteristic, not a future outcome.
- **Availability at prediction time:** Clearly pre-outcome in conceptual meaning; actual measurement timing is unverified.
- **Leakage assessment:** **LOW CONCERN**. No direct label/event encoding; timing and implausible-value caveats remain.
- **Known transformation:** shared capping/imputation; no log1p; scaling for logistic only.
- **Important caveats:** Zero is implausible for this borrower context. No observation date establishes when age was measured.

## NumberOfTime30_59DaysPastDueNotWorse

- **Source definition:** Count of 30–59-day delinquency occurrences during the preceding two years.
- **Our interpretation:** Historical delinquency if the lookback ends at the scoring snapshot.
- **Availability at prediction time:** Ambiguous in the local evidence; usable before the outcome only if its snapshot/lookback is verified at t0.
- **Leakage assessment:** **REVIEW**. Potentially post-outcome if behavioral counts include any part of the future label window; historical delinquency is not inherently leakage.
- **Known transformation:** shared capping/imputation; log1p and scaling for logistic; neither for XGBoost.
- **Important caveats:** Canonical alias of NumberOfTime30-59DaysPastDueNotWorse. Codes near 98 are sentinel-like, not authenticated missing codes.

## DebtRatio

- **Source definition:** Monthly debt, alimony and living expenses relative to gross monthly income.
- **Our interpretation:** Financial burden ratio; its expense scope differs from a simple loan-payment ratio.
- **Availability at prediction time:** Ambiguous in the local evidence; usable before the outcome only if its snapshot/lookback is verified at t0.
- **Leakage assessment:** **REVIEW**. Financial/household state could have been refreshed after the scoring point; no timestamps demonstrate alignment.
- **Known transformation:** shared capping/imputation; log1p and scaling for logistic; neither for XGBoost.
- **Important caveats:** Huge ratios and zero/missing-income denominator treatment are unexplained; no currency is verified.

## MonthlyIncome

- **Source definition:** Borrower monthly income.
- **Our interpretation:** Monthly monetary income under source units, not an assumed annual salary.
- **Availability at prediction time:** Ambiguous in the local evidence; usable before the outcome only if its snapshot/lookback is verified at t0.
- **Leakage assessment:** **REVIEW**. Financial/household state could have been refreshed after the scoring point; no timestamps demonstrate alignment.
- **Known transformation:** shared capping/imputation; log1p and scaling for logistic; neither for XGBoost.
- **Important caveats:** Currency, gross/net reporting and verification status are unknown. Missing and zero income are distinct observations.

## NumberOfOpenCreditLinesAndLoans

- **Source definition:** Number of open installment loans and credit lines.
- **Our interpretation:** Aggregate count of facilities at the assumed snapshot.
- **Availability at prediction time:** Ambiguous in the local evidence; usable before the outcome only if its snapshot/lookback is verified at t0.
- **Leakage assessment:** **REVIEW**. Financial/household state could have been refreshed after the scoring point; no timestamps demonstrate alignment.
- **Known transformation:** shared capping/imputation; log1p and scaling for logistic; neither for XGBoost.
- **Important caveats:** Includes heterogeneous products; no account identities, balances or histories accompany the count.

## NumberOfTimes90DaysLate

- **Source definition:** Number of delinquency occurrences at 90 days or beyond.
- **Our interpretation:** Prior severe delinquency only if timestamped before the outcome window.
- **Availability at prediction time:** Ambiguous in the local evidence; usable before the outcome only if its snapshot/lookback is verified at t0.
- **Leakage assessment:** **REVIEW**. Potentially post-outcome if behavioral counts include any part of the future label window; historical delinquency is not inherently leakage.
- **Known transformation:** shared capping/imputation; log1p and scaling for logistic; neither for XGBoost.
- **Important caveats:** Reproduced source definition gives no explicit lookback for this field. Do not assign one or use it to identify current default. Sentinel-like 98 values need explanation.

## NumberRealEstateLoansOrLines

- **Source definition:** Number of mortgage/real-estate facilities, including home-equity lines.
- **Our interpretation:** Real-estate-related facility count, not property valuation.
- **Availability at prediction time:** Ambiguous in the local evidence; usable before the outcome only if its snapshot/lookback is verified at t0.
- **Leakage assessment:** **REVIEW**. Financial/household state could have been refreshed after the scoring point; no timestamps demonstrate alignment.
- **Known transformation:** shared capping/imputation; log1p and scaling for logistic; neither for XGBoost.
- **Important caveats:** No collateral values, security ranking or recovery records; maximum 54 is unusual, not a demonstrated source error.

## NumberOfTime60_89DaysPastDueNotWorse

- **Source definition:** Count of 60–89-day delinquency occurrences during the preceding two years.
- **Our interpretation:** Historical delinquency if its lookback ends before the outcome window.
- **Availability at prediction time:** Ambiguous in the local evidence; usable before the outcome only if its snapshot/lookback is verified at t0.
- **Leakage assessment:** **REVIEW**. Potentially post-outcome if behavioral counts include any part of the future label window; historical delinquency is not inherently leakage.
- **Known transformation:** shared capping/imputation; log1p and scaling for logistic; neither for XGBoost.
- **Important caveats:** Canonical alias of NumberOfTime60-89DaysPastDueNotWorse. Sentinel-like counts need source coding documentation.

## NumberOfDependents

- **Source definition:** Family dependents, excluding the borrower.
- **Our interpretation:** Reported household dependency count, not a direct measure of expenditure.
- **Availability at prediction time:** Ambiguous in the local evidence; usable before the outcome only if its snapshot/lookback is verified at t0.
- **Leakage assessment:** **REVIEW**. Financial/household state could have been refreshed after the scoring point; no timestamps demonstrate alignment.
- **Known transformation:** shared capping/imputation; log1p and scaling for logistic; neither for XGBoost.
- **Important caveats:** Missing is not zero; definitions/verification are unknown and unusually high counts may be real.

## Non-predictor fields

`SeriousDlqin2yrs` is the outcome, not a model feature. The blank CSV index becomes `source_row_id` for lineage; it is not a stable borrower identifier, temporal ordering or predictor. The two alias normalizations change names, not the underlying records. Neither identifier nor target should enter an audit profile grouping by accident: duplicate comparisons must declare their columns.

No feature has verified row-level pre-outcome timestamps. REVIEW is an evidence request, not proof of contamination; no predictor has been dropped or changed as a result of this document.

# Model card: origination delinquency research model

Documentation revision: Phase 13, 2026-10-04. Software package: 0.12.0.
Status: **research only; no production champion or lending approval**. Model owner,
independent validator, policy approver and deployment approver are unassigned.
This card consolidates recorded lab evidence; it is not an independent validation
opinion. [Governance register](governance_register.json) contains exact values and
identities; [governance process](../docs/model_governance.md) defines review gates.

## Purpose, population and target

The model estimates the probability of the inherited SeriousDlqin2yrs label for
records matching the source feature contract. Its intended use is offline credit
risk research, comparison of ranking/probability quality and demonstrations of
risk grades and hypothetical policy. `/score` and `/decision` expose this research
candidate with research_only=true. That flag is descriptive, not an access control.

The inherited Give Me Some Credit-style CSV has 150,000 records and 10,026 positive
labels (6.684%). Original source acquisition, eligible application population,
acceptance process, observation dates and true borrower identities are not
independently verified. The inherited file/notebook provenance mismatch is in the
[Phase 1 audit](phase1_audit.md). The V1 pickle remains historical evidence and is
never the selected model. The new experiment retained all source rows and fitted
its own full pipelines.

Target: `Inherited SeriousDlqin2yrs; source two-year delinquency label`. The name describes a two-year delinquency outcome;
its equivalence to contractual default is unconfirmed. No observation date or
censoring/performance-window reconstruction is possible from this CSV. This is
not the synthetic next-12-month target, a portfolio default forecast or an
accounting ECL model. External use requires a verified target and population.

Source SHA-256: `9d863f88e2ef2190ed608c15c3dd362d1833d940e35f4f2bf22660e04b5f05cb`.

## Inputs and model construction

Ten canonical numeric predictors are required by the pipeline; nullable values
are allowed. API counts must be nonnegative integers representable exactly by
float64, age must be whole years, and infinite/negative/unknown fields are rejected.
Targets, source IDs and future outcomes cannot enter the scoring feature frame.

| Canonical predictor | Interpretation / material qualification |
| --- | --- |
| RevolvingUtilizationOfUnsecuredLines | Reported revolving utilization; values above one flagged |
| age | Reported age; zero or over 110 flagged, not automatically dropped |
| NumberOfTime30_59DaysPastDueNotWorse | Reported 30–59-day delinquency count |
| DebtRatio | Reported debt ratio; denominator/affordability basis unverified |
| MonthlyIncome | Reported income; currency/time-unit provenance unverified |
| NumberOfOpenCreditLinesAndLoans | Reported open credit count |
| NumberOfTimes90DaysLate | Reported severe delinquency count |
| NumberRealEstateLoansOrLines | Reported real-estate credit count |
| NumberOfTime60_89DaysPastDueNotWorse | Reported 60–89-day delinquency count |
| NumberOfDependents | Reported dependent count |

Unknown lookback semantics and sentinel-like delinquency values need domain review.
Training-fitted 0.001/0.999 quantile caps retain rows. Median imputation and missing
indicators fit only training; an entirely missing training feature cannot acquire
a meaningful effect merely because new values appear at serving time. Logistic
regression adds log1p transforms except age, standard scaling and L2 regularization.
XGBoost uses caps/imputation without logs/scaling: 250 hist trees, depth 3,
learning rate .05, row/column subsampling .8, L2 5, seed 42, two CPU threads.
No class rebalancing or hyperparameter search was used.

Base/preprocessing fit, calibration fit and candidate selection are separate:

<!-- evidence:partitions -->
| Partition | Rows | Events | Purpose |
| --- | ---: | ---: | --- |
| train | 67,562 | 4,514 | Base model / preprocessing fit |
| development | 22,483 | 1,504 | Candidate / calibration selection |
| calibration | 29,964 | 2,005 | Calibrator fit |
| test | 29,991 | 2,003 | Consumed final evaluation |
<!-- /evidence:partitions -->

Exact-predictor duplicates stay together, including opposite-label duplicates.
Source row IDs provide lineage, not borrower identity. This reduces duplicate
leakage but cannot prove borrower independence or temporal generalization.
Calibrators fit only calibration rows; development log loss selects methods,
then Brier and declared method order break ties. Logistic selected isotonic;
XGBoost selected a positive-slope sigmoid of raw log odds. All variants use
1e-6 probability clipping. Selection preceded final-test scoring in Phase 5.
The persistent test-consumption lock remains in place: no new selection may use
this consumed test population. See [validation report](validation_report.md).

## Recorded performance and calibration

The following is the previously recorded 29,991-row final holdout, not a fresh
Phase 13 evaluation. It contains 2,003 events (6.6787%) and is grouped
cross-sectional validation, not out-of-time or external validation.

<!-- evidence:final_metrics -->
| Selected model | AUC | Gini | KS | Average precision | Brier | Log loss | ECE | O/E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression / isotonic | 0.834776 | 0.669551 | 0.512370 | 0.345232 | 0.050967 | 0.188287 | 0.004050 | 1.026412 |
| xgboost / sigmoid | 0.868152 | 0.736304 | 0.577380 | 0.408630 | 0.048545 | 0.176030 | 0.002820 | 1.023134 |
<!-- /evidence:final_metrics -->

O/E means observed events divided by summed predicted probability. XGBoost's
1.023134 O/E indicates roughly 2.3% aggregate event underprediction relative to
expected events. Its selected sigmoid reduces log loss from .176073 to .176030
but worsens Brier from .048538 to .048545; that very small raw-versus-sigmoid gain
has no demonstrated statistical significance. AUC/AP are unchanged. ECE depends
on bins and does not establish calibration in every band or subgroup.

For selected XGBoost, the conditional 95% group-bootstrap intervals are AUC
[.859273, .875408], Brier [.046743, .050395], log loss [.170154, .182341]. The
200 paired draws hold estimators/calibrators fixed and resample exact-predictor
groups; they exclude training uncertainty and unresolved borrower dependence.
Reliability Wilson intervals assume independent rows. Metrics at diagnostic
threshold .10 are not validated underwriting thresholds.

Test income missingness is 20.046%, dependents missingness 2.694%. Missing-income
XGBoost mean PD is 5.48% versus 5.82% observed; utilization above one has 695 rows,
37.41% observed events and AUC .7538. Age segments differ materially. These are
subgroup diagnostics, not fairness certification or grounds for post-test tuning.
Details and figures remain in the [Phase 5 evidence](phase5_validation_summary.json).

## Artifact identity and reproducibility

| Item | SHA-256 |
| --- | --- |
| Selected XGBoost/sigmoid wrapper | `17030fc4d0a5cf0e905f11933e97542b0afcccba789fdfcc481c0bc9347d7d75` |
| Logistic/isotonic benchmark wrapper | `4687a557b52a0242bc38bfb17c14afff0fefbf0e6b2bbba7a6038fa56ed135b3` |
| selection.json artifact bytes | `7bd1c00846f90aab0452a3ed5e128cba950ece2ad7bad0f8e9ee9c34cf237c12` |
| Selection lock content identity | `717064395e6f5f7a87efbc2c904f695cff5e853b1c35d2a0810a2e9efc7b57c7` |

The file hash and lock identity measure different content and must not be
interchanged. API model version is xgboost:sigmoid:17030fc4d0a5; full artifact hash
is returned separately. Base training package was 0.4.0; calibration package was
0.5.0; current software version does not change those historical identities.
Frozen dependencies include NumPy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1,
XGBoost 3.2.0 and joblib 1.6.0. Python 3.11 is the verified reuse baseline.
Explicit CRLF Git attributes preserve five frozen source byte hashes. Changing model
code, dependencies, preprocessing or calibration requires a new reviewed
experiment; bypassing manifests or deleting the test lock is unacceptable.

Only trusted locally generated joblib files may be loaded. Checksums detect
changes, not malicious artifact authorship. Data/models remain ignored, external
inputs. MLflow copies frozen evidence locally without fitting or promoting it.
See [API/tracking](../docs/api_tracking.md) and
[engineering verification](phase12_engineering_report.md).

## Decision use, monitoring and exclusions

Serving uses the baseline demonstration policy: approve only below .03; review
from .03 to below .10; decline at/above .10. Missing/zero income, missing/high
proxy ratios and inadequate limits can move otherwise-approved records to
review. Predictive imputation never supplies policy affordability inputs.
`APPROVE` is an illustrative label, not authorization to extend credit.
[Credit policy](credit_policy.md) documents limits, assumed LGD/EAD and units.

The historical development reference and controlled perturbation demonstrate
feature/PD/score/missingness alerts. They do not show prospective performance
monitoring, concept drift or increased realized defaults. Recalibration/retraining
requires validated, dated mature outcomes and new independent evaluation.
[Monitoring report](monitoring_report.md) defines proposed escalation and review.

Excluded uses include automated real lending, borrower-facing adverse-action
explanations, regulatory capital or IFRS 9 calculations, verified affordability,
profit optimization, protected-group fairness certification and extrapolation to
unverified products/populations. No scorecard WOE/IV library, standalone rolling
behavioral-feature transformer, causal explanation or recovery/CCF model has been
implemented. Coefficients, gain importances and policy reason codes do not provide
complete borrower-level explanations.

## Related inventory and outstanding findings

The separate [portfolio benchmark](phase7_expected_loss_report.md) is a synthetic,
uncalibrated, state-only Markov model of recorded absorbing DEFAULT over 12 months,
fitted through 2024-12-31. It cannot share the origination label, data or calibration
claims. LGD/CCF/scenario multipliers are assumptions, not observed loss estimates.
[Reject-inference experiments](reject_inference_report.md) use synthetic oracle
truth to explore MAR/MNAR and overlap; rejected repayment outcomes remain unknown
in actual lending. Neither component is a promoted production model.

All nine findings in the [risk register](model_risk_register.md) remain open.
Ownership, independent challenge and approvals are pending. This documentation
records evidence and limits; it does not close findings or approve deployment.

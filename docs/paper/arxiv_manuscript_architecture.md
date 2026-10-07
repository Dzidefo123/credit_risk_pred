# Task 14 arXiv manuscript architecture

Planning outline only. No abstract, manuscript introduction or conclusions written.

## 1. Introduction — motivation slots only

- Claim IDs: MACRO_DEVELOPMENT_M1_JOINT_LOG_LOSS, MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS, MACRO_PRIMARY_M1_JOINT_LOG_LOSS, MACRO_PRIMARY_M2_JOINT_LOG_LOSS, MACRO_DELTA_FACILITY_JOINT_LOG_LOSS, MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS, MACRO_PRIMARY_M1_PAYOFF_AUC, MACRO_PRIMARY_M2_PAYOFF_AUC, MACRO_PRIMARY_M1_PAYOFF_BRIER, MACRO_PRIMARY_M2_PAYOFF_BRIER, MACRO_CAL_M1_PAYOFF, MACRO_CAL_M2_PAYOFF, FANNIE_PROPOSED, FANNIE_PROVENANCE
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 2. Related Work — search plan only

- Claim IDs: None; literature search pending, no empirical claim
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 3. Problem Formulation

- Claim IDs: See subsection claim bindings
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 3.1. Longitudinal mortgage risk

- Claim IDs: PD_LOGISTIC_ROC_AUC, PD_LOGISTIC_MEAN_PROBABILITY, PD_LOGISTIC_OBSERVED_RATE, PD_CALIBRATION_INTERCEPT, PD_CALIBRATION_SLOPE
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 3.2. Default and payoff as competing events

- Claim IDs: SURV_60_NAIVE_NET_DEFAULT, SURV_60_DEFAULT_CIF
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 3.3. Temporal transport

- Claim IDs: MACRO_DEVELOPMENT_M1_JOINT_LOG_LOSS, MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS, MACRO_PRIMARY_M1_JOINT_LOG_LOSS, MACRO_PRIMARY_M2_JOINT_LOG_LOSS, MACRO_DELTA_FACILITY_JOINT_LOG_LOSS, MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS, FANNIE_PROPOSED, FANNIE_PROVENANCE
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 3.4. Point-in-time information constraint

- Claim IDs: MACRO_DEVELOPMENT_M1_JOINT_LOG_LOSS, MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS, MACRO_PRIMARY_M1_JOINT_LOG_LOSS, MACRO_PRIMARY_M2_JOINT_LOG_LOSS, MACRO_DELTA_FACILITY_JOINT_LOG_LOSS, MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 4. Data

- Claim IDs: PD_SAMPLE, SURV_COUNTS
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 4.1. Freddie source

- Claim IDs: COHORT_COMPONENTS
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 4.2. Cohort construction

- Claim IDs: PD_COUNTS, SURV_COUNTS, COHORT_RECOVERY, COHORT_ROWS
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 4.3. Event definitions

- Claim IDs: SURV_DESIGN
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 4.4. Censoring

- Claim IDs: SURV_DESIGN
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 4.5. Macro data

- Claim IDs: PIT_SERIES, PIT_FEATURES, PIT_PROVENANCE, PIT_SUPPORT
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 4.6. Data limitations

- Claim IDs: PIT_GAPS, PD_LIMITS
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 5. Methods

- Claim IDs: See subsection claim bindings
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 5.1. 12-month PD baseline

- Claim IDs: PD_FEATURES
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 5.2. Discrete-time competing risks

- Claim IDs: SURV_DESIGN
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 5.3. M0/M1/M2 ladder

- Claim IDs: MACRO_ACTUAL_FEATURES
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 5.4. Temporal validation

- Claim IDs: MACRO_DESIGN_COUNTS, MACRO_LEDGER
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 5.5. Calibration

- Claim IDs: PD_CALIBRATION_INTERCEPT, MACRO_CAL_M1_PAYOFF, MACRO_CAL_M2_PAYOFF
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 5.6. CIF estimation

- Claim IDs: MACRO_CIF_24_SUPPORT
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 5.7. Cluster bootstrap

- Claim IDs: MACRO_DELTA_FACILITY_JOINT_LOG_LOSS, MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 5.8. Support diagnostics

- Claim IDs: DIAG_SUPPORT
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 6. Results

- Claim IDs: See subsection claim bindings
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 6.1. Expanded PD results

- Claim IDs: PD_LOGISTIC_ROC_AUC, PD_LOGISTIC_MEAN_PROBABILITY, PD_LOGISTIC_OBSERVED_RATE, PD_CALIBRATION_INTERCEPT, PD_CALIBRATION_SLOPE
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 6.2. Competing-risk foundation

- Claim IDs: SURV_60_NAIVE_NET_DEFAULT, SURV_60_DEFAULT_CIF
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 6.3. Macro development performance

- Claim IDs: MACRO_DEVELOPMENT_M1_JOINT_LOG_LOSS, MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS, MACRO_PRIMARY_M1_JOINT_LOG_LOSS, MACRO_PRIMARY_M2_JOINT_LOG_LOSS, MACRO_DELTA_FACILITY_JOINT_LOG_LOSS, MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 6.4. Temporal deterioration

- Claim IDs: MACRO_DEVELOPMENT_M1_JOINT_LOG_LOSS, MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS, MACRO_PRIMARY_M1_JOINT_LOG_LOSS, MACRO_PRIMARY_M2_JOINT_LOG_LOSS, MACRO_DELTA_FACILITY_JOINT_LOG_LOSS, MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 6.5. Discrimination versus calibration

- Claim IDs: MACRO_PRIMARY_M1_PAYOFF_AUC, MACRO_PRIMARY_M2_PAYOFF_AUC, MACRO_PRIMARY_M1_PAYOFF_BRIER, MACRO_PRIMARY_M2_PAYOFF_BRIER, MACRO_CAL_M1_PAYOFF, MACRO_CAL_M2_PAYOFF
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 6.6. Cumulative incidence

- Claim IDs: MACRO_CIF_24_OBSERVED_PAYOFF, MACRO_CIF_24_M1_PAYOFF, MACRO_CIF_24_M2_PAYOFF, DIAG_CIF_COMPONENTS
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 7. Diagnostic Analysis

- Claim IDs: DIAG_SUPPORT, DIAG_COMPOSITION, DIAG_SURVIVORS, DIAG_PAYOFF_VARIANCE, DIAG_NO_EVENT, DIAG_ANNUAL, DIAG_COEFFICIENTS, DIAG_ORACLE, DIAG_CIF_COMPONENTS, DIAG_ASSESSMENT
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 7.1. Covariate support

- Claim IDs: DIAG_SUPPORT, DIAG_RANGE_UNEMPLOYMENT_LEVEL
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 7.2. Composition shift

- Claim IDs: DIAG_COMPOSITION, DIAG_SURVIVORS
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 7.3. Payoff instability

- Claim IDs: DIAG_PAYOFF_VARIANCE, DIAG_ZERO_UNEMPLOYMENT_LEVEL
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 7.4. Calendar regimes

- Claim IDs: DIAG_ANNUAL
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 7.5. Limits of attribution

- Claim IDs: DIAG_SUPPORT, DIAG_COEFFICIENTS, DIAG_ASSESSMENT
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 8. Exploratory refinancing incentive

- Claim IDs: REFI_EXPLORATORY_P1_JOINT_LOG_LOSS, REFI_EXPLORATORY_P2_JOINT_LOG_LOSS, REFI_EXPLORATORY_P1_PAYOFF_AUC, REFI_EXPLORATORY_P2_PAYOFF_AUC, REFI_PAIRED, REFI_DECISION, REFI_GAP, REFI_PAIRED
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 9. Proposed external replication — no results

- Claim IDs: FANNIE_PROPOSED, FANNIE_PROVENANCE
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 10. Discussion — bounded implications slots

- Claim IDs: MACRO_DEVELOPMENT_M1_JOINT_LOG_LOSS, MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS, MACRO_PRIMARY_M1_JOINT_LOG_LOSS, MACRO_PRIMARY_M2_JOINT_LOG_LOSS, MACRO_DELTA_FACILITY_JOINT_LOG_LOSS, MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS, MACRO_PRIMARY_M1_PAYOFF_AUC, MACRO_PRIMARY_M2_PAYOFF_AUC, MACRO_PRIMARY_M1_PAYOFF_BRIER, MACRO_PRIMARY_M2_PAYOFF_BRIER, MACRO_CAL_M1_PAYOFF, MACRO_CAL_M2_PAYOFF, DIAG_SUPPORT, DIAG_COEFFICIENTS, DIAG_ASSESSMENT, REFI_EXPLORATORY_P1_JOINT_LOG_LOSS, REFI_EXPLORATORY_P2_JOINT_LOG_LOSS, REFI_EXPLORATORY_P1_PAYOFF_AUC, REFI_EXPLORATORY_P2_PAYOFF_AUC, REFI_PAIRED, REFI_DECISION
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 11. Limitations

- Claim IDs: MACRO_LIMITS, PD_LIMITS, PIT_APC, REFI_LIMITS, FANNIE_PROVENANCE
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 12. Reproducibility and governance

- Claim IDs: MACRO_LEDGER, FANNIE_PROVENANCE
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## 13. Conclusion — verified claim slots only

- Claim IDs: SURV_60_NAIVE_NET_DEFAULT, SURV_60_DEFAULT_CIF, MACRO_DEVELOPMENT_M1_JOINT_LOG_LOSS, MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS, MACRO_PRIMARY_M1_JOINT_LOG_LOSS, MACRO_PRIMARY_M2_JOINT_LOG_LOSS, MACRO_DELTA_FACILITY_JOINT_LOG_LOSS, MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS, MACRO_PRIMARY_M1_PAYOFF_AUC, MACRO_PRIMARY_M2_PAYOFF_AUC, MACRO_PRIMARY_M1_PAYOFF_BRIER, MACRO_PRIMARY_M2_PAYOFF_BRIER, MACRO_CAL_M1_PAYOFF, MACRO_CAL_M2_PAYOFF
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## Appendix. Supplement — sensitivities and provenance

- Claim IDs: B3_FEASIBILITY, SURV_DURATION, MACRO_REDUCED_SENS, MACRO_RATE_SENS
- Gate: retain source population, clock, estimand, uncertainty and exploratory status.

## Task 15 drafting constraints

- Fannie: proposed external replication awaiting provenance closure and protocol freeze; DRAFT_NOT_YET_AUTHORIZED.
- Temporal Freddie deterioration only; no demonstrated cross-provider failure.
- Task 6 60-month evidence: descriptive net-risk/CIF contrast, not supported model forecast.
- Task 10/12 CIF: historical rolling PIT paths, not known-at-entry prospective paths.
- Task 11 diagnostics and Task 12 exploration stay labeled; no causal explanation.
- Literature novelty and publication permission remain review items.

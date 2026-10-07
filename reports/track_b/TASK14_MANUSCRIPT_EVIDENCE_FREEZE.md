# Task 14 manuscript evidence freeze

## Decision

**ARXIV MANUSCRIPT EVIDENCE FREEZE READY WITH MATERIAL LIMITATIONS**.

Task 14 consumes existing public frozen aggregates and design evidence only. No new loan data, outcome access, fitting, calibration, resampling, bootstrap, predictions or headline metric calculation. No abstract, manuscript introduction, discussion narrative or conclusions drafted. Hash freeze denotes a paper evidence inventory, not dataset pre-registration or external replication sealing.

## Central question and contributions

- Ranked question 1: Does the frozen PIT macro feature set improve temporal competing-risk probability quality relative to the mortgage baseline in the evaluated Freddie population?
- Contribution scope: design-specific negative temporal result; ranking versus probability quality; conditional competing-event estimands; bounded diagnostics; exploratory refinancing; auditable evidence lineage.
- Original methodological novelty: not established; literature search deferred to Task 15.
- Primary supported result slots: H2 descriptive net-risk/CIF contrast; H3 development gain versus temporal deterioration; H4 payoff ranking/proper-score divergence; H5 rolling historical PIT payoff CIF discrepancy, with diagnostic propagation qualification.
- Secondary: expanded PD benchmark/calibration, horizon-supported competing-risk validation, fixed sensitivities and support design.
- Exploratory: Task 11 POST_VALIDATION_DIAGNOSTIC and Task 12 EXPLORATORY_ONLY. No diagnostic repair promoted and no new independent evidence created.

## Headline audit

| Headline | Audit | Qualification |
| --- | --- | --- |
| H1 | QUALIFY | Meaningful ranking and material underprediction in Task 5, not deployable calibration |
| H2 | QUALIFY | Different estimands: descriptive net risk versus observed competing CIF, conditional entry |
| H3 | VERIFY | Task 10 design-specific development gain and temporal deterioration |
| H4 | VERIFY | Payoff ranking gain accompanies worse proper scores/calibration in this design |
| H5 | QUALIFY | Large payoff CIF discrepancy; component substitution traces numerical propagation, not a causal mechanism |
| H6 | QUALIFY | Post-hoc diagnostic support only |
| H7 | QUALIFY | EXPLORATORY_ONLY; small payoff Brier gain is not robust to calendar block uncertainty |
| H8 | VERIFY | Cross-provider transport has not been tested |

## Exact frozen headline values

Values below retain source JSON precision. Probabilities are fractions, not percentage labels. Every number is an exact evidence lookup, not a fresh calculation.

| Metric | Exact value | Claim ID |
| --- | --- | --- |
| Task 5 logistic temporal AUC | `0.7833771782234903` | `PD_LOGISTIC_ROC_AUC` |
| Task 5 logistic AP | `0.1763022671246067` | `PD_LOGISTIC_AVERAGE_PRECISION` |
| Task 5 logistic Brier | `0.007422536403713394` | `PD_LOGISTIC_BRIER` |
| Task 5 logistic log loss | `0.04233034893925971` | `PD_LOGISTIC_LOG_LOSS` |
| Task 5 XGBoost temporal AUC | `0.7222974999941579` | `PD_XGBOOST_ROC_AUC` |
| Task 5 XGBoost AP | `0.017483339178524127` | `PD_XGBOOST_AVERAGE_PRECISION` |
| Task 5 mean predicted risk | `0.003958756535786404` | `PD_LOGISTIC_MEAN_PROBABILITY` |
| Task 5 observed risk | `0.008322206005651648` | `PD_LOGISTIC_OBSERVED_RATE` |
| Task 5 CITL | `0.9871119332588771` | `PD_CALIBRATION_INTERCEPT` |
| Task 5 calibration slope | `0.7680755130883121` | `PD_CALIBRATION_SLOPE` |
| Task 6 60m naive net default | `0.052061664461949264` | `SURV_60_NAIVE_NET_DEFAULT` |
| Task 6 60m observed default CIF | `0.03100281090705663` | `SURV_60_DEFAULT_CIF` |
| Task 10 development M0 log loss | `0.09170962895471475` | `MACRO_DEVELOPMENT_M0_JOINT_LOG_LOSS` |
| Task 10 development M1 log loss | `0.09015128103398809` | `MACRO_DEVELOPMENT_M1_JOINT_LOG_LOSS` |
| Task 10 development M2 log loss | `0.08959281435366433` | `MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS` |
| Task 10 temporal M1 log loss | `0.08920130868490905` | `MACRO_PRIMARY_M1_JOINT_LOG_LOSS` |
| Task 10 temporal M2 log loss | `0.1053229451127057` | `MACRO_PRIMARY_M2_JOINT_LOG_LOSS` |
| Task 10 M1 payoff AUC | `0.5654257013618629` | `MACRO_PRIMARY_M1_PAYOFF_AUC` |
| Task 10 M2 payoff AUC | `0.625867954528649` | `MACRO_PRIMARY_M2_PAYOFF_AUC` |
| Task 10 M1 payoff Brier | `0.015127305620779442` | `MACRO_PRIMARY_M1_PAYOFF_BRIER` |
| Task 10 M2 payoff Brier | `0.01982496438444564` | `MACRO_PRIMARY_M2_PAYOFF_BRIER` |
| Task 10 24m observed payoff CIF | `0.33673487196277013` | `MACRO_CIF_24_OBSERVED_PAYOFF` |
| Task 10 24m M1 payoff CIF | `0.258686480614572` | `MACRO_CIF_24_M1_PAYOFF` |
| Task 10 24m M2 payoff CIF | `0.7581426348757327` | `MACRO_CIF_24_M2_PAYOFF` |
| Task 12 development P1 log loss | `0.09015128103398809` | `REFI_DEVELOPMENT_P1_JOINT_LOG_LOSS` |
| Task 12 development P2 log loss | `0.08967110795138739` | `REFI_DEVELOPMENT_P2_JOINT_LOG_LOSS` |
| Task 12 exploratory P1 log loss | `0.08920130868490905` | `REFI_EXPLORATORY_P1_JOINT_LOG_LOSS` |
| Task 12 exploratory P2 log loss | `0.09578292521181422` | `REFI_EXPLORATORY_P2_JOINT_LOG_LOSS` |
| Task 12 P1 payoff Brier | `0.015127305620779442` | `REFI_EXPLORATORY_P1_PAYOFF_BRIER` |
| Task 12 P2 payoff Brier | `0.015115946076881624` | `REFI_EXPLORATORY_P2_PAYOFF_BRIER` |
| Task 12 P1 payoff AUC | `0.5654257013618629` | `REFI_EXPLORATORY_P1_PAYOFF_AUC` |
| Task 12 P2 payoff AUC | `0.6245545634498562` | `REFI_EXPLORATORY_P2_PAYOFF_AUC` |

- Task 10 paired M2-minus-M1 log loss: `0.01612163642779664`; facility percentile 95 interval `[0.01540973362343688, 0.016783525058395806]`; calendar-year interval `[0.000053258694600802695, 0.041502093619988564]`. Claims MACRO_DELTA_FACILITY_JOINT_LOG_LOSS / MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS.
- Task 12 exploratory paired P2-minus-P1 log loss: `0.006581616526905171`; facility interval `[0.005800916048663267, 0.007324344797756831]`; calendar-year interval `[0.0007903907546276087, 0.017209066007567036]`. Claim REFI_PAIRED.
- Task 12 payoff Brier delta facility interval excludes zero, but calendar-year interval `[-0.00005098802206086796, 0.00003759846512088142]` includes zero. Do not present the tiny payoff Brier gain as robust improvement.
- Task 5 calibration intercept is calibration-in-the-large, distinct from its joint intercept. Do not exchange the two.
- Task 6 60-month contrast is descriptive nonparametric conditional-entry evidence; the frozen structural model has insufficient training-duration support at 60 months.
- Task 10 CIF uses historical rolling PIT macro paths, not prospective paths known at entry. Task 12 has the same path restriction.

## Population and design freeze

- Expanded sample: 20000 selected 2010-vintage facilities; cohort has 1241045 eligible landmarks / 19590 facilities / 7153 positive landmarks / 618 defaulting facilities. Development 478796 landmarks / 13801 facilities; evaluation 128812 landmarks / 2423 facilities. Primary cohort hash and split counts are pinned in PD_COUNTS and dataset facts.
- Task 6 conditional-entry global cohort: 19606 facilities / 1254425 risk intervals; development 13813 facilities / 471976 intervals; evaluation 2423 facilities / 132409 intervals. Default/payoff/censor counts and risk-set hashes remain source-linked in SURV_COUNTS.
- Recovered seven-vintage source cohort: 140000 selected facilities and 7741663 canonical rows across 2006, 2008, 2010, 2014, 2018, 2020 and 2022. These are not the fitted-model population.
- Task 10 primary development: 42609 facilities / 1568661 risk intervals / 1904 defaults / 25822 payoffs. Primary temporal seen-vintage evaluation: 5619 facilities / 248939 intervals / 280 defaults / 3823 payoffs. Unseen-vintage supplement remains separate.
- Task 10 development September 2010 through December 2017; purge calendar 2018; evaluation January 2019 through February 2026. Primary vintages 2006/2008/2010/2014, with 2018/2020/2022 separate unseen-vintage restriction.
- Task 5 and Task 6 development ends December 2014; calendar 2015 purged; evaluation starts January 2016. Internal Task 5 groups are facility-disjoint, not independently time-held-out.
- Candidate macro acquisition includes UNRATE, DGS10, MORTGAGE30US, USSTHPI, CPIAUCSL and GDPC1. Eligibility uses eight features; actual M2 uses seven and excludes the redundant spread. Reduced sensitivity uses five. Source series are not synonymous with model predictors.
- All population facts, transformations, event/censor definitions, supported windows and source/sample/risk-set hashes resolve in dataset_facts_registry.json and experiment_design_freeze.json. Missing quantities remain null, not inferred.

## Statistical units and prior exposure

- Task 5: 1000 fixed-model facility/loan cluster draws, 2423 evaluation facilities. Overlapping monthly landmarks are not independent observations.
- Task 6: 400 facility horizon bootstrap draws; censoring/Aalen-Johansen reestimated within original draws, models fixed. Monthly uncertainty is separately recorded. Calendar-window entry differs in seasoning and survivor selection.
- Tasks 10 and 12: 1000 paired facility draws, 5619 clusters; 1000 calendar-year draws, eight annual blocks including partial 2026. Facility intervals condition on the realized macro path; calendar uncertainty is coarse.
- No training uncertainty or unknown cross-facility borrower dependence is recovered by these intervals. Calibration and diagnostic-refit coefficient intervals remain null where absent.
- Task 5 temporal aggregate outcomes and nested Task 3 predictions were previously inspected. Task 6 shares prior Task 4/5 exposure. Task 10 is locked evaluation after support design, not a virgin holdout. Task 12 explicitly uses inspected Tasks 10/11 outcomes.
- The exact proxy age-period-cohort identity and rank deficiency are preserved; national macro coefficients are constrained predictive associations, not unrestricted causal identification.

## Diagnostics and mechanism gate

- DIAG_SUPPORT records 77 multivariate extrapolative months; the frozen distinct-month evaluation count is 86 in DIAG_RANGE_UNEMPLOYMENT_LEVEL. Univariate range frequencies, composition, variance accounting, no-event excess loss, ablations, 2020/2023 calibration, coefficient geometry/condition numbers, oracle and component substitutions remain individually source-linked.
- Support extrapolation, coefficient instability and payoff amplification: supported diagnostically; separate causal contributions unidentified.
- Composition/survivor shift, macro collinearity and default mapping instability: partially supported.
- Some intercept drift/regime association is visible, but the original hypotheses that intercept drift dominates or the pandemic alone explains failure are NOT_SUPPORTED. The same-sample oracle is optimistic and independently unvalidated.
- No ablation is causal feature importance; no post-hoc root-cause ranking becomes an identified mechanism.

## Fannie and rejected language

- FANNIE SOURCE PROVENANCE REQUIRES PROVIDER CONFIRMATION.
- Protocol exactly DRAFT_NOT_YET_AUTHORIZED; Task 13A unauthorized; no external outcome analysis.
- Allowed: proposed external replication awaiting provenance closure and protocol freeze.
- Reject positive descriptions of Fannie as sealed, registered, pre-registered, replicated or externally validated.
- Reject cross-provider transport failure, industry-wide generalization, causal macro effects, observed borrower refinancing thresholds, full origination-lifetime risk and regulatory/production validation claims.

## Figures, tables and reproducibility

- Eight figure plans: architecture; competing-risk states; development/temporal scores; support map; payoff ranking/calibration; CIF; exploratory refinancing; proposed external extension.
- Nine table plans: cohorts; specifications; expanded PD; competing-risk validation; macro ladder; diagnostics; exploratory refinancing; boundaries; proposed Fannie mapping.
- Each figure and table resolves claim IDs and exact source pointers. No figure/table generated in Task 14. Frozen aggregates suffice for planned formatting/redrawing; new raw-data curves, bins or bands are DO_NOT_GENERATE.
- Formatting from public frozen aggregates is reproducible; re-running empirical research requires provider access and private data. Proposed Fannie results are not currently reproducible because none exist.
- Preprint positioning is provisional; category fit, endorsement and eligibility not determined. Related-work search concepts and inclusion criteria are defined, no citations invented.

## Generalization and publication boundaries

- Thirteen boundaries cover mortgage/consumer, Freddie/US population, facility/borrower, EAD/CCF, severity/LGD, SICR, regulatory PD, conditional/lifetime, national/geographic macro, temporal/external, association/causality, macro/mortgage knowledge time and historical/prospective paths.
- Current research is not an IFRS 9, IRB, regulatory PD/LGD/EAD, production bank, regulator validated or audit approved model.
- Raw records and loan identifiers remain PRIVATE_NON_COMMITTABLE. Public metadata and source hashes are not permission to redistribute data.
- PUBLICATION_REVIEW_REQUIRED applies to empirical counts, metrics, coefficients and figures. Existing sparse-cell suppression must carry into manuscript tables; no permission conclusion is made from artifacts already being public.
- Applicable Freddie publication permissions and unresolved Fannie accepted terms need review before public submission. Task 15 drafting readiness does not imply publication clearance.

## Evidence conflicts and resolutions

- C_MANIFEST: earlier 60000-facility STOP versus later complete 140000-facility recovery — documented supersession, both historical artifacts retained.
- C_BRIER_PRECISION: Task 10 score versus Task 11 decomposition differs at floating-point accounting precision — preserve both; primary frozen Task 10 score governs headline.
- C_FANNIE_TERMINOLOGY: human sealed/pre-registered description versus draft machine status — higher-priority status governs.
- No unresolved PRIMARY evidence conflict. The conflicts registry retains both evidence sources and resolutions.

## Task 15 gate

READY_WITH_MATERIAL_LIMITATIONS for separately authorized drafting: PRIMARY claims verified, quantitative headline pointers pinned, exploration labeled, Fannie and generalization/regulatory gates present, conflicts explicitly resolved, no new empirical calculations. This task does not execute Task 15. Literature novelty and publication permissions remain open. Conclusions must follow the verified registry, not preselect a positive story.

## Verification and preservation

The machine-readable task14_verification.json records final tests and preservation. Prior research files, artifacts, sources and ledgers are hashed only; no model loading, scoring or ledger consumption. Track A historical AUC 0.868152 / Brier 0.048545 / log loss 0.176030 remain retained historical evidence, not new evaluation.

## Files created

- `docs/paper/arxiv_manuscript_architecture.md`
- `docs/paper/arxiv_positioning.md`
- `docs/paper/dataset_facts_registry.json`
- `docs/paper/evidence_conflicts.json`
- `docs/paper/experiment_design_freeze.json`
- `docs/paper/figure_registry.json`
- `docs/paper/generalization_boundaries.json`
- `docs/paper/headline_claim_audit.json`
- `docs/paper/manuscript_section_map.json`
- `docs/paper/mechanism_claim_gate.json`
- `docs/paper/paper_contribution_candidates.json`
- `docs/paper/paper_evidence_manifest.json`
- `docs/paper/publication_data_boundary.json`
- `docs/paper/related_work_search_plan.md`
- `docs/paper/reproducibility_matrix.json`
- `docs/paper/statistical_unit_audit.json`
- `docs/paper/table_registry.json`
- `docs/paper/task14_audit_coverage.json`
- `docs/paper/task14_preservation_manifest.json`
- `docs/paper/track_b_claim_evidence_registry.json`
- `scripts/build_paper_evidence.py` — stdlib aggregate-copy builder, rejects existing evidence.
- `scripts/check_paper_evidence.py` — exact pointer/hash/reference/terminology gates.
- `tests/test_paper_evidence.py` — mutation and preservation regressions.
- `reports/track_b/TASK14_MANUSCRIPT_EVIDENCE_FREEZE.md` — this audit report.
- `reports/track_b/task14_verification.json` — final verification record.

## Git boundary

Start: aa6eb3a001429d362c7fd76a45b9affc4995846f. Intended commit: docs: freeze Track B manuscript evidence. No push; STOP after reporting.

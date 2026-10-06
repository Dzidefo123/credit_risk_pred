# Track B dataset search strategy

Status: Task 0 plan only. No dataset search/selection/download was performed for this task. Official accounting/governance guidance was consulted only to bound the design. Track A stays unchanged.

## Next task and discovery scope

Exactly one next task: **Candidate Dataset Discovery and Comparative Suitability Assessment**. Begin with read-only discovery of publisher pages, dictionaries, schemas, licensing and coverage documentation. Record provenance and component-specific evidence using the [framework](DATASET_SUITABILITY_FRAMEWORK.md) and [contract](LONGITUDINAL_DATA_CONTRACT.md). Do not accept license agreements, register accounts, pay for access, download record-level data or fit models without separately authorized scope. A metadata shortlist is not a chosen or empirically validated source.

## Source categories to investigate later

| Category | Potential contribution | Main evidence to inspect | Frequent limitations to challenge |
| --- | --- | --- | --- |
| Public credit datasets | Repeated credit snapshots or event-only PD | Identity, dates, future event/follow-up and provenance | Static labels, unclear observation time, anonymized row IDs |
| Academic/research panels | Hazard/transition and methodological comparisons | Sampling, repeated units, censoring, event derivation and reproducibility | Restricted access, small event counts, narrow populations |
| Central-bank/supervisory sources | Credit-register panels or economic context | Micro vs aggregate unit, access/license, histories and definitions | Confidential microdata, aggregated releases not loan panels |
| Mortgage performance sources | Amortizing PD/lifetime/EAD; possibly loss/workouts | Loan-month linkage, schedules, defaults, prepayments, recoveries and collateral geography | Servicing/selection changes, censored loans and product-specific applicability |
| Consumer-credit performance panels | Behavioral PD/transitions or revolving EAD | Drawn/undrawn balances, limit histories, event exposure, payment/exit data | Missing recovery histories or origination risk |
| Peer-to-peer lending sources | Term-loan outcomes and selected repayment histories | Snapshot vs evolving status, issue/availability dates, charged-off definitions and loan selection | Rewritten statuses, incomplete recoveries, issued-loan selection and future-outcome leakage |
| Public macro/release archives | Historical context and forecast-vintage/scenario research | Reference/release dates, revisions, geography, forecasts-as-of and reuse terms | Latest revisions only, realized values substituted for historical forecasts |
| Synthetic sources | Architecture, interface and edge-case tests | Explicit generation assumptions and synthetic provenance | Cannot establish empirical performance or replace economic loss observations |

No listed category guarantees any component. Sources that publish only delinquency classifications cannot be presumed to contain loss cash flows. Baseline/upside/downside names do not imply available calibrated scenarios or weights.

## Discovery procedure and planned outputs

1. Read publisher/maintainer documentation and record source ownership, exact version, license/access terms and dictionary links. Separate authorized public metadata from restricted records; use no guessed undocumented fields.
2. Complete one dossier per candidate/version covering accessibility, temporal depth, stable entity identity, outcome definition, exposure history, recoveries, macro linkage and reproducibility. Mark unknowns rather than filling them with assumptions.
3. Apply global gates and all eight component decisions independently. Record narrower admissible scope and hard rejection reasons. Metadata-supported feasibility remains provisional until separately authorized record-level checks.
4. Compare options by supported components, critical gaps, lawful access and replication feasibility. Do not compute an overall score, choose a dataset from reputation, or treat a large row count as temporal evidence.
5. Recommend a documented component-level shortlist and explicit next authorization needed. Discovery delivers a comparative assessment, not acquisition or model training. No empirical metrics are expected from this task.

The later comparison should include evidence-backed SUPPORTED/PARTIALLY SUPPORTED/UNSUPPORTED conclusions where documentation permits, and UNASSESSED pending evidence otherwise. A missing core gate cannot receive a supported modeling recommendation.

## Multi-dataset design and common interfaces

A defensible possible arrangement is Dataset A for dated PD/lifetime risk, Dataset B for default/workout LGD, Dataset C for revolving EAD and a separately versioned public macro archive. These are roles, not selected sources. Independently assess each population and estimand. Do not join unrelated facilities/borrowers using coincident IDs or profiles.

Future interfaces would carry dataset/version namespace, entity mapping, t0/information vintage, product/currency, default definition, eligibility/coverage, horizon/grid, output meaning, scenario/discount basis and uncertainty/provenance. PD interfaces must distinguish conditional hazard, marginal first-default mass and cumulative probability. LGD/EAD interfaces must declare default-time conditioning and amount bases. Scenario interfaces must separate reference, release, forecast-as-of and future periods. Validation interfaces must expose exclusions, censoring and group/time overlap.

Cross-source component reuse requires documented target harmonization, population/product/economic overlap, selection/transportability assessments and conditional dependence assumptions. Without linked or externally validated transportability, an assembled loss example remains cross-dataset research integration/sensitivity, not one observed portfolio's empirical ECL. Public macro linkage is geographic/calendar/vintage alignment, not evidence that micro datasets describe the same customers.

## Proposed future package architecture - not implemented

```text
src/credit_risk/track_b/
    data/          # authorized adapters, entity/PIT/coverage contracts
    pd/            # incident 12-month/default-risk research
    survival/      # hazards, competing exits and term structures
    lgd/           # episode-linked workouts and censored recovery
    ead/           # term/revolving exposure and drawdown research
    sicr/          # comparable origination/current risk experiments
    staging/       # declared policy/heuristic classification evidence
    ecl/           # time/scenario/discount integration and reconciliation
    macro/         # release/forecast vintages and scenario interfaces
    stress/        # justified stress/sensitivity experiments
    validation/    # temporal/entity, component and integration checks
    governance/    # provenance, experiment access, findings and simulated approvals
```

This is a proposal only: no empty package/modules are created in Task 0. Add implementations only when a later task establishes appropriate data and scope. Reuse generic utilities after scientific review, not Track A models/targets or any assumption that its holdout is fresh. Future Track B holdout identity/reservation rules need their own dated/entity-aware design; do not reset or repurpose the existing consumed registry.

## Research questions and defensible comparisons

| Question | Proposed investigation after data acceptance | Evidence/design needed |
| --- | --- | --- |
| RQ-B1 | How does risk evolve across future horizons and competing exits? | Coherent conditional/marginal/cumulative estimates, coverage and calendar validation |
| RQ-B2 | Do hazard/survival methods outperform multi-horizon binary baselines for the same estimand? | Common PIT features/population/horizons, censoring-aware metrics and fixed chronological evaluation; no presumed superiority |
| RQ-B3 | How does workout loss vary with borrower, facility and economic context? | Matched EAD/recoveries/costs, completed/censored workouts and selection sensitivity; association is not causation |
| RQ-B4 | How does exposure evolve before default across products? | PIT balances/limits/draws, schedules and matched default exposure; consider default-conditioned selection |
| RQ-B5 | How sensitive are SICR classifications to alternative declared research policies? | Comparable original/current risk, qualitative evidence, reasons and maturity/model-version controls; no accounting optimality claim |
| RQ-B6 | How do justified macro forecast scenarios change term structures and lifetime loss estimates? | Dated forecast vintages, compatible component conditioning/discounting and transparent weights; no fabricated scenarios |
| RQ-B7 | How stable are PD/LGD/EAD estimates across observed economic regimes and stress assumptions? | Regime/event coverage, true out-of-time tests and separate empirical tests from unobserved-shock sensitivity |
| RQ-B8 | How do parameter, censoring, model, scenario and integration uncertainties propagate into loss estimates? | Dependence-aware component uncertainty, alternative assumptions and interval calibration; no unwarranted independent-error aggregation |

Model comparisons and final evaluation rules must be prespecified after accepted data scope. No RQ presumes regulatory validity, causal identification or deployment readiness.

## Governance handoff

The contract records required evidence for Model Development -> Independent Validation -> Findings -> Developer Response -> Remediation -> Approval Simulation -> Monitoring -> Recalibration/Redevelopment Trigger. Preserve reviewer independence and evidence of challenge, not merely a passing demo. Simulated approvals do not constitute independent bank validation. Before acquisition/modeling, separately resolve license, sensitive-data handling, component eligibility, PIT lineage and untouched temporal validation populations.

Task 0 stops at these three documents and their documentation/integrity tests. Track A is unchanged. The only next recommendation is **Candidate Dataset Discovery and Comparative Suitability Assessment**.

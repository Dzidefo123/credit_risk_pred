# Track B Task 1: dataset selection decision

Decision date: 2026-10-06. Status: prespecified research design before inspection of loan records. No download, registration, license acceptance, panel ingestion, model fitting or empirical performance assessment is authorized or performed by this task.

## Formal selection and boundary

Select **Freddie Mac Standard** as the development source for the restricted mortgage research program below. Pin the Release 47 schema family; verify the actual acquired release/file layout later rather than assuming every historical file has that format. Do not include Non-Standard, reperforming-loan products or the full historical collection in the initial pilot. Fannie Mae Primary remains a future external replication candidate, not a second development/tuning source.

This is a conditional selection for named research estimands, not blanket acceptance of the [77-field contract](LONGITUDINAL_DATA_CONTRACT.md). The [crosswalk](FIELD_DICTIONARY_COMPARISON.md) and [comparative assessment](DATASET_COMPARATIVE_ASSESSMENT.md) remain the recorded evidence of field gaps. Their PARTIAL/UNSUPPORTED conclusions for full accounting/borrower scope are not overwritten by a narrower permission to proceed with research design.

The latest-release panel is initially a **retrospective loan-month study with unverified knowledge-time vintages**. It cannot claim historical real-time scoring or prove that every revised value was available at t0. Later authentic release archives or validated source availability evidence would be required for that stronger claim. Track A is frozen at [track-a-v1.0](https://github.com/Dzidefo123/credit_risk_pred/tree/track-a-v1.0); neither its predictions nor consumed observations are reused or reinterpreted.

## Component decision matrix

PROCEED means proceed to the separately authorized data-engineering gate, not fit a model now. Each restricted interpretation is binding.

| Research component | Decision | Binding interpretation |
| --- | --- | --- |
| 12-month mortgage PD | PROCEED | Probability of the first observed composite adverse mortgage event before payoff, within 12 monthly intervals |
| Lifetime PD / survival | PROCEED | Monthly hazards/cumulative incidence with delayed entry, coverage, exits and extrapolation limitations |
| Competing risk of prepayment | PROCEED | Code 01 combines voluntary payoff/maturity; study the combined payoff endpoint, not pure discretionary prepayment |
| Mortgage EAD | PROCEED WITH DEFINED INTERPRETATION | Monthly principal at an observed research event; not full accounting exposure or revolving EAD |
| Workout LGD | PROCEED WITH LIMITATIONS | Aggregate disposition-loss severity proxy only; full timed workout LGD remains unsupported |
| SICR research | PROCEED AS RESEARCH POLICY SIMULATION | Later explicit heuristic/comparability study, not observed institutional policy validation |
| IFRS 9 staging | LATER / POLICY-DEPENDENT | Recognition, impairment and policy evidence must be separately established |
| ECL | LATER | After compatible PD/exposure/loss foundations and justified scenario/discount assumptions |
| Revolving CCF/EAD | OUT OF SCOPE | No revolving commitment/drawdown observations |
| Borrower-level portfolio modeling | UNSUPPORTED | Loan identifiers do not establish obligor identities across facilities |

The name Workout LGD in the requested matrix does not authorize a full workout claim. The permitted initial object is a disposition-loss proxy; transaction dates, completed recoveries, accounting EIR and downturn adjustments are not recovered by naming a ratio LGD.

## Decisions fixed before records

The [research design](MORTGAGE_RESEARCH_DESIGN.md) and [machine-readable protocol](mortgage_research_protocol.json) fix t0, event definitions, payoff/censoring distinctions, 12-month labels, uncertainty boundaries and the small pilot plan. No event counts, score results or loss distributions informed these choices. A change requires an issue/reason, versioned protocol diff and acknowledgment of which data/outcomes had already been seen; it cannot be presented retrospectively as prespecified.

The primary research event is first observed severe-delinquency/REO state or a specified credit-related terminal event. It is explicitly a composite proxy, not exact 90-day default, legal/accounting default or default at origination. Sales, defects and unknowns have separate rules. Same-month incompatible causes are quarantined rather than silently ordered.

## Acquisition and implementation gates

Task 2 must separately confirm lawful access/usage and repository sharing rules before obtaining a single official sample-vintage archive. No raw/sample loan records, vendor PDFs or credentials belong in Git by default. Pin source/release/layout, acquisition date, source hashes and revision provenance. If availability, parsing or source completeness fails a core gate, stop the affected component; a PROCEED decision cannot repair missing evidence.

Use the fixed pilot: official 2010 Standard sample, at most 1,000 retained loans, chosen by deterministic identifier hashing without outcomes. Retain their entire available histories, including exits and gaps. Do not expand vintages or change the subset to obtain more defaults. An official sample archive may contain more loans than are retained; a separately agreed acquisition/streaming/storage budget must be checked before retrieval. This task authorizes neither that retrieval nor the whole historical dataset.

Engineering acceptance precedes modeling: schema and masks/sentinels, unique loan-month keys, event/exit date consistency, coverage and revisions, correct target windows, no future-feature joins, reproducible selection, and preserved Track A evidence. Small pilot results establish software behavior only, not portfolio representativeness or validated predictive performance.

## Research sequence and next task

First identify the 12-month event probability for the declared facility process. Then compare coherent hazard/survival and competing-payoff formulations on that same estimand. No XGBoost challenger, calibration search, policy threshold or accounting model is introduced by Task 1. Model families, evaluation cohorts and final access rules need a separate prespecified modeling task after engineering acceptance.

Exactly one next task: **Track B Task 2 - authorized small-sample acquisition and longitudinal-panel engineering**, with extensive temporal leakage and censoring tests. No data or models implemented now.

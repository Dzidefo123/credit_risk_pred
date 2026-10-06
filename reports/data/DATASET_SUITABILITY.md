# Dataset suitability assessment

Task 2, 2026-10-05. This is a dataset capability assessment, not a regulatory validation or an approval of lending use. It relies on [documented provenance/target semantics](../../docs/TARGET_DEFINITION.md), [feature definitions](../../docs/DATA_DICTIONARY.md), [leakage findings](LEAKAGE_REVIEW.md) and retained/new descriptive audits. No new model performance was computed.

## Current research uses

| Use | Suitability | Basis and conditions |
| --- | --- | --- |
| Existing PD classification research | **Conditionally suitable** | Useful for benchmark/challenger classification of the inherited source outcome. Describe it as serious-delinquency research; preserve grouped partitions and train-only preprocessing. Does not establish lender applicability, observed originations, representative rejects or future-vintage performance |
| Probability calibration research | **Conditionally suitable** | Can compare predicted probabilities against the binary source label with independent fit/selection partitions. Calibration refers to this outcome and population, not verified contractual default. Follow-up/sampling gaps limit transportability. The original final test is consumed and cannot support fresh independent calibration comparisons |
| Regulatory PD research | **Unsuitable for empirical regulatory PD development/validation** | The event/window equivalence, stable obligor identity, dated population, representativeness and governance evidence are absent. The file may illustrate techniques, but cannot establish regulatory estimates |

The official [Basel CRE36 requirements](https://www.bis.org/committees/bcbs/basel-framework/standard/cre/36/inforce/2023-01-01/published/2022-12-08) distinguish default adjudication and risk-estimation requirements from a competition classifier label. This assessment does not infer compliance from shared delinquency terminology.

## IFRS 9 component assessment

| Component | Support from this dataset | Missing evidence/data |
| --- | --- | --- |
| 12-month PD | **Unsuitable** | Dated observations/events and twelve-month follow-up/censoring rules; a two-year aggregate label does not reveal first-year events |
| Lifetime PD | **Unsuitable** | Contractual/expected lives, dated event/survival histories, closures/prepayments and censoring; two years is not the life of each facility |
| Stage 1 | **Unsuitable** | Instrument scope/recognition date, current versus initial risk and absence of credit impairment; source label is not current staging |
| Stage 2 | **Unsuitable** | Comparable initial/current lifetime risk, instrument history, forward-looking evidence and a justified SICR policy |
| Stage 3 | **Unsuitable** | Current credit-impaired status and dated adjudication; historical occurrence counts and future binary labels cannot establish it |
| SICR | **Unsuitable** | Risk at initial recognition and reporting date, updated information and supportable assessment; no repeated dated instruments |
| LGD | **Unsuitable** | Default exposures, recoveries, collateral/workout costs, timing and economic loss; none are measured here |
| EAD | **Unsuitable** | Currency balances, limits, scheduled/drawn exposures and drawdowns at default; utilization alone is not exposure or a conversion factor |
| ECL | **Unsuitable** | Instrument cash flows, exposure/loss data, discounting, scenario-conditioned risk and staged horizons; multiplication of proxies is not empirical IFRS 9 ECL |

These conclusions follow from the observed data gaps. The standard's scope and impairment framework are described by the [IFRS Foundation's IFRS 9 page](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/). [OSFI's official IFRS 9 guidance](https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/ifrs-9-financial-instruments-disclosures) explains assessment relative to initial recognition and twelve-month versus lifetime loss allowances. [IFRS Foundation implementation material](https://www.ifrs.org/news-and-events/news/2016/07/25-webcast-on-ifrs-9/) discusses forward-looking information and multiple scenarios. OSFI is cited as public explanatory guidance, not as the project's jurisdiction or evidence of local compliance.

Twelve-month ECL is not simply cash losses paid during the next twelve months. Its default-event horizon and subsequent cash-shortfall measurement must be distinguished. Regardless of terminology, this file supplies neither the event timing nor those cash flows. Splitting the two-year probability, assuming a constant hazard, assigning stages from past-event counts or importing assumed LGD/limits would create new assumptions, not recover missing observations.

## Future data requirements

A different, authorized longitudinal loan/account source is required for empirical IFRS 9 work. It should provide stable obligor/facility keys, origination/recognition and reporting dates, dated default/credit-impairment events, eligibility and follow-up, actual balances/limits/draws, contractual schedules/maturities, recoveries/costs/collateral and discount-rate information. Forward-looking scenarios must be separately justified and linked to the observation calendar. Supplier definitions, sampling, reporting lags and units belong in a versioned data contract.

Existing synthetic portfolio, vintage/roll-rate and assumed-loss demonstrations remain educational examples. Their dates and exposures cannot be joined to these anonymous profiles or used to claim real dataset support. No future model, IFRS 9 stage or distributed component was implemented in Task 2.

## One next implementation task

Add a source-and-sample holdout registry so a new experiment directory cannot label already consumed observations as fresh final validation. Preserve compatibility with trusted historical evidence and test using fixtures, without new original-holdout prediction access. This addresses the remaining integrity mechanism identified by the audit; it does not cure missing borrower/date provenance or authorize new fits. Do not begin it without a separate instruction.

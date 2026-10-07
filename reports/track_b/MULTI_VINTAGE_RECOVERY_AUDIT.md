# Track B Task 8C — multi-vintage cohort recovery

MULTI-VINTAGE COHORT READY WITH MATERIAL LIMITATIONS

Original Task 8 STOP, Task 8A unresolved audit and Task 8B design remain unchanged. Policy B changes research eligibility only. Four exact source observations remain raw and unresolved; no token is coerced or identifier/cohort rewritten.

## Gate and sample evidence

| Vintage | Status | Eligible universe | Quarantine | Sample SHA256 |
| --- | --- | --- | --- | --- |
| 2006 | READY_WITH_LIMITATIONS | 1193543 | 1 | 1b39c08d96da214fd5e33273c6d38e0a88940f59afe96537625bbeaaf20af371 |
| 2008 | READY_WITH_LIMITATIONS | 1233145 | 1 | 91286c8885dbc90b8c91bb7a8300eda1df0cf63e45f43f72f7ea1e697c27e26d |
| 2010 | READY_WITH_LIMITATIONS | Not reconstructed; inherited frozen sample | 0 | e719a6b4d23caca8ac55eeaa6fae54da23c95887dd83b3ac22cd925d63902832 |
| 2014 | READY_WITH_LIMITATIONS | 1142999 | 0 | d26b2ba3e08837f82c940ecefbaa18c4bd201d64a0a114760eafeeb80a8126f5 |
| 2018 | READY_WITH_LIMITATIONS | 1285434 | 0 | b3e2952bb5bc0d0a551105cecfa964b18481b3764442edb8362245f4f17f921e |
| 2020 | READY_WITH_LIMITATIONS | 3913740 | 1 | 978c32ebac99980286cf83e86051316dad6080f9cc167c017900e15cd7e2a3f1 |
| 2022 | READY_WITH_LIMITATIONS | 1580752 | 1 | 8b0e0fee2f6921c2c6a04363b01cb234be7ccb7cd93a350765e6faac5db66f59 |


Frozen samples are not completed cohorts. Combined frozen facilities: 140000; completed audited facilities: 140000; completed monthly rows: 7741663. No borrower uniqueness claimed.

## Fail-closed findings

## Events

| Vintage | Default | Payoff/maturity | Administrative | Ambiguous | Active/unknown |
| --- | --- | --- | --- | --- | --- |
| 2006 | 2876 | 16750 | 38 | 46 | 290 |
| 2008 | 1956 | 17625 | 39 | 36 | 344 |
| 2010 | 623 | 18147 | 29 | 33 | 1168 |
| 2014 | 800 | 16018 | 33 | 28 | 3121 |
| 2018 | 1051 | 15488 | 42 | 9 | 3410 |
| 2020 | 352 | 6913 | 28 | 21 | 12686 |
| 2022 | 678 | 3492 | 86 | 41 | 15703 |


Descriptive first observed raw endpoints under frozen Task2 definitions; not estimated incidence or model performance.

## Follow-up and calendar

| Vintage | Raw span min/median/max | Analytical follow-up min/median/max | Calendar first/last |
| --- | --- | --- | --- |
| 2006 | 0.0/50.0/241.0 | 0.0/44.0/241.0 | 2006-01 / 2026-03 |
| 2008 | 0.0/38.0/218.0 | 0.0/33.0/218.0 | 2008-01 / 2026-03 |
| 2010 | 0.0/53.0/194.0 | 0.0/51.0/194.0 | 2010-01 / 2026-03 |
| 2014 | 0.0/64.0/146.0 | 0.0/62.0/146.0 | 2014-01 / 2026-03 |
| 2018 | 0.0/29.0/98.0 | 0.0/27.0/98.0 | 2018-01 / 2026-03 |
| 2020 | 0.0/63.0/74.0 | 0.0/63.0/74.0 | 2020-01 / 2026-03 |
| 2022 | 0.0/44.0/50.0 | 0.0/44.0/50.0 | 2022-01 / 2026-03 |


Natural intervals from first observation, with right censoring; no common truncation or inferred daily origination date.

## Horizon support

| Vintage | 12 | 24 | 36 | 60 | 84 | 120 |
| --- | --- | --- | --- | --- | --- | --- |
| 2006 | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED |
| 2008 | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED |
| 2010 | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED |
| 2014 | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED |
| 2018 | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED | UNSUPPORTED |
| 2020 | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED | UNSUPPORTED | UNSUPPORTED |
| 2022 | SUPPORTED | SUPPORTED | SUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |


Unchanged thresholds: SUPPORTED requires >=1000 at risk and >=10000 known status; LIMITED >=100 and >=2000; otherwise UNSUPPORTED. Earlier default/payoff outcomes can be known; unknown/gap/administrative censoring is not filled.

## Missingness, distributions and release mapping

Machine evidence includes original score/LTV/DTI/UPB/rate/term quantiles, purpose/occupancy distributions and field missingness for completed cohorts only. SOURCE_QUARANTINED is counted separately at the source-record level; selected-field SOURCE_QUARANTINED counts are zero because exclusions precede selection. Entirely blank fields remain MISSING_IN_SOURCE, never presumed STRUCTURALLY_UNAVAILABLE. No feature selection or modeling.

## Age-period and pandemic support

| Calendar year | Contributing vintages |
| --- | --- |
| 2006 | 1 |
| 2007 | 1 |
| 2008 | 2 |
| 2009 | 2 |
| 2010 | 3 |
| 2011 | 3 |
| 2012 | 3 |
| 2013 | 3 |
| 2014 | 4 |
| 2015 | 4 |
| 2016 | 4 |
| 2017 | 4 |
| 2018 | 5 |
| 2019 | 5 |
| 2020 | 6 |
| 2021 | 6 |
| 2022 | 7 |
| 2023 | 7 |
| 2024 | 7 |
| 2025 | 7 |
| 2026 | 7 |


| Provider age band | Calendar years represented | First/last year |
| --- | --- | --- |
| <= 12 | 21 | 2006 / 2026 |
| <= 24 | 20 | 2007 / 2026 |
| <= 36 | 19 | 2008 / 2026 |
| <= 60 | 18 | 2009 / 2026 |
| <= 84 | 16 | 2011 / 2026 |
| <= 120 | 14 | 2013 / 2026 |
| <= 180 | 11 | 2016 / 2026 |
| <= 240 | 6 | 2021 / 2026 |
| >240 | 1 | 2026 / 2026 |


Full age × year × vintage counts and overlap density are in the companion JSON. These inherited prefix counts include terminal/censor boundaries. Completed seven-vintage support can inform the preferred constrained predictive design; regime coverage, macro availability, censoring and evaluation gates remain required. Multiple occupied cells do not establish unrestricted APC identification.

Provider age and the first-payment elapsed clock are not interchangeable. A separate full proxy-clock matrix and exact provider-age-24 counts appear in JSON. No age reset mechanism is inferred or source age corrected.

| Vintage | 2020 active facilities | 2021 active facilities |
| --- | --- | --- |
| 2006 | 831 | 637 |
| 2008 | 989 | 775 |
| 2010 | 4534 | 3309 |
| 2014 | 9753 | 6858 |
| 2018 | 15622 | 8614 |
| 2020 | 17619 | 18983 |
| 2022 | 0 | 0 |


Active pandemic counts require a known no-event state inside the analytical prefix; they exclude endpoints, unknowns and later post-endpoint rows.

Old source performance archives were not rescanned. Canonical completed CSVs were read for newly required active-pandemic facility counts and clock diagnostics; their bytes remain unchanged.

## APC diagnostics

{
  "synthetic": {
    "rows": 35,
    "columns": 4,
    "rank": 3,
    "null_vector": [
      0,
      1,
      -1,
      -1
    ],
    "maximum_identity_residual": 0.0,
    "scope": "Exact-clock illustration; does not assert exact mortgage origination month"
  },
  "empirical_proxy": {
    "unique_clock_rows": 18738,
    "columns": 4,
    "rank": 3,
    "maximum_identity_residual": 0.0,
    "scope": "First-payment proxy clocks; not exact origination date or provider age"
  },
  "structural_identity": "period = cohort + age",
  "unrestricted_apc_identified": false,
  "causal_macro_effects_identified": false,
  "future_constraints": [
    "smooth duration basis",
    "parsimonious cohort representation",
    "national macro replacing unrestricted period FE",
    "prespecified interactions only"
  ]
}

Overlap supports assessment of a future constrained predictive design, not unrestricted APC or causal macro identification.

## Resources

| Vintage | Performance/audit seconds | Peak process MiB | Output MiB |
| --- | --- | --- | --- |
| 2006 | 488.0 | 159.5 | 771.1 |
| 2008 | 388.9 | 159.5 | 700.3 |
| 2010 | 139.8 | 154.8 | 207.0 |
| 2014 | 508.0 | 154.8 | 763.9 |
| 2018 | 395.4 | 154.8 | 657.3 |
| 2020 | 779.8 | 159.5 | 1550.4 |
| 2022 | 335.8 | 159.5 | 760.3 |


Peak memory is process high-water. Stage timing excludes origination where documented; reused rows retain historical measurements. ZIP materialization is measured; SQLite temporary sort spill is not instrumented.

## Limitations

- Current-release retrospective disclosure: operational knowledge time and historical revisions remain unverified.

- ZIP timestamps and matching field counts do not independently attest the exact source release.

- Facility identity is not borrower identity; cross-facility or cross-vintage borrower independence is not established.

- First payment month is not exact origination or accounting recognition; no fabricated daily dates or exact DPD.

- Monthly research default is a proxy; payoff includes maturity, not uniquely voluntary prepayment.

- Loss amounts are signed aggregate disclosures, not timed recovery cash flows, accounting LGD or ECL.

- Modification and assistance disclosures do not identify every intervention or pandemic forbearance.

- Natural follow-up is unequal; no outcome extrapolation, adaptive replacement or global truncation.

- Overlap does not identify unrestricted age, period and cohort effects or causal macro coefficients.

- Source semantics for A1-A4 remain unresolved; exact-record eligibility exclusion does not establish provider error or harmlessness.

- Inherited 2010 frozen sample has no reconstructed full eligible-universe count; do not present identifier census as full eligibility certification.

- Source-wide structural performance column checks; semantic typing covers selected histories, not every unselected performance record.

- Provider-reported age diverges from the immutable first-payment elapsed clock on some histories. Both support matrices are retained separately; no reset mechanism is inferred and no provider age is rewritten. Broad provider-age coverage does not establish original-duration overlap.

## Verification

Final tests and preservation are recorded separately in multi_vintage_recovery_verification.json. Historical AUC0.868152, Brier0.048545 and log loss0.176030 remain retained results; no holdout evaluation or artifact regeneration.

## Exactly one next task

Point-in-time macro acquisition and feature engineering, constrained to demonstrated periods/horizons and APC restrictions

# Track B Task 7 — Macro Data Provenance

## Purpose

Establish information architecture before fitting macro-conditioned competing risks. Decision: **MACRO DESIGN READY WITH MATERIAL IDENTIFICATION LIMITATIONS**. No mortgage joins, model fitting, prediction regeneration, consumed-ledger access, stress calculation or new mortgage acquisition took place. Track A and Tasks 2–6 stay frozen. The eight candidates are hypotheses, not validated causal channels or a mandate to include all eight in regression.

Companion specifications: [feature registry](macro_feature_registry.json), [research protocol](macro_research_protocol.json), [identification design](MACRO_IDENTIFICATION_DESIGN.md), [observation schema](macro_observation.schema.json), [acquisition schema](macro_acquisition_manifest.schema.json), [scenario schema](macro_scenario.schema.json).

## Point-in-Time Principle

Economic reference period is the time being measured. Original agency release date is the first publication of that observation. Revision date is the agency publication of a particular replacement value. Vintage date is the version's availability in the chosen archive. Retrieval time is when this research downloaded evidence, potentially years later; it is not historic knowledge time. Forecast origin and scenario publication/as-of dates are separate clocks.

The primary information set is **latest known as of t0**: reference-period end, original release, version revision and archive vintage must all be <= t0. Initial-release sensitivity requires certified revision_sequence=0; absence of an initial vintage must never be filled with today's history. Current-revised values live in a separate descriptive representation and the predictor selector rejects them. Retrieval after t0 is legitimate for a genuine historically archived vintage, not proof that current data were known historically.

The implementation has date resolution and assumes assessment at the end of t0. It does not establish an intraday publication timestamp. If dates are uncertain, require documented conservative upper bounds from an archive or exclude the version. An assumed mean lag is never evidence of a release date. Mortgage knowledge time itself remains UNVERIFIED in the inherited retrospective panel; vintage-aware macro data do not make the entire mortgage information set historically operational.

## Candidate Series

| Series | Definition/source | Native frequency; units; adjustment | Reference history begins | Role and mechanism hypotheses |
| --- | --- | --- | --- | --- |
| UNRATE | BLS CPS U-3 unemployment | Monthly; percent; SA | 1948-01 | Primary: repayment capacity; employment conditions/mobility affect payoff |
| PAYEMS | BLS CES total nonfarm jobs | Monthly; thousands; SA | 1939-01 | Reserve alternative labor signal; income/jobs and transaction activity |
| FEDFUNDS | Board effective funds rate, monthly daily average | Monthly; percent; NSA | 1954-07 | Reserve rate regime; differs from FOMC target and existing fixed mortgage coupon |
| DGS10 | Board 10-year constant-maturity Treasury market yield | Daily; percent; NSA | 1962-01-02 | Primary long-rate/refinancing environment; not a traded bond's realized return |
| MORTGAGE30US | Freddie PMMS 30-year fixed mortgage average | Weekly ending Thursday; percent; NSA | 1971-04-02 | Conditional primary mortgage/refinancing environment; not borrower-specific offer |
| USSTHPI | FHFA national all-transactions repeat-sales HPI | Quarterly; 1980:Q1=100; NSA | 1975-Q1 | Conditional primary equity/default incentives and mobility; includes appraisal/refinancing information |
| GDPC1 | BEA real GDP level at annual rate | Quarterly; chained 2017 dollars currently; SA annual rate | 1947-Q1 | Primary activity/income and housing transaction environment |
| CPIAUCSL | BLS all-items CPI-U city average | Monthly; 1982–84=100; SA | 1947-01 | Primary real-income/rate channel; payoff effect indirect and sign uncertain |

Reference starts are from linked official FRED table metadata in the registry. They describe current history, NOT vintage completeness. GDP historical releases may use older dollar bases; parser must preserve metadata per version and harmonize explicitly before growth calculations. No cross-base arithmetic. PMMS changed methodology on 17 November 2022: a new study must document that break rather than silently pool measurement regimes. [Freddie series notes](https://fred.stlouisfed.org/series/MORTGAGE30US).

## Source Authority

Hierarchy: agency original release plus historical release calendar -> official vintage archive of values (ALFRED) -> official current history for descriptive checking only. Agency publication and archive availability must both be established for each admitted version. No third-party CSV mirrors.

Authoritative definitions, source links, units, frequency and adjustment were inspected for all eight series using the official FRED/ALFRED pages listed in the registry. [ALFRED real-time documentation](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html) explains the difference between present FRED knowledge and past information sets. [Observation API documentation](https://fred.stlouisfed.org/docs/api/fred/series_observations.html) supplies real-time fields and initial/all-revision output options; an API request without an explicit historical vintage is not acceptable as a predictor acquisition.

Small source-probe evidence: [metadata probe](../../reports/track_b/macro_source_probe.json) and [release-date probe](../../reports/track_b/macro_release_lag_probe.json). A direct HTTP probe is bounded to 16 requests, 25-second timeout per request and 2 MB per response; failures are recorded. Official web-source inspection is distinct from successful raw HTTP acquisition. There is no claim that a failed download was verified by a file hash. The separately hashed web-tool extraction covers three official metadata pages and is labeled as extracted response JSON, not original HTTP/API bytes; no observations from it are admitted to predictors. Any current historical table encountered is descriptive evidence only; it is never parsed into predictor observations.

## Vintage Availability

All eight have **VINTAGE-AWARE AVAILABLE** official archive metadata. This classification is source feasibility, not certification of every desired historical version. Observed ALFRED metadata histories begin: UNRATE 1960-03-15; PAYEMS 1955-05-06; FEDFUNDS 1996-12-03; DGS10 2005-06-28; MORTGAGE30US 2010-06-17; USSTHPI 2010-08-25; GDPC1 1991-12-04; CPIAUCSL 1972-07-21. These are metadata history endpoints, not assumed exact first-observation release calendars.

The two 2010 onsets are material for the proposed pre-2010 cohorts. Primary use before verified archive coverage is blocked. Official historical releases could make gaps **RELEASE-DATE RECONSTRUCTABLE**, but only after actual value/version/calendar evidence is acquired. We have not certified such a reconstruction. Existing coverage cannot be projected backward. If only current history is obtained, classify that representation **CURRENT-REVISED ONLY**; if no dated source is established, **INSUFFICIENT PROVENANCE**. Neither is admitted to primary predictors. First-release calculations require complete initial-release evidence, not merely the earliest record in a truncated download.

## Release Lags

Lag = original publication date minus inclusive reference-period end, in calendar days. Revision lag and archive delay must be reported separately. Dates below are a deliberately small documented agency-release sample, not distributional estimates across 2010–2026.

| Series | Sample n | Median days | Range days | Evidence scope |
| --- | ---: | ---: | --- | --- |
| UNRATE | 3 | 4 | 2–6 | March–May 2014 original Employment Situation releases |
| PAYEMS | 3 | 4 | 2–6 | Same release dates, different underlying survey |
| CPIAUCSL | 3 | 15 | 15–17 | March–May 2014 CPI releases |
| GDPC1 | 1 | 30 | 30–30 | Q1 2014 advance release only |
| USSTHPI | 0 | unavailable | unavailable | FHFA Q1 HPI release-family benchmark 57 days; exact all-transactions version not certified |
| MORTGAGE30US | 0 | unavailable | unavailable | Weekly release dating must be certified across historical methodology/holiday regimes |
| DGS10 | 0 | unavailable | unavailable | Daily market observation versus H.15 publication/archive date must be reconciled |
| FEDFUNDS | 0 | unavailable | unavailable | Completed monthly average, published afterward; exact historical calendar not acquired |

The [March labor release](https://www.bls.gov/news.release/archives/empsit_04042014.htm), [April labor release](https://www.bls.gov/news.release/archives/empsit_05022014.htm) and [May labor release](https://www.bls.gov/news.release/archives/empsit_06062014.htm) document 4 April, 2 May and 6 June. The [March CPI](https://www.bls.gov/news.release/archives/cpi_04152014.htm), [April CPI](https://www.bls.gov/news.release/archives/cpi_05152014.htm) and [May CPI](https://www.bls.gov/news.release/archives/cpi_06172014.htm) document 15 April, 15 May and 17 June. [BEA Q1 advance](https://www.bea.gov/news/2014/gross-domestic-product-1st-quarter-2014-advance-estimate) was 30 April. [FHFA Q1 release](https://www.fhfa.gov/news/news-release/fhfa-house-price-index-rises-for-eleventh-consecutive-quarter) was 27 May; the accompanying purchase-only headline must not be mislabeled USSTHPI. Full per-version lag distributions remain an acquisition acceptance check.

## Geographic Coverage

Choose **national US** as the primary geography before modeling. The pinned Freddie layout includes property_state and MSA; the retained panel carries property_state but not MSA. Existence of a geography column is not evidence of complete/accurate historic mapping. Task 6 documents sparse defaults and avoided geographic modeling. No new loan geography inspection is needed for this national design; actual state/MSA missingness and stability are unmeasured in Task 7.

National coverage is common across the eight candidates, avoids identity linkage and an unverified state/MSA vintage registry, and is computationally modest. A national macro observation applies to every US facility even if its state is missing. The mapper rejects state/MSA/postal requests. State features would require dated LAUS/HPI coverage, mappings including boundary changes, missingness/event-support diagnostics and a separate amendment. MSA would add sparse support and changing definitions; regional aggregation would need prespecified boundaries and weights. Neither is justified by finer granularity alone. National variables cannot identify geographic effects.

## Transformations

Primary candidates: unemployment level and three-month percentage-point change; latest released mortgage and Treasury levels; their difference in percentage points with separate native reference dates; HPI YoY percent growth; CPI SA YoY percent growth; GDP quarter-on-quarter nonannualized percent growth. PAYEMS YoY growth and effective funds level are reserve alternatives, not extra feature mining. Do not automatically fit all correlated rate/labor variables.

YoY = 100*(known_level(q)/known_level(q-12 months)-1). GDP QoQ = 100*(known_level(q)/known_level(q-3 months)-1), explicitly nonannualized despite GDP level being SAAR. Both operands are resolved in the SAME t0 information set and carry hashes/release/vintage/revision metadata. First-release sensitivity uses each operand's initial value separately. An unavailable lag is unavailable, not interpolated. Monthly and quarterly endpoint alignment is required. Quarterly information is held only after release; no unreleased within-quarter interpolation. Daily/weekly primary rate levels use latest released native observation, not an incomplete-month average. A rate spread with nonmatching native dates retains both dates and must pass freshness checks.

Registry freshness caps are prespecified research choices: D 7 days, W 14 days, M 62 days, Q 183 days since reference end. They are not release-date evidence. Future feature engineering must pass these caps through the selector for the current feature period and each rate-spread operand, report stale/missing rates and prohibit backward filling. An exact YoY/QoQ lag denominator is intentionally older: it must satisfy knowledge-time rules but does not inherit the current-period age cap. The primitives are unit-tested; no mortgage-sized feature table is built now.

## Join Semantics

`select_asof` explicitly filters completed reference periods and all three publication/version dates before choosing the newest reference period and its latest eligible version. Exact-period lag selection uses the same filter. An initial-release selector never substitutes a revision for revision_sequence=0. Duplicates and inconsistent units/frequency/source/adjustment fail clearly. Missing input returns None. It is a correctness primitive over a small version collection; a future scalable batch join must establish equivalence before processing real mortgage risk intervals.

Required examples are tested: March31 cannot see March released April; April30 sees the initial March version; May cannot see the June revision; July can see it under latest-known, while first-release still keeps the initial version. An archive revision dated later than t0 is prohibited even if its original observation release date was earlier. Unknown dates are schema errors.

## Leakage Risks

Reject current history in predictor input; future versions/periods/forecasts; implicit reference-month equality joins; backfilled initial releases; transformations using later revised denominators; mixed dollar bases; dates inferred from average lags; future observed delinquency paths; and silently using today's mortgage disclosure as a verified historic operational snapshot. Missingness, freshness and regime shifts require aggregate diagnostics. No licensed loan geography or IDs enter public reports.

Raw evidence uses exclusive creation and SHA256 verification. Acquisition manifest must contain source, series, geography, acquisition timestamp, URL, raw response hash/path, vintage/reference/release coverage, units, adjustment, license, parser version and date-evidence description. Initial-release use also requires complete_revision_history=True plus verified initial observation coverage.

Zones under the existing ignored data/track_b tree: macro/raw immutable responses; macro/interim normalized version rows retaining source hashes; macro/processed features with t0 and operand lineage; macro/manifests acquisition and processing specifications/hashes. Raw never overwritten. A deterministic sorted-key JSON hash identifies processed content; processing metadata must include registry/protocol/parser versions and parent response hashes. None of these private zones requires committing actual macro or mortgage rows.

## Scenario Architecture

Represent baseline/adverse/research-downside/upside as coherent future monthly paths, not a static shock. Each scenario records as-of, forecast origin, publication date, provider, provenance, horizon, variable units/frequency, conversion assumptions, internal-consistency review and raw source hashes. Each variable must have ordered complete horizons1..H at future month ends. A forecast must have actually been issued by as-of; an observed future point cannot be relabeled forecast. Metadata validation cannot prove a falsely labeled source; raw provider publication evidence is a mandatory later approval gate.

Historical replay uses a previously observed path shifted to future scenario months, preserves original reference/release dates, is descriptive and carries no probability. It is not a forecast historically available at the old replay origin. Forecast scenarios preserve their actual issuer clocks. Hypothetical paths are labeled researcher-defined, including percentage points versus percent change, persistence and cross-variable relationships. No arbitrary shock values or executed scenario paths are supplied here. A research downside must never be labeled an official regulatory stress.

Weights are optional. If supplied, require source/method evidence, all scenarios weighted, finite values in[0,1] and total1. No invented 60/30/10 weights. Numerical/schema checks do not establish economic coherence or defensible probabilities; a named future risk/research reviewer must sign off.

## Identification Limitations

One 2010 cohort intertwines age, calendar environment, cohort underwriting and survivor selection. Millions of monthly rows do not create millions of independent macro environments. Default before payoff is a competing-risk quantity; macro changes payoff composition as well as default hazard. See the dedicated identification document for the age-period-cohort rank problem and proposed restrictions. Pandemic labor shocks/forbearance/interventions and PMMS measurement change need explicit regime handling.

## Recommended Design

Exactly one preferred design: **prespecified multiple origination vintages, national vintage-aware macro information, predictive joint competing hazards with explicit constrained duration/cohort representation**. This improves overlap, without claiming unrestricted age-period-cohort or causal identification. No model is fitted. Acquisition and modeling gates are in the protocol. The next task is **Track B Task 8 — Multi-Vintage Mortgage Cohort Acquisition and Harmonization**, not implementation of macro stress or ECL.

# Track B Task 9 — Point-in-Time Macro Data Audit

## Executive Summary

PIT MACRO SUPPORT INSUFFICIENT

243 unique reporting months, 140,000 facilities and 7,741,663 rows retained. Validated numeric vintage versions: 7358. No model fitting or outcome-based selection.

## Research Boundary

National US; six Task7 primary series, eight frozen features; reserves unchanged. Mortgage sample identity and prior evidence stay frozen.

## Series Registry

| Series | Provider | Frequency | Units | Freshness days |
| --- | --- | --- | --- | --- |
| UNRATE | BLS | M | Percent | 62 |
| DGS10 | Federal Reserve Board | D | Percent | 7 |
| MORTGAGE30US | Freddie Mac | W | Percent | 14 |
| USSTHPI | FHFA | Q | Index 1980:Q1=100 | 183 |
| GDPC1 | BEA | Q | Billions of Chained 2017 Dollars | 183 |
| CPIAUCSL | BLS | M | Index 1982-1984=100 | 62 |


Registry SHA256 (LF): `0e06c5afad76edc0c6ea0ade459181265a6cc319398c51cd9129f11b0030aea1`. Frozen before predictor requests.

## Source Provenance

| Series | Acquired provenance | Versions | Status |
| --- | --- | --- | --- |
| UNRATE | VINTAGE_AWARE_AVAILABLE | 499 | DATED_VERSIONS_ACQUIRED |
| DGS10 | INSUFFICIENT_PROVENANCE | 0 | NO_PREDICTOR_OBSERVATIONS |
| MORTGAGE30US | VINTAGE_AWARE_AVAILABLE | 1160 | DATED_VERSIONS_ACQUIRED |
| USSTHPI | VINTAGE_AWARE_AVAILABLE | 3437 | DATED_VERSIONS_ACQUIRED |
| GDPC1 | VINTAGE_AWARE_AVAILABLE | 856 | DATED_VERSIONS_ACQUIRED |
| CPIAUCSL | VINTAGE_AWARE_AVAILABLE | 1406 | DATED_VERSIONS_ACQUIRED |


[ALFRED help](https://alfred.stlouisfed.org/help) distinguishes archive dating from exact provider release timing. [Real-time export documentation](https://alfred.stlouisfed.org/help/downloaddata) defines value validity intervals. Exact dates stay null unless independently certified; archive availability provides a conservative upper bound, never an invented agency date.

## Acquisition

FRED_API_KEY configured: True. Each attempted request is bounded, recorded and credential-redacted. Raw payloads use exclusive writes. Successful HTML metadata is never parsed as predictor observations.

| Series | Endpoint | HTTP status | Failure |
| --- | --- | --- | --- |
| UNRATE | https://api.stlouisfed.org/fred/series | 200 |  |
| UNRATE | https://api.stlouisfed.org/fred/series/vintagedates | 200 |  |
| UNRATE | https://api.stlouisfed.org/fred/series/observations | 200 |  |
| DGS10 | https://api.stlouisfed.org/fred/series | 200 |  |
| DGS10 | https://api.stlouisfed.org/fred/series/vintagedates | 200 |  |
| DGS10 | https://api.stlouisfed.org/fred/series/observations | 400 | HTTPError |
| MORTGAGE30US | https://api.stlouisfed.org/fred/series | 200 |  |
| MORTGAGE30US | https://api.stlouisfed.org/fred/series/vintagedates | 200 |  |
| MORTGAGE30US | https://api.stlouisfed.org/fred/series/observations | 200 |  |
| USSTHPI | https://api.stlouisfed.org/fred/series | 200 |  |
| USSTHPI | https://api.stlouisfed.org/fred/series/vintagedates | 200 |  |
| USSTHPI | https://api.stlouisfed.org/fred/series/observations | 200 |  |
| GDPC1 | https://api.stlouisfed.org/fred/series | 200 |  |
| GDPC1 | https://api.stlouisfed.org/fred/series/vintagedates | 200 |  |
| GDPC1 | https://api.stlouisfed.org/fred/series/observations | 200 |  |
| CPIAUCSL | https://api.stlouisfed.org/fred/series | 200 |  |
| CPIAUCSL | https://api.stlouisfed.org/fred/series/vintagedates | 200 |  |
| CPIAUCSL | https://api.stlouisfed.org/fred/series/observations | 200 |  |


Warmup reference window January2004–March2026 covers 12-month operands plus quarterly freshness/release margins. No current-revised shadow was acquired.

## Knowledge-Time Rules

End-of-day date resolution; reference complete and documented publication/revision/archive bounds <=t0. Unknown exact provider dates remain null.

For reporting/risk month m use t0=end of m-1. Prevents a macro release during the outcome month entering that month hazard. No claim mortgage loan state was operationally known then.

LATEST_KNOWN_AS_OF_T0 selects the newest completed reference period whose real-time validity interval contains t0. INITIAL_RELEASE requires certified initial evidence and remains separate. CURRENT_REVISED is rejected. Input ordering cannot change selection.

## Release Lags

Exact historical provider lag distributions are unmeasured for all six series (n=0). Task7 small probes remain historical evidence, not a new full-history distribution.

## Vintage Coverage

Task7 metadata hints: PMMS begins June2010, HPI August2010. These remain feasibility hints until dated value exports establish actual coverage; no pre2010 backfill. [GDP archived metadata](https://alfred.stlouisfed.org/series?seid=GDPC1) supplies dollar-base regimes used by the adapter; it does not itself supply predictor values.

## Transformations

| Feature | Series | Transform | Lag months |
| --- | --- | --- | --- |
| unemployment_level | UNRATE | level | 0 |
| unemployment_change_3m | UNRATE | difference | 3 |
| treasury_10y_level | DGS10 | level | 0 |
| mortgage_30y_level | MORTGAGE30US | level | 0 |
| mortgage_treasury_spread | MORTGAGE30US | spread | 0 |
| hpi_yoy | USSTHPI | growth_pct | 12 |
| cpi_yoy | CPIAUCSL | growth_pct | 12 |
| gdp_qoq | GDPC1 | growth_pct | 3 |


Every available value retains source hashes, native reference endpoints, archive validity, exact/null provider dates, bounds, units/regime, t0, operands and feature hash. GDP growth is nonannualized. Paired operands share t0 and policy; mixed units/regimes fail closed. Exact older denominators are knowledge-constrained but exempt from the current operand freshness cap.

## Revision Analysis

No empirical revision magnitude can be reported without certified paired versions. Earliest retained does not imply initial. No current-revised denominator is permitted.

## Feature Availability

| Feature | Months in scope | Populated | Unavailable | Stale | Provenance unavailable |
| --- | --- | --- | --- | --- | --- |
| unemployment_level | 243 | 243 | 0 | 0 | 0 |
| unemployment_change_3m | 243 | 242 | 1 | 0 | 0 |
| treasury_10y_level | 243 | 0 | 243 | 0 | 243 |
| mortgage_30y_level | 243 | 189 | 54 | 0 | 0 |
| mortgage_treasury_spread | 243 | 0 | 243 | 0 | 243 |
| hpi_yoy | 243 | 187 | 56 | 0 | 0 |
| cpi_yoy | 243 | 243 | 0 | 0 | 0 |
| gdp_qoq | 243 | 243 | 0 | 0 | 0 |


Full feature × calendar-month evidence is in pit_macro_data_audit.json. Missing information remains unavailable, without statistical imputation.

## Mortgage-Month Join

| Vintage | Unique months | Rows retained | Facilities retained |
| --- | --- | --- | --- |
| 2006 | 243 | 1280801 | 20000 |
| 2008 | 219 | 1016145 | 20000 |
| 2010 | 195 | 1400360 | 20000 |
| 2014 | 147 | 1352589 | 20000 |
| 2018 | 99 | 819246 | 20000 |
| 2020 | 75 | 1034572 | 20000 |
| 2022 | 51 | 837950 | 20000 |


A normalized month-key relation and tested deterministic left join retain all facilities. Only the reporting-month column was counted from canonical panels; no outcomes were inspected for feature selection and no existing panel was rewritten.

## Common Support

| First risk month | Last risk month | Months | Status |
| --- | --- | --- | --- |
| 2006-01 | 2026-03 | 243 | PARTIAL_MACRO_SUPPORT |


Mortgage follow-up alone does not establish macro-modelable horizons. Combined support remains unestablished wherever the full PIT vector is unavailable; no mortgage facility is dropped.

## APC Constraint

period = cohort + age; rank3 with four columns. Prespecified future design: fixed duration bands, seven archive-vintage indicators (2006 reference), national PIT features and no unrestricted period effects or interaction search. No model fitted; actual design rank/conditioning and calendar-block uncertainty remain future gates.

## Pandemic Period

Descriptive reporting-year indicators for 2020 and 2021 only. No outcome-selected boundary, pandemic causal effect, scenario probability or complete intervention-policy claim.

## Limitations

- Official metadata feasibility is distinct from acquired, certified observation coverage.

- No exact provider publication lag is inferred from archive dates or historical average lags.

- Earliest retained archive values are not certified initial releases; initial-release sensitivity unavailable without independent evidence.

- HPI/PMMS pre2010 vintage hints are not backfilled; no alternate series introduced.

- GDP rebasing and PMMS 2022-11-17 measurement change remain explicit; unlike regimes cannot enter paired transformations.

- Macro date-only evidence supports end-of-day use; intraday timing is unsupported.

- PIT macro data do not certify the operational knowledge time of retrospective mortgage disclosures.

- Monthly aggregate join retains all facilities; expanded 7.7-million-row panel is not materialized or modified.

- Multiple vintages and macro variables do not identify unrestricted APC or causal macro coefficients.

## Decision

PIT MACRO SUPPORT INSUFFICIENT

Exactly one next task: Track B Task 9A — Obtain and certify authoritative ALFRED real-time exports, then rerun the PIT coverage gates without changing mortgage samples

Final tests and preservation are recorded separately in pit_macro_verification.json. No completion commit is permitted for insufficient support.

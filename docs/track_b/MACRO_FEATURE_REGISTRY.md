# Track B Task 9 macro feature registry

This registry prespecifies national US macro features. Its machine-readable
equivalent is [pit_macro_feature_registry.json](pit_macro_feature_registry.json);
source definitions and historical measurement regimes are frozen in
[pit_macro_series_registry.json](pit_macro_series_registry.json).
The source registry was hashed before predictor requests:
`0e06c5afad76edc0c6ea0ade459181265a6cc319398c51cd9129f11b0030aea1`.

**Original unauthenticated attempt: all eight features were unavailable.** Six bounded official
ALFRED export-form requests timed out; no API key was configured and no validated
vintage observations were acquired. Synthetic tests establish implementation
behavior, not historical coverage. No current revised history was substituted.

**Authenticated follow-up:** the six series were subsequently acquired as
metadata, vintage-date indexes and vintage-aware observations. The full eight
features have support September 2010–February 2026, with partial support outside
that window. See [authenticated acquisition](PIT_MACRO_API_ACQUISITION.md).
The frozen machine-readable definitions and registry hash are unchanged.

| Feature | Provider / series | Native frequency | Definition / units | Lag | Current-reference freshness |
| --- | --- | --- | --- | --- | --- |
| unemployment_level | BLS / UNRATE | Monthly, SA | Latest known unemployment rate; percent | None | 62 days |
| unemployment_change_3m | BLS / UNRATE | Monthly, SA | Latest known rate minus same-series rate three reference months earlier; percentage points | 3 months | 62 days |
| treasury_10y_level | Federal Reserve Board / DGS10 | Daily, NSA | Latest known 10-year constant-maturity Treasury rate; percent | None | 7 days |
| mortgage_30y_level | Freddie Mac / MORTGAGE30US | Weekly, NSA | Latest known 30-year fixed mortgage survey rate; percent | None | 14 days |
| mortgage_treasury_spread | Freddie Mac + Federal Reserve Board / MORTGAGE30US − DGS10 | Native weekly + daily | Mortgage rate minus Treasury rate; percentage points | None | Each operand: 14 / 7 days |
| hpi_yoy | FHFA / USSTHPI | Quarterly, NSA | 100 × (national all-transactions HPI / same quarter a year earlier − 1); percent | 12 months | 183 days |
| cpi_yoy | BLS / CPIAUCSL | Monthly, SA | 100 × (CPI / same month a year earlier − 1); percent | 12 months | 62 days |
| gdp_qoq | BEA / GDPC1 | Quarterly, SA annual-rate level | 100 × (real GDP / prior quarter real GDP − 1); **nonannualized** percent growth | 3 months | 183 days |

All eight are primary, prespecified calendar-condition covariates for future
research. None was selected against mortgage outcomes. PAYEMS and FEDFUNDS stay
reserve; no regional or substitute series was acquired.

## Availability convention

Reporting/risk month `m` uses `t0` at the end of the preceding month, under the
Task 7 date-only end-of-day convention. An eligible observation must have a
completed reference period and publication, revision and archive-availability
bounds no later than `t0`. Latest-known selection also requires its real-time
validity interval to contain `t0`. Exact release/revision dates remain null when
unknown. An ALFRED archive date can supply a conservative availability upper
bound; it is not relabeled as an agency publication date.

The primary rule is `LATEST_KNOWN_AS_OF_T0`. `INITIAL_RELEASE` is a distinct
sensitivity requiring independent certification of the initial release.
Earliest retained vintage does not certify an initial release. `CURRENT_REVISED`
is excluded. See [research protocol](pit_macro_research_protocol.json).

Freshness is elapsed days from the completed reference period to `t0`.
Lower-frequency observations may carry forward only while eligible and within
the cap; this deterministic as-of selection is not statistical imputation.
Historical growth/difference denominators use the exact required reference
period and the same information set; their age is intentional and exempt from
the current-reference cap. Missing or future denominators remain unavailable.

## Lineage and limitations

Each available engineered feature retains its operands, source hashes, native
reference dates, real-time intervals, exact-or-null release/revision dates,
availability bounds, retrieval time, units, measurement regime, `t0`, rule,
engine version and deterministic lineage hash. Missing and stale features retain
an explicit status rather than a fabricated value. Mortgage-month joins retain
facilities and do not overwrite the canonical panel.

GDP rebasing cannot mix dollar bases within a growth calculation. PMMS changed
methodology on 17 November 2022. Mortgage/Treasury spread operands can have
different native dates; each must pass its own freshness rule. No monthly
average is silently substituted for a native rate observation.

Historical metadata hints for PMMS/HPI archives do not establish pre-2010
coverage. No empirical release-lag or revision-magnitude result is available.
The retrospective mortgage disclosure's operational knowledge time remains a
separate limitation. Macro variables do not resolve the age-period-cohort
identity or establish causal effects.

Official evidence: [ALFRED help](https://alfred.stlouisfed.org/help),
[real-time exports](https://alfred.stlouisfed.org/help/downloaddata), and
[FRED real-time API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html).

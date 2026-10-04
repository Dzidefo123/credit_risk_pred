# Vintage and roll-rate analytics

Phase 6 analyzes longitudinal account-month histories. It does not turn the
inherited origination dataset into a portfolio or connect unrelated synthetic
accounts to origination scores. Modules accept the canonical portfolio contracts
and run their integrity checks before calculating metrics.

## Reproduce

From the repository root:

```console
uv sync --locked
uv run --locked python scripts/portfolio.py --accounts data/raw/phase3-demo/accounts.csv --history data/raw/phase3-demo/history.csv --source-manifest data/raw/phase3-demo/manifest.json --config configs/portfolio.yaml --output-dir artifacts/phase6-portfolio-001
```

If the ignored Phase 3 demo is absent, generate it first with
`uv run --locked credit-risk-lab generate-portfolio --config configs/synthetic_portfolio.yaml --output-dir data/raw/phase3-demo`.
Use a new analytics output directory for repetitions. `--source-manifest` checks
input hashes and provenance where supplied; the canonical source CSVs are always
validated. `--as-of YYYY-MM-DD` controls information availability, defaults to
the latest source snapshot, and rejects time components/timezones. Future rows
are excluded from calculations; the supplied source is integrity-checked in full.

The CLI command is also available as `credit-risk-lab analyze-portfolio`.
Runtime needs pandas, NumPy and Matplotlib. DuckDB is a development-only dependency
for verifying the SQL companions; no Spark or Databricks runtime is required.

## Vintage metrics

`vintage_table(accounts, history, config, as_of)` creates origination month x MOB
cells, with booking month = 0. Account master entries define the original cohort
size, including booked accounts with no available history. Future bookings after
the cutoff are excluded. `max_months_on_book` defaults to 35.

| Field | Definition |
| --- | --- |
| cohort_accounts | Original booked accounts in the cohort |
| observed_accounts | Available snapshots in that cohort/MOB cell |
| snapshot_coverage | Observed snapshots / original cohort, for matured cells |
| bad_rate | Recorded default OR DPD >= bad threshold (90 default), divided by observed accounts |
| delinquency_rate | Recorded default OR DPD >= delinquency threshold (30 default), divided by observed accounts |
| recorded_default_rate | Snapshot recorded default flags / observed accounts |
| balance_exposure | Sum of observed closing balances in arbitrary source currency units |
| credit_limit_exposure | Sum of observed snapshot limits |
| complete_history_accounts | Accounts with every snapshot from MOB 0 through this MOB |
| cumulative_observed_default_count | Distinct accounts with an observed recorded default at or before this MOB |
| cumulative_default_rate | Cumulative recorded defaults / original cohort, only if every account has complete history through the MOB |
| cumulative_default_lower_bound | Known observed defaults / original cohort; lower bound under incomplete history, not an estimated rate |

Bad/delinquency rates overlap and include recorded defaults even if the source's
recorded default criterion differs from DPD. The absorbing recorded flag defines
cumulative default. An analytic 90+ DPD event can cure and is not automatically
a recorded default. This differs from Phase 3's forward 12-month analytic target.

Snapshot rates use observed-row denominators; missing accounts are explicit in
coverage. Zero observed accounts give unknown rates and zero observed exposure,
which does not mean the full cohort has zero exposure. Immature cells have missing
counts/exposure/rates, so their triangles do not appear as good performance.
A gap or left truncation makes the official cumulative rate unavailable for that
and later MOBs. The original denominator is never reduced to survivors. This
conservative rule avoids claiming a complete cumulative default curve from partial
histories; it is not a survival estimator or an informative-censoring correction.

Cohorts smaller than `minimum_cohort_size` (20 default) are flagged, not dropped.
Plots show them; analysts must interpret that support flag when using curves.
Within a fully observed cohort cumulative recorded-default rates cannot decrease.
Comparing MOB 35 curves across the demo would compare only its earliest cohort;
the checkpoint summary explicitly reports which cohorts/accounts are complete.

## Roll-rate metrics

`roll_rates(history, as_of, accounts)` uses only observations of the same account
at consecutive calendar month-ends, both within the cutoff. It never bridges a
gap. A terminal observation whose next month is after the cutoff is not yet due;
one whose next month is at/before the cutoff but absent is missing follow-up.
Neither is inferred to be a cure, default, closure, or continued good performance.

Order: CURRENT, DPD_1_29, DPD_30_59, DPD_60_89, DPD_90_PLUS, DEFAULT.
Pooled and per-origin-month matrices retain all six states. Account-weighted
cells divide pair counts by the origin-state count. Balance-weighted cells use
**origin closing balance**, not destination balance, limits or expected loss.
Unsupported count rows and zero-total-balance rows have unknown probabilities,
not invented identity rows. Supported rows sum to one.

| Metric | Meaning |
| --- | --- |
| roll_forward | Any higher ordinal state, including entry to DEFAULT; deterioration |
| roll_back | Any lower ordinal state, including partial improvement |
| cure | Nondefault delinquent origin returning fully to CURRENT; subset of roll-back |
| new_default | Nondefault origin entering recorded DEFAULT |
| stay | Same source/destination state, including default persistence |

Per-state summaries divide each metric by that state's observed pairs, with both
count and balance weights. Monthly forward/default summaries divide by observed
nondefault-origin pairs; monthly roll-back/cure summaries divide by delinquent
nondefault-origin pairs. The overall new-default rate uses nondefault pairs too.
Default persistence is therefore not counted as a newly defaulting account.

Coverage diagnostics count all origins whose next month is due by cutoff,
consecutive pairs, missing next months, not-yet-due origins and gap links excluded.
Pair coverage is not guaranteed to be representative: missing follow-up can bias
observed transitions. Pooled matrices combine ages and calendar periods; they
are descriptive migrations, not constant causal probabilities or forecasts.

## Exports, SQL and evidence

Outputs include `vintages.csv`, ignored account-level `transition_pairs.csv`,
pooled transition count/probability/balance CSVs, state and monthly summaries,
long-format `monthly_transition_matrices.csv`, two PNG figures and `analytics.json`.
The JSON records definitions, config, cutoff, hashes, package/library versions,
generator provenance, checkpoint denominators, coverage and limitations. Missing
rates serialize as JSON null. The manifest hashes every generated artifact
except itself. Account-level source and outputs remain ignored local files.

`sql/vintage_analysis.sql` and `sql/roll_rates.sql` provide equivalent core
aggregates against registered tables. Supply validated `portfolio_accounts` and
`portfolio_history`, one-row `analytics_parameters`, a complete `mob_calendar`
integer grid 0..max MOB, and `delinquency_states(state, ordinal)` in canonical
order. The queries do not replace Python source validation. Date/window patterns
are intended to be straightforward to adapt to Spark; equivalence is executed
and tested in DuckDB. Databricks runtime compatibility is not yet certified.

Tests use small hand-calculated fixtures and compare SQL/Python results for full,
gapped, left-truncated and immature histories. They verify exposure/count
conservation, rates summing to one, absent-state/zero-exposure handling, cure
versus partial improvement, terminal follow-up, default absorption, cutoff
isolation, missing-denominator handling, deterministic exports and no overwrite.

Measured synthetic evidence is in [the Phase 6 report](../reports/phase6_portfolio_report.md).
The demo has no closures, recoveries or macroeconomic shocks. Its high default
prevalence is generated by illustrative mechanics, not observed credit portfolio
performance. Phase 7 will add expected loss and further portfolio analytics.

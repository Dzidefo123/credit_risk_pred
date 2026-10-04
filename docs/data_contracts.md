# Phase 3: data contracts and target definitions

## Two distinct data problems

The origination adapter preserves the inherited Give Me Some Credit label and
population. It does not manufacture an observation date or next-12-month target.
The synthetic portfolio is an independent demonstration of monthly account risk,
not an expansion of the Kaggle dataset or evidence of real account performance.
The two data sources must not be joined as though they described the same people.

### Origination input

`load_origination_csv` accepts the ten known numeric predictors, an optional
source_row_id, and the observed binary SeriousDlqin2yrs label (required by default).
The two hyphenated delinquency names and the inherited CSV index are normalized.
An index is provenance, not a customer identifier or candidate feature. The
ORIGINATION_FEATURES constant defines the predictor allowlist; labels, IDs and
unexpected future-outcome columns are excluded from it.

Validation reports missingness, exact duplicate predictor/label records and
suspicious age/utilization/delinquency values. It rejects negative, infinite,
non-numeric and fractional count fields, malformed labels and ambiguous aliases.
It does not impute, drop duplicates, filter outliers or use labels to remove rows.
Age zero, utilization above one, and sentinel-like counts are reported for review,
not automatically discarded. Fit preprocessing only on a later training partition.
Duplicates require split-aware handling to avoid contaminating test sets.

### Canonical portfolio tables

`accounts` has one row per account. Required fields:

| Fields | Meaning and validation |
| --- | --- |
| account_id | Nonempty string, unique |
| origination_date | Timezone-naive date with no time component |
| age, monthly_income | Nonnegative values; age integral |
| origination_limit | Positive limit at booking |
| is_synthetic | Explicit boolean provenance |

`history` has one row per account and month-end. Required fields:

| Fields | Meaning and validation |
| --- | --- |
| account_id, origination_date, is_synthetic | Match account master; provenance and booking date constant |
| observation_date | Timezone-naive month-end; unique account-month |
| months_on_book | Integral calendar-month difference from origination; booking month is 0 |
| dpd, state, default_flag | Nonnegative whole DPD; mapped band unless recorded default is flagged |
| opening_balance, draws, interest | Nonnegative monetary values for the monthly reconciliation |
| payment, scheduled_payment, write_off | Actual payment, amount due, and write-off |
| balance, credit_limit, utilization | Nonnegative balance, positive limit; utilization equals balance / limit |

Recorded DEFAULT is absorbing. The six states are CURRENT (0 DPD), DPD_1_29,
DPD_30_59, DPD_60_89, DPD_90_PLUS, and DEFAULT. A recorded default flag can
represent a source default criterion other than DPD; it therefore overrides the
DPD band. The synthetic generator assigns DEFAULT DPD between 120 and 180.
An analytical 90-DPD target is deliberately distinct from recorded DEFAULT.

Balance reconciliation is opening + draws + interest - payment - write_off;
tolerance is 0.011 currency units for cent rounding. Opening balances must equal
prior closing balances for consecutive months. Utilization can exceed 1 after
interest accrual or limit reductions; it is not clipped. Gaps are permitted in
real histories and explicitly handled by the target builder, not filled as good
performance. Adapters parse dates and validate without silently changing outcomes.
A real source can replace the simulation by supplying these tables, including
is_synthetic=False, and mapping its default/delinquency semantics explicitly.

## Synthetic generator and limitations

`SyntheticPortfolioConfig` controls account count, observation calendar, booking
window, seed, start month, annual rate and the six-state transition matrix.
Rows must be probability distributions, with absorbing DEFAULT. The generator
uses NumPy default_rng with a local seed; it does not change global RNG state.

Fixed illustrative assumptions in the generator: incomes are lognormal with
median 3,500 and log standard deviation 0.5; booking limits are 0.5-2 times income,
clipped to 500-25,000; ages range 21-75; initial balances use 10-80% of limit.
An internal Beta(2,8) latent risk multiplies deterioration probabilities by
1 + 3 * risk before renormalization. The latent value is not exported as a feature.
The base transition matrix is therefore not the unconditional observed matrix.

For later months, interest is charged on opening balances outside DEFAULT.
Draws occur in CURRENT and DPD_1_29; payments are a state-dependent fraction of
scheduled due, capped by available balance. A transition into 30+ delinquency
from a lower band cuts the limit 10%, with a 500 floor. First recorded default
writes off 20% of opening balance; subsequent default states stop interest,
draws and payments. Month 0 is the booking snapshot without cash movements.
These mechanics demonstrate data relationships rather than contractual billing,
collections or empirically fitted portfolio dynamics. There are no closures,
recoveries, macroeconomic cycles or currency conversion. Units are arbitrary
currency units; results are educational and not regulatory ECL calculations.

The default run books 500 synthetic accounts across 2022's 12 months and observes
month-ends through 2024-12-31. The high default-state prevalence is a simulation
artifact chosen to exercise analytical states, not a plausible portfolio forecast.
CSV outputs and manifests remain ignored. Manifests record configuration,
package version, counts and SHA-256 hashes. Exports refuse existing paths.
Determinism is verified with the locked dependency versions; floating-point or
random-library changes may alter outputs with a future lockfile.

## Forward target: first default within the next 12 months

`TargetConfig` defaults: horizon_months=12, default_dpd_threshold=90 and
indeterminate_dpd_threshold=30. The source flag or DPD threshold qualifies a
default event. This target is for the longitudinal contract only.

For a month-end observation t, the performance interval is (t, t+12 month-ends].
An event at t is pre-existing; one exactly at the upper boundary is included.
`as_of` excludes all observations after the information cutoff when labeling.
No unobserved early default is inferred between snapshots.

Statuses are evaluated in this order:

1. preexisting_default: any qualifying event at or before t, including cured DPD
   after a prior 90+ event. Label is missing; this is a first-default risk set.
2. history_incomplete: missing snapshots between booking and t, including
   left-truncated accounts. Eligibility cannot be established; label is missing.
3. bad: an observed future default inside the window. Label 1 can be established
   before full maturity, including a window with a missing future month.
4. censored: no observed future default and fewer than all horizon month-ends.
   Missing follow-up, disappearance, and calendar-end censoring never become good.
5. indeterminate: complete window without default but at least one month at or
   above the configurable intermediate DPD threshold. Label is missing.
6. good: complete window with no default or indeterminate event. Label 0.

Set indeterminate_dpd_threshold=null for a pure default/nondefault target among
complete windows. Excluding intermediate delinquency changes the estimation
population; report this selection and do not call the resulting score an
unconditional population PD without further justification.

Targets are a separate account-date keyed table with nullable integer label,
status, performance_end, observed_months and first_default_date (the earliest
observed qualifying event in the forward interval, including diagnostic events
on excluded rows). These future-looking fields must never be predictors. Build
features using observations available at t only. Repeated account snapshots have
overlapping performance windows: later modeling needs account-aware splits and
time-window separation/embargo, not random independent row splitting.

Observed bads can be labeled early while censored nonbads cannot. Evaluation of
PD therefore needs a common mature cohort or an explicit survival/censoring
method; naive pooling of early bads and mature goods introduces selection bias.
This phase implements transparent labels, not a solution to informative censoring
or approval-selection bias. As-of governs performance availability; source-history
integrity is validated on the supplied history before labeling.

## Reproduce Phase 3

From the repository root after `uv sync --locked`:

```console
uv run --locked credit-risk-lab check-config --config-dir configs
uv run --locked credit-risk-lab validate-origination --csv cs-training.csv
uv run --locked credit-risk-lab generate-portfolio --config configs/synthetic_portfolio.yaml --output-dir data/raw/phase3-demo
uv run --locked credit-risk-lab build-targets --accounts data/raw/phase3-demo/accounts.csv --history data/raw/phase3-demo/history.csv --config configs/target.yaml --output data/processed/phase3-demo/targets.csv
uv run --locked pytest
```

Use a new output directory on subsequent runs; overwriting is refused. The
origination command needs an authorized local CSV and is not required for the
synthetic demo. A fresh clone can run all tests and the synthetic demo without
Kaggle data, V1 model artifacts, or the original environment. The ignored local
CSV is unmodified. Target manifests include input hashes, config, cutoff, version,
status counts and output hash. No data generation occurs merely by importing a
module. Vintage, roll-rate, behavioral feature and model implementations follow
in later phases.

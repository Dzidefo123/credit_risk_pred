# Expected loss and portfolio scenarios

Phase 7 adds model-derived longitudinal PDs, modular loss arithmetic and portfolio
summaries. This is an educational analytical framework, not an IFRS 9/ECL
regulatory implementation or validated provisioning method.

## Reproduce

```console
uv sync --locked
uv run --locked python scripts/expected_loss.py --accounts data/raw/phase3-demo/accounts.csv --history data/raw/phase3-demo/history.csv --source-manifest data/raw/phase3-demo/manifest.json --config configs/expected_loss.yaml --output-dir artifacts/phase7-loss-002
```

Generate the ignored demo first if absent, using the Phase 3 command in
[portfolio_analytics.md](portfolio_analytics.md). The CLI is also available as
`credit-risk-lab expected-loss`. Use a new output directory for each run;
overwriting previous experiments is rejected. No source/model is loaded on import.

`--as-of` defaults to the source's latest observation. The snapshot is the exact
latest month-end at or before the cutoff, not each account's latest available row.
Account master entries booked by that month-end define the expected inventory.
Missing snapshot accounts are not closed, forward-filled or assigned zero risk.
Complete snapshot coverage is required by default. Explicit
`require_complete_snapshot: false` permits a report on observed inventory only,
with missing-account exclusions and coverage; it still cannot estimate the unknown
exposures. Real and synthetic accounts cannot be pooled.

## Where model PD comes from

The synthetic accounts do not contain the ten origination predictors needed by
the Phase 4/5 applicant models. Those models also target inherited two-year
delinquency rather than this portfolio's recorded default. No artificial join or
feature mapping is made between these unrelated sources, and no final holdout is
rescored or used to choose the portfolio model.

`TransitionPDModel` is a separate empirical Markov benchmark. It fits count-weighted
monthly transition probabilities from consecutive observed pairs at/before the
snapshot date, using the canonical six-state order from Phase 6. Nondefault-state
rows require at least `minimum_state_pairs` (20 default); sparse rows cause a clear
error rather than invented transition probabilities. DEFAULT absorption is a
structural requirement of the validated source contract, even without observed
DEFAULT-to-DEFAULT pairs.

For a horizon h (12 months default), the forecast is the DEFAULT column of P^h.
Because default is absorbing, this is the probability of recorded default by the
horizon. A defaulted account has probability 1 but is already defaulted inventory,
not a new-default forecast. The target is recorded DEFAULT, not Phase 3's analytic
90-DPD target or the inherited origination target. The model JSON exports the
matrix, state support, horizon PDs, cutoff, coverage and provenance.

This benchmark assumes time homogeneity and sufficient risk information in the
current state. It is uncalibrated and has no prospective accuracy validation.
Fitting source migrations does not establish that P^h forecasts real account
performance. Missing follow-up, repeated accounts, changes over time and account
heterogeneity can invalidate that assumption. No confidence intervals or model
promotion are claimed. A future validated behavioral model can provide keyed PD
scores to the loss engine without changing the loss arithmetic.

## Loss arithmetic and assumptions

`calculate_expected_loss(probability, lgd, ead)` supports numeric scalars/vectors
and rejects missing, nonfinite, negative or invalid probability/severity inputs.
`exposure_at_default` uses:

```text
undrawn = max(credit_limit - balance, 0)
EAD = balance + credit_conversion_factor * undrawn
forward EL = model PD * LGD * EAD
```

Over-limit accounts retain their full balance without negative undrawn exposure.
Zero balances can still have EAD if unused lines may be drawn. CCF 0 uses drawn
balance only; CCF 1 adds the whole positive unused line. CCF and LGD are assumptions,
not estimated recovery or future utilization models. All amounts use source
currency units; currencies must be reconciled upstream before pooling.

`loss_table(snapshot, pd_scores, config)` joins PD scores one-to-one on
account_id/observation_date. Missing, duplicate, extra or invalid scores cause an
error, rather than positional scoring or zero fill. The score population must
match the observed exposure population, and recorded-default scores must be 1.

Nondefault inventory (including DPD_90_PLUS) receives forward expected loss.
For already-defaulted inventory, undrawn lines are assumed unavailable and EAD
is observed balance. Its separate residual-loss assumption is LGD * balance;
forward new-default EL is zero. `combined_loss_proxy` adds these two components
for arithmetic comparison, but is not a regulatory ECL/provision. Historical
write-offs are excluded from this residual-stock measure. There are no expected
recovery timings, cash-flow discounts or realized-loss validation.

## Scenarios and portfolio summaries

| Scenario | PD odds multiplier | LGD | CCF |
| --- | ---: | ---: | ---: |
| base | 1.0 | 0.45 | 0.50 |
| adverse | 1.5 | 0.60 | 0.75 |
| severe | 2.5 | 0.75 | 1.00 |

These are illustrative sensitivities, not forecasts or calibrated stress tests.
PD scenarios transform p as `p / (p + (1-p)/multiplier)`, preserving endpoints
without clipping invalid probabilities. The base scenario must preserve model
PD. Scenario names are unique; LGD/CCF stay in [0,1]. No scenario probabilities
are assigned and scenario losses are not averaged into one figure. These shocks
are deterministic assumptions applied after the benchmark fit, not stressed
transition matrices or macroeconomic models.

Scenario summaries report exposure, forward loss, separate defaulted-stock loss,
expected new defaults among nondefault accounts, mean PD, EAD-weighted PD and
forward EL/EAD. These quantities use the configured horizon and must not be
annualized from a different horizon without additional modeling. Account EAD HHI
and top-n EAD/loss shares describe concentration; they do not measure dependence,
connected-borrower concentration or unexpected loss. Zero-exposure or zero-loss
denominators give unavailable rates, not arbitrary zero ratios.

State and origination-month summaries reconcile to portfolio totals within each
dimension; never sum overlapping dimensions together. The engine validates finite
aggregate totals and does not silently round accounts before aggregation.

## Outputs, SQL and validation

Ignored local artifacts include snapshot/exclusion tables, model PD scores,
account-level losses, scenario/segment summaries and portable `pd_model.json`.
`expected_loss.json` records definitions, config, versions, coverage and
source/model-code/artifact hashes. Figures and reports contain aggregate data.
The manifest does not include its own hash.

`sql/portfolio_snapshot.sql` preserves booked accounts with unknown snapshot
exposure; `sql/delinquency_analysis.sql` retains an UNOBSERVED category and uses
observed-row denominators. SQL/Python coverage and totals are tested in DuckDB.
Supply the validated tables and one-row analytics_parameters (as_of and DPD
thresholds). SQL is a companion to source validation, not a replacement for it.
Spark/Databricks execution remains unverified.

Tests cover hand-calculated multi-period PDs, no future transitions in fitting,
structural default, sparse-state refusal, scalar/vector loss boundaries, score
alignment, over-limit/zero exposure, defaulted stock separation, scenario ordering,
segment/portfolio conservation, missing snapshot coverage and reproducible exports.
Measured results are in [the Phase 7 report](../reports/phase7_expected_loss_report.md).

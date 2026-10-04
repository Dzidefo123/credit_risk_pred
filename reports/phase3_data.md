# Phase 3: data contracts, synthetic portfolio and targets

Date: 2026-10-04 (America/Sao_Paulo). Branch: feature/risk-modeling-lab.
Package version: 0.3.0. Tested interpreter: Python 3.11.4.

## Implemented

Origination CSV normalization and non-destructive quality diagnostics; canonical
account/history contracts and adapters; configurable seeded synthetic generation;
forward-window first-default targets with information cutoff, indeterminate
performance, incomplete history and censoring; data preparation CLI commands.
Added NumPy and pandas to locked dependencies. No predictive model was trained.
The V1 notebook, README, applications, templates, CSV and pickle are unchanged.

See docs/data_contracts.md for field contracts, definitions, assumptions,
replacement of synthetic data by real histories, and leakage/censoring limitations.

## Actual local evidence

The inherited CSV adapter processed 150,000 rows, retaining all rows and missing
values. MonthlyIncome has 29,731 missing; NumberOfDependents has 3,924.
It reported 609 duplicate predictor/label records, 1 age-zero-or-over-110 record,
3,321 utilization-over-one records and 269 records with delinquency counts >=90.
These are diagnostic flags, not automatic exclusions or model results.

The seed-42 synthetic demo generated 500 accounts and 15,142 monthly snapshots.
Observation range: 2022-01-31 through 2024-12-31. State snapshot counts:
CURRENT 8,808; DPD_1_29 946; DPD_30_59 556; DPD_60_89 349;
DPD_90_PLUS 297; DEFAULT 4,186. The synthetic default prevalence is illustrative
and is not calibrated to a real portfolio.

The configured 12-month/90-DPD target produced 15,142 rows:
3,511 good; 2,874 bad; 2,484 censored; 1,414 indeterminate;
4,859 pre-existing-default observations. All excluded/unmatured nonbad labels
remain missing, not zero. These counts describe repeated, overlapping account
observations, not distinct account bad rates. Full config and output hashes are
in phase3_demo_summary.json. Generated CSVs and local manifests remain ignored.

## Validation gate

The full suite currently has 52 passing tests. Data tests cover aliases,
missingness retention, nonmutating validation, finite numeric fields, source
leakage columns, deterministic seeds, cures, deterioration, absorbing default,
monthly continuity, accounting reconciliation, corrupt contracts, CSV round trips,
provenance and overwrite protection. Target tests cover horizon boundaries,
pre-existing default after cure, recorded-default criteria, partial follow-up,
left truncation, missing months, indeterminate settings and as-of availability.
CLI tests exercise generation and target export, including overwrite rejection.

Before commit: complete Ruff lint/format checks, Phase 1 preservation checks,
locked sync, packaging/import verification and diff inspection. Python 3.12-3.14
remain declared but unverified. No real-world performance or compliance claim
is made. Phase 4 will implement the logistic baseline and tree challenger.

Completed final checks: Ruff lint and format, Phase 1 preservation, locked sync,
wheel and source-distribution builds, artifact exclusion and demo-hash reconciliation.
The isolated wheel imported all 15 submodules, generated a small portfolio and
built its targets successfully. Staged diff and generated-data ignore rules passed.

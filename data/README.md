# Data provenance and local storage

The inherited `cs-training.csv` remains at the repository root locally, but is
no longer versioned. The inherited API still refers to that location. Its
SHA-256 and audited schema are recorded in `reports/phase1_inventory.json`.
The original Git baseline also retains it; this cleanup does not rewrite history.

The V1 README attributes the data to Kaggle Give Me Some Credit:
https://www.kaggle.com/competitions/GiveMeSomeCredit/overview
Obtain future copies from the authorized source, review its terms, and document
any transformation. Do not treat the inherited file as a verified pristine copy:
its 150,000 rows and renamed delinquency columns differ from the notebook's
stored 125,113-row initial sample. The CSV uses `NA` for missing values.

`SeriousDlqin2yrs` is the inherited two-year delinquency label, not evidence of a
newly constructed 12-month default target. Source target semantics, observation
population, and sampling need independent verification before modeling.
The inherited file contains no account identifier or dated monthly history.
It cannot support genuine vintage, behavioral, or out-of-time analysis.

Later phases will separate origination data from explicitly synthetic account
histories using replaceable data contracts. Proposed ignored locations are
`data/raw/`, `data/interim/`, and `data/processed/`. No synthetic portfolio or
new modeling is implemented in Phase 1. A fresh clone will not include local
raw data or model artifacts; the legacy API is historical evidence and is not
a reproducible serving release.

## Phase 3 data preparation

The origination loader preserves the inherited label and all valid input rows;
it normalizes field names and reports quality without learned preprocessing.
The separate synthetic generator emits accounts.csv, history.csv and a manifest
under an ignored run directory. Account and history tables explicitly contain
is_synthetic=True. Real replacements must conform to the documented contracts
and explicitly set false. Forward targets are stored separately from features.

Use the commands and definitions in ../docs/data_contracts.md. The audited
Phase 3 demo is in data/raw/phase3-demo locally, with forward targets in
data/processed/phase3-demo/targets.csv. Generated CSVs are not committed.
A small measured run summary is recorded in reports/phase3_demo_summary.json.

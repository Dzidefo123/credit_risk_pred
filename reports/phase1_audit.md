# Phase 1: repository audit and cleanup

Audit date: 2026-10-04 (America/Sao_Paulo).
Repository: https://github.com/Dzidefo123/credit_risk_pred
Baseline: `811a2fe` (full identity in `phase1_inventory.json`).
Working branch: `feature/risk-modeling-lab`.
Scope: audit and cleanup only. No training, model redesign, or API refactor.

## Inspection method and inventory

Reviewed the complete tracked-path and blob-size inventory, README, all 73
notebook cells and their stored outputs, both serving modules, both templates,
CSV schema/counts/missing tokens, environment metadata, and static pickle strings.
There are no repository AGENTS.md instructions, tests, packaging manifest, CI,
Dockerfile, or dependency lock in the baseline. Bundled third-party libraries
were inventoried by package metadata rather than reviewed as authored source.
No vendored interpreter, application, or serialized model was executed.

The baseline has 11,497 tracked files. Of these, 11,488 are in `venv/`, totaling
422,053,480 bytes; one is `__pycache__/app.cpython-38.pyc` (1,844 bytes).
Authored evidence is README.md (3,883 bytes), app.py (1,936), main.py (1,558),
credit_risk(1).ipynb (240,248), and two templates (1,944 combined).
The CSV is 7,414,964 bytes and combined_model.pkl is 339,656 bytes.
Detailed counts, hashes, and discovered package versions are in the inventory.

## What V1 actually does

The Colab notebook loads `/content/cs-training.csv`, drops the index, renames
past-due columns, deduplicates, fills missing values, filters records, trains
XGBoost, and evaluates it first on the same data. It then makes an 80/20 random
split (`random_state=42`), scales features for a 128/64-unit neural network,
trains for 10 epochs, refits XGBoost on the training split, and averages the two
models' test probabilities at a 0.5 decision threshold.

Crucially, cell 71 runs `joblib.dump(model, 'combined_model.pkl')`: it saves only
the XGBoost variable, not the probability ensemble, neural network, or scaler.
Static pickle inspection begins with `xgboost.sklearn.XGBClassifier`; parsing
stops at a joblib binary array payload. This corroborates the model type but
is not a complete artifact reconstruction or performance verification.

`main.py` loads this artifact and the entire unused CSV at import time, accepts
ten numeric JSON fields at `/predict/`, and returns class-1 probability with
input fields. `app.py` duplicates the schema and model loading, serves an HTML
form, and renders a binary prediction. Neither implements health checks,
versioned PD output, calibration, grades, underwriting rules, or limits.

## Dataset and provenance findings

The local CSV has 150,000 records, 12 columns including a blank-name index,
10,026 label-1 records, and 139,974 label-0 records (6.684% label-1 rate).
Missing values use `NA`: MonthlyIncome has 29,731 (19.821%); NumberOfDependents
has 3,924 (2.616%). It is a cross-sectional dataset without account or monthly
dates. Its already-renamed delinquency columns and row count do not match the
notebook's initial stored shape `(125113, 11)` after index removal. We cannot
claim the committed artifact was trained on this exact committed dataset.
Source verification and an explicit target contract are required later.

The target name is SeriousDlqin2yrs. A configurable next-12-month target must
not be retroactively asserted for these records. Observation date, performance
window, bad/good/indeterminate rules, censoring, default definition, and leakage
controls remain undocumented. Synthetic longitudinal data must be separate
and clearly identified; none exists in V1.

## Technical debt, ordered by risk

1. **Validation contamination:** notebook cells 50-55 fit and evaluate XGBoost
   on the same data. Stored accuracy 0.9481 and class-1 recall 0.29 are historical
   in-sample outputs, not independently validated results. Later stored NN and
   ensemble test accuracy are both 0.9350; they have not been rerun or endorsed.
   No ROC-AUC, PR-AUC, KS, calibration, uncertainty, or segment validation is
   implemented. The initial majority-class comparison is already about 93%.
2. **Outcome-dependent selection:** cells 42-44 use SeriousDlqin2yrs in a rule
   dropping high-DebtRatio rows where the label equals MonthlyIncome. This
   alters the modeled population using outcomes and cannot be applied at scoring.
   Income imputation and filtering happen before train/test splitting. Fit
   learned preprocessing on training data only; justify exclusions independently
   of the outcome. Deduplication does not resolve selection or acceptance bias.
3. **Serving/training mismatch:** the saved artifact is XGBoost alone, despite
   the README's ensemble description. No fitted preprocessing, feature contract,
   artifact lineage, or version accompanies it. README also says median income
   imputation for both groups, but cell 14 imputes zero for the missing-dependents
   group. Raw requests bypass notebook imputation and cleaning.
4. **Broken form contract:** home.html submits URL-encoded form fields while
   app.py's Pydantic body expects JSON. It has no form-to-JSON adapter. This is
   a static contract finding; the legacy server was not launched.
5. **Reproducibility:** hardcoded Colab paths, mutable notebook state, no lock,
   no TensorFlow/NN seed, deprecated DataFrame.append, and slice assignment.
   The bundled Python environment targets 3.8.2 and contains pandas 2.0.3,
   whose DataFrame.append removal conflicts with cell 22. Its metadata lacks
   TensorFlow, seaborn, and matplotlib despite notebook imports. Captured
   versions are evidence, not a reliable install specification.
6. **Engineering:** duplicate entry points, import-time relative paths and
   unused CSV reads, unused imports, unconstrained numeric schemas, absent
   tests/CI/logging/configuration/packaging. Model deserialization is not needed
   for this audit and must not be used merely to inspect an unknown artifact.
7. **Risk science/governance:** no logistic benchmark, calibration holdout,
   out-of-time evidence, behavioral data, vintage/roll rates, EL assumptions,
   selection-bias experiment, decision policy, monitoring, or governance reports.
   No fairness/representativeness assessment or license is present. Do not infer
   production readiness, policy suitability, or regulatory compliance.

## Preserve, untrack, and replace

Preserve byte-for-byte in this phase: README.md, the original notebook, app.py,
main.py, and templates/home.html and result.html. They remain tracked in their
original paths, so readers can inspect V1 without a disruptive relocation.

Untrack `venv/`, `__pycache__/`, `cs-training.csv`, and `combined_model.pkl` using
`git rm --cached`. Retain every local copy; add ignore rules. Record data/model
hashes before cleanup. Original files remain recoverable at the baseline commit.
No Git history rewrite, remote push, environment execution, or destructive file
removal is performed. Historical Git objects still contain these blobs, so the
initial clone size is not reduced by an index-only cleanup.

Later replace notebook-centric training, duplicate serving modules, and opaque
artifact persistence with small modules, explicit contracts, and versioned
pipelines. Keep V1 as historical evidence rather than treating it as a champion.
Phase 14 will rewrite the business-facing README; it remains unchanged here.

## Proposed migration and phase gates

| Phase | Scope and verification gate |
| --- | --- |
| 2 | src/credit_risk separation, packaging, config; clean-install/import checks |
| 3 | loaders, quality and target contracts; deterministic synthetic histories, censoring checks |
| 4 | logistic baseline and practical boosted-tree challenger; leakage-safe splits |
| 5 | independent calibration selection and final validation; ranking and PD metrics |
| 6 | vintage denominators and consecutive-month roll rates; absorbing default tests |
| 7 | modular PD/LGD/EAD loss and scenarios; units/horizon/exposure checks |
| 8 | configurable grades, decisions, affordability and limits; policy boundary checks |
| 9 | selection-bias experiment; propensity overlap and IPW assumptions/limitations |
| 10 | fixed-reference feature/PD/missingness drift; configurable thresholds |
| 11 | versioned score/decision API and optional local MLflow; schema/lineage checks |
| 12 | meaningful pytest, Ruff, CI and Docker; reproducible build/start checks |
| 13 | model card, validation and monitoring reports with measured evidence |
| 14 | business-first README, Mermaid and interview story; complete final audit |

Origination modeling and behavioral analytics will have separate data and
feature contracts. Static Give Me Some Credit supports a cross-sectional
origination illustration, not real longitudinal portfolio validation. Synthetic
behavioral results will never be reported as realized real-world performance.
No later phase is implemented or committed as part of this task.

## Phase 1 verification

Run `python scripts/check_phase1.py`. It checks the cleanup branch, preserved
source and local artifact hashes, removal from the Git index, ignore behavior,
notebook JSON, and Python syntax without running model code or loading pickle.
This is a phase-specific preservation check, not a model-validation suite.
Revisit or retire it when a later phase intentionally migrates V1 paths.

Before commit, inspect `git diff --cached --stat`, `git diff --cached --check`,
and the added audit documentation. The committed environment is not reused.
No pre-existing test suite exists. Runtime compatibility and V1 model results
are explicitly unverified; no new performance claims are made.

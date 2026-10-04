# Credit Risk Modeling & Decisioning Lab: architecture

The lab separates the applicant decision (origination PD, calibration, and
underwriting) from account evolution (behavioral risk and portfolio analytics).
Phase 2 established packaging and configuration. Phase 3 adds data contracts,
synthetic histories and censored forward targets. Phase 4 adds logistic and
XGBoost candidates with development-only scoring. Phase 5 fits held-out calibration,
locks development-selected choices and evaluates the final holdout with uncertainty
and segment diagnostics. Phase 6 adds coverage-aware vintages and consecutive-month
account/balance roll rates. Phase 7 adds a longitudinal transition PD benchmark,
expected loss and configurable scenarios. Phase 8 adds configurable underwriting,
hypothetical limits and development policy comparisons. Phase 9 adds separate synthetic
selection-bias experiments with masked outcomes and propensity weighting.
Phase 10 adds frozen-reference feature, PD, score and missingness monitoring.
Phase 11 adds a research scoring/decision API and optional local MLflow tracking.
Phase 12 adds clean-checkout CI, release gates and an external-artifact Docker runtime.
Phase 13 consolidates model evidence, open findings and procedural governance gates.
Phase 14 provides the business-first README and final repository audit. The root notebook,
app.py, main.py, and templates remain unchanged V1 evidence; they are excluded from the distributable package.

## Package boundaries

| Package | Intended responsibility |
| --- | --- |
| data | Source adapters, data contracts, synthetic histories and targets |
| features | Train-only origination transformations; rolling behavioral layer remains deferred |
| models | PD models, calibration and interpretability |
| validation | Discrimination, calibration, backtesting and stability |
| portfolio | Vintages, monthly transitions, migration and expected loss |
| decisioning | Configurable risk grades, underwriting, limits and selection bias |
| monitoring | Feature, score, PD and missingness drift |
| utils | Typed config and structured logging |

The data package now implements adapters, validation, synthetic histories and
forward targets. Origination features, initial metrics and PD candidates are implemented in Phase 4.
Calibration and final-holdout validation are implemented in Phase 5.
The portfolio package implements vintage and roll-rate analytics in Phase 6,
then model-driven expected loss and exposure/concentration summaries in Phase 7.
Decisioning, monitoring, API and optional tracking now implement the responsibilities
documented below. The longitudinal PD benchmark uses current delinquency state,
not a trained rolling behavioral-feature model. Imports never load the inherited CSV or model. Configurations are
explicit external inputs, rather than embedded machine-specific paths.

## Reproducible development

Python 3.11 is the development baseline (`.python-version`). The package declares
Python >=3.11,<3.15; only the interpreter used in the phase checks is verified.
Install uv separately, then run these commands from the repository root:

```console
uv sync --locked
uv run --locked credit-risk-lab check-config --config-dir configs
uv run --locked pytest
uv run --locked ruff check src tests scripts api
uv run --locked ruff format --check src tests scripts api
uv build
```

The same commands work on Windows without make. Makefile provides convenience
targets for environments with GNU make. uv.lock records resolved dependencies;
`uv sync --locked` refuses to silently update it. The project uses uv copy mode because OneDrive rejects environment hardlinks.
A fresh .venv is separate from
the ignored V1 venv. Dependencies support the implemented algorithms. NumPy and pandas support the data layer; scikit-learn and XGBoost support the
candidates. SciPy supports sigmoid fitting and Matplotlib generates validation figures.
No TensorFlow or remote MLflow server is required.

## Configuration contracts

`development.yaml` controls seed, environment, log level and paths. Relative paths
resolve against the parent of the supplied configuration directory; absolute
paths remain absolute. Loading and validation do not create output directories.
`model.yaml` reserves independent training, calibration, and test partitions.
The final test set must remain untouched by model/calibration selection.
`decision_policy.yaml` holds illustrative PD thresholds, ordered grade bounds,
LGD and credit-limit assumptions. These are demonstration settings rather than
validated policy recommendations. Grade assignment and threshold inclusion
behavior are implemented in Phase 8; see credit_strategy.md.

Safe YAML loading rejects executable tags. Typed contracts reject unknown keys,
invalid probabilities, inconsistent partitions, unordered grades and reversed
limit bounds. JSON logging writes to stderr with UTC timestamps; CLI result JSON
writes to stdout. Imports do not configure global logging or read configuration.

The wheel contains only credit_risk modules. YAML examples, tests, V1 code and
local artifacts are not wheel resources. Supply --config-dir when invoking the
configuration checker outside the checkout. See the [README](../README.md) for
the current walkthrough and [final audit](../reports/final_repository_audit.md)
for delivered scope, justified deviations and outstanding evidence.

Phase 3 commands, source contracts and target definitions are in
[data_contracts.md](data_contracts.md).

Phase 4 training discipline and commands are in [pd_modeling.md](pd_modeling.md).

Phase 5 calibration, holdout discipline and commands are in
[calibration_validation.md](calibration_validation.md).

Phase 6 vintage definitions, roll-rate denominators and SQL counterparts are in
[portfolio_analytics.md](portfolio_analytics.md).

Phase 7 model PD provenance, expected-loss arithmetic and scenario assumptions are in
[expected_loss.md](expected_loss.md).

Phase 8 underwriting, hypothetical limits and appetite comparisons are in
[credit_strategy.md](credit_strategy.md).

Phase 9 simulation assumptions, propensity weighting and identification limits are in
[reject_inference.md](reject_inference.md).

Phase 10 reference freezing, drift definitions and alert governance are in
[monitoring.md](monitoring.md).

Phase 11 serving contracts, lifecycle and optional experiment export are in
[api_tracking.md](api_tracking.md).

Phase 12 test coverage, CI lanes, release checks and Docker mounts are in
[testing_ci_docker.md](testing_ci_docker.md).

Phase 13 model inventory, intended use, approval status and change controls are in
[model_governance.md](model_governance.md) and the [model card](../reports/model_card.md).

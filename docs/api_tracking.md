# Scoring API and optional MLflow

The package now serves the frozen Phase 5 calibrated PD wrapper and the Phase 8
policy through FastAPI. The original V1 app.py/main.py/templates are preserved.
The authoritative application is `credit_risk.api.main:create_app`; root api/main.py
and api/schemas.py are checkout aliases. The wheel contains the packaged API.

Install the core environment and launch on loopback from the checkout:

```powershell
uv sync --locked
.venv\Scripts\python.exe -m uvicorn credit_risk.api.main:create_app --factory --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for the generated contract. Set
`CREDIT_RISK_SERVING_CONFIG` to another YAML path to choose a different explicit
configuration; the default is configs/serving.yaml. Paths inside that YAML resolve
relative to its directory, and the configured policy name must exist in
credit_strategy.yaml. API import and factory creation do not read CSVs or load
models; lifespan startup verifies source/artifact checksums, base source hashes,
frozen dependency versions and the calibration lock, then loads the selected
wrapper once. Prediction uses a per-service lock and does not fit/recalibrate.
The [FastAPI lifespan documentation](https://fastapi.tiangolo.com/advanced/events/)
describes the startup/shutdown pattern used here.

| Route | Result |
| --- | --- |
| GET /health | Readiness, package version, model version; HTTP 503 if initialization is unavailable |
| POST /score | Calibrated PD, grade, missing/suspicious inputs, target semantics, artifact identity |
| POST /decision | Score fields plus policy decision, reasons, hypothetical limit, assumed EAD/loss proxy |

Requests contain an application_id and a nested features object. IDs use
letters/digits/underscore/dot/hyphen and are limited to 100 characters. Feature
names are the ten canonical origination names; missing or omitted values become
null and use the frozen model's imputation only for prediction. Raw missing income
still blocks an automatic offer in the policy engine. Count fields require JSON
integers within the exact float64 integer range; age must contain whole years.
Other continuous fields accept JSON numbers. Numeric values must be finite
and nonnegative. Booleans, strings used as numbers, unknown fields, target labels,
model artifacts and source_row_id are rejected. Application IDs are caller
correlation IDs, not borrower identifiers.

An invented input example:

```json
{
  "application_id": "illustrative-001",
  "features": {
    "RevolvingUtilizationOfUnsecuredLines": 0.2,
    "age": 43,
    "NumberOfTime30_59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.2,
    "MonthlyIncome": 5000,
    "NumberOfOpenCreditLinesAndLoans": 3,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60_89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1
  }
}
```

The example returns PD 0.0162208012, grade G2, APPROVE and a hypothetical limit
5,400 source income units under baseline. Omitting income returns PD 0.0134437256
but MANUAL_REVIEW with zero offered limit. These are model outputs for invented
inputs, not observed repayment performance. PD retains the inherited two-year
delinquency interpretation; it is not independently established contractual
default. Responses explicitly mark research_only=true. The current benchmark
has not been promoted as a production champion. Model version uses candidate,
calibration method and artifact hash; full model and policy hashes are also
returned. Package version describes serving code, not a newly trained model.

Invalid input produces HTTP 422 with validation details that omit submitted
values. Reserved original final-test predictor profiles are also rejected before
prediction. Artifact initialization/model-output failures produce HTTP 503 and
never fall back to the inherited V1 model or emit a decision. The API does not
write request payloads to tracking. Its operational errors log exception types
without private model paths or applicant values. An unavailable readiness state
must be fixed through configuration/artifact diagnosis before using scoring.

## Optional local experiment tracking

Core training/validation/portfolio/API commands do not import or require MLflow.
Install the extra only when recording evidence:

```powershell
uv sync --locked --extra tracking
.venv\Scripts\python.exe -m credit_risk.cli track-experiment --kind origination --run-dir artifacts/phase4-origination-001 --config configs/tracking.yaml --run-name phase4-frozen-training
.venv\Scripts\python.exe -m credit_risk.cli track-experiment --kind validation --run-dir artifacts/phase5-validation-001 --config configs/tracking.yaml --run-name phase5-frozen-validation
```

The tracking extra uses mlflow-skinny plus SQLAlchemy/Alembic for the local SQLite
backend. No tracking server is required. TrackingConfig controls the local
SQLite file, local artifact root, experiment name and whether joblib models are
copied. Default paths are inside ignored artifacts/mlflow. An explicit
MlflowClient owns this SQLite URI; the exporter does not change the global fluent
tracking URI or active run. MLflow telemetry is disabled for this local export.
The [client API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)
supports the create_run/log_batch/log_artifact operations used here, and the
[local database tutorial](https://www.mlflow.org/docs/latest/ml/tracking/tutorials/local-database/)
explains SQLite tracking. A separate MLflow server/UI is optional and is not
launched by this exporter.

This integration records already-completed runs. It verifies all declared
artifact hashes before creating a tracking run and records source manifest and
package identity. It logs flattened configuration parameters, development
metrics, calibration choices/results and prior final-holdout aggregate metrics.
Final-holdout metrics are explicitly named historical_final_holdout: logging
existing JSON does not rescore test or revisit model selection. JSON aggregate
manifests, figures, selection metadata and optional joblib model artifacts are
copied. CSV row predictions/labels are verified but not copied. Models are logged
as existing joblib artifacts, not MLflow pyfunc deployments or registry promotions.

Repeating an export intentionally creates another run in the same experiment.
Existing experiment artifact location must match the configured local root.
Successful exports become FINISHED; a logging failure becomes FAILED and yields
an actionable error. Failed runs may retain partial metrics/artifacts, making
failure visible rather than rolling back history. Checksums detect modification;
only trusted locally generated model files are supported. Tracking never loads
joblib models or modifies source manifests.

Reproduce the verified API/model/SQLite integration without opening a listening
port, using actual ASGI requests and the verified model:

```powershell
.venv\Scripts\python.exe scripts/check_phase11.py --output-dir artifacts/phase11-integration-003
.venv\Scripts\python.exe -m pytest -q
```

Use a fresh output directory. The script creates new local tracking records,
checks recorded counts/statuses, and verifies that the original experiment and
final-test lock files are unchanged. Tracking integration tests skip when the
extra is absent; API and evidence-preparation tests still run.

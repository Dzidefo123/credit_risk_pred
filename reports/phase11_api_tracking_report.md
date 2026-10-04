# Phase 11: API and MLflow integration

FastAPI now serves GET /health, POST /score and POST /decision using the frozen
XGBoost/sigmoid wrapper and baseline Phase 8 policy. Startup performs the existing
source/model/calibration integrity checks once. No request fits a model, tunes a
threshold, or uses the inherited V1 pickle. Original V1 source remains intact.

The verified model identity is xgboost:sigmoid:17030fc4d0a5. For an invented
applicant, actual ASGI requests returned PD 1.6221%, grade G2, APPROVE and a
hypothetical limit of 5,400 source income units. Removing monthly income returned
PD 1.3444%, MANUAL_REVIEW and zero automatic offer. This confirms that predictive
imputation is separate from the raw affordability checks. These example results
are not observed portfolio performance or a lender policy recommendation.

The API returns model/policy hashes, calibration method, target semantics and
research-only status. Strict request contracts reject targets, unknown fields,
nonfinite or negative numbers, invalid count types and unsupported identifiers.
Validation messages omit submitted values. Missing artifacts or invalid model
outputs fail with 503; reserved original holdout profiles fail with 422 before
prediction. Health reports readiness rather than merely process liveness.

Optional MLflow tracking uses local SQLite metadata and local model/evidence
artifacts. The actual exports were read back and verified:

| Evidence | Run ID | Parameters | Metrics | Artifacts |
| --- | --- | ---: | ---: | ---: |
| Phase 4 origination | 124e9eec95a048c7b9507b652d5c09d2 | 23 | 36 | 3 |
| Phase 5 validation | e14692b5fb8c4b08b5c2ef6f85b2285f | 16 | 144 | 6 |

Both runs finished successfully in experiment 1. Origination exports include
both fitted candidate joblib artifacts; validation exports include the selected
calibrated wrappers, selection metadata and calibration/discrimination figures.
Calibration methods and all recorded development/final aggregate metrics are
available in the tracking records. Historical final metrics are exported from
the existing manifest, not recomputed. Row prediction/label CSVs are omitted.
Tracking telemetry is disabled, and no remote server or model registry is used.

A failed artifact write is tested to leave a FAILED run, making partial export
history explicit. Repeated export creates a new run; source manifests and model
files are never changed. Optional tracking absence gives a clear installation
instruction without creating a database. Core serving and analytics work without
the tracking extra. The frozen modeling dependency versions remain unchanged.

[Portable integration evidence](phase11_integration_summary.json) records actual
responses, tracking IDs/counts and hashes; raw local integration/store artifacts
remain ignored. [API and tracking instructions](../docs/api_tracking.md) include
startup, request schema, interpretation, export and reproduction commands.

Validation: 249 tests passed with the optional tracking extra installed, including
API lifespan/route contracts, endpoint boundaries, missing income, leakage fields,
reserved holdout protection, failure behavior, checksum/path checks and local
SQLite round trips. The actual ASGI/model/tracking integration also passed.
Two upstream deprecation warnings remain in Starlette's httpx compatibility path
and MLflow's SQLAlchemy loader code; neither prevented these tested operations.
Lint, formatting, configuration, V1 preservation, package build and core-only
installed-wheel smoke checks passed. No original final-test scoring, new model
training, model promotion or external publication occurred.

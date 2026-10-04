# Phase 12: tests, CI and Docker

Package version: 0.12.0. Work remains on feature/risk-modeling-lab in the dedicated
credit-risk-lab checkout. No day-trading project was changed.

Added complete-bundle serving integration tests using newly generated synthetic
fixtures. They verify frozen model loading, matching score/decision probabilities,
no fitting during serving, unchanged source evidence, reserved-profile rejection,
and source/model tampering rejection before deserialization. Release tests and
an archive gate reject datasets, serialized artifacts, caches, secrets and V1
serving code in package distributions. The source release has explicit included
paths; the wheel still contains only the credit_risk package and metadata.

The expanded Ruff gate covers checkout scripts and API aliases. Existing thin
scripts were formatted without changing their commands or model behavior. The
local historical-preservation script remains unchanged and outside Ruff.

Git attributes preserve CRLF only in the five base/calibration files whose raw
byte hashes were frozen on Windows. A real Git checkout-index export with
core.autocrlf=false reproduced all five byte hashes. No frozen model source,
calibration code, source data or experiment lock was changed.

GitHub Actions now defines seven Python/dependency/OS test lanes and a separate
Linux container job. Baseline Python 3.11 runs on Windows/Linux with core/tracking;
Linux also covers Python 3.12–3.14 with core dependencies. Every test lane checks
lint, formatting, configs, tests, release contents and a separately installed core
wheel. Actions use verified full release commit hashes and read-only repository
permissions. The workflow passed actionlint 1.7.12, downloaded from its official
release and verified against its published archive checksum. Hosted jobs have
not run: this phase was committed locally and not pushed.

Docker support comprises a multi-stage Python 3.11/uv build, a non-editable locked
core installation, libgomp, a non-root runtime, readiness health check and explicit
external research mounts. The build context allows only package build inputs;
original data/models/legacy code never enter it. Container instructions use
read-only evidence mounts, a read-only application filesystem and writable /tmp.
A smoke script checks actual UID, read-only root filesystem and 503 responses
without artifacts, then removes only its own uniquely named container. CI invokes
that script after a real image build and checks the core package/configuration.

## Verification recorded locally

| Check | Observed result |
| --- | --- |
| Full suite with optional tracking | 261 passed; two upstream deprecation warnings |
| Clean source release + installed wheel, core-only | 259 passed, two expected MLflow skips; one upstream warning |
| New serving/release tests | 12 passed |
| Ruff lint / format | Passed; 85 Python files formatted |
| Example configuration validation | Passed, package 0.12.0 |
| uv lock consistency | Passed; model dependencies unchanged |
| Wheel and source archive build/content gates | Passed |
| Isolated core installed-wheel smoke | Passed; MLflow absent and editable checkout not imported |
| Linux-style Git export of frozen source files | All five original byte hashes preserved |
| V1 preservation / local retained artifacts | Passed |
| Git diff whitespace review | Passed |
| GitHub workflow static validation | actionlint passed; shellcheck not installed/enabled locally |
| Local Docker build/runtime | Not executed: Linux Docker engine unavailable |
| Hosted CI, including Linux/new-Python lanes | Configured; not executed locally |

Docker CLI 29.1.3 is installed, but the desktop-linux engine pipe was unavailable.
Starting Docker Desktop in the background and a bounded 30-second desktop start
attempt did not produce an engine; the latter timed out. No image build or
container smoke pass is claimed. Docker build/runtime verification remains for a
working local engine or the committed Linux CI job. The test matrix describes
future hosted checks, not evidence of observed Linux/Python 3.12–3.14 results.

The clean-package run lacked the original dataset, historical artifacts and V1
environment. Tests only fitted/scored their own temporary synthetic fixtures;
no original final-test scoring, model promotion, image publication or deployment
occurred. Starlette/httpx and MLflow/SQLAlchemy warnings are inherited upstream
deprecations already documented in Phase 11. Installed-wheel checks use uv copy
mode to avoid OneDrive hardlink failures.

See [testing_ci_docker.md](../docs/testing_ci_docker.md) for coverage mapping,
commands, container mounts and limitations. Phase 13 is model governance
documentation; README replacement remains Phase 14.

# Testing, CI and Docker

Run commands from the credit-risk-lab checkout. Python 3.11 is the baseline for
reusing the frozen Phase 4/5 model artifacts; the lock has different dependency
branches on newer interpreters. Compatibility tests on newer Python versions
create their own synthetic experiments and do not certify old model portability.

## Local checks

```console
uv sync --locked --extra tracking
uv run --no-sync pytest --strict-config --strict-markers -q
uv run --no-sync ruff check src tests scripts api
uv run --no-sync ruff format --check src tests scripts api
uv run --no-sync credit-risk-lab check-config --config-dir configs
uv build
uv run --no-sync python scripts/check_release.py --dist-dir dist
```

For the core-only lane, use `uv sync --locked` without the extra. The two SQLite
tracking integration tests then skip; dependency-absence and evidence checksum
tests still run. Tests use fresh temporary data and fitted fixtures, never the
original dataset, original calibrated model or original final holdout. No global
network service is needed by pytest. Dependencies must first be installed.

The existing risk tests use hand-calculated cases and invariants:

| Risk requirement | Evidence in tests |
| --- | --- |
| Gini, KS, discrimination versus calibration | test_metrics.py, test_calibration.py |
| PSI and frozen monitoring support | test_monitoring.py |
| PD × LGD × EAD and portfolio reconciliation | test_expected_loss.py |
| Risk bands, inclusive boundaries, affordability | test_credit_strategy.py |
| Vintage denominators, missing follow-up, exposure | test_portfolio.py |
| Consecutive-month transitions, cures, matrix rows | test_portfolio.py |
| Observation cutoffs and future-transition exclusion | test_targets.py, test_expected_loss.py |
| Train/calibration/development/test separation | test_pd.py, test_validation.py |
| Whole verified bundle through API startup/scoring | test_serving_integration.py |
| Private/generated artifacts excluded from releases | test_release.py, check_release.py |

The longitudinal model is currently a state-transition benchmark. There is no
standalone rolling behavioral-feature transformer yet; cutoff tests cover the
implemented target and transition pipelines, not an unimplemented feature layer.

New serving integration tests train and calibrate a small, explicitly synthetic
fixture in a temporary directory. They verify actual model deserialization,
matching score/decision probabilities, protected evidence hashes, rejection of
that fixture's reserved profiles, and tampering rejection before deserialization.
They never retrain or rescore the original research experiment.

The installed-wheel smoke check is intentionally run outside the checkout with
`python -I`, using an isolated environment that omits MLflow:

```powershell
uv export --locked --no-dev --no-editable --no-emit-project --output-file dist/runtime-requirements.txt > $null
$projectPath = (Get-Location).Path
uv run --isolated --no-project --link-mode copy --directory $env:TEMP --python 3.11 --with-requirements "$projectPath/dist/runtime-requirements.txt" --with "$projectPath/dist/credit_risk_lab-0.12.0-py3-none-any.whl" python -I "$projectPath/scripts/smoke_installed.py"
```

## GitHub Actions

`.github/workflows/ci.yml` runs on pushes, pull requests and manual dispatch.
It tests Windows/Linux on Python 3.11, with both core and optional tracking
installations, plus Linux core compatibility on Python 3.12, 3.13 and 3.14.
Each lane installs the committed lock, checks formatting/lint/configuration,
runs the full tests, builds/inspects releases and installs the core wheel outside
the checkout. Core wheel checks omit optional tracking even in a tracking lane.
Actions are pinned to verified release commit hashes. The workflow has read-only
repository permissions, cancels superseded runs, sets timeouts and disables
MLflow telemetry. The Docker job builds and runs locally on its Linux runner;
it does not publish an image or deploy a service. These jobs execute after the
commit is pushed to GitHub; a local actionlint pass is not a hosted CI run.

The five base/calibration source files whose raw byte hashes are recorded in the
Windows-frozen experiment explicitly use CRLF in `.gitattributes`. This preserves
the recorded bytes in Linux checkouts without editing code or weakening checks.
Other sources and V1 evidence have no new line-ending policy. The original
`check_phase1.py` remains a separate local historical-preservation check because
its retained V1 environment, cache and original data deliberately do not exist
in clean CI checkouts. It is excluded from Ruff; other checkout scripts and API
aliases now pass the same lint/format checks as the package.

## Container

```console
docker build --tag credit-risk-lab:local .
python scripts/smoke_container.py --image credit-risk-lab:local
```

The multi-stage Dockerfile installs the core non-editable package from uv.lock,
uses Python 3.11, includes libgomp for numerical libraries, and runs as UID/GID
10001. The final image contains the installed environment and example configs;
it contains no original data, model artifacts, tracking database, tests or V1
runtime. `.dockerignore` allows only build inputs. The pinned base/uv version
tags can be overridden/reviewed when rebuilding; OS repository packages and tag
contents are not pinned by digest, so this is not a bit-identical image guarantee.
The upstream Linux XGBoost wheel brings an NCCL dependency even though the lab
uses CPU inference; GPU hardware is not required and the image is consequently
larger than a basic FastAPI image.

Default serving config expects these three trusted, read-only mounts. In
PowerShell, from the checkout:

```powershell
$projectPath = (Get-Location).Path
docker run --rm --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges -p 127.0.0.1:8000:8000 --mount "type=bind,source=$projectPath/cs-training.csv,target=/research/cs-training.csv,readonly" --mount "type=bind,source=$projectPath/artifacts/phase4-origination-001,target=/research/phase4-origination-001,readonly" --mount "type=bind,source=$projectPath/artifacts/phase5-validation-001,target=/research/phase5-validation-001,readonly" credit-risk-lab:local
```

`/tmp` permits plotting-library configuration/cache while the application
filesystem stays read-only. To use another bundle, mount its YAML and set
`CREDIT_RISK_SERVING_CONFIG`; use the identical source/code/dependency versions
required by its manifests. Checksums do not authenticate an untrusted pickle.
The image is a research service; model artifacts remain external requirements.

Without mounts, Uvicorn starts but `/health`, `/score` and `/decision` return 503.
Docker's readiness health check reports unhealthy in that case. The smoke script
uses an ephemeral localhost port, verifies that failure behavior and cleans up
only its own uniquely named container. It does not manufacture a model to make
readiness green. A successful mounted-model run requires the local evidence
bundle and is separate from the artifact-free CI container check.

Design references: [uv Docker integration](https://docs.astral.sh/uv/guides/integration/docker/),
[Dockerfile reference](https://docs.docker.com/reference/dockerfile/),
[setup-uv](https://github.com/astral-sh/setup-uv),
[GitHub Actions Python testing](https://docs.github.com/en/actions/tutorials/build-and-test-code/python).

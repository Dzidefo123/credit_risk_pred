# Phase 2: project architecture and packaging

Date: 2026-10-04 (America/Sao_Paulo).
Branch: feature/risk-modeling-lab.

Implemented src/credit_risk package boundaries, a versioned console entry point,
Hatchling wheel packaging, uv dependency management, Makefile convenience targets,
typed YAML contracts and JSON logging. Development dependencies are pytest and
Ruff. Only configuration dependencies are runtime requirements in this phase.

The configuration checker validates example development, experiment and policy
settings without reading data, deserializing artifacts or making decisions.
Detailed setup and package boundaries are documented in docs/architecture.md.
Legacy source remains unchanged. No Phase 3 data contracts or modeling have
been implemented.

Validation gate: run the configuration and malformed-input tests, Ruff lint and
format checks, the Phase 1 preservation check, build a wheel, and import its
installed package in an isolated environment outside the source import path.
Inspect the phase diff and verify the working tree after committing. Record
actual command results before treating this phase as complete.

## Verification evidence

Checks executed on Python 3.11.4: pytest (19 configuration/CLI/logging cases),
Ruff lint and format checks, and the Phase 1 source/artifact preservation check.
The configuration CLI validated all three example files. uv built both a source
distribution and wheel. The wheel was installed with exported locked runtime
dependencies into an isolated uv environment outside the checkout; all 11
submodules imported and the CLI validated external configuration successfully.
No inherited CSV or pickle was loaded. All checks passed before commit.

OneDrive initially rejected uv hardlinks, leaving a partial PyYAML installation.
The affected new-environment read-only flags were cleared and PyYAML was
reinstalled using copy mode. The project now sets tool.uv.link-mode=copy.
A subsequent locked sync and isolated wheel installation verify the dependency
setup; the historical venv was not altered. Python 3.12-3.14 are declared but
were not exercised in this phase.

# Track B Task 2: authorized mortgage panel engineering

## Current acquisition status and scope

The initial run had no authorized archive and recorded NOT_ACQUIRED. The user subsequently supplied an authenticated-download original, historical_data_2010.zip. Metadata-only preflight now verifies its root hash and four nested quarterly ZIPs, but stops before loan rows: the bundle is not the prespecified sample pair and exceeds current budgets. No source copy, extraction, rename or recompression was performed.

That metadata-only STOP is preserved in FREDDIE_2010_INITIAL_PREFLIGHT_STOP.json. The approved Task 2A amendment subsequently enabled the real annual run. The [current audit](../../reports/track_b/FREDDIE_2010_DATA_AUDIT.md) and [JSON evidence](../../reports/track_b/freddie_2010_data_audit.json) now record ACQUIRED and PROCEED WITH CONDITIONS, with genuine empirical counts. Initial and annual-run private manifests remain separate; the original archive is immutable.

The [Task 1 protocol](mortgage_research_protocol.json), [research design](MORTGAGE_RESEARCH_DESIGN.md), vintage, event definitions, six-month history and maximum 1,000 loans are unchanged. No model or ECL calculation is implemented. The initial representation is nominal-time retrospective, with historical availability unverified.

## Authorized local input interface

There is no downloader or authentication/session handling. Obtain one 2010 Standard sample archive through the official mechanism after accepting its applicable terms. The interface expects the Release 47 layout and exactly sample_orig_2010.txt plus sample_perf_2010.txt, not a full historical dataset or different vintage. Unsupported names/layouts stop ingestion; do not rename/shift columns to make another source fit.

Create a private authorization JSON with exactly these keys (values below describe requirements, not accepted terms or real acquisition evidence):

| Key | Requirement |
| --- | --- |
| source_kind | authorized_freddie for an actual source; synthetic_fixture reserved for explicit engineering fixtures |
| source_organization | Freddie Mac for official data |
| vintage | 2010 |
| source_release | String 47 |
| layout | freddie-standard-r47-july-2026 |
| source_sha256 | Actual ZIP byte SHA-256 |
| acquired_at | Actual timezone-aware acquisition timestamp, not a future time |
| official_source_reference | Public official HTTPS reference; no query/session/credential URL |
| terms_accepted | True only after the user actually accepted applicable terms |
| license_redistribution_status | prohibited, restricted or permitted after documented review |
| aggregate_publication_permitted | Boolean; public aggregate reporting requires an explicit affirmative decision |

This is a user attestation, not automated certification of a license. Do not put cookies, tokens, passwords or authentication files in it. Unknown/extra keys are rejected. Archive hash is verified before parsing and after reading/copying. Resource guards currently stop archives over 200 MiB compressed, 2 GiB expanded over 60,000 origination IDs or over 250,000 retained performance records; review transfer/storage scope separately if the official file exceeds these limits. Do not automatically fetch more data.

```console
uv run --no-sync python scripts/build_track_b_panel.py
uv run --no-sync python scripts/build_track_b_panel.py --source data/track_b/raw/sample_2010.zip --authorization data/track_b/manifests/authorization.json --loans 1000
```

The first command writes the honest NOT_ACQUIRED audit only and exits with code 2, so missing acquisition cannot look like a successful panel build. Integrity/cohort failures exit nonzero after private diagnostics are written. The second is for a future already-authorized local archive. --publish-aggregates is an additional explicit publication request, accepted only if the authorization permits it; fixtures can never publish as empirical Freddie evidence. Default real/fixture outputs stay private.

## Release and parsing contract

The [official July 2026 workbook](https://www.freddiemac.com/fmac-resources/research/pdf/file_layout_july_2026.xlsx) was inspected in memory: 31 origination positions and 35 performance positions. Documentation SHA-256 is ce054271c42b7ad5f173a045c73368d997a2ac99253dcb312a45ccef43a4b13e. Canonical position mappings and parser version are recorded in each local manifest; dictionaries are not redistributed.

The parser validates width, loan ID/vintage, month format, numeric finiteness/integral fields, fixed-rate product and required linkage/date fields. Blank and documented field-specific sentinels are missing, never arbitrary numeric values. Monthly state strings remain source strings; no exact DPD is manufactured. R47 loss signs are retained. The workbook permits alphanumeric net proceeds while the glossary describes amounts: an unexplained alphabetic disclosure is preserved privately, numerically missing and flagged, without guessing a meaning.

Every row contributes to source/valid/malformed counters. Reject reasons contain no licensed record values. Malformed origination/performance rows, duplicate static keys or unmatched performance rows stop the workflow; a private structured rejection audit gives counts and reasons. No last-row-wins schema repair or outcome-based reselection. UTF-8/corrupt archive failures also stop, never produce a partial successful panel.

## Identifier-only sampling and linking

Select at most 1,000 unique valid origination IDs by ascending SHA256 of UTF-8 freddie_mortgage_research_v1 + colon + loan_id, ties by ID. Selection is completed before performance inspection. Salt, selected-ID-set digest and source identities are recorded; labels, default, loss, balance and completeness do not determine inclusion. Missing performance for a selected loan is reported, never replaced by another loan. Loan IDs mean mortgages/facilities, not borrowers.

All source records are parsed/validated, but only selected histories are retained. Source parser counts are distinct from selected-cohort counts. Origination/performance unmatched counts have explicit scopes. Raw selected records remain private. The panel orders loan/month observations; raw order and chronological gaps are reported. Duplicate months become a flagged quarantine placeholder rather than a chosen or merged record; original duplicate rows remain in the private interim file. Gap/unknown/terminal history prevents subsequent primary re-entry.

## Nominal-time landmarks and outcome separation

Every selected observed loan/month contributes a panel/audit landmark, including insufficient-lookback, prevalent-event, missing-state and censored cases. Eligible landmarks meet Task 1's clean continuous prefix and lookback. Future maturity/completeness does not determine t0 inclusion.

FEATURE is a prespecified whitelist of origination/static and contemporary monthly fields. OUTCOME contains outcome status, binary twelve-month indicator, event offset and ascertainable follow-up. AUDIT contains loan identity, t0, eligibility/reason and explicit unverified knowledge-time status. feature_frame rejects identities, outcomes, recovery fields, future-macro values and arbitrary unregistered columns; it does not train or select features. A modeling cohort decision is still required later.

The twelve-month loop uses offsets 1..12. Month-12 default is positive; month-13 default does not change a completed earlier horizon. Prevalent/default/unknown/gap history makes t0 ineligible. Payoff/maturity is a competing cause, not all non-default terminations or administrative censoring. Administrative exit, missing months or cutoff preserve unknown binary outcomes. Same-month payoff/default and date contradictions are ambiguous. After a verified payoff, later default rows do not create an earlier default outcome; unexpected continued records are separately flagged and block modeling until reviewed.

First future unknown with no ascertainable interval is insufficient_followup; later unknown/cutoff is right_censored. Both remain unknown binary values. The complete-event-free proportion and all-ascertained-outcome proportion have different meanings and use all eligible landmarks as denominator, not only matured survivors. Positive/payoff outcomes may be ascertainable early; they are not called twelve months of survival. All ineligible/censored/ambiguous rows remain in outputs.

## Private outputs and manifests

All data/track_b/** is Git-ignored, including:

- raw/sample_2010.zip: immutable authorized source copy.
- interim/selected_origination.csv and selected_performance.csv: licensed selected records and source disclosures.
- processed/panel.csv: all landmarks with controlled roles and explicit unknown outcomes.
- manifests/freddie_2010_manifest.json: source/member sizes/hashes, release/layout, parser, license/attestation, linkage counts and selection digest.
- manifests/panel_manifest.json: protocol/source/acquisition-manifest hashes, code hashes/version, counts/range, roles, creation time and output hash.
- manifests/freddie_2010_data_audit.json and Markdown: private aggregate quality/follow-up/loss inspection; fixtures explicitly labeled FIXTURE_ONLY.
- manifests/ingestion_rejection.json: private rejection diagnostics when a schema/linkage gate fails.

The initial NOT_ACQUIRED state created no acquired-source manifest. The continuation creates only a genuine metadata preflight acquisition manifest; no processed-panel manifest exists. Writes refuse overwriting prior panel/interim/manifests, enforce the private zone and do not extract ZIP members to arbitrary paths. Authorized aggregates can be promoted to reports only after an explicit publication decision.

The audit covers selected records, date ranges, per-loan history length, feature/source missingness, duplicates/gaps, state/exit categories, principal availability/distribution, label/follow-up counts, ambiguous records, parser/linkage findings and signed loss-field availability. No LGD ratio, EAD model, PD score, calibration statistic or ECL is calculated. Signed/anomalous disclosures are review flags, not silently repaired values.

## Verification and scientific gate

Tests use invented synthetic records generated in code, never licensed records. They exercise schema rejects, masks, signed values, sampling/order invariance, calendar boundaries, censoring, prevalent/default cases, duplicates/gaps/termination, future-mutation invariance, firewall, source/protocol hashes, private output limits and no fitting/loading. Existing Track A evidence/registry/frozen-source checks remain in force; its tag is not altered.

No source contradiction can currently be inferred because no actual records were inspected. Any future mismatch is reported and requires a separate protocol/layout amendment, not a convenient event change. After receipt of an authorized archive, run the same pipeline and empirical audit, then evaluate all Task 2 stopping rules before designing Task 3 cohorts/models.

Exactly one next task: **Review and approve a versioned annual-2010-bundle sampling-frame/parser/resource amendment.** Keep event/horizon definitions unchanged and inspect no loan rows until approved. Do not advance to a PD baseline yet.

## Metadata-only preflight continuation

The original ZIP is read in place. Its hash is computed first and verified again after inspection. Stored nested ZIPs are viewed through bounded seekable slices; only central directories are decoded, no loan-member payloads or files extracted. Names, sizes, CRCs and timestamps are recorded without treating CRCs as SHA-256 or timestamps as release/acquisition proof.

```console
uv run --no-sync python scripts/build_track_b_panel.py --preflight --attest-official-download --source <original-zip-path>
```

Exit code 3 means metadata acquisition was recorded but compatibility blockers prohibit processing. This flag records the user's official-download attestation; it cannot prove the actual row layout or override the governing sample frame. The current parser and selection/event logic are unchanged. An annual bundle is not automatically accepted because its underlying names resemble Release 47 files.


## Approved annual continuation and empirical result

Task 2A approved the [versioned annual amendment](ANNUAL_BUNDLE_ACQUISITION_AMENDMENT.md) before any loan rows were inspected. The [initial stop](../../reports/track_b/FREDDIE_2010_INITIAL_PREFLIGHT_STOP.json) remains preserved. Annual processing uses bounded stored-member views; the compressed-wrapper fallback has one private temporary quarter and tested cleanup. No entire archive is copied or text extracted.

```console
uv run --no-sync python scripts/build_track_b_panel.py --annual --attest-official-download --source <original-annual-zip-path>
uv run --no-sync python scripts/build_track_b_panel.py --annual-report-only
```

The annual workflow refuses an adaptive redraw if a frozen sample manifest exists. The report-only command recomputes aggregate follow-up selection diagnostics from the existing private panel and does not change sample or labels. Annual run source, sample and panel manifests live under data/track_b/manifests/annual_2010_v1; the panel lives under data/track_b/processed/annual_2010_v1. All remain Git-ignored.

The [empirical audit](../../reports/track_b/FREDDIE_2010_DATA_AUDIT.md) now supersedes the preflight STOP operationally, without erasing it: PROCEED WITH CONDITIONS for PD/survival cohort design, limited aggregate loss evidence, and unverified historical availability. No models were fitted. The 1,000 ID set is frozen for subsequent experiments, regardless of event support. The next task is Track B Task 3 - Cohort Design and 12-Month PD Baseline.

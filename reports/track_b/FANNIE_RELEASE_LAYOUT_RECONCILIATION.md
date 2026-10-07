# Track B Task 13R — Fannie Release/Layout Reconciliation

## Executive Summary

**FANNIE RELEASE LAYOUT RECONCILED WITH MATERIAL LIMITATIONS.**

Official documents explain a legitimate 113-position layout and a later 114-position
layout. The additional position is **114: Origination VantageScore 4.0**, announced for
the October 2026 SF dataset publication. It is not a frozen primary replication predictor.
There is no evidence here of archive corruption or a need to pad a blank column.

This reconciles the documentation discrepancy, **not the archive's exact publication
identity**. The archive has a documented candidate ordering but remains unbound to a
provider-confirmed release. Its governing-release gate stays closed, accepted-license
provenance remains unresolved, and Task 13A is not authorized. No outcomes were interpreted.

## Task 13 Stop Condition

Task 13 correctly stopped against the then-pinned current 114-position glossary.
Its report, crosswalk, protocol, archive receipt and code are unchanged. Task 13R adds
new evidence rather than rewriting that earlier conclusion. The external protocol
remains **DRAFT_NOT_YET_AUTHORIZED**.

The reconciliation uses official historical documentation and supplements, not a
third-party schema, guessed missing column or inspected loan values. The
[evidence JSON](fannie_release_layout_reconciliation.json) separates documentary ordering,
observed width, semantic SF inventory and unresolved archive binding.

## Archive Identity

| Property | Evidence |
| --- | --- |
| User-supplied archive | `2010Q1.zip` |
| SHA256 | `09735f1dd50b3a3046117a6cdab12fe003d7aebd43f9a058bdeafbd24789c415` |
| Archive size | 364466968 bytes |
| ZIP member | `2010Q1.csv` |
| Member sizes | 364466690 compressed;5714098123 declared uncompressed bytes |
| Encoding/delimiter | ASCII prefix compatible with UTF-8;pipe |
| Line endings | LF |
| Header | Official FAQ says headerless; header contents were not interpreted |
| Acquisition quarter | 2010Q1; not origination year or publication version |
| Performance observation period | Not inspected |
| Download date | Unverified; no download receipt provided |
| Portal label | User attests Primary acquisition/performance archive, authenticated Data Dynamics |
| Exact portal release label | Unverified |
| ZIP timestamp | 4 May 2026; not a release/download receipt |

The archive was neither extracted, renamed nor modified. Its byte hash was checked
again. Do not infer release identity from filename, download year or ZIP timestamp.

## Observed Physical Layout

The same preauthorized bounded routine reconfirmed **512 opaque records with 113 fields
each**, exactly reproducing Task 13's receipt. It counted delimiters and checked encoding,
without splitting out or interpreting field values. No event function was called.
See [new structural receipt](fannie_release_archive_structure.json).

The official FAQ and February 2021 tutorial describe one combined record layout containing
static acquisition characteristics and monthly performance information. Separate logical
acquisition/performance *files* belong to the earlier format. There is no documented
modern record-type switch that would justify treating these 512 lines as different-width
acquisition and performance records. The bounded check does not establish full-file quality.

## Current Official Layout

The pinned September 2026 PDF and newly retrieved official XLSX were independently
extracted programmatically. Their positions, field names and SF applicability agree after
recovering a PDF text-overflow label at position 105 from full-page text and visual inspection.
That extraction issue is not a provider field rename.

| Inventory | Current count | Meaning |
| --- | --- | --- |
| Shared documented physical positions | 114 | Preserve every ordinal slot |
| SF-applicable checkmarks | 73 | Excludes 41 entries marked SF NA |
| SF semantic inventory | 72 | Also excludes 109, which always reports not applicable for SF |

These are **documentation counts**, not loan-value/nonmissing counts. Date-bound fields
remain conditional; 72 does not mean 72 observable values in any loan record or month.
In particular 24/25 cease population under the newer score regime, while 114 has a
later publication boundary. Never drop CRT-only/NA placeholders to force raw width
to match the semantic inventory. Conversely, never claim 114 usable SF variables.

The [release registry](../../docs/track_b/fannie_release_layout_registry.json) records every
current position, field name, applicability, date mentions, source/page and physical expectation.

## Historical Layout Evidence

| Official evidence | Layout conclusion |
| --- | --- |
| June 2015 layout | Two files:23 acquisition positions and 26 performance positions |
| June 2016 layout, effective July | Two files:24 acquisition and 29 performance positions |
| December 2016 layout, January 2017 enhancement | Two files:25 acquisition and 31 performance positions |
| September 2020 announcement and February 2021 tutorial | Enhanced combined108-position format |
| July 2024 retained glossary | Full ordering through 110 |
| November 2025 announcement and its linked `1225` amendment | Update32;append111/112/113, SF publication April 2026 |
| August 2026 announcement and linked definitions workbook | Append114, SF publication October 2026 |

A full historical 113-row binary glossary was not located. Instead, the reconstruction
uses a hash-pinned official full110-row glossary plus a hash-pinned official amendment
with explicit positions 32/111/112/113. The current first113 field names and SF applicability
match that composite ordering exactly. Its name explicitly includes **reconstructed**.

A search-index extraction shows a February 2026 glossary ending at113, but opening the
same URL now returns September 2026's114-position PDF. That supports the warning about
mutable URLs; it is **not** used as frozen historical binary proof. A directory named
with an old date is likewise insufficient: document bytes and footer/version govern.

## 2020 Enhanced Format

The September 2020 announcement states a combined108-field layout, with acquisition
and performance information through Q2 2020. The October 14 notice identifies the
enhanced release as October 29. The tutorial explicitly describes historical acquisition
quarters, including old cohorts, in the combined format; a 2010 package therefore need
not retain the format that existed in 2010. This is documentary evidence of historical
repackaging, not empirical validation of every republished file.

The [legacy movement map](../../docs/track_b/fannie_2020_legacy_field_mapping.json) maps all
25 acquisition and 31 performance positions from the last retained legacy layout.
Their duplicate loan-ID position collapses into55 unique mapped positions. Fifteen new
SF concepts bring that documentary SF inventory to70; the shared physical layout has108 slots.

Examples: loan ID 1 becomes 2; original rate acquisition 4 becomes 8; current rate performance 4
becomes 9; delinquency11 becomes 40; zero-balance code 13 becomes 44. Product Type becomes
Amortization Type in the combined dictionary. Later label changes such as HomeReady to
Special Eligibility Program are separately dated, not attributed to October 2020.

The official format-comparison document confirms month/date reformatting to MMYYYY,
original UPB display precision changing to two decimals, and delinquency moving from
one-digit/X notation to two-digit/XX notation. Format changes do not establish exact DPD
or remove the documented source rounding limitation. The108 layout is registered as
**width documented, adapter disabled** until its complete contemporaneous ordering is pinned.

## 2026 Disclosure Changes

The November 17,2025 announcement schedules SF changes for April 2026 and CRT changes
for February25,2026. Only origination Classic FICO111 discloses a new SF score; issuance112
and current113 are not disclosed in SF. All three belong to the shared physical ordering.

The August17,2026 announcement schedules the next SF update for October 2026 and CRT
files on/after October26. Its workbook's first sheet lists111–113, while the **Definitions**
sheet also contains114. Inspecting only the first sheet would miss the added position.
The current PDF and XLSX confirm114 and its SF applicability.

As of this task's 7 October 2026 review, the published September glossary is documentation
for an upcoming October release. It must not automatically govern an earlier downloaded
file. Nor does a historical loan's acquisition quarter decide its release schema.

## Field-Level Difference

| Comparison | Classification | Research consequence |
| --- | --- | --- |
| Reconstructed113 versus documented 114 | FIELD_ADDED_LATER:position 114 only | NO_RESEARCH_IMPACT for frozen primary predictors/events |
| Older110 versus reconstructed113 | FIELD_ADDED_LATER:111/112/113 |111 affects score-regime provenance;112/113 SF NA |
| Older MSA label at32 | FIELD_RENAMED_ONLY:MSA/MSDA amendment | No width change; geography version matters |
|24/25 population retirement | CONDITIONAL_DISCLOSURE | REPLICATION_FEATURE_IMPACT; no column removal |
|112/113 and 110 SF support | SF_NOT_APPLICABLE | Physical slots remain; no SF variable support claim |
| Archive113 against future glossary114 | DOCUMENTATION_VERSION_MISMATCH | PROVENANCE_ONLY, conditional on verified archive binding |

No first113 position movement or removal was identified in the reconstructed/current
ordering comparison. This does not prove those names govern every slot in the private
archive: its release binding still needs a provider receipt or confirmation.

## Date-Bound Semantics

| Position | SF publication boundary | Activity applicability | Qualification |
| --- | --- | --- | --- |
|32 | April 2026 update | December 2025 | Geography enhancement; no additional column |
|109/110 | January 2024 enhancement announced | Shared October 2023 notes |109 always NA for SF;110 explicitly SF NA |
|111 | April 2026 update | December 2025 | Earlier activity not presumed populated |
|112/113 | No SF score disclosure | CRT December 2025 | Physical shared slots do not imply SF availability |
|114 | October 2026 update | SF May 2026;CRT August 2026 | Publication waits for eligible loans/cutoff; no observed values |
|24/25 | Exact first affected SF file unverified | No population from March 2026 per current glossary | Announcement's May transition must not be confused with SF publication |

Publication date, observation month and acquisition quarter are separate clocks.
Physical presence permits blank or not-applicable historical cells. No field values,
missingness rates or older-activity backfills were checked.

## Sample-File Comparison

The official public sample link returned **HTTP403**. A metadata-only HEAD request also
returned403. No sample body was acquired and no sample schema, header, width or positions
were inferred. Third-party samples were not substituted. The Task13R policy would permit
only a bounded structural comparison of that specific official public sample, privately,
without interpreting outcomes; it does not permit another quarterly archive.

## Release-Specific Parser Design

Keep provider record → explicit release/order gate → canonical Fannie metadata → research
crosswalk. Registered IDs distinguish legacy acquisition/performance,2020 enhanced108,
documented 110, reconstructed April 2026 candidate 113, and October 2026 documented 114.
Only layouts with complete cited ordering can pass a **structural** width check.

There is no auto-detection from field count, no arbitrary-width acceptance and no113→114
blank insertion. Unavailable VantageScore in a candidate 113 canonical metadata view is
**STRUCTURAL_ABSENCE**, not a fabricated provider field or generic missing value.

Before research, require provider-verified release, ordering and archive-hash binding.
The Task13R research gate refuses activation even if those checks are later supplied:
research needs a separately authorized task. No ingestion/model parser is activated here.
The optional document extractor uses the bundled PDF/XLSX runtime without changing
project dependencies, configuration or earlier scripts.

## Crosswalk Impact

Task13's19 frozen primary predictors and event variables do not use VantageScore114.
Its absence alone therefore has **NO_RESEARCH_IMPACT** on that ladder. Do not substitute
it for FICO24/25/111. The same-width score-regime transition remains a feature/provenance
issue already captured by Task13's draft. Historical disclosure timing remains unverified.

Task13 crosswalk/protocol hashes are unchanged. New release metadata is a sidecar,
not a retroactive amendment or authority to bypass their existing release gates.

## Vintage Availability

Years2006/2008/2010/2014/2018/2020/2022 lie within the official Primary historical
scope: notes from 1999, acquisitions from 2000 through the current published period.
The tutorial documents downloadable acquisition-quarter packages. None of the seven
broader populations or package manifests was downloaded or opened in this task.

This is **documented temporal source support**, not verified eligible population size
or complete origination-year coverage. Acquisition-quarter packages are not note-year
cohorts. The exact quarter universe or explicitly bounded acquisition population still
must be frozen before identifier selection; no hindsight sampling rule is introduced.

## License Provenance

Workflow provenance is **USER_ATTESTED_AUTHENTICATED_MANUAL_DATA_DYNAMICS_DOWNLOAD**.
The user's supplied filename/path and authorization are recorded without personal portal
details, credentials, cookies or tokens. Exact accepted terms remain
**LICENSE_PROVENANCE_UNRESOLVED** until a non-sensitive version/receipt is supplied.

The frozen Task13 registry identifies the published royalty-free/internal-use document
v2.0 effective28 October 2015, footer20 April 2017, reviewed7 October 2026. Relevant clauses
are3.2(a) distribution,3.2(b) conditional non-commercial research publication,3.2(c)
individual identification and 4.2/4.3 notices/references. This records clauses; it makes
no legal determination about the intended publication. Retain
**LEGAL_OR_LICENSE_REVIEW_REQUIRED**. No licensed rows or raw documents are committed.

## Remaining Uncertainty

The physical-width discrepancy is explained by official schema evolution, but the
archive's exact release/download label and governing-order confirmation remain unverified.
The historical113 ordering is an official composite, not a recovered full113 binary.
Only512 lines were structurally checked; full CRC, whole-file width, identifiers, temporal
quality and semantic values were not validated. Public sample retrieval failed.
Origination-year acquisition coverage and accepted-license provenance remain unresolved.

If needed, request from Fannie support, without sending records: “For Primary2010Q1.zip
with the recorded SHA256, member 2010Q1.csv and 113 structural positions, which publication
and glossary/order version govern it? Does its physical layout retain NA positions 112/113
and precede the October 2026 addition114? Is an official archived113-position layout or
release manifest available?” Separately obtain the applicable acceptance-version receipt
from the downloader. These are drafted questions; no message was sent.

## Decision

**FANNIE RELEASE LAYOUT RECONCILED WITH MATERIAL LIMITATIONS.**

The legitimate113/114 evolution is explained. Operational research use remains stopped
until archive binding, license provenance and population coverage are settled. Recommend
separate authorization for **Track B Task13A — Fannie Sealed External Cohort Acquisition**
only after those gates and protocol acceptance. Do not implement or authorize it now.

Task13R code, metadata and documentation are public; source documents and data remain
private. All prior evidence is preserved. No push is authorized.

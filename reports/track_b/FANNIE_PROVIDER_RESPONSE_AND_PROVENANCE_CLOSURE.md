# Fannie Provider Response and Provenance Closure — Task 13T

## Executive Summary

**FANNIE SOURCE PROVENANCE STILL REQUIRES PROVIDER CONFIRMATION.**

The user supplied the substantive provider email and three downloaded resources.
These establish provider-directed documentary evidence. They do not identify the
publication that produced this exact ZIP or bind a historical terms version to its
acceptance. Task 13A authorization is **NO**. This is a valid provenance-only task,
not evidence that the source archive is malformed or unusable.

## Prior Stop Condition

Task 13S identified a reconciled 113-position candidate layout, but unresolved archive
release identity and accepted-license provenance. Task 14 froze the Freddie paper
evidence with Fannie proposed and unexecuted. Both historical records remain unchanged.

## Provider Support Request

The request asked which SF Loan Performance publication produced `2010Q1.zip`, which
layout governs it, whether 113 physical positions are expected, and how to identify the
applicable terms and accepted-version evidence. The prior request is preserved in
[the support artifact](FANNIE_SOURCE_PROVIDER_SUPPORT_REQUEST.md).

## Provider Response

The user supplied the email body attributed to Fannie Mae Investor Relations. It directs
the user to Data Dynamics Resources, the SF/CRT glossary, the Data Refresh Calendar and
the CRT Terms click-through agreement. It does **not** directly confirm an archive release.
The response date is unavailable; no independent email authentication was performed.
The retained transcription omits the greeting, personal identity, address and signature;
the original body typo is preserved. No email headers, account IDs or sessions are stored.
See [response evidence](fannie_provider_response_evidence.json).

## Provider-Directed Resources

The [document registry](../../docs/track_b/fannie_provider_document_registry.json) pins:

| Document | Bytes | SHA256 |
| --- | ---: | --- |
| crt-file-layout-and-glossary.xlsx | 41690 | debc2a9ae2573ae73b57e4e52d99ca6825b90062208ffcab792a95190ddc43e5 |
| data-dynamics-data-refresh-calendar.pdf | 90606 | b148824cf8b0818f9ff1f53b0f9cb4f5d5afab6ff47d92e59383dcb26a37e233 |
| crt terms.pdf | 121408 | cfff52a66a805a8a24b189eb79ba9ee68a053fa69324ef29fdf9ee3ab84111f9 |

Authenticated download route is user-attested. Exact download URLs and receipt dates
were not retained. Local file timestamps are separately labeled evidence of local presence.
Sources were read in place and not modified or committed. PDF text and rendered pages
are private; no original PDF metadata author or tracking fields are included in the public
registry. Relevant PDF pages were visually checked alongside text extraction.

## SF Glossary and File Layout

The workbook has one Combined Glossary sheet, 114 sequential documented positions and
the same byte hash as the current official workbook already pinned in Task 13R. Names
match the prior registry after whitespace normalization; SF applicability matches exactly.
Seven raw-name differences are only trailing/repeated whitespace at positions
27, 71, 86, 87, 88, 90 and 108. No provider record is normalized or interpreted.

Position 113 is Current Classic FICO and is not disclosed for SF. Position 114 is
Origination VantageScore 4.0, SF applicable from May 2026 activity; its CRT activity date
is August 2026. Those are activity dates, not dataset publication dates. The workbook has
no explicit numbered version or publication date; the frozen September 2026 PDF remains
the dated associated documentary evidence. Do not assign its date to this ZIP.

## Data Refresh Calendar

Page 1 lists **Single-Family Loan Performance Data Dashboard**, quarterly, between the
20th and 31st of the first month. January/April/July/October are a calendar-quarter
interpretation of that rule, not individually reported 2026 publication dates. Weekend
and holiday dates move to the following business day; the document allows schedule
variation and delays. It does not provide archive build IDs or download receipts.

The calendar names the dashboard; the provider directs users to it for SFLPD timing.
It is not independent proof that a particular quarterly ZIP was rebuilt on a particular
day. Its September 3, 2026 PDF modification metadata and 2025 copyright are not provider
publication dates. CRT monthly schedules must not be applied to SFLPD.

## CRT Terms

The downloaded document is **Fannie Mae CRT Click-Through Usage Agreement**, dated as
of October 20, 2017. No numbered version is stated. Its preamble describes effectiveness
upon affirmative click acceptance or written acceptance, not a dated acceptance record
for this user. Investor Relations specifically identifies CRT Terms as a click-through
agreement; that resource identification is stronger than an inferred link.

Its CRT-investment scope and conditions for Updated Credit Scores differ from the
previously identified SFLPD royalty-free/internal-use terms, version 2.0 effective
October 28, 2015. They are separate documents; do not silently substitute one for the
other. Sections 13–16 and 19 describe conditions involving updated-score use, model
validation/benchmarking/calibration and disclosure. These merit applicability review.
The refreshed Equifax CRT scores and SFLPD origination score fields must not automatically
be treated as the same data or contractual scope. No legal applicability or permission
conclusion is made. **LEGAL_OR_LICENSE_REVIEW_REQUIRED** remains.

## Authenticated Portal Evidence

The portal revisit and use/terms notices are user-attested context in the task instructions.
The provider email identifies the CRT Terms link in the left navigation. No non-sensitive
portal screenshot, exact notice transcription, acceptance receipt or account history was
supplied. The agent did not enter an authenticated portal or collect browser/session data.
Do not upgrade this to independently evidenced authentication or version-specific consent.

The supplied portal summary says use constitutes review/agreement; the PDF describes an
affirmative acceptance condition. Their relationship is unestablished. This is an
applicability question, not an agent interpretation that either condition supersedes the other.

## 113-Field Layout

**YES_BY_DOCUMENTED_VERSION_RECONSTRUCTION.** Task 13R combines the full 110-position
glossary with the official amendment introducing positions 111–113 in the April 2026
SF publication. The October 2026 SF update adds position 114. Current portal documentation
is compatible with that evolution; it is not a newly obtained historical 113-position file.
The October 26 publication threshold applies to CRT files, not an established exact SF date.
No position-114 padding or deletion of physical NA slots is permitted.

## Archive-Specific Release Binding

`2010Q1.zip`: 364466968 bytes; SHA256
`09735f1dd50b3a3046117a6cdab12fe003d7aebd43f9a058bdeafbd24789c415`.
One member, `2010Q1.csv`, compressed 364466690 bytes, declared uncompressed 5714098123
bytes. Prior schema-only evidence records 512 prefix lines of width 113. Task 13T
verifies hashes and directory metadata only; no member body is opened.

**RELEASE_IDENTITY_PARTIALLY_SUPPORTED / PARTIALLY_SUPPORTED** for exact archive binding.
The chain connects a user-attested portal acquisition, provider-directed documents and
compatible structural evolution. It lacks a release identifier tied to this hash. General
cadence and a compatible width cannot discriminate specific refreshes or archive builds.

The existing ZIP filesystem creation and last-write evidence is October 7, 2026 at
15:53:20 UTC and 15:56:39 UTC respectively (fractional seconds retained in JSON). This is
**LOCAL_DOWNLOAD_TIMING_EVIDENCE**, not a receipt; copying or synchronization can affect it.
The internal ZIP timestamp May 4, 2026 is also not provider release proof. A 2010 Q1 label
identifies an acquisition quarter, not a 2026 publication. No timestamps were altered.

## Terms Acceptance Provenance

**ACCEPTANCE_VERSION_UNRESOLVED**; authenticated acquisition remains user-attested.
The currently obtained CRT agreement can be identified and hashed, but this alone does
not establish which document applied or was accepted when the ZIP was acquired.
Neither an independently evidenced portal notice nor a version-bound acceptance receipt
exists in supplied evidence. This missing link, together with unresolved SFLPD/CRT terms
applicability, prevents a scientifically defensible authorization upgrade here.

## Seven-Vintage Coverage

The prior documented temporal scope includes 2006/2008/2010/2014/2018/2020/2022, with
the existing qualifications unchanged. The three provider-directed documents contain
no complete historical quarterly package inventory. Date-bound field notes are not a
downloadable-cohort inventory. No year is newly certified for complete acquisition-quarter
or origination-year coverage; no additional package is downloaded.

## Publication Boundary

**PUBLICATION_REVIEW_REQUIRED.** Fannie loan-level records remain PRIVATE and
NON-COMMITTABLE. Raw provider documents and licensed data do not enter the public diff.
This task does not decide arXiv publication permission for aggregate results, imply that
CRT Updated Credit Score restrictions necessarily govern SFLPD origination attributes,
or declare publication prohibited. The new Fannie evidence establishes no change to
Freddie publication permissions and no new issue that blocks Task 15 drafting under
Task 14. Existing publication review remains separate from drafting readiness.

## Remaining Limitations

- Exact archive publication, governing historical layout and accepted-version evidence remain missing.
- Current resource direction is not archive-specific confirmation; date and authentication of the email are not independently established.
- Portal notice/label screenshots are absent; local timestamps cannot replace receipts.
- CRT/SFLPD terms applicability and seven-vintage package coverage remain unresolved.

A follow-up could ask: “Which SF publication/build produced the frozen 2010 Q1 archive
identified by the supplied SHA256? Which historical glossary governs its 113 positions?
Do the October 2017 CRT agreement, the SFLPD royalty-free terms, or both apply to Primary
SFLPD acquisition and the proposed non-commercial aggregate research? How can a
non-sensitive accepted-version record be obtained?” **Prepared only; not sent.**

## Task 13A Readiness

| Gate | Status |
| --- | --- |
| Schema reconciled | PASS |
| Provider response obtained | PASS_WITH_MATERIAL_LIMITATION |
| Governing documentation identified | PASS_WITH_MATERIAL_LIMITATION |
| 113-field layout documented | PASS_WITH_MATERIAL_LIMITATION |
| Exact archive release binding | PASS_WITH_MATERIAL_LIMITATION |
| Applicable terms document identified | PASS_WITH_MATERIAL_LIMITATION |
| Acceptance provenance | FAIL |
| Private-data governance | PASS |
| Seven-vintage source availability | PASS_WITH_MATERIAL_LIMITATION |
| Fresh-evidence protection | PASS |
| Protocol status | FAIL — deliberately draft |
| Publication review | FAIL — not cleared by this task |
| Task 13A authorization | FAIL — NO |

The gate does not demand unattainable certainty or insist on an email identifying the ZIP
when adequate documentary binding exists. Here the documents still do not identify its
specific publication, and the historical applicable-terms/acceptance chain is independently
unresolved. Resource direction alone does not close those gaps.

## Decision

**FANNIE SOURCE PROVENANCE STILL REQUIRES PROVIDER CONFIRMATION.**
**TASK 13A AUTHORIZATION = NO.** Protocol remains DRAFT_NOT_YET_AUTHORIZED; no sealing,
pre-registration, outcome analysis or external replication. Task 14 paper evidence and
reports remain immutable. Task 15 remains independently governed and is not executed.
See [assessment](fannie_provenance_final_assessment.json) and
[verification](fannie_provider_verification.json). Focused commit only; no push. STOP.

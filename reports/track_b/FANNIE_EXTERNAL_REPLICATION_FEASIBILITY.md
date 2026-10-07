# Track B Task 13 — Fannie External Replication Feasibility

## Executive Summary

**FANNIE EXTERNAL REPLICATION REQUIRES SOURCE CLARIFICATION.**

Scientific feasibility is promising, but the authorized archive has 113 physical
positions in its bounded prefix whereas the pinned current glossary specifies 114.
Its matching official layout and release identity are unverified. Do not guess a
legacy adapter from field count or ZIP timestamp. The draft also requires an explicit
acquisition-quarter coverage definition before calling a sample an origination-year cohort.

This is a documentation and bounded structural inspection task. No loan values were
interpreted, no outcomes counted, no models fitted, no samples created and no wider
loan-data acquisition performed. Tasks10–12 conclusions remain frozen.

## Official Source

Use Fannie Mae **Primary Single-Family Loan Performance Data**, not CAS/CIRT reference
pools or the separate HARP refinance dataset. The
[source registry](../../docs/track_b/fannie_source_registry.json) records official URLs,
retrieval times, versions, available byte hashes and unavailable hashes explicitly.
Documentation retrieved 7 October2026 includes the dataset page updated31 July2026,
FAQ dated10 March2026 and glossary dated10 September2026. Official documentation bytes
and PDF visual checks remain private. Glossary SF applicability was checked visually,
especially termination and2026 credit-score tables.

## Access and Licensing

Registration and acceptance of applicable terms are required. The user reports manual
authenticated acquisition and explicitly authorizes this archive for schema validation.
A versioned terms-acceptance receipt has not been captured. No credentials or browser
session data were accessed.

The official royalty-free license v2.0 describes internal/non-commercial use and
restricts external data/derived-product distribution in3.2(a). Section3.2(b) provides
a non-commercial academic/research exception subject to non-reconstruction and
non-identification conditions. Section3.2(c) targets identification of individuals;
it does not establish a blanket ban on ordinary macro association. Notices and
nominative-reference conditions also apply. These are documented terms, not legal advice.

**LEGAL_OR_LICENSE_REVIEW_REQUIRED:** establish which terms were actually accepted,
whether the planned portfolio/publication context qualifies, and whether proposed
aggregates or derived outputs can reveal/reconstruct data. Do not infer that keeping
raw data private alone resolves publication rights. No licensed records are committed.

## Dataset Population

Official documentation describes selected conventional, fully documented, fully
amortizing fixed-rate mortgages, notes from 1999 and acquisitions from 2000. Government
insured, ARM, balloon, interest-only, prepayment-penalty and various nonstandard/Alt-A
products are excluded; additional delivery/recourse exclusions are described in FAQQ1.
The current page announces Q12026 acquisition/performance updates released31 July2026.
That does not prove the supplied archive belongs to that release.

The page describes terms30 years or less; FAQQ1 says at least5 and less than35 years.
Preserve this difference. Any common-term restriction requires an explicit source-based
decision before the protocol is sealed. Freddie Standard and Fannie Primary are selected
GSE populations, not identical lender populations or universal US mortgages.

HARP refinance loans are supplied separately. Exclude that separate product from primary
replication. Retain original Primary facilities that later refinance through HARP; excluding
them using future refinance information would introduce selection bias. Mapping keys
link mortgages, not borrowers, and do not identify the reason for all ordinary payoffs.

## Release and Revision Process

Quarterly releases add acquisition and monthly performance information and can correct,
add or remove historical records. Preserve archive identifier/hash, actual download time,
official release date, layout version, matching documentation hashes and amendment notes.
Currently revised historical data are historical event evidence, not verified knowledge-time
records. General disclosure lag cannot recreate the first availability of corrected cells.
The official timing guide illustrates different MBS/CRT/historical publication calendars.

## Schema History

FAQQ11/Q15/Q20 describes merged, headerless quarterly acquisition/performance files
since October2020, replacing separate files. The `.csv` extension does not establish
comma delimitation. The authorized archive was neither renamed nor extracted.

| Structural property | Authorized bounded check |
| --- | --- |
| Archive | `2010Q1.zip`,364466968 bytes |
| Members | One: `2010Q1.csv` |
| Compression | ZIP deflate;364466690 compressed bytes |
| Declared member size | 5714098123 uncompressed bytes |
| Prefix checked | 512 records, maximum 65536 bytes per record |
| Delimiter/encoding | Pipe; ASCII prefix compatible with UTF-8 |
| Line endings / slots | LF;113 fields in each checked record |
| Pinned current glossary | 114 positions,10 September2026 |
| ZIP timestamp |4 May2026; not proof of release date |
| Header | FAQ documents no header; no field values interpreted |

See [schema evidence](fannie_schema_validation.json) and the
[pre-inspection boundary](../../docs/track_b/fannie_schema_inspection_policy.json).
No full-member CRC, whole-file field consistency, semantic slot alignment or record quality
claim is made. A 113-slot prefix does not prove it is the first 113 current glossary fields.
The release-selection gate deliberately rejects the unmatched layout.

Positions 24/25 cease population from March2026 activity;111 Origination Classic FICO
starts December 2025. Position 114 Origination VantageScore4.0 starts May2026 SF activity.
Positions 112/113 issuance/current FICO are **NA for SF**. Do not mix score families,
fill old t0 features with later disclosures, or infer score availability from origination year.
Other changes include assistance102 from April2020, resolution106/108 from July2020,
MSDA geography from December 2025 and special-program enumerations from January2023 release.
Slots marked NA remain slots; never drop placeholders and shift subsequent positions.

## Freddie/Fannie Field Crosswalk

The [machine-readable crosswalk](../../docs/track_b/freddie_fannie_field_crosswalk.json)
contains 56 unique canonical concepts, provider fields/positions, units, temporal status,
missingness, modification behavior, availability, citations and explicit transformations.
It uses all six requested compatibility classifications. Its current-glossary positions
are documentation claims, not verified alignment of the supplied113-slot archive.

| Component | Freddie Standard | Fannie Primary | Replication implication |
| --- | --- | --- | --- |
| Population | Selected conventional Standard | Selected fully documented fixed-rate Primary | Structural selection differences |
| Vintage packaging | Frozen Freddie annual origination source | Fannie acquisition-quarter files | Freeze coverage; never relabel packages |
| Note month | Not explicit in pinned Standard |14 note date | Preserve provider distinction |
| First-payment clock | Scheduled first payment |15 scheduled first payment | Shared derived duration; age audit separately |
| FICO | Representative origination score |24/25 legacy,111 all-borrower minimum | Proxy/regime provenance |
| Delinquency | Numeric bands plus R REO |40 numeric00–99/XX | Conceptual default replication |
| Exit reasons | Source-specific code set |44 including06 repurchase | Explicit code mapping |
| Modifications | Flags/current rate/balance fields | Persistent 42, current9, separate deferrals | No blanket superiority claim |
| Loss fields | Aggregate realized-loss/recoveries |54–64 expense/proceeds;77/78 NA SF | Information only; no loss modeling |
| Updates | Release47 frozen source | Quarterly corrected historical files | Hash/version pinning both providers |
| Historical PIT | Not fully verified | Not fully verified | No real-time backtest claim |

## Origination Compatibility

Rate8 in percent and term13 in months translate directly. Original balance10 is rounded;
LTV20 may be blank above97 or unknown; DTI23 has range/unknown suppression. Preserve
provider missingness and train preprocessing only on development data. Legacy FICO uses
the minimum of available borrower/co-borrower scores, with incomplete-score provenance;
111 is separately versioned. No VantageScore substitution or future-score backfill.

Purpose27 preserves P purchase/C cash-out/R refinance/U unspecified refinance.
The glossary does not explicitly label R as Freddie no-cash-out N: retain a distinct
refinance-other-than-explicit-cashout category and classify the comparison as a proxy. Occupancy30 maps U to explicit unknown. Property28, units29,
channel4 and geography preserve provider unknowns and sentinels. Seller5/servicer6 are
grouped under provider-specific thresholds and are excluded from replication predictors.
Facility IDs stay private and provider-prefixed; no borrower-level independence claim.

## Performance Compatibility

Reporting3 is servicer-period month, not disclosure time. Current balance12 masks the
first six life months and includes noninterest modification/deferral amounts. Current
rate9 reflects modified terms where applicable. Legal remaining17 differs from
prepayment-adjusted remaining18, which is blank after modification. Provider age16
has FAQ/glossary convention nuances; use the frozen first-payment proxy clock.

Foreclosure52 is completion/liquidation; disposition53 is end of property interest.
Standalone repurchase date47, interest-bearing balance110, and net credit-event loss77/78
are CRT-only NA for SF. Their existence in the shared layout must not create support claims.
Expenses/proceeds54–64 are aggregate buckets, not complete timed workout cash flows.
They offer useful descriptions but no unconditional improvement over Freddie loss evidence.
No LGD, EAD, ECL or staging work is authorized.

## Default Semantics

Draft research proxy: first known monthly delinquency03–99 or termination02/03/09.
This preserves the90+DPD band concept, not exact DPD or a regulatory default label.
Fannie lacks Freddie's explicit R REO delinquency state. Later foreclosure/disposition
dates will not be added to the primary definition to make the sources appear identical.
Consequently label material event differences **CONCEPTUAL_REPLICATION_ONLY**.

| Fannie Primary code | Documented reason | Draft state absent previous default |
| --- | --- | --- |
| blank | No termination | AT_RISK only with known00–02 delinquency |
|01 | Prepaid or matured | PAYOFF_OR_MATURITY |
|02/03/09 | Third-party sale / short sale / deed-in-lieu or REO disposition | DEFAULT_PROXY |
|06 | Repurchased | ADMINISTRATIVE_EXIT |
|15 | Nonperforming note sale | AMBIGUOUS_EXIT unless known severe delinquency establishes default |
|16/96 | Reperforming note sale / non-credit removal | ADMINISTRATIVE_EXIT |
|97/98 | Legacy CAS credit-event codes | Reject: not Primary applicable |

Date conflicts and severe-delinquency-plus-payoff conflicts are ambiguous first.
Known severe delinquency remains default even with administrative/note-sale codes.
XX/blank delinquency without explicit event ends usable history; unknown codes fail closed.
First event absorbs: no cure reentry. These rules are tested synthetically only.

## Payoff Semantics

Code01 cannot separate voluntary payoff from maturity and cannot establish refinance
motivation. Keep PAYOFF_OR_MATURITY, not a measured refinance label. Repurchase 06,
note sales15/16 and non-credit96 are distinct exits. Do not merge them into payoff.
HARP links can support a separately authorized product sensitivity, not redefine all 01 exits.

## Modification Semantics

Flag42 stays Y after the first legal modification, with subsequent term changes reflected
in monthly fields. Deferrals are not legal modifications;106/108 describe separate relief.
This is not a complete modification transaction history. Freddie also supplies current
interest and modification flags, so Fannie is not automatically stronger on that basis.
Fannie-specific deferral details may help later diagnostics, subject to activity support.

## Refinancing-Incentive Compatibility

Preserve Task 12 primary **ORIGINAL_CONTRACT_RATE_PROXY_GAP**: original rate8 minus
unchanged PIT MORTGAGE30US level, in percentage points; positive/negative pieces remain
the primary representation. This is borrower-relative coupon incentive, not a causal
refinancing effect. Ordinary payoff remains a mixed endpoint.

Current rate9 could support a **CURRENT_CONTRACT_RATE_GAP** using only the t0 row,
with source knowledge-time limitations. It is not silently promoted to primary and is
not authorized by this draft. It requires a separate prespecified extension before
outcome access. No modification/outcome-based sample exclusions or feature selection.

## Macro Compatibility

Reuse frozen `task9_api_v5` observations, metadata, vintage validity, transformations
and freshness checks. M2 uses unemployment level/change3m, Treasury10y, mortgage30y,
HPIyoy, CPIyoy and GDPqoq; primary excludes redundant spread exactly as Task 10 did.
No API acquisition, current revised backfill, provider-driven series substitution or
extension of macro support. Risk month m is the target; assessment t0 is end of month m−1.

## Calendar Support

| Specification | Existing frozen risk-month support | Implication |
| --- | --- | --- |
| Full Task 10 M2 |2010-09 through 2026-02 | Early-vintage delayed entry; exact cell gates reused |
| Reduced historical sensitivity |2006-02 through 2026-02 | Fixed subset without PMMS/HPI; never primary promotion |
| Task 12 P2 comparable experiment |2010-09 through 2026-02 | Same primary support and cutoff |
| PMMS level alone |2010-07 through 2026-03 | Does not authorize widening the P2 comparison |

Source coverage is documentary at this stage. No Fannie follow-up or events were measured.
Vintage2022 cannot generally provide60 months by2026-02; unsupported horizons are
suppressed on calendar/follow-up rules, not selected by favorable event performance.

## Candidate Vintages

Candidate note years2006/2008/2010/2014/2018/2020/2022 span pre-crisis, crisis,
post-crisis, later expansion, pandemic/refinancing and rate reversal regimes. Official
historical scope supports considering them; actual complete eligible universes remain
unverified. Freddie overlap is a comparability reason, not proof of equal populations.
2010Q1 is an acquisition partition, not a representative2010 origination population.

## Sampling Design

Draft target20000 facilities per supported vintage,140000 only if all seven pass.
First freeze acquisition-quarter coverage and static eligibility, including any term
restriction. A bounded acquisition-lag population must be explicitly named; never
claim full note-year coverage without evidence. No arbitrary lag cutoff is adopted here.

Then rank UTF-8 `salt:provider:origination_year:loan_id` by SHA256 digest ascending,
breaking ties by private ID; salt `track-b-fannie-external-v1`, provider FANNIE_PRIMARY.
Select first 20000, using no outcomes, modifications or future follow-up for sampling.
Repeated static descriptors in merged monthly rows require consistency checks and
deduplication; duplicate loan-month or conflicting facility attributes fail closed.
Shortfall requires an explicit pre-outcome amendment, not event oversampling.

A single quarter declares5.714GB uncompressed. Stream with top-k/external sorting and
private joins; do not load the entire historical population into pandas. All IDs and
selected registry records remain private. No sample has been created.

## Censoring

Use independent administrative censoring only as an unverified research assumption;
repurchases/note sales can be informative. Unknown/gap/date-conflict histories end at
the last reliable interval, without later reentry. Freeze end-of-data at2026-02 for
comparability, separately recording source performance cutoff. Payoff is a competing
event, never treated as independent censoring. Six contiguous known active observations including t0 and five earlier months,
plus shared support, create delayed entry; report the conditional population honestly.

## CIF Feasibility

Monthly first-event histories can support Aalen–Johansen and12/24/36/60-month CIFs,
conditional on contiguous follow-up, enough risk sets/events and censor survival.
Retain Task 10 gates: risk set200, censor survival0.1, cause-event minimum20.
Historical rolling PIT macro paths are not prospective paths known at entry.
Neither lifetime coverage nor independent censoring has been empirically established.

## Replication Targets

The [protocol draft](../../docs/track_b/fannie_external_replication_protocol.json)
maps all 19 predictors across M0/M1/M2/P0/P1/P2, including EXACT, TRANSFORMED and PROXY
replication classifications. No required primary predictor is documentary UNAVAILABLE;
release/population gates nevertheless prevent operational activation.

| Model | Frozen translation |
| --- | --- |
| M0/P0 | Duration bands plus vintage cohort |
| M1/P1 | Above plus score,LTV,DTI,log originalUPB,original rate,term,purpose,occupancy |
| M2 | M1 plus unchanged seven-feature PIT macro vector |
| P2 | P1 plus original-coupon REFI_POS and REFI_NEG |

Keep multinomial logistic family, solver/regularization, preprocessing, uncertainty and
proper-score-first decisions. Fit the same specifications independently on Fannie
development; external hypothesis replication does not require equal coefficients.
Development2010-09–2017-12,2018 purge, evaluation2019-01–2026-02; facility roles are
independent provider-ID hashes. Seen-vintage primary and unseen-vintage diagnostics
remain separate. Hash roles retain the frozen70/30 development/evaluation allocation.
All calendar windows refer to target risk months, with t0 one month earlier.
Evaluation is sealed and consumed once in a new namespace.

Macro success retains Task 10 joint-log-loss facility AND calendar confidence gates
with cause-Brier/calibration guardrails. P2 requires joint improvement, noninferior
payoff Brier, improved payoff calibration, coherent CIF, default guardrails and stability
using the unchanged numeric criteria in Task 12. AUC alone cannot pass. Only fixed
reduced/rate/leave-vintage and LINEAR/P3 sensitivities are planned; none can replace primary.

## Fresh-Evidence Protection

The schema checker counts delimiters and verifies encoding; it never selects,
parses or summarizes outcome fields. Coding semantics come from documentation and
synthetic fixtures. No event/count/rate or performance evidence was obtained from Fannie.
Physical bytes were necessarily read for hashing and the bounded structural check;
therefore say **outcomes not interpreted**, not literally “performance bytes never opened.”
The user authorized precisely this exception before external protocol activation.

Before broader acquisition: accept feasibility, resolve source/layout/coverage/license
gates, freeze final protocol/software/config/hashes, freeze identifier-only sample and
roles, then seal evaluation. Static decisions may not be amended after outcome viewing.
Task 10 remains NO RELIABLE TEMPORAL MACRO INCREMENT DEMONSTRATED; Task 11 remains
MACRO FAILURE MECHANISMS PARTIALLY IDENTIFIED; Task 12 remains
REFINANCING INCENTIVE HYPOTHESIS MIXED EXPLORATORILY.

## Licensed-Data Governance

Existing `/data/track_b/**` ignore rule covers private Fannie raw/interim/processed/
registry/documentation. Original archive stays in Downloads. No extraction, rename,
copy into public paths, ID join output, sample-row logs or raw records were produced.
Public evidence contains structural metadata, official semantic definitions and hashes.

The Task 13 checker rejects licensed archive/row formats in Task 13 paths and public
identifier/row-like content; regression tests verify rejection/redaction. Existing
repository governance remains intact. It checks Task 13 tracked outputs, not arbitrary
future untracked exports; run before any commit and extend deliberately with acquisition.
Future publication needs applicable-term and non-reconstruction review; hashes do not
make individual row outputs safe. Provider notices remain with private copies.

## Material Semantic Differences

The material differences are population selection, note-year versus acquisition-quarter
coverage, rounded/static suppressed values, FICO regime, provider loan-age conventions,
REO/default granularity, mixed payoff/maturity, repurchase/nonperforming sale treatment,
modification/deferral definitions, masked balances, CRT-only unavailable fields and
historical revision/knowledge-time uncertainty. Do not compress these into an arbitrary
replication-distance score. Cross-provider success would strengthen transport evidence
within two selected GSE populations, not establish universal validity or causality.

## Limitations

Only512 structural records from one quarter were checked. Full CRC, semantic field
alignment, identifiers, duplicates, dates, rates, balances, temporal ordering and code
validity are future fail-closed gates, not current empirical passes. The archive's release
and accepted terms version are unknown. Direct terms-PDF byte retrieval returned403;
official web extraction was read and no binary hash invented. No official sample loan file
was acquired. The unmatched 113/114 layout prevents freezing an operational adapter.

Acquisition coverage, FAQ/page term wording and historical disclosure time remain
unresolved. Nominal event history can support conceptual research but not regulatory
PD, borrower-level inference, timed LGD/EAD, ECL, IFRS9 or prospective macro forecasts.

## Decision

**FANNIE EXTERNAL REPLICATION REQUIRES SOURCE CLARIFICATION.**

Resolve the113-position archive's official matching release/layout, preserve 2026 changes,
document applicable accepted terms and freeze acquisition-quarter population coverage.
Then recommend **Track B Task 13A — Fannie Mae Sealed External Cohort Acquisition** for
separate authorization: finalize/seal protocol, acquire privately, select deterministic
IDs, pass source/data-quality gates and preserve sealed outcomes. Do not implement it now.

The draft status is **DRAFT_NOT_YET_AUTHORIZED**. No push is authorized.

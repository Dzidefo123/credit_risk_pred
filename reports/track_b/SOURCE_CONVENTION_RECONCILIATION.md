# Track B Task 8A — source conventions: unresolved

**Decision: SOURCE CONVENTIONS UNRESOLVED.** This is an investigation record, not an approved compatibility amendment. The original Task 8 STOP evidence remains unchanged. No new samples, performance access, models, macro acquisition, completion commit or push.

## A–B. Official identifier and file-quarter semantics

The current Release 47 July 2026 guide (pp. 7–8 and 16) and January 2026 guide (pp. 8 and 15) assign origination year/quarter to the identifier and group origination/performance files by origination quarter. Archive vintage and member quarter are enclosing cohort labels; the embedded quarter is a loan attribute. First payment is a scheduled due month and can reset for special mortgages. Exact origination or acquisition dates are not supplied in the audited origination layout. These concepts are not interchangeable. There is no evidence that the enclosing Q4 denotes acquisition/purchase quarter.

## Documentation inventory

| Source | Version/date | Location | Finding |
|---|---|---|---|
| [disclosure-changes-summary.pdf](https://www.freddiemac.com/fmac-resources/research/pdf/disclosure-changes-summary.pdf) | v1.2, June 2026, effective July 2026 Release 47 | Origination fields 13 and 20; disclosure changes | No new dot-rate enumeration or identifier-quarter exception established. |
| [file_layout.xlsx](https://www.freddiemac.com/fmac-resources/research/pdf/file_layout.xlsx) | Pre-July-2026 layout, linked with January guide; workbook has no release attestation | Origination rows 13 and 20 | Same rate and identifier formats; 32 origination fields, so do not apply this historical column map to the supplied archives. |
| [file_layout_july_2026.xlsx](https://www.freddiemac.com/fmac-resources/research/pdf/file_layout_july_2026.xlsx) | July 2026, Release 47 | Origination rows 13 and 20 | Rate Numeric - 6,3, length 6; identifier PYYQnXXXXXXX, length 12; 31 origination fields. |
| [general_user_guide_july_2026.pdf](https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf) | Release 47, July 2026 | pp. 3, 4, 6, 7-8, 16 | Identifier includes origination year/quarter; quarterly files group originations; scheduled first payment is not an origination date; no dot-rate enumeration. |
| [release-47-sample-files.zip](https://www.freddiemac.com/fmac-resources/research/docs/release-47-sample-files.zip) | Official Release 47 illustrative sample | Sample Files/origination_sample_file.txt | 1,000 origination rows; numeric original rates throughout; zero blank, dot, other nonnumeric rates. Performance sample payload not opened. |
| [release_notes.pdf](https://www.freddiemac.com/fmac-resources/research/pdf/release_notes.pdf) | Release 47, July 2026; release table July 29, 2026 | Release table and Release 47/earlier release summaries | Origination and performance cutoff March 31, 2026; no dot-rate or file-quarter exception identified in reviewed notes. |
| [user_guide.pdf](https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf) | January 2026, pre-July layout | pp. 2-3, 7-8, 15 | Same origination-quarter grouping and identifier meaning; original rate numeric literal decimal; special mortgages can have reset scheduled dates. |
| [rpl_loan_id_match_faq.pdf](https://www.freddiemac.com/fmac-resources/research/pdf/rpl_loan_id_match_faq.pdf) | Release 47, July 2026 | pp. 1-2 | Explains matching reperforming-loan identifiers; does not resolve origination rate or quarter exceptions. No RPL records acquired. |

All downloaded public reference files are privately retained and SHA-256 pinned in the companion JSON. Prior documentation means January 2026, not a release from 2006/2008. Supplied archive release identity remains structurally compatible with R47 rather than independently attested. No unofficial mirror establishes a convention. Absence of an enumeration in reviewed materials is not proof that none exists elsewhere.

## C. Complete prefix census

| Archive | Member | Embedded quarter | Count | % of member | Mismatch |
|---|---|---|---:|---:|---|
| 2006 | orig_2006Q1.txt | 2006Q1 | 298,574 | 100.000000000 | False |
| 2006 | orig_2006Q2.txt | 2006Q2 | 308,208 | 100.000000000 | False |
| 2006 | orig_2006Q3.txt | 2006Q3 | 283,975 | 100.000000000 | False |
| 2006 | orig_2006Q4.txt | 2006Q4 | 302,787 | 100.000000000 | False |
| 2008 | orig_2008Q1.txt | 2008Q1 | 421,635 | 100.000000000 | False |
| 2008 | orig_2008Q2.txt | 2008Q2 | 370,982 | 100.000000000 | False |
| 2008 | orig_2008Q3.txt | 2008Q3 | 221,752 | 100.000000000 | False |
| 2008 | orig_2008Q4.txt | 2008Q4 | 218,776 | 99.999542914 | False |
| 2008 | orig_2008Q4.txt | 2009Q1 | 1 | 0.000457086 | True |
| 2010 | orig_2010Q1.txt | 2010Q1 | 360,856 | 100.000000000 | False |
| 2010 | orig_2010Q2.txt | 2010Q2 | 363,854 | 100.000000000 | False |
| 2010 | orig_2010Q3.txt | 2010Q3 | 503,408 | 100.000000000 | False |
| 2010 | orig_2010Q4.txt | 2010Q4 | 592,072 | 100.000000000 | False |
| 2014 | orig_2014Q1.txt | 2014Q1 | 221,854 | 100.000000000 | False |
| 2014 | orig_2014Q2.txt | 2014Q2 | 290,992 | 100.000000000 | False |
| 2014 | orig_2014Q3.txt | 2014Q3 | 326,457 | 100.000000000 | False |
| 2014 | orig_2014Q4.txt | 2014Q4 | 303,696 | 100.000000000 | False |
| 2018 | orig_2018Q1.txt | 2018Q1 | 296,816 | 100.000000000 | False |
| 2018 | orig_2018Q2.txt | 2018Q2 | 364,502 | 100.000000000 | False |
| 2018 | orig_2018Q3.txt | 2018Q3 | 336,669 | 100.000000000 | False |
| 2018 | orig_2018Q4.txt | 2018Q4 | 287,447 | 100.000000000 | False |
| 2020 | orig_2020Q1.txt | 2020Q1 | 517,072 | 100.000000000 | False |
| 2020 | orig_2020Q2.txt | 2020Q2 | 926,253 | 100.000000000 | False |
| 2020 | orig_2020Q3.txt | 2020Q3 | 1,187,782 | 100.000000000 | False |
| 2020 | orig_2020Q4.txt | 2020Q4 | 1,282,633 | 99.999922035 | False |
| 2020 | orig_2020Q4.txt | 2020Q3 | 1 | 0.000077965 | True |
| 2022 | orig_2022Q1.txt | 2022Q1 | 583,706 | 100.000000000 | False |
| 2022 | orig_2022Q2.txt | 2022Q2 | 437,886 | 100.000000000 | False |
| 2022 | orig_2022Q3.txt | 2022Q3 | 336,779 | 100.000000000 | False |
| 2022 | orig_2022Q4.txt | 2022Q4 | 222,381 | 99.999550323 | False |
| 2022 | orig_2022Q4.txt | 2022Q3 | 1 | 0.000449677 | True |

All 28 origination members were streamed; 12,169,807 records were inspected. Zero wrong column counts, malformed identifier syntaxes, wrong product prefixes, and duplicate identifiers within each annual universe. This is a census of specified conventions, not a replacement full eligibility gate. All source identifiers remain unchanged and private.

### Boundary and cross-field evidence

| Member | Embedded | Count | Quarter delta | First payment | Maturity | Term | Schedule coherent |
|---|---|---:|---:|---|---|---:|---|
| 2008Q4 | 2009Q1 | 1 | 1 | 200903 | 203902 | 360 | True |
| 2020Q4 | 2020Q3 | 1 | -1 | 202012 | 205011 | 360 | True |
| 2022Q4 | 2022Q3 | 1 | -1 | 202212 | 205211 | 360 | True |

All three discrepancies are isolated adjacent-quarter cases. Maturity minus first payment plus one equals 360 in each. First payment is two months after the embedded quarter start for 2008, and five months after it for 2020/2022. A coherent schedule supports alignment; it cannot identify an exact origination date or validate the enclosing file placement. No systematic boundary population or documented release-specific exception was established.

## D. Prefix decision and exact existing gates

**PREFIX_STRICT — retained, not newly introduced.** The original implementation enforced prespecified origination-cohort integrity: `schema.parse` requires the archive year in the identifier; `study.sample` rejects `loan[4] != str(q)`; `study.retain` applies the corresponding selected-performance quarter gate. See the unchanged [schema](../../src/credit_risk/track_b/multivintage/schema.py) and [study](../../src/credit_risk/track_b/multivintage/study.py). Both guides support the cohort expectation. Neither reviewed guide documents the observed crossings as permitted.

Do not conclude that the records are corrupt merely from their strings: syntax and schedules are coherent and the provider warns that source data can contain errors. Their placement remains unexplained after documentation and cross-field review. The existing zero-unresolved-record protocol cannot be relaxed on three rare cases without an explicit compatible interpretation. No record is rewritten, excluded, replaced or sampled. The survey preserves archive vintage, member quarter, embedded quarter and mismatch separately in the matrix; these are audit evidence, not amended canonical fields.

## E–G. Original interest rate

Both layouts specify field 13 as Numeric - 6,3, length 6. The guides describe the note rate and do not enumerate dot as missing. Disclosure changes, release notes and the reviewed FAQ do not establish a dot-rate convention. The official illustrative Release 47 sample has 1,000 numeric rates and zero blank/dot/other tokens; its performance member was not opened.

| Vintage | Total | Numeric | Blank | Dot | Other nonnumeric | Numeric min | Numeric max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2006 | 1,193,544 | 1,193,543 | 0 | 1 | 0 | 3.0 | 10.97 |
| 2008 | 1,233,146 | 1,233,146 | 0 | 0 | 0 | 2.625 | 10.5 |
| 2010 | 1,820,190 | 1,820,190 | 0 | 0 | 0 | 2.375 | 7.25 |
| 2014 | 1,142,999 | 1,142,999 | 0 | 0 | 0 | 2.25 | 7.0 |
| 2018 | 1,285,434 | 1,285,434 | 0 | 0 | 0 | 0.0 | 7.0 |
| 2020 | 3,913,741 | 3,913,741 | 0 | 0 | 0 | 1.5 | 7.125 |
| 2022 | 1,580,753 | 1,580,753 | 0 | 0 | 0 | 1.75 | 9.5 |

The JSON includes exact per-vintage percentages and all 28 member rate counts/minima/maxima. Numeric zero in 2018 is reported as observed, without reinterpretation or a new gate.

2006Q3 line 107274: exactly 31 columns and 30 delimiters; dot occupies rate position 13; LTV numeric, channel recognized, amortization FRM. First payment 200609, maturity 203608, term 360: coherent schedule. No column shift is indicated. The raw dot remains in the immutable hashed source archive. No loan identifier is published.

**MALFORMED_ISOLATED_VALUE.** This classification concerns the documented numeric format, not a claim to know the provider’s intended meaning. A single aligned nonnumeric token is insufficient evidence of a recurring provider missing convention. No dot occurs in other numeric origination fields across these archives. Canonical null conversion and a null reason code are therefore not authorized. Existing blank-to-null handling remains unchanged.

## H–K. Amendment and recovery gates

No compatibility amendment frozen; no parser/harmonization version increment. Parser remains `multivintage-r47-v1.0.0`. Evidence version `source-convention-review-v1.0.0` identifies this STOP investigation only. The companion JSON is hashed in the final response. No full origination gate rerun, eligible-universe declaration, compatibility-token acceptance, recovery, new sample hash, or new performance layout finding follows an unresolved decision. Stopped vintages remain 2006/2008/2020/2022 with no frozen samples or performance retention. Completed 2010/2014/2018 checkpoints remain unchanged; their performance datasets were not rescanned.

## L–O. Event, horizon, overlap and APC support

Seven-vintage event counts, horizon support at 12/24/36/60/84/120 months, age × period coverage and APC rank diagnostics were not recomputed: that requires the prohibited new performance access. The original three-vintage evidence remains historical partial support. No new Task 8 readiness decision is made. Unrestricted age-period-cohort identification remains structurally constrained; source reconciliation does not solve it.

## P. Tests

21 new regression cases cover equal-quarter identifiers; cross-year and same-year adjacent boundary audit evidence; preservation of embedded quarter; malformed/missing/wrong-product identifiers; numeric/blank/dot/alphabetic/nonfinite rates; unchanged dot rejection and absent invented null reason; wrong column count and delimiter shift; schedule coherence without inferred origination dates; origination-only census with duplicate counting; checkpoint hash tampering and partial-checkpoint refusal; original Task 8 public-evidence preservation and completed sample/panel hash preservation. No synthetic test claims a legitimate documented crossing, because that evidence was not found.

Targeted suite: 69 passed, one intentional duplicate-ZIP warning. Full suite: 805 passed, four existing warnings, 68.07 seconds. Governance, repository hygiene and configuration checks passed. Lint and formatting passed (177 Python files checked).

## Q. Preservation

[Preservation verification](source_convention_preservation.json): 342 pre-existing public files byte-identical; 17 protected private Track A/earlier panel files byte-identical; completed cohort panel/checkpoint/sample files byte-identical; seven archive hashes unchanged; consumed ledgers checked by opaque byte hashing only, with no consumption API or outcome evaluation. Original STOP report unchanged. Track A annotated tag and peeled commit unchanged.

Frozen sample hashes remain:

| Vintage | Sample SHA-256 |
|---|---|
| 2010 | `e719a6b4d23caca8ac55eeaa6fae54da23c95887dd83b3ac22cd925d63902832` |
| 2014 | `d26b2ba3e08837f82c940ecefbaa18c4bd201d64a0a114760eafeeb80a8126f5` |
| 2018 | `b3e2952bb5bc0d0a551105cecfa964b18481b3764442edb8362245f4f17f921e` |

Retained Track A XGBoost AUC 0.868152, Brier 0.048545, log loss 0.176030 are historical results, not new evaluations. No locked holdout was consumed or reevaluated, and no frozen model artifact was regenerated.

## R–T. Decisions and next task

**SOURCE CONVENTIONS UNRESOLVED.** Original Task 8 decision remains **STOP — HARMONIZATION INVALID**; full execution did not complete and no new readiness decision is issued. No completion commit and no push.

**Exactly one next task:** obtain authoritative Freddie Mac clarification of the isolated dot-rate token and three quarter-placement discrepancies, using retained archive hashes and private record references, before reconsidering a parser amendment. This report does not authorize sending a message to the provider.

# Track B sample-expansion feasibility

## Executive Summary

**EXPANSION SUPPORT ADEQUATE**. The 20,000-loan frozen nested sample supports 246 development and 95 evaluation defaulting loans. No model fitted or expanded predictive/calibration metrics computed.

## Motivation

Task 3 had 13 development and five temporal evaluation event loans. This study expanded by identifier ranking, not by outcomes, while preserving the original baseline.

## Phase A Planning

Nested prediction: observed k stays fixed; new events ~ BetaBinomial(N-1000,k+0.5,1000-k+0.5); Jeffreys prior. Original events remain fixed. Uncertainty and conditional evaluation eligibility are included; per-selected-ID evaluation yield uses 5/1,000, not 5/130 applied to every selected loan.

| N | Overall expected [95% predictive] | Development expected [95% predictive] | Evaluation expected [95% predictive] | Joint lower probability |
|---|---|---|---|---|
| 5000 | 156.9 [113, 209] | 66.9 [39, 103] | 27.0 [11, 51] | 0.000 |
| 10000 | 314.2 [220, 425] | 134.4 [76, 210] | 54.5 [20, 106] | 0.000 |
| 20000 | 628.9 [436, 856] | 269.2 [149, 425] | 109.4 [40, 215] | 0.913 |

## Event-Support Objective

50 evaluation defaults gives illustrative rare-event CITL halfwidth ~0.28 rather than ~0.88 at five; AUC halfwidth at assumed 0.8 ~0.07 rather than ~0.23. Development 150 supplies materially more support for later modest fixed models. Neither threshold is a validity/regulatory minimum; dependence and calibration slope still require empirical validation.

## Selected Expansion Size

Exactly 20,000: the smallest considered candidate with a conservative >=90% joint probability lower bound for >=150 development and >=50 temporal evaluation event loans. Jeffreys and uniform-prior calculations were frozen before new outcomes; no resizing followed.

## Nested Sampling

Same 1,820,190 annual IDs and SHA256(salt:ID) order. First 1,000 match cryptographically and remain in first 20,000. Quarters Q1/Q2/Q3/Q4: 3,942 / 3,947 / 5,613 / 6,498. Expanded sample SHA256: `e719a6b4d23caca8ac55eeaa6fae54da23c95887dd83b3ac22cd925d63902832`. Amendment SHA256: `3a61eea974b2f2df1f49bfa5d2cbe0d5dfa77916d6d7f4c072fbe475e2d24f16`. Full algorithm, original/source hashes and Phase A test attestation are in JSON.

## Phase B Empirical Results

All 20,000 selected histories found; zero replacements, missing histories, gaps reported or conflicting duplicates. Qualifying raw default records occur on 623 facilities. Eligible positive landmarks represent 618 facilities. Five raw-default facilities never have an eligible landmark (lookback/prefix rules); they are not silently added to development.

| Cohort | Landmarks | Loans | Positive landmarks | Default loans |
|---|---:|---:|---:|---:|
| panel | 1400360 | 20000 | 7153 | 618 |
| eligible | 1255653 | 19608 | 7153 | 618 |
| primary | 1241045 | 19590 | 7153 | 618 |
| development | 478796 | 13801 | 2560 | 246 |
| evaluation | 128812 | 2423 | 1072 | 95 |
| purged_2015 | 108804 | 9881 | 503 | 86 |
| unused_other | 524633 | 11611 | 3018 | 273 |

Eligible outcome status counts: 7,153 default-positive; 1,029,959 complete event-free; 203,933 payoff/maturity; 14,238 right-censored; 370 ambiguous. Unknown status remains unknown.

## Temporal Event Support

Task 3 hash groups and monthly landmarks are unchanged. Development ends 2014-12, evaluation starts 2016-01; 2015 is purged. Development/evaluation loans do not overlap. Purged and unused row sets are disjoint, but their loan/event-loan counts overlap and must not be added. Loan-level fractions: development 246/13,801=1.78%; evaluation 95/2,423=3.92%. These are descriptive event-support fractions across landmark horizons, not new calibrated twelve-month probabilities.

## Planning vs Observed

| Quantity | Expected | 95% predictive range | Observed | Position |
|---|---:|---|---:|---|
| overall | 628.9 | [436, 856] | 623 | within |
| development | 269.2 | [149, 425] | 246 | within |
| evaluation | 109.4 | [40, 215] | 95 | within |

Overall comparison uses raw qualifying-record loans (623), matching the planning proxy. Usable incident-landmark default loans (618) are reported separately. All comparisons fall within the frozen ranges; no adaptive follow-up sample was selected.

## Resource Impact

One source-performance pass: 127,232,321 rows scanned, 1,400,360 retained. Source scan 645.0 seconds; total panel/audit run 1,180.9 seconds. Peak process working set 482.4 MiB (versus Task 2 675.9 MiB). Panel 285,888,374 bytes; retained cache 222,453,760 bytes; combined ~484.8 MiB. No full corpus extraction. ZIP materialization peak zero; SQLite journal/index-sort transient disk peak not instrumented.

Cached Git-ignored panel and selected-row spool have immutable manifests/hashes. Future research can avoid rescanning the population. Cached loading/model-matrix memory and fit runtime have not been benchmarked; do not infer them from streaming memory. Fresh full rebuilding cost was ~19.7 minutes in this environment.

## Preservation

Original source, sample, Task 2/3 evidence and Track A unchanged. All 1,000 original panel histories reproduce after type normalization; original frozen bytes remain unchanged. The machine report records hashes for all previous protected files. No loan IDs or raw rows are published. Locked holdout untouched and historical model performance not re-evaluated.

## Limitations

- Feasibility is distinct-facility event support, not proof of independence or predictive validity
- Historical operational knowledge time remains UNVERIFIED
- Same 2010 vintage and fixed calendar split; survival selection and drift persist
- Planning exchangeability/posterior assumptions and unknown borrower clustering limit probability interpretation
- Calibration stability and actual metric uncertainty require later validation; no metrics fitted here
- SQLite transient scratch peak was not measured; zero temporary_source_bytes refers only to ZIP materialization

## Decision

EXPANSION SUPPORT ADEQUATE: research event support clears the fixed 150/50 gate. This is potential for stronger validation, not established model accuracy/calibration or a regulatory minimum.

Next task: Track B Task 5 — Expanded-Cohort 12-Month PD Development and Temporal Validation. Not implemented.

Planning references: [SciPy beta-binomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html), [Hanley and McNeil (1982)](https://doi.org/10.1148/radiology.143.1.7063747).

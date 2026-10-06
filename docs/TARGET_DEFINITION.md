# Target definition and dataset provenance

Task 2 investigation: 2026-10-05. Scope: documentary provenance and descriptive training-data audit; no model fitting, calibration, locked holdout scoring or prediction inspection.

## Repository evidence first

The preserved [V1 README](history/README_v1.md) explicitly attributes the dataset to Kaggle and links Give Me Some Credit. [Data instructions](../data/README.md) retain that attribution. The Phase 1 inventory identifies the retained CSV by SHA-256 `9d863f88e2ef2190ed608c15c3dd362d1833d940e35f4f2bf22660e04b5f05cb`. Prior source inspection recorded 150,000 rows, ten predictors and 10,026 positive labels (6.684%). The local delinquency column names have been normalized; the original notebook's initial stored shape is 125,113 rows. There is no authenticated original download manifest or pristine-file comparison. Therefore identity is **repository-attributed Give Me Some Credit training data**, with corroborating schema/counts, not proof of an untouched official file.

## External evidence and its limits

The original [Kaggle competition overview](https://www.kaggle.com/c/GiveMeSomeCredit/overview/description) identifies Credit Fusion as host, cites Credit Fusion and Will Cukierski (2011), describes historical borrower data and a two-year financial-distress prediction task. Kaggle is the publication platform; the underlying supplying lender, extraction process, sampling mechanism, country/currency and observation calendar remain unverified. Competition launch dates are not borrower observation dates.

The [competition data page](https://www.kaggle.com/c/GiveMeSomeCredit/data) lists training/test CSVs and Data Dictionary.xls; access is subject to competition rules. This investigation did not accept rules, download a replacement dataset or retrieve the original dictionary workbook. Definitions were cross-checked against the [Empulse maintainer's published data-description table](https://empulse.readthedocs.io/en/stable/guide/datasets/give_me_some_credit.html#data-description) and a [published empirical study's dataset table](https://pmc.ncbi.nlm.nih.gov/articles/PMC9041569/). These are external reproductions of dataset definitions, not repository files or independently authenticated original documentation. Empulse's processed dataset has a different row count; it is not the lab's source, and its assumed costs/LGD/credit limits must not be imported as observations. No claim of pristine-file equivalence follows from matching descriptions.

## Event, horizon and observability

Source target name: **SeriousDlqin2yrs**. Published descriptions define label 1 as serious delinquency at the 90-day threshold or a worse credit outcome; label 0 denotes absence of that labeled event. Kaggle explicitly gives a **next-two-years horizon**. This combines the event definition with the competition's forward-looking task, rather than inferring a window solely from the field name. Neither description specifies detailed adjudication of “worse”, charge-off, bankruptcy, materiality, cures or repeat events.

The target is serious delinquency, **not a verified contractual/regulatory default or a loss amount**. At an assumed snapshot t0, the conceptual window is the following two years. A positive label could become knowable when an event is observed; a negative label would normally require completed follow-up under a documented labeling policy. Actual t0, outcome dates, reporting lags and completeness of individual follow-up are absent, so row-level maturity and censoring cannot be verified. Do not infer that all rows had the same collection date.

## Observation unit and assumed scoring point

The intended competition unit is a borrower/person profile, corroborated by the publisher's description and feature dictionary. The local file provides only a record index, not a stable borrower key. One physical row is one source record; one-row-per-unique-borrower, repeat borrowers and true independence cannot be established. The predictors aggregate loans/credit lines rather than provide an account-level performance table. Duplicate profiles can represent different people or repeated records; conflicting labels do not establish either explanation.

The research scoring assumption is an existing borrower financial snapshot before the prospective outcome window. It is not verified loan origination or an approval application. All historical behavior counts would need to end at or before t0. The actual snapshot timestamps are unknown. See the [feature dictionary](DATA_DICTIONARY.md) and [leakage review](../reports/data/LEAKAGE_REVIEW.md).

## Available time structure

There are **no observation, origination, account, performance-end or outcome-date columns**, no contractual maturity and no legitimate temporal ordering. Row order/source IDs are not dates. A relative outcome horizon and two predictors' stated historical lookbacks do not create a calendar or permit out-of-time validation. The 90-day-late count's reproduced definition does not specify its own lookback; do not silently assign it the same two-year lookback. Separate synthetic account histories have dates, but cannot repair the Kaggle records or prove real temporal performance.

## Consistency with current code

`data/loaders.py` and the frozen manifests preserve the inherited source two-year delinquency label; that is consistent with the competition description. Canonical aliases are documented in the dictionary. No target is relabeled and no API contract changes. Names such as PD, `pd` and `origination` are research abstractions and do not certify a regulatory event or a verified observation point. `data/targets.py` constructs distinct synthetic forward-window labels; its configurable twelve-month window does not apply to this source.

## Regulatory Interpretation

| Interpretation | Conclusion |
| --- | --- |
| Generic binary credit-risk outcome | Justified as research on the inherited label, subject to provenance/sampling limitations |
| Model-estimated delinquency probability | Best-supported wording: probability of the documented two-year source outcome, conditional on source population and model |
| Model-estimated default probability | Only an explicitly declared serious-delinquency proxy; not verified contractual default |
| 12-month PD | Unsupported: the two-year aggregate label provides no event times to reconstruct twelve-month labels |
| Lifetime PD | Unsupported: no contractual lifetimes, survival/censoring histories or full event trajectory |
| Regulatory PD | Unsupported: event equivalence, population/timing, validation and governance requirements are not established |

The [Basel IRB minimum requirements, CRE36](https://www.bis.org/committees/bcbs/basel-framework/standard/cre/36/inforce/2023-01-01/published/2022-12-08) define default through both payment-status and unlikely-to-pay criteria, with materiality/context requirements. The source's short delinquency description does not demonstrate equivalent adjudication. This is a limitation assessment, not a jurisdiction-specific compliance opinion.

A two-year probability cannot be converted into a measured twelve-month PD by division or an unverified constant-hazard formula. Such calculations would add assumptions rather than identify a new observed target. The project's strongest honest description is **credit-risk classification and calibration research using model-estimated serious-delinquency probabilities**.

## Evidence still required

Obtain an authorized download/source manifest, original versioned dictionary, supplier/extraction and sampling details, stable borrower/account keys, dated observation/outcome lineage, label adjudication, follow-up/censoring policy, units and reporting lags. These gaps remain even though competition identity and intended horizon are supported. The [suitability assessment](../reports/data/DATASET_SUITABILITY.md) explains which use cases require a different dataset.

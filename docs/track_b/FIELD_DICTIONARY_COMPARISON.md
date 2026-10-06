# Freddie Mac and Fannie Mae: Track B field dictionary comparison

Review date: 2026-10-06. Documentation only; no loan records, sample files, model fitting, dataset selection or terms acceptance. Read alongside the unchanged [77-field contract](LONGITUDINAL_DATA_CONTRACT.md) and [suitability gates](DATASET_SUITABILITY_FRAMEWORK.md).

## Scope and evidence locators

Freddie: Standard Dataset Release 47. O/P identify origination/performance positions in the official change notice. Fannie: Primary Single-Family Loan Performance column only; numbered locators refer to its shared glossary. NA in that column is not supplied data. An absence means no matching applicable field in the reviewed dictionaries, not a claim about every internal GSE system.

| Source ID | Official documentation | Version |
| --- | --- | --- |
| FM_GUIDE | [FM_GUIDE](https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf) | Release 47, July 2026 |
| FM_CHANGES | [FM_CHANGES](https://www.freddiemac.com/fmac-resources/research/pdf/disclosure-changes-summary.pdf) | v1.2, June 2026, effective July 2026 |
| FM_PAGE | [FM_PAGE](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset) | page reviewed 2026-10-06 |
| FN_DICT | [FN_DICT](https://capitalmarkets.fanniemae.com/resources/file/credit-risk/pdf/crt-file-layout-and-glossary.pdf) | 2026-09-10; applicability restricted to SF Loan Performance column |
| FN_FAQ | [FN_FAQ](https://capitalmarkets.fanniemae.com/resources/file/credit-risk/pdf/sf-loan-performance-dataset-faqs.pdf) | 2026-03-10 |
| FN_PAGE | [FN_PAGE](https://capitalmarkets.fanniemae.com/credit-risk-transfer/fannie-mae-single-family-loan-performance-data) | 2026-07-31 |



Freddie guide/change-notice hashes record the documentation bytes inspected. Fannie glossary was reviewed through official web extraction; direct retrieval returned 403, so no binary hash was invented. [Machine-readable review](field_dictionary_comparison.json) records these distinctions. No vendor PDF or loan data is redistributed.

## Classification

- **SUPPORTED**: A documented applicable field represents the named concept at its disclosed granularity, not verified record quality or full PIT readiness.

- **DERIVABLE**: Transparent future transformation or local metadata possible from documented inputs; condition/definition must be declared. No values derived in this task.

- **PARTIAL**: Relevant information exists but falls short of the field definition, precision, completeness or provenance.

- **ABSENT**: No applicable field in the reviewed public mortgage dictionaries; outside-scope CRT fields excluded.

- **UNKNOWN**: Documentation does not establish the required property; do not treat as supported.

These field classifications are different from component acceptance. SUPPORTED fields do not establish borrower identity, feature availability or completed follow-up. DERIVABLE entries describe possibilities, not performed transformations.

## Complete contract crosswalk

| Entity / field | Freddie status | Freddie locator | Fannie status | Fannie locator | Qualification |
| --- | --- | --- | --- | --- | --- |
| Lineage / source version / `dataset_id` | DERIVABLE | FM_PAGE; release namespace | DERIVABLE | FN_PAGE; release namespace | Local provider/product/release namespace, not a supplied row-level field. |
| Lineage / source version / `source_record_id` | DERIVABLE | FM_CHANGES O20/P1/P2 | DERIVABLE | FN_DICT 2/3 | Loan/month composite audit key; not a transaction or borrower identifier. |
| Lineage / source version / `available_at` | UNKNOWN | FM_GUIDE p16; FM_PAGE | UNKNOWN | FN_PAGE corrections; FN_FAQ publication lag | Public release context does not establish first availability of each historical row/attribute. |
| Lineage / source version / `effective_from` | PARTIAL | FM_CHANGES P2/P10 | PARTIAL | FN_DICT 3/45 | Reporting/event months are coarsened proxies, not complete valid-time version history. |
| Lineage / source version / `effective_to` | PARTIAL | FM_CHANGES P2/P10 | PARTIAL | FN_DICT 3/45 | Next report/termination can bound an interval only after coverage and modification checks. |
| Lineage / source version / `revision_id` | PARTIAL | FM_GUIDE p2/p17 | PARTIAL | FN_PAGE corrections | Local release/hash can be recorded; individual value revision lineage is not provided. |
| Lineage / source version / `definition_version` | DERIVABLE | FM_GUIDE Release 47; FM_CHANGES | DERIVABLE | FN_DICT dated 2026-09-10 | Pin dictionary version and declared research derivation separately; do not assume old layouts match. |
| Borrower / `borrower_id` | ABSENT | FM_GUIDE origination/performance glossary | ABSENT | FN_DICT complete SF field register | Loan identity and borrower count do not establish cross-loan obligor identity. |
| Borrower / `borrower_type` | PARTIAL | FM_CHANGES O23 | PARTIAL | FN_DICT 21 | Borrower count is not a full individual/business legal-entity classification. |
| Borrower / `relationship_start_date` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | No lender relationship-start field; first payment/acquisition are different concepts. |
| Borrower / `segment_attributes` | PARTIAL | FM_CHANGES O3/O8/O10 | PARTIAL | FN_DICT 22/26/30 | Selected mortgage attributes only; no complete demographic/business register. |
| Facility / contractual version / `facility_id` | SUPPORTED | FM_CHANGES O20/P1 | SUPPORTED | FN_DICT 2 | Loan-level key only; refinance mappings do not supply borrower identity. |
| Facility / contractual version / `borrower_id` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | No stable cross-facility borrower key. |
| Facility / contractual version / `product_type` | SUPPORTED | FM_CHANGES O16/O18; FM_GUIDE p2 | SUPPORTED | FN_DICT 28/35; FN_PAGE scope | Mortgage/product coding is documented; this comparison is Standard/Primary fixed-rate scope. |
| Facility / contractual version / `origination_date` | PARTIAL | FM_CHANGES O2/O20; FM_GUIDE p4 | SUPPORTED | FN_DICT 14 | Freddie has first-payment month and origination quarter, not an exact note-date field; Fannie supplies month-level note date. |
| Facility / contractual version / `initial_recognition_date` | UNKNOWN | FM_GUIDE acquisition coverage | UNKNOWN | FN_PAGE acquisition coverage | Accounting recognition is not established by booking/first payment or portfolio acquisition. |
| Facility / contractual version / `maturity_date` | SUPPORTED | FM_CHANGES O4/P6 | DERIVABLE | FN_DICT 3/17 | Month-level legal maturity; Fannie requires a documented calendar transformation and modification checks. |
| Facility / contractual version / `currency` | PARTIAL | FM_GUIDE dollar amount definitions | PARTIAL | FN_DICT dollar amount definitions | Monetary units are described as dollars; no row-level currency/FX register. |
| Facility / contractual version / `original_balance` | SUPPORTED | FM_CHANGES O11 | SUPPORTED | FN_DICT 10 | Booking principal is masked/rounded; not unrounded accounting exposure. |
| Facility / contractual version / `credit_limit` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | Original mortgage principal is not a revolving commitment limit. |
| Facility / contractual version / `contractual_schedule` | PARTIAL | FM_CHANGES O2/O4/O11/O13/O22 | PARTIAL | FN_DICT 8/10/13/15/17; 48/50 SF NA | Terms permit only assumption-based amortization; no full due/payment/fee ledger. |
| Facility / contractual version / `contractual_rate_terms` | PARTIAL | FM_CHANGES O13/P11 | PARTIAL | FN_DICT 8/9 | Note/current coupons do not provide every fee/reset/contract version. |
| Facility / contractual version / `effective_interest_rate` | ABSENT | FM_GUIDE rate fields | ABSENT | FN_DICT 8/9 | Note coupon is not accounting EIR. |
| Facility / contractual version / `collateral_reference` | PARTIAL | FM_CHANGES O12/O17/O18/P26 | PARTIAL | FN_DICT 20/28/31/32/53 | Collateral characteristics/proxy ratios exist, not complete valuation/priority/cash-flow lineage. |
| Facility / contractual version / `impairment_scope` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | No accounting recognition/POCI/simplified-approach adjudication. |
| Observation / snapshot / `facility_id` | SUPPORTED | FM_CHANGES P1 | SUPPORTED | FN_DICT 2 | Stable disclosed loan key, not borrower key. |
| Observation / snapshot / `observation_date` | SUPPORTED | FM_CHANGES P2; FM_GUIDE p16 | SUPPORTED | FN_DICT 3 | Reporting month is documented; not exact scoring timestamp or row availability. |
| Observation / snapshot / `credit_state` | DERIVABLE | FM_CHANGES P4/P9 | DERIVABLE | FN_DICT 40/44 | Construct declared monthly research states; termination is not automatically default. |
| Observation / snapshot / `outstanding_balance` | SUPPORTED | FM_CHANGES P3/P12/P32 | SUPPORTED | FN_DICT 12/63/108 | Defined principal balance; masking, deferral and accrued amounts require reconciliation. |
| Observation / snapshot / `available_limit` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | No revolving undrawn commitment. |
| Observation / snapshot / `days_past_due` | PARTIAL | FM_CHANGES P4 | PARTIAL | FN_DICT 40/41 | Delinquency bands/months do not identify exact integer days; missing/special codes are not current. |
| Observation / snapshot / `arrears_amount` | PARTIAL | FM_CHANGES P28 | ABSENT | FN_DICT 85 SF NA | Freddie loss-stage accrued interest is not t0 total arrears; Fannie CRT-only interest field is excluded. |
| Observation / snapshot / `payment_history` | PARTIAL | FM_CHANGES P13; balance/state series | PARTIAL | FN_DICT 41/51; 48/50 SF NA | Status/last-paid-due-month history is not a complete actual-payment/due-event ledger. |
| Observation / snapshot / `score_or_rating` | PARTIAL | FM_CHANGES O1/O31 | PARTIAL | FN_DICT 24/25/111/114; 71/113 SF NA | Origination scores are disclosed with vintage changes; a public current-score series is not established. |
| Observation / snapshot / `forbearance_status` | PARTIAL | FM_CHANGES P8/P25/P30 | PARTIAL | FN_DICT 42/102/106 | Selected assistance/modification measures; time bounds and full source-policy coverage remain limited. |
| Observation / snapshot / `qualitative_risk_flags` | PARTIAL | FM_CHANGES P7/P29/P30 | PARTIAL | FN_DICT 102/106 | Selected flags do not constitute the full qualitative-risk adjudication required by a staging policy. |
| Default episode / impairment event / `default_episode_id` | DERIVABLE | FM_CHANGES P1/P4/P9 | DERIVABLE | FN_DICT 2/40/44 | A local episode key requires declared research entry/cure rules; no supplied bank episode identifier. |
| Default episode / impairment event / `event_entity_id` | SUPPORTED | FM_CHANGES P1 | SUPPORTED | FN_DICT 2 | Supported at facility scope only; borrower contagion cannot be identified. |
| Default episode / impairment event / `default_date` | DERIVABLE | FM_CHANGES P2/P4/P9/P10 | DERIVABLE | FN_DICT 3/40/44/45 | First qualifying month under a separately declared proxy; not a supplied universal default date. |
| Default episode / impairment event / `default_definition_id` | DERIVABLE | FM_GUIDE states; local policy | DERIVABLE | FN_DICT state codes; local policy | Pin an explicit research-definition identifier; do not imply regulatory equivalence. |
| Default episode / impairment event / `default_derivation` | DERIVABLE | FM_CHANGES state/event inputs | DERIVABLE | FN_DICT state/event inputs | Versioned rule and lineage can be designed later; no labels calculated here. |
| Default episode / impairment event / `default_reason` | PARTIAL | FM_CHANGES P4/P9 | PARTIAL | FN_DICT 40/44 | Delinquency/exit codes are evidence, not complete default-reason adjudication. |
| Default episode / impairment event / `credit_impaired_status` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | Research delinquency/REO is not accounting credit-impaired status. |
| Default episode / impairment event / `cure_date` | PARTIAL | FM_CHANGES P2/P4 | PARTIAL | FN_DICT 3/40/41 | Return-to-current can proxy a transition; institutional cure/probation rules are unavailable. |
| Default episode / impairment event / `write_off_date` | PARTIAL | FM_CHANGES P9/P10 | PARTIAL | FN_DICT 80/44/45 | Exit/amount-reporting month does not isolate a verified write-off occurrence date. |
| Default episode / impairment event / `exposure_at_default` | PARTIAL | FM_CHANGES P3/P27/P28 | PARTIAL | FN_DICT 12/46; 85 SF NA | Removal exposure is not necessarily first-default exposure; reconcile amount/event timing and masking. |
| Recovery / workout cash flow / `cash_flow_id` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | No individual transaction/reversal identifiers. |
| Recovery / workout cash flow / `default_episode_id` | DERIVABLE | FM_CHANGES P1/P4/P9 | DERIVABLE | FN_DICT 2/40/44 | Link aggregate loss records to declared research episode only with consistent rules. |
| Recovery / workout cash flow / `cash_flow_date` | PARTIAL | FM_GUIDE p17; FM_CHANGES P10 | PARTIAL | FN_DICT 53/54:62 | Disposition/reporting dates are proxies, not dates of each recovery/cost transaction. |
| Recovery / workout cash flow / `cash_flow_amount` | PARTIAL | FM_CHANGES P14:21/P35 | PARTIAL | FN_DICT 54:62 | Aggregate loss components, not complete timed cash-flow ledger. |
| Recovery / workout cash flow / `cash_flow_type` | PARTIAL | FM_CHANGES P14:21/P35 | PARTIAL | FN_DICT 54:62 | Aggregate categories can be mapped; no per-transaction type/reversal trail. |
| Recovery / workout cash flow / `cash_flow_currency` | PARTIAL | FM_GUIDE dollar units | PARTIAL | FN_DICT dollar units | No individual currency/FX transaction register. |
| Recovery / workout cash flow / `workout_end_date` | PARTIAL | FM_GUIDE p17 trailing updates | PARTIAL | FN_PAGE corrections; FN_DICT 53 | Disposition is not proof ultimate recovery is complete; later updates require frozen-release review. |
| Recovery / workout cash flow / `discount_basis` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | Accounting discount convention and actual recovery timing are not supplied. |
| Coverage / exit / `entity_id` | SUPPORTED | FM_CHANGES P1 | SUPPORTED | FN_DICT 2 | Loan-scope coverage key, not obligor identity. |
| Coverage / exit / `coverage_start_date` | DERIVABLE | FM_GUIDE acquisition start; P2 | DERIVABLE | FN_PAGE acquisition start; FN_DICT 3 | Earliest observed month identifies dataset entry, not origination or continuous pre-entry history. |
| Coverage / exit / `followup_end_date` | PARTIAL | FM_GUIDE cutoff/termination; P2 | PARTIAL | FN_PAGE cutoff; FN_DICT 3/45 | Last month/release cutoff is a coverage proxy; record-level gap/completeness checks remain unperformed. |
| Coverage / exit / `exit_date` | SUPPORTED | FM_CHANGES P10 | SUPPORTED | FN_DICT 45 | Coarsened termination month; event definition separate from default. |
| Coverage / exit / `exit_reason` | SUPPORTED | FM_CHANGES P9 | SUPPORTED | FN_DICT 44 | Decode each provider's current codebook; equal numeric codes need not mean identical events. |
| Coverage / exit / `coverage_gaps` | DERIVABLE | FM_CHANGES P1/P2 | DERIVABLE | FN_DICT 2/3 | Missing expected loan-months can be flagged later; true event-ascertainment gaps remain unknown. |
| Origination risk / policy reference / `facility_id` | SUPPORTED | FM_CHANGES O20/P1 | SUPPORTED | FN_DICT 2 | Loan identity with scope-specific refinance mapping. |
| Origination risk / policy reference / `initial_risk_reference` | PARTIAL | FM_CHANGES O1/O31 | PARTIAL | FN_DICT 24/25/111/114 | Origination score is not a complete archived recognition-time PD term structure. |
| Origination risk / policy reference / `current_risk_reference` | PARTIAL | FM_CHANGES P4/P30 | PARTIAL | FN_DICT 40/102; 71/113 SF NA | Current states provide research inputs, not a comparable current probability/rating series. |
| Origination risk / policy reference / `sicr_policy_version` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | Institution/accounting SICR policy not disclosed; a future heuristic must be labeled research. |
| Origination risk / policy reference / `stage_reason_record` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | No accounting stage decisions, reasons or overrides. |
| Macroeconomic observation / release vintage / `series_id` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | Requires separate macro provider/version, not mortgage field. |
| Macroeconomic observation / release vintage / `geography` | PARTIAL | FM_CHANGES O5/O17/O19 | PARTIAL | FN_DICT 31/32/33 | Property geography supports a coarse regional macro link; no macro observation itself. |
| Macroeconomic observation / release vintage / `reference_period` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | Loan reporting month is not an external economic-series reference period. |
| Macroeconomic observation / release vintage / `release_date` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | No macro release calendar in loan files. |
| Macroeconomic observation / release vintage / `macro_vintage_id` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | No historical macro revision archive. |
| Macroeconomic observation / release vintage / `macro_value` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | No unemployment/GDP/rate/house-price series. |
| Scenario forecast / valuation run / `scenario_id` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | External justified scenario design/source required. |
| Scenario forecast / valuation run / `forecast_as_of` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | No historical forecast issuance record. |
| Scenario forecast / valuation run / `forecast_period` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | Contractual maturity is not a macro forecast horizon. |
| Scenario forecast / valuation run / `forecast_value` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | No baseline/upside/downside macro paths. |
| Scenario forecast / valuation run / `scenario_weight` | ABSENT | FM_GUIDE glossary | ABSENT | FN_DICT SF register | No scenario probability weights; do not invent them. |
| Scenario forecast / valuation run / `valuation_horizon_basis` | PARTIAL | FM_CHANGES O4/P6/P9 | PARTIAL | FN_DICT 13/17/44 | Legal term/exit evidence supports mortgage-life research, not complete accounting/behavioral-life policy. |



## Interpretation boundary

Monthly delinquency permits a declared research event definition, not a supplied regulatory default. Coarsened months are not exact days. Aggregate proceeds/costs are not a dated transaction ledger. Review conclusions are provisional until authorized record-level quality, coverage and vintage inspection.

See the [comparative assessment](DATASET_COMPARATIVE_ASSESSMENT.md) for material gaps and next decision. No formal development-source selection is made here.

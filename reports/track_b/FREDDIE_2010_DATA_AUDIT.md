# Freddie 2010 empirical longitudinal audit

## Decision and evidence boundaries

**PROCEED WITH CONDITIONS** for mortgage PD/survival cohort design. No models, calibration, LGD/EAD estimates or ECL were calculated. Historical feature availability remains unverified in this revised release.

The [initial stop](FREDDIE_2010_INITIAL_PREFLIGHT_STOP.json) is preserved. The [annual amendment](../../docs/track_b/ANNUAL_BUNDLE_ACQUISITION_AMENDMENT.md) changes only input frame/resources. The original scientific rules remain fixed.

## Source, sampling and scan

Original ZIP SHA-256: `a82bc0f1efffccfb494b2b33f61877428bb4a6443c1a73d44bc7ee24d77aa45d`. No extraction or source copy.

Annual universe: **1,820,190 IDs**; selected **1000** before performance access. No quotas or replacements.

Salt: `freddie_mortgage_research_v1`; sample-set SHA-256: `b40de7596b0d4fbb189c1ceec5176f4461f7d290117250c1a7ec15b605ce7b11`.

| Quarter | Orig IDs | Selected | Perf scanned | Retained |
| --- | --- | --- | --- | --- |
| 1 | 360856 | 195 | 23293531 | 13961 |
| 2 | 363854 | 191 | 23119788 | 12073 |
| 3 | 503408 | 278 | 35128486 | 20508 |
| 4 | 592072 | 336 | 45690516 | 25690 |

Total performance scanned: **127,232,321**; retained: **72,232** (0.05677% hit rate). Nonselected attributes discarded after ID. Selected records passed typed parsing; nonselected attributes were not fully validated. Zero unmatched IDs.

## Longitudinal integrity

Panel landmarks: **72,232**; date range 2010-01 to 2026-03.

| Integrity/availability measure | Value |
| --- | --- |
| duplicate_loan_month_rate | 0.0 |
| gap_interval_count | 0 |
| post_terminal_unexplained_rows | 0 |
| selected_loans_with_history | 1000 |
| selected_loans_without_history | 0 |
| eligible_principal_missing | 0 |
| eligible_principal_zero | 0 |
| all_observation_principal_zero | 936 |

| History-length quantile | Months |
| --- | --- |
| 0.0 | 2.0 |
| 0.25 | 28.0 |
| 0.5 | 58.0 |
| 0.75 | 108.25 |
| 1.0 | 194.0 |

No duplicates, gaps or unexpected post-terminal rows were observed in selected histories. This is not a guarantee of borrower independence or full-source cleanliness.

## Twelve-month follow-up

Eligible landmarks: **65,064**; ineligible: **7,168**. All are retained. Overlapping landmarks are not independent defaults; denominator is all eligible t0 rows.

| Protocol status | Count | Eligible fraction |
| --- | --- | --- |
| ambiguous_event_order | 52 | 0.0799% |
| competing_payoff | 10288 | 15.8121% |
| negative_survived_horizon | 53648 | 82.4542% |
| positive_default | 368 | 0.5656% |
| right_censored | 708 | 1.0882% |

| First observed loan endpoint | Loans |
| --- | --- |
| active_or_unknown | 58 |
| administrative | 1 |
| ambiguous | 5 |
| default | 31 |
| payoff | 905 |

| Qualifying-record loan count | Loans |
| --- | --- |
| selected_loans_with_observed_qualifying_record | 31 |

Payoff/maturity cannot be separated into voluntary prepayment alone. Ambiguous windows remain unknown. Censoring was not converted to a negative.

## Complete-follow-up selection effect

Physical 12-month record coverage retains **54,786** of **65,064** landmarks and excludes **10,278 (15.80%)**.

| Group | Landmarks | Loans | Median age | Median principal |
| --- | --- | --- | --- | --- |
| all_eligible | 65064 | 983 | 48.0 | 128728.81 |
| physical_full12 | 54786 | 902 | 47.0 | 126977.62 |
| not_physical_full12 | 10278 | 960 | 54.0 | 139307.575 |

| Excluded protocol status | Count |
| --- | --- |
| competing_payoff | 9480 |
| right_censored | 707 |
| positive_default | 43 |
| ambiguous_event_order | 48 |

This restriction removes early payoff/default/censoring cases and changes the population. It is not an approved modeling filter. Physical presence is not event-free survival or proof of genuine information availability.

| Year | Eligible | Protocol status counts |
| --- | --- | --- |
| 2010 | 1254 | competing_payoff: 126; negative_survived_horizon: 1122; positive_default: 6 |
| 2011 | 9631 | ambiguous_event_order: 4; competing_payoff: 1459; negative_survived_horizon: 8114; positive_default: 46; right_censored: 8 |
| 2012 | 9731 | competing_payoff: 2072; negative_survived_horizon: 7590; positive_default: 65; right_censored: 4 |
| 2013 | 7595 | competing_payoff: 873; negative_survived_horizon: 6672; positive_default: 50 |
| 2014 | 6679 | competing_payoff: 875; negative_survived_horizon: 5776; positive_default: 28 |
| 2015 | 5783 | competing_payoff: 924; negative_survived_horizon: 4834; positive_default: 25 |
| 2016 | 4851 | competing_payoff: 764; negative_survived_horizon: 4071; positive_default: 16 |
| 2017 | 4071 | ambiguous_event_order: 5; competing_payoff: 612; negative_survived_horizon: 3449; positive_default: 5 |
| 2018 | 3449 | ambiguous_event_order: 7; competing_payoff: 414; negative_survived_horizon: 3009; positive_default: 19 |
| 2019 | 3009 | competing_payoff: 521; negative_survived_horizon: 2438; positive_default: 50 |
| 2020 | 2438 | competing_payoff: 612; negative_survived_horizon: 1780; positive_default: 46 |
| 2021 | 1780 | competing_payoff: 364; negative_survived_horizon: 1416 |
| 2022 | 1416 | competing_payoff: 192; negative_survived_horizon: 1220; positive_default: 4 |
| 2023 | 1220 | competing_payoff: 125; negative_survived_horizon: 1087; positive_default: 8 |
| 2024 | 1087 | ambiguous_event_order: 7; competing_payoff: 184; negative_survived_horizon: 896 |
| 2025 | 896 | ambiguous_event_order: 29; competing_payoff: 171; negative_survived_horizon: 174; right_censored: 522 |
| 2026 | 174 | right_censored: 174 |

## Exposure, missingness and losses

| Eligible principal statistic | Value |
| --- | --- |
| available | 65064 |
| max | 710000.0 |
| median | 128728.81 |
| min | 288.33 |

Monthly principal is supported only as the approved proxy, not accounting/regulatory EAD. Terminal zero balances do not establish zero EAD at first default.

| Origination field | Missing selected loans |
| --- | --- |
| amortization_type | 0 |
| channel | 0 |
| first_payment_month | 0 |
| first_time_buyer | 0 |
| harp_indicator | 0 |
| interest_only | 0 |
| loan_id | 0 |
| loan_purpose | 0 |
| maturity_month | 0 |
| mi_percentage | 0 |
| msa | 185 |
| number_of_borrowers | 0 |
| number_units | 0 |
| occupancy_status | 0 |
| orig_cltv | 0 |
| orig_credit_score | 0 |
| orig_dti | 289 |
| orig_interest_rate | 0 |
| orig_ltv | 0 |
| orig_upb | 0 |
| original_loan_term | 0 |
| postal_prefix | 0 |
| pre_harp_loan_id | 711 |
| prepayment_penalty | 0 |
| property_state | 0 |
| property_type | 0 |
| seller_name | 0 |
| special_program | 1000 |
| super_conforming | 0 |
| valuation_method | 0 |
| vantage_score | 1000 |

| Aggregate loss field | Available observations | Min | Median | Max |
| --- | --- | --- | --- | --- |
| actual_loss | 8 | -1339.03 | 49472.235 | 149346.08 |
| bankruptcy_cramdown_costs | 413 | 0.0 | 0.0 | 0.0 |
| delinquent_accrued_interest | 8 | 4734.27 | 14250.414999999999 | 21855.3 |
| legal_costs | 8 | 0.0 | 2382.75 | 5968.15 |
| mi_recoveries | 8 | -27771.31 | 0.0 | 0.0 |
| misc_expenses | 8 | -1339.03 | 445.0 | 1341.45 |
| net_sale_proceeds | 8 | -301081.95 | -154638.99 | -74133.73 |
| non_mi_recoveries | 8 | -3020.61 | -880.325 | 0.0 |
| preservation_costs | 8 | 0.0 | 51.25 | 12135.17 |
| removal_upb | 936 | 288.33 | 146489.47 | 688566.39 |
| taxes_insurance | 8 | 0.0 | 3506.3050000000003 | 13947.75 |
| total_expenses | 8 | -1339.03 | 4897.305 | 28957.3 |

Very sparse actual-loss disclosures support inspection, not an LGD model. Signed gains and net expense credits are retained; no clipping or silent normalization. No transaction timing or ultimate-workout completion is recovered from aggregates. Blank/no-plan conventions in assistance/modification flags need explicit encoding review.

## Protocol verification and empirical gates

| Assumption | Assessment | Qualification |
| --- | --- | --- |
| annual_frame_and_identifier_sampling | CONFIRMED | No outcome-based selection |
| pinned_31_35_column_structure | CONFIRMED WITH QUALIFICATION | Typed selected rows and structural probes; not all unselected attributes |
| facility_identity_linkage | CONFIRMED WITH QUALIFICATION | Not borrower identity |
| monthly_time_and_delinquency_bands | CONFIRMED WITH QUALIFICATION | No precise daily or knowledge time |
| historical_feature_availability | NOT TESTABLE | Revised release only |
| first_payment_equals_origination | NOT TESTABLE | Inference explicitly prohibited |
| all_termination_dates_align_with_report_month | CONTRADICTED | Ambiguous records quarantined, no date reconciliation |
| payoff_maturity_combined_cause | CONFIRMED WITH QUALIFICATION | Cannot isolate voluntary prepayment |
| monthly_principal_exposure_proxy | CONFIRMED WITH QUALIFICATION | Not accounting EAD |
| recovery_gain_signs | CONFIRMED WITH QUALIFICATION | Signed values retained; expense credit requires review |
| timed_workout_ledger | NOT TESTABLE | No dated transaction-level recoveries |

| Gate | Assessment |
| --- | --- |
| A | PASS with release-vintage and acquisition-time qualifications |
| B | PASS pinned mapping, no column shifts |
| C | PASS source ID linkage; no borrower independence |
| D | PASS monthly chronology; knowledge time unverified |
| E | CONDITIONAL: ambiguous windows retained as unknown |
| F | PASS statuses distinguish censored/competing/event-free |
| G | PASS nominal-time firewall; historical PIT unverified |
| H | PASS monthly principal proxy only |
| LGD | LIMITED: sparse actual-loss observations, no timed workouts |

## Resources and manifests

| Engineering measure | Value |
| --- | --- |
| elapsed_scan_seconds | 465.125 |
| full_text_extracted | False |
| non_selected_attributes_retained | False |
| peak_memory_bytes | 708743168 |
| temporary_bytes_peak | 0 |
| temporary_files_created | 0 |

Private panel/IDs/manifests remain Git-ignored. Source, sample, protocol, amendment and transformation hashes permit reproduction. Reporter did not mutate the panel.

## Next task

**Track B Task 3 - Cohort Design and 12-Month PD Baseline.** Freeze calendar/entity splits and censoring/competing-risk treatment first. The same 1,000 IDs remain fixed despite sparse defaults/loss observations. No modeling implemented here; Track A remains unchanged.

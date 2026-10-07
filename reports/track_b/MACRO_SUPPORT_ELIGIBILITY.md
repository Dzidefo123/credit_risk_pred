# Track B Task9A — Macro-Support Eligibility


## Executive Summary


MACRO ELIGIBILITY DESIGN READY WITH MATERIAL LIMITATIONS.

| Design | Calendar window | Months | Facilities | Intervals | Defaults | Payoffs |
| --- | --- | --- | --- | --- | --- | --- |
| PRIMARY | 2010-09 through 2026-02 (186 months) | 186 | 119629 | 5541179 | 5617 | 76729 |
| REDUCED | 2006-02 through 2026-02 (241 months) | 241 | 136389 | 6498466 | 8104 | 90946 |



## Task 9 Boundary


PIT MACRO SUPPORT INSUFFICIENT for original full window; unchanged. Successful frozen acquisition and all earlier failed/partial audits remain unchanged. This task intersects support with unchanged mortgage risk intervals; it does not declare Task9 complete. No API reacquisition or credential access.

Frozen source run: `task9_api_v5`; 20,009 numeric version/validity segments, not revision-event counts. Archive cutoff `2026-03-31`; reporting boundary `2026-03`. Retrieval times are frozen in the specification and JSON evidence.


## Full Feature Support


| Feature | Source | Native frequency | First eligible risk month | Last | Months |
| --- | --- | --- | --- | --- | --- |
| unemployment_level | UNRATE | M | 2006-01 | 2026-03 | 243 |
| unemployment_change_3m | UNRATE | M | 2006-01 | 2026-02 | 242 |
| treasury_10y_level | DGS10 | D | 2006-02 | 2026-03 | 242 |
| mortgage_30y_level | MORTGAGE30US | W | 2010-07 | 2026-03 | 189 |
| mortgage_treasury_spread | MORTGAGE30US | W | 2010-07 | 2026-03 | 189 |
| hpi_yoy | USSTHPI | Q | 2010-09 | 2026-03 | 187 |
| cpi_yoy | CPIAUCSL | M | 2006-01 | 2026-03 | 243 |
| gdp_qoq | GDPC1 | Q | 2006-01 | 2026-03 | 243 |

Exact transformations, operands, units and assessment-date limits are in the JSON definitions and unchanged [feature registry](../../docs/track_b/pit_macro_feature_registry.json). Risk month m uses end-of-month m−1 information; all required operands must be known then, with completed reference periods. Archive validity is a conservative knowledge bound, not an independently certified exact original release timestamp. Frozen freshness caps: UNRATE/CPI 62 days; Treasury 7; PMMS 14; HPI/GDP 183. Exact lag operands are not subject to current-value freshness, but must share the t0 information/metadata regime.


## Coverage Bottlenecks


| Feature | Unavailable risk months, out of 243 |
| --- | --- |
| unemployment_level | 0 |
| unemployment_change_3m | 1 |
| treasury_10y_level | 1 |
| mortgage_30y_level | 54 |
| mortgage_treasury_spread | 54 |
| hpi_yoy | 56 |
| cpi_yoy | 0 |
| gdp_qoq | 0 |

These counts overlap. The JSON contains all 243 × 8 cells, original selector status and deterministic reasons. Pre2010 PMMS/HPI archive support prevents full coverage. January2006 Treasury exceeds the frozen freshness cap. March2026 requires October2025 unemployment as the exact three-month-change operand of January2026; the raw API has a missing token. [BLS confirms October2025 CPS observations were not collected](https://www.bls.gov/cps/methods/2025-federal-government-shutdown-impact-cps.htm). No interpolation or future revision is admitted. Missing source observation, missing operand, unreleased/unarchived version, insufficient provenance and stale current value remain separate machine-readable reasons.


## Primary Common-Support Design


2010-09 through 2026-02 (186 months)

All eight features must be available for every eligible interval. All 140,000 sampled facilities remain in the registry. Early-vintage loans still active later may enter with delayed entry; no synthetic pre-entry exposure.


## Reduced Historical Sensitivity


2006-02 through 2026-02 (241 months). Features: unemployment_level, unemployment_change_3m, treasury_10y_level, cpi_yoy, gdp_qoq

One fixed sensitivity excludes PMMS level, mortgage/Treasury spread and HPI YoY because their dated housing archives restrict early coverage. It retains labor-market, yield, inflation and output information. Economic and provenance rationale was frozen before counts; no performance search or promotion to primary is permitted.


## Historical Archive Investigation


The [FHFA 2008Q2 release](https://www.fhfa.gov/reports/house-price-index/2008/Q2), published 26August2008, contains a same-publication national all-transactions pair: 2007Q2=387.45 and 2008Q2=380.82 (1980Q1=100), printed page52. This isolated pair is PIT_RECONSTRUCTED_AUTHORITATIVELY; it is not admitted into frozen features. It does not establish a complete release chain or semantic bridge to the frozen USSTHPI versions. Original PDF bytes are retained privately with SHA256 `eb091fa542152d6449e17264156117faeb5b9a92920742173279524f5315d8a7`. The [official PMMS archive](https://www.freddiemac.com/pmms/archive) offers compiled history, but the bounded investigation did not establish complete pre2010 publication/revision provenance. Its classification is HISTORICAL_VALUE_KNOWN_BUT_RELEASE_PROVENANCE_INSUFFICIENT. This is not proof no earlier releases exist. Complete extension: UNAVAILABLE; no current revised backfill or fixed-lag assumptions.


## Mortgage Eligibility


| Design | Vintage | Retained | Contributing | Intervals | First | Last |
| --- | --- | --- | --- | --- | --- | --- |
| PRIMARY | 2006 | 20000 | 9210 | 378206 | 2010-09 | 2026-02 |
| PRIMARY | 2008 | 20000 | 12558 | 477041 | 2010-09 | 2026-02 |
| PRIMARY | 2010 | 20000 | 19593 | 1251957 | 2010-09 | 2026-02 |
| PRIMARY | 2014 | 20000 | 19574 | 1197706 | 2014-07 | 2026-02 |
| PRIMARY | 2018 | 20000 | 19626 | 655052 | 2018-07 | 2026-02 |
| PRIMARY | 2020 | 20000 | 19434 | 890432 | 2020-07 | 2026-02 |
| PRIMARY | 2022 | 20000 | 19634 | 690785 | 2022-07 | 2026-02 |
| REDUCED | 2006 | 20000 | 19430 | 1011241 | 2006-07 | 2026-02 |
| REDUCED | 2008 | 20000 | 19085 | 800002 | 2008-07 | 2026-02 |
| REDUCED | 2010 | 20000 | 19606 | 1253248 | 2010-07 | 2026-02 |
| REDUCED | 2014 | 20000 | 19574 | 1197706 | 2014-07 | 2026-02 |
| REDUCED | 2018 | 20000 | 19626 | 655052 | 2018-07 | 2026-02 |
| REDUCED | 2020 | 20000 | 19434 | 890432 | 2020-07 | 2026-02 |
| REDUCED | 2022 | 20000 | 19634 | 690785 | 2022-07 | 2026-02 |

Intervals are (currentmonth, nextmonth], both in the unchanged contiguous analytical prefix. Six known pre-t0 months, current active none state, consecutive target and nonnegative scheduled-first-payment proxy age are required. No reentry after an endpoint, gap or unknown prefix. The original 7,741,663 rows are read only; no samples redrawn or facilities deleted.


## Event Support


| Design | Vintage | Unique defaults | Unique payoffs | Censored | Exit reasons |
| --- | --- | --- | --- | --- | --- |
| PRIMARY | 2006 | 1193 | 7684 | 333 | {'payoff': 7684, 'default': 1193, 'MACRO_SUPPORT_END': 297, 'AMBIGUOUS_CENSOR': 18, 'ADMINISTRATIVE_CENSOR': 18} |
| PRIMARY | 2008 | 1088 | 11081 | 389 | {'payoff': 11081, 'MACRO_SUPPORT_END': 345, 'default': 1088, 'AMBIGUOUS_CENSOR': 24, 'ADMINISTRATIVE_CENSOR': 20} |
| PRIMARY | 2010 | 618 | 17740 | 1235 | {'payoff': 17740, 'MACRO_SUPPORT_END': 1179, 'default': 618, 'AMBIGUOUS_CENSOR': 30, 'ADMINISTRATIVE_CENSOR': 26} |
| PRIMARY | 2014 | 792 | 15587 | 3195 | {'payoff': 15587, 'MACRO_SUPPORT_END': 3146, 'default': 792, 'ADMINISTRATIVE_CENSOR': 22, 'AMBIGUOUS_CENSOR': 27} |
| PRIMARY | 2018 | 1044 | 15118 | 3464 | {'payoff': 15118, 'MACRO_SUPPORT_END': 3428, 'default': 1044, 'AMBIGUOUS_CENSOR': 7, 'ADMINISTRATIVE_CENSOR': 29} |
| PRIMARY | 2020 | 248 | 6416 | 12770 | {'MACRO_SUPPORT_END': 12737, 'payoff': 6416, 'default': 248, 'ADMINISTRATIVE_CENSOR': 17, 'AMBIGUOUS_CENSOR': 16} |
| PRIMARY | 2022 | 634 | 3103 | 15897 | {'MACRO_SUPPORT_END': 15815, 'payoff': 3103, 'default': 634, 'AMBIGUOUS_CENSOR': 35, 'ADMINISTRATIVE_CENSOR': 47} |
| REDUCED | 2006 | 2854 | 16208 | 368 | {'payoff': 16208, 'default': 2854, 'MACRO_SUPPORT_END': 297, 'ADMINISTRATIVE_CENSOR': 29, 'AMBIGUOUS_CENSOR': 42} |
| REDUCED | 2008 | 1914 | 16761 | 410 | {'payoff': 16761, 'default': 1914, 'MACRO_SUPPORT_END': 345, 'AMBIGUOUS_CENSOR': 34, 'ADMINISTRATIVE_CENSOR': 31} |
| REDUCED | 2010 | 618 | 17753 | 1235 | {'payoff': 17753, 'MACRO_SUPPORT_END': 1179, 'default': 618, 'AMBIGUOUS_CENSOR': 30, 'ADMINISTRATIVE_CENSOR': 26} |
| REDUCED | 2014 | 792 | 15587 | 3195 | {'payoff': 15587, 'MACRO_SUPPORT_END': 3146, 'default': 792, 'ADMINISTRATIVE_CENSOR': 22, 'AMBIGUOUS_CENSOR': 27} |
| REDUCED | 2018 | 1044 | 15118 | 3464 | {'payoff': 15118, 'MACRO_SUPPORT_END': 3428, 'default': 1044, 'AMBIGUOUS_CENSOR': 7, 'ADMINISTRATIVE_CENSOR': 29} |
| REDUCED | 2020 | 248 | 6416 | 12770 | {'MACRO_SUPPORT_END': 12737, 'payoff': 6416, 'default': 248, 'ADMINISTRATIVE_CENSOR': 17, 'AMBIGUOUS_CENSOR': 16} |
| REDUCED | 2022 | 634 | 3103 | 15897 | {'MACRO_SUPPORT_END': 15815, 'payoff': 3103, 'default': 634, 'AMBIGUOUS_CENSOR': 35, 'ADMINISTRATIVE_CENSOR': 47} |

Count one first endpoint per facility and design; no overlapping-landmark event multiplication. Default uses unchanged delinquency≥3/REO or credit termination02/03/09 research semantics; payoff01 includes maturity. Administrative15/16/96, ambiguous termination timing and unknown states censor before unascertainable target intervals. Administrative and other censor reasons are reported separately; count totals need not match raw Task8 endpoints because earlier failures/lookback are excluded.


## Duration Support


Both proxy duration and provider age are reported separately per vintage in JSON. Primary future clock: completed scheduled first-payment proxy months at t0; bands0–12,13–24,25–36,37–60,61–84,85–120,121–180,181–240,241+. No outcome-selected knots.

| Design | Vintage | Proxy age: eligible interval counts |
| --- | --- | --- |
| PRIMARY | 2006 | {'37-60': 106241, '61-84': 115974, '85-120': 77589, '121-180': 58209, '181-240': 19956, '25-36': 131, '13-24': 82, '0-12': 24} |
| PRIMARY | 2008 | {'25-36': 111493, '37-60': 153993, '61-84': 71915, '85-120': 60627, '121-180': 47369, '181-240': 11350, '13-24': 20237, '0-12': 57} |
| PRIMARY | 2010 | {'0-12': 166994, '13-24': 200016, '25-36': 155717, '37-60': 242930, '61-84': 175308, '85-120': 179728, '121-180': 127332, '181-240': 3932} |
| PRIMARY | 2014 | {'0-12': 163143, '13-24': 198994, '25-36': 168284, '37-60': 276192, '61-84': 185428, '85-120': 152185, '121-180': 53480} |
| PRIMARY | 2018 | {'0-12': 162926, '13-24': 169119, '25-36': 97863, '37-60': 115286, '61-84': 92566, '85-120': 17292} |
| PRIMARY | 2020 | {'0-12': 163427, '13-24': 188670, '25-36': 175238, '37-60': 324197, '61-84': 38900} |
| PRIMARY | 2022 | {'0-12': 172563, '13-24': 219942, '25-36': 204072, '37-60': 94208} |
| REDUCED | 2006 | {'0-12': 159535, '13-24': 199801, '25-36': 164470, '37-60': 215707, '61-84': 115974, '85-120': 77589, '121-180': 58209, '181-240': 19956} |
| REDUCED | 2008 | {'0-12': 157312, '13-24': 170927, '25-36': 126509, '37-60': 153993, '61-84': 71915, '85-120': 60627, '121-180': 47369, '181-240': 11350} |
| REDUCED | 2010 | {'0-12': 168285, '13-24': 200016, '25-36': 155717, '37-60': 242930, '61-84': 175308, '85-120': 179728, '121-180': 127332, '181-240': 3932} |
| REDUCED | 2014 | {'0-12': 163143, '13-24': 198994, '25-36': 168284, '37-60': 276192, '61-84': 185428, '85-120': 152185, '121-180': 53480} |
| REDUCED | 2018 | {'0-12': 162926, '13-24': 169119, '25-36': 97863, '37-60': 115286, '61-84': 92566, '85-120': 17292} |
| REDUCED | 2020 | {'0-12': 163427, '13-24': 188670, '25-36': 175238, '37-60': 324197, '61-84': 38900} |
| REDUCED | 2022 | {'0-12': 172563, '13-24': 219942, '25-36': 204072, '37-60': 94208} |



## Vintage Overlap


Calendar-year counts below; every quarter and its contributing vintage set are included in JSON. Counts refer to risk target months, not origination years.

| Design | Year | Intervals | Facilities | Contributing vintages |
| --- | --- | --- | --- | --- |
| PRIMARY | 2010 | 99544 | 27849 | [2006, 2008, 2010] |
| PRIMARY | 2011 | 382983 | 38160 | [2006, 2008, 2010] |
| PRIMARY | 2012 | 341256 | 32236 | [2006, 2008, 2010] |
| PRIMARY | 2013 | 243256 | 23694 | [2006, 2008, 2010] |
| PRIMARY | 2014 | 213789 | 24079 | [2006, 2008, 2010, 2014] |
| PRIMARY | 2015 | 344019 | 34075 | [2006, 2008, 2010, 2014] |
| PRIMARY | 2016 | 330102 | 29703 | [2006, 2008, 2010, 2014] |
| PRIMARY | 2017 | 275531 | 24562 | [2006, 2008, 2010, 2014] |
| PRIMARY | 2018 | 260320 | 28976 | [2006, 2008, 2010, 2014, 2018] |
| PRIMARY | 2019 | 397559 | 38285 | [2006, 2008, 2010, 2014, 2018] |
| PRIMARY | 2020 | 340977 | 37308 | [2006, 2008, 2010, 2014, 2018, 2020] |
| PRIMARY | 2021 | 376240 | 39936 | [2006, 2008, 2010, 2014, 2018, 2020] |
| PRIMARY | 2022 | 372244 | 41055 | [2006, 2008, 2010, 2014, 2018, 2020, 2022] |
| PRIMARY | 2023 | 515681 | 46074 | [2006, 2008, 2010, 2014, 2018, 2020, 2022] |
| PRIMARY | 2024 | 504371 | 43270 | [2006, 2008, 2010, 2014, 2018, 2020, 2022] |
| PRIMARY | 2025 | 468848 | 40408 | [2006, 2008, 2010, 2014, 2018, 2020, 2022] |
| PRIMARY | 2026 | 74459 | 37337 | [2006, 2008, 2010, 2014, 2018, 2020, 2022] |
| REDUCED | 2006 | 21237 | 7304 | [2006] |
| REDUCED | 2007 | 179978 | 18876 | [2006] |
| REDUCED | 2008 | 225125 | 28290 | [2006, 2008] |
| REDUCED | 2009 | 336430 | 34168 | [2006, 2008] |
| REDUCED | 2010 | 294061 | 32026 | [2006, 2008, 2010] |
| REDUCED | 2011 | 382983 | 38160 | [2006, 2008, 2010] |
| REDUCED | 2012 | 341256 | 32236 | [2006, 2008, 2010] |
| REDUCED | 2013 | 243256 | 23694 | [2006, 2008, 2010] |
| REDUCED | 2014 | 213789 | 24079 | [2006, 2008, 2010, 2014] |
| REDUCED | 2015 | 344019 | 34075 | [2006, 2008, 2010, 2014] |
| REDUCED | 2016 | 330102 | 29703 | [2006, 2008, 2010, 2014] |
| REDUCED | 2017 | 275531 | 24562 | [2006, 2008, 2010, 2014] |
| REDUCED | 2018 | 260320 | 28976 | [2006, 2008, 2010, 2014, 2018] |
| REDUCED | 2019 | 397559 | 38285 | [2006, 2008, 2010, 2014, 2018] |
| REDUCED | 2020 | 340977 | 37308 | [2006, 2008, 2010, 2014, 2018, 2020] |
| REDUCED | 2021 | 376240 | 39936 | [2006, 2008, 2010, 2014, 2018, 2020] |
| REDUCED | 2022 | 372244 | 41055 | [2006, 2008, 2010, 2014, 2018, 2020, 2022] |
| REDUCED | 2023 | 515681 | 46074 | [2006, 2008, 2010, 2014, 2018, 2020, 2022] |
| REDUCED | 2024 | 504371 | 43270 | [2006, 2008, 2010, 2014, 2018, 2020, 2022] |
| REDUCED | 2025 | 468848 | 40408 | [2006, 2008, 2010, 2014, 2018, 2020, 2022] |
| REDUCED | 2026 | 74459 | 37337 | [2006, 2008, 2010, 2014, 2018, 2020, 2022] |



## APC Implications


| Design | Clock rows | Columns | Rank | Identity residual |
| --- | --- | --- | --- | --- |
| PRIMARY | 16393 | 4 | 3 | 0.0 |
| REDUCED | 17648 | 4 | 3 | 0.0 |

Calendar t0 = scheduled first-payment cohort + proxy age exactly. Overlap does not identify unrestricted age, period and cohort effects. Freeze six vintage indicators with2006 reference, constrained duration bands and eight primary macro features. No unrestricted period fixed effects, trend or interactions; no causal coefficients. Archive vintage is also not exact origination/first-payment cohort.

Separately, mortgage rate − Treasury yield equals their spread: maximum absolute residual is 2.220446049250313e-16 over 189 complete rate months. These three unrestricted coefficients are not separately identifiable. All eight features remain required for eligibility. Before fitting, Task10 must seal an identifiable coefficient constraint/basis; no performance-driven feature deletion or claim of three independent rate effects.


## Horizon Support


Conditional elapsed months from one active entry per facility/design differ from Task8's first-observation origin. Known status means observed to horizon or an earlier first event. Combined support requires original mortgage horizon SUPPORTED and conditional macro risk≥1000/known≥10000 (LIMITED≥100/≥2000); it cannot upgrade originally unsupported mortgage tails. The compact table gives combined labels; denominators and separate statuses are in JSON.

| Design | Vintage | 12 | 24 | 36 | 60 | 84 | 120 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PRIMARY | 2006 | MORTGAGE_SUPPORTED_MACRO_UNSUPPORTED | MORTGAGE_SUPPORTED_MACRO_UNSUPPORTED | MORTGAGE_SUPPORTED_MACRO_UNSUPPORTED | MORTGAGE_SUPPORTED_MACRO_UNSUPPORTED | MORTGAGE_SUPPORTED_MACRO_UNSUPPORTED | MORTGAGE_SUPPORTED_MACRO_UNSUPPORTED |
| PRIMARY | 2008 | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_UNSUPPORTED |
| PRIMARY | 2010 | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED |
| PRIMARY | 2014 | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED |
| PRIMARY | 2018 | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_UNSUPPORTED |
| PRIMARY | 2020 | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_UNSUPPORTED | MORTGAGE_UNSUPPORTED |
| PRIMARY | 2022 | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_UNSUPPORTED | MORTGAGE_UNSUPPORTED | MORTGAGE_UNSUPPORTED |
| REDUCED | 2006 | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED |
| REDUCED | 2008 | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED |
| REDUCED | 2010 | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED |
| REDUCED | 2014 | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED |
| REDUCED | 2018 | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_UNSUPPORTED |
| REDUCED | 2020 | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_UNSUPPORTED | MORTGAGE_UNSUPPORTED |
| REDUCED | 2022 | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_SUPPORTED_MACRO_SUPPORTED | MORTGAGE_UNSUPPORTED | MORTGAGE_UNSUPPORTED | MORTGAGE_UNSUPPORTED |

Observed monthly hazards do not require future macro paths at entry. Scenario forecasting and cumulative prospective risks require separately governed future macro paths; none are generated here.


## Pandemic Coverage


| Design | Year | Intervals | Facilities | Vintages |
| --- | --- | --- | --- | --- |
| PRIMARY | 2020 | 340977 | 37308 | [2006, 2008, 2010, 2014, 2018, 2020] |
| PRIMARY | 2021 | 376240 | 39936 | [2006, 2008, 2010, 2014, 2018, 2020] |
| REDUCED | 2020 | 340977 | 37308 | [2006, 2008, 2010, 2014, 2018, 2020] |
| REDUCED | 2021 | 376240 | 39936 | [2006, 2008, 2010, 2014, 2018, 2020] |

Historical economic-regime coverage only; no pandemic or GFC effect estimated.


## Financial-Crisis Coverage


| Design | Year | Intervals | Facilities | Vintages |
| --- | --- | --- | --- | --- |
| PRIMARY | 2007 | 0 | 0 | [] |
| PRIMARY | 2008 | 0 | 0 | [] |
| PRIMARY | 2009 | 0 | 0 | [] |
| REDUCED | 2007 | 179978 | 18876 | [2006] |
| REDUCED | 2008 | 225125 | 28290 | [2006, 2008] |
| REDUCED | 2009 | 336430 | 34168 | [2006, 2008] |

Historical economic-regime coverage only; no pandemic or GFC effect estimated.


## Future Task 10 Population


[Frozen specification](../../docs/track_b/macro_support_eligibility_spec.json), LF SHA256 `f1bd2f1082574b26f3ca969764e51d7228445c247fa89e80c2ded4bac74a95fe`. The pre-count freeze and interval-key hashes are in JSON. Validation candidates were chosen before counts and are adopted only after feasibility checks; no block changed in response to outcomes. Development uses70% facility hash roles through 2017December, excludes2018, and evaluates the disjoint30% roles in 2019January–2026February. Both pooled primary blocks must have≥100 unique defaults and≥500 payoffs.

| Design | Block | Facilities | Intervals | Defaults | Payoffs |
| --- | --- | --- | --- | --- | --- |
| PRIMARY | development | 42609 | 1568661 | 1904 | 25822 |
| PRIMARY | temporal_evaluation | 22971 | 901447 | 868 | 11146 |
| REDUCED | development | 54359 | 2241245 | 3652 | 35789 |
| REDUCED | temporal_evaluation | 22971 | 901447 | 868 | 11146 |

The pooled candidate temporal block contains unseen 2018/2020/2022 vintages. Development only estimates 2006/2008/2010/2014 cohort effects. The [final validation design](../../docs/track_b/macro_support_validation_design.json) therefore separates primary seen-vintage temporal metrics from explicitly restricted unseen-vintage extrapolation. Dates and hash roles are unchanged.

| Temporal population | Facilities | Intervals | Defaults | Payoffs |
| --- | --- | --- | --- | --- |
| Primary seen vintages | 5619 | 248939 | 280 | 3823 |
| Separate unseen-vintage sensitivity | 17352 | 652508 | 588 | 7323 |

The seen-vintage block must itself meet the frozen event minimums. For unseen-vintage sensitivity, its unestimated coefficient is explicitly fixed to zero relative to the 2006 training reference and flagged UNSEEN_VINTAGE; never pool its metrics with primary temporal results. Leave-vintage-out applies the same explicit reference-effect restriction; if 2006 is held out, training reference becomes 2008. This tests conditional extrapolation, not learned unseen-cohort effects.

Leave-vintage-out is prespecified across seven vintages; sparse causes must be reported as infeasible where applicable. Vintage endpoint counts are feasibility, not measured model performance. Task10 must seal a new consumption ledger and complete modeling protocol before fitting. Previously inspected outcomes cannot become a virgin holdout.


## Limitations


- Primary cannot evaluate 2007–2009 macro regimes; reduced is one fixed sensitivity only.
- Mortgage records are current-release retrospective disclosures; macro PIT does not certify historical mortgage operational knowledge time.
- Survival into the support window creates delayed entry and a conditional population; old-vintage survivors are not the full origination cohorts.
- Default is a monthly research proxy; payoff includes maturity. Unknown and administrative states censor before unascertainable exposure; censoring independence is not established.
- Scheduled first-payment proxy is not exact origination age; provider age remains separate.
- Facility disjointness is not borrower disjointness. Monthly rows and macro periods are dependent; prospective validation must account for facility/calendar clustering.
- Unrestricted age-period-cohort effects remain unidentified. Future coefficients would be predictive conditional associations, not causal economic effects.
- Prior outcomes have been inspected. Future temporal evaluation is prespecified reused research evidence, not a virgin holdout; Task10 needs a new sealed ledger before fitting.
- No complete authoritative pre2010 HPI/PMMS release chain admitted. Isolated historical HPI evidence does not extend frozen feature support.
- Observed interval support does not provide future macro scenario paths or authorize unsupported tail forecasts, IFRS9 staging or ECL claims.
- Earlier development cannot estimate later-vintage fixed effects. Primary temporal metrics must use seen vintages; unseen and held-out vintages require explicitly flagged reference-effect extrapolation sensitivity, never learned unseen-cohort effects.
- Mortgage rate, Treasury yield and their spread are algebraically redundant. All eight features remain required for eligibility; Task10 must prespecify an identifiable coefficient constraint/basis before fitting, with no unrestricted three-rate effects.


## Decision


MACRO ELIGIBILITY DESIGN READY WITH MATERIAL LIMITATIONS.

Next: Track B Task 10 — Prespecified Macro-Conditioned Competing-Risk Modeling and Temporal/Vintage Validation. Not implemented. Actual second full canonical pass reproduced all aggregate counts, interval fingerprints and 14 private facility eligibility files. Preservation PASSED: 395 prior public LF hashes, 94 Task9 private byte hashes, 51 earlier private hashes and 371 prior tracked byte hashes; frozen tag, samples, source ZIPs and consumed ledgers remain unchanged. Complete details are in JSON.

Retained Track A AUC 0.868152 / Brier 0.048545 / log loss 0.176030 are historical results only, not newly evaluated. Test execution evidence is recorded separately in `macro_support_verification.json`.

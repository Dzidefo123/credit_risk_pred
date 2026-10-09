# STAGE 2 — INDEPENDENT EVIDENCE VERIFICATION

**Manuscript:** v0.3, "Population- and Regime-Dependent Transport of Vintage-Aware Macroeconomic Features in Mortgage Competing-Risk Models"
**Frozen commit (as stated by coordinator):** 1a42d505c6652fecf384d00b2d19d737c4b8524a
**Package root:** /home/claude/cold_review_stage2/input/

**Scope.**
- I read only files inside /home/claude/cold_review_stage2/.
- I did not use the web or open other directories or git history.
- I did not edit the manuscript or Stage 1. Nothing was refit or regenerated.
- Python was used only for arithmetic on files already in the package. The scripts live in a scratch directory outside the package and read the package files without changing them.

**Verdict vocabulary for Stage 1 items:** CONFIRMED / STRENGTHENED / WEAKENED / RESOLVED / REFUTED / UNVERIFIABLE.

---

## B. Stage 1 hash verification and review independence

| Item | Result |
|---|---|
| `stage1/STAGE1_PROVISIONAL_ASSESSMENT.md` SHA-256 | `63aa6f5a6801474249ab934079d1ba333a3ab55cc1a2cf46b74165659362ff58` |
| Identical to my original Stage 1 output file (`/home/claude/cold_review_stage1/output/…`) | Yes, byte-identical hash. Stage 1 is unchanged. |
| Stage 1 modified in Stage 2 | No |
| Manuscript `paper/main_v0.3.md` SHA-256 | `dd7738c5…5f17a`, same as the Stage 1 input |
| `paper/references.bib` | Byte-identical to the Stage 1 bibliography |
| Package manifest (`PACKAGE_MANIFEST.sha256`) | All 369 listed files verify (`sha256sum -c`, run from inside `input/`). 0 failures, 0 extra files. |

**REVIEW_CONTAMINATION_RISK (recorded under §24 of the prompt).**

1. **`reports/paper/task19_claim_evidence_map.json`.**
   - Each binding carries the fields `task17_primary_claim_verdicts`, `task17_status`, `task18_status` and `correction_status`. These are Task 17/18 review classifications.
   - Before I recognized this, I saw them for exactly one binding (Q0001: "SURVIVES", "PROPOSED", "NOT_EXECUTED_STRENGTHENING_ANALYSIS", "CORRECTED_SCOPE_OR_RETAINED_VALUE").
   - From then on, my scripts read only these fields: `binding_id`, `fact_key`, `source_artifact`, `source_field`, `exact_value`, `displayed_value`, `display_format`, `source_sha256_lf`, `frozen_status`, `manuscript_section`.
   - No review classification was used in any judgement below.
2. **`reports/paper/task19_evidence/*`.**
   - The local-closure registration and verification files contain process metadata, e.g. "User explicitly authorized corrected script preparation and execution after the external-review audit", plus Task 18 analysis identifiers (SA01/SA06).
   - They contain no reviewer conclusions. I used only their numeric outputs, code and hashes.
3. **`paper/main_v0.3.md`** itself refers to the Task 17 erratum and Task 18 corrections. This was already visible in Stage 1 and is not new contamination.

No other package file contains Task 17–20 reviewer text. A grep for task17–20, hostile, reviewer, erratum and external-review found only generic governance wording in four `docs/track_b` design documents.

---

## A. Evidence package inventory

369 files. All match the manifest hashes. Grouped by role:

| Path (group) | Files | Purpose | Referenced by manuscript / evidence map | Frozen / hash status | Content type |
|---|---:|---|---|---|---|
| `paper/main_v0.3.md`, `paper/references.bib` | 2 | Manuscript and bibliography | The manuscript itself | Hash = Stage 1 | Interpretation |
| `reports/paper/task19_claim_evidence_map.json` | 1 | Index of 257 quantitative statements and their source fields | Index only. Base commit 3098da7…; source LF hashes recorded | Hashes of the bound sources match the package | Index, plus embedded review-status fields (contamination risk) |
| `reports/paper/task19_evidence/local_closure_{registration.json, registration.sha256, script.py, output.json, verification.json}`, `admission.json`, `derived_metadata.json`, `.gitattributes` | 8 | Post-hoc month-level AUC decomposition (SA01) and landmark-entry distribution (SA06), computed on frozen arrays | 22 + 3 bindings | Hash chain verified: registration sha = sealed sha = `dc011baa…`; script `188f239f…`; output `7a1be6c5…`. The registration's public input hashes are the **CRLF variants** of the package's LF files; I verified this by converting LF to CRLF and re-hashing, and both match exactly. Private array hashes (M1/M2 predictions) equal `prediction_hashes` in the Task 10 report. | **Primary aggregate** (post-hoc) plus code and provenance |
| `reports/track_b/macro_competing_risk_validation.json` | 1 | **Frozen Task 10 experiment result**: scores, calibration (with reliability deciles), paired facility/calendar intervals, per-year and per-vintage stability, unseen-vintage scores, CIF (observed AJ monthly table plus model horizon metrics), protocol, model parameters, ledger, prediction hashes | 124 bindings | LF hash `ae401cad…` = map binding hash. Embedded protocol equals `docs/track_b/macro_competing_risk_protocol.json` (dict equality True). | **Primary aggregate results**, plus model coefficients and provenance |
| `src/credit_risk/track_b/macro_hazard/*.py` (protocol, data, models, ledger, metrics, cif, study) | 7 of 153 src | Task 10 implementation | Not cited | **LF hashes of all 7 equal the `code_sha256_lf` recorded in the consumed Task 10 ledger** | Code |
| Other `src/credit_risk/track_b/**` (macro_support eligibility, survival math/risk, data panel/freddie/schemas/sampling, pit_macro engine, multivintage core) | ~60 | Target construction, eligibility, PIT macro joining, AJ/IPCW maths | Not cited | Manifest hash only (not individually ledger-bound) | Code |
| Other `src/**` (Track A credit scoring, API, monitoring, etc.) | ~90 | Unrelated to this manuscript | No | Manifest | Code (not relevant) |
| `reports/track_b/macro_signal_attribution_stability.json` (+ `.md`, `macro_signal_verification.json`) | 3 | **Post-validation diagnostics**: macro shift and support, correlations, per-year macro summaries, per-year calibration, error partitions, joint-loss-by-realized-event accounting, frozen-coefficient ablations | 6 bindings | Hash matches the map | **Primary aggregate (post-hoc diagnostic)** |
| `reports/track_b/macro_support_eligibility.json`, `docs/track_b/macro_support_validation_design.json`, `macro_support_eligibility_spec.json` | 3+ | Task 9A support/eligibility design and **per-vintage event counts** (including unseen 2018/2020/2022) | Indirect | Spec LF hash `f1bd2f10…` recorded in protocol | Primary aggregate counts plus protocol |
| `reports/track_b/multi_vintage_recovery_audit.json` (+ data audit, manifests) | ~10 | Sampling (20k per vintage), field missingness incl. assistance/deferral flags, **calendar-year × vintage active-row support** | Indirect | Manifest | Primary aggregate (data audit) |
| `docs/track_b/mortgage_research_protocol.json` | 1 | Event (target) policy | 4 bindings | Hash matches | Protocol |
| `docs/track_b/pit_macro_*`, `macro_feature_registry.json`, `MACRO_DATA_PROVENANCE.md`, `PIT_MACRO_API_ACQUISITION.md`; `reports/track_b/pit_macro_*`, `macro_release_lag_probe.json`, `macro_source_probe.json` | ~15 | Macro series, transforms, PIT rules, release-lag probe | Indirect | Manifest | Protocol, provenance and aggregate audit |
| `docs/track_b/*` design/contract `.md`/`.json` (LONGITUDINAL_DATA_CONTRACT, MACRO_IDENTIFICATION_DESIGN, SURVIVAL protocol, field comparability, etc.) | ~45 | Design intent, field semantics, limitations | Indirect | Manifest | Interpretation and protocol (secondary) |
| `docs/track_b/fannie_*`, `reports/track_b/FANNIE_*`, `fannie_*` | ~35 | Proposed external replication. Status: not executed. | Mentioned as "not authorized" | Manifest | Provenance (not relevant to results) |
| `reports/track_b/` other (expanded PD, PD baseline, survival validation, refinancing, sample expansion, source conventions, TASK14 freeze) | ~30 | Earlier tasks and exploratory refinancing study | Refinancing mentioned as EXPLORATORY_ONLY | Manifest | Secondary aggregates and interpretation (not inspected in depth) |
| `reports/track_b/figures/**` | 16 | PNG figures | No | Manifest | Visual summaries (not inspected; numbers were verified from the JSON instead) |
| `configs/*`, `scripts/*`, `pyproject.toml`, `data/README.md` | 34 | Configs, run scripts, dependencies; README covers the Track A Kaggle file only | No | Manifest | Code and provenance |

**Not in the package:**
- private arrays `evaluation.npy`, `development.npy`, `*_evaluation.npy`, `cif_*.npy` and the model `.joblib` files (only their hashes are present);
- raw Freddie files;
- `macro_month_table.json`;
- `reports/paper/task18_*.json` (withheld by the coordinator).

**Primary versus secondary.**
- Primary empirical evidence means frozen aggregate outputs computed by the ledger-bound code: `macro_competing_risk_validation.json`, the closure output, the diagnostics JSON and the support/audit counts.
- Everything else is protocol, code, provenance or interpretation.
- The claim-evidence map is treated **only as an index**.

---

## C. Claim-evidence verification

**Method.**
- Every one of the 257 bindings was resolved by JSON pointer into its source artifact.
- Each `exact_value` was compared with the source value, `displayed_value` with the Q-tagged manuscript text, and each source file's LF SHA-256 with `source_sha256_lf`.
- For the 98 bindings that target withheld `task18_*.json` files, I recomputed the value **from primary sources** (mostly `stability.calendar`, `primary`, `unseen_vintage` in the Task 10 report).

| Group | Bindings | Value check | Source hash | Display vs manuscript | Discrepancy |
|---|---:|---|---|---|---|
| `macro_competing_risk_validation.json` | 124 | 124 exact match | match | all match | NONE |
| `local_closure_output.json` | 22 | 22 exact | match | all match | NONE |
| `macro_signal_attribution_stability.json` | 6 | 6 exact | match | match | NONE |
| `mortgage_research_protocol.json` | 4 | 4 exact | match | match | NONE |
| `derived_metadata.json` | 3 | 3 exact. Each also reproduced from primary: 6 positive non-2020 years, 31 excluded months, dominant 24-month path end 2020-12. | match | match | NONE |
| `task18_sa02_calendar_decomposition.json` (withheld) | 79 | **77 reproduced from primary sources to ≤1e-15.** Two (Q0145/Q0146, ex-2020 year-block interval) cannot be reproduced exactly because the seed and procedure are withheld; my approximation gives [−0.00150, +0.00671] against bound [−0.00151, +0.00671]. | WITHHELD | match | NONE for 77; **UNVERIFIABLE (plausible)** for 2 |
| `task18_sa01_within_period_auc.json` (withheld) | 18 | **18 reproduced exactly** from per-year AUCs and pair counts (cases × non-cases) | WITHHELD | match | NONE |
| `task18_results.json` (withheld) | 1 | Q0207 unseen Δ = 0.069811276 − 0.070630509 = −0.000819233, reproduced | WITHHELD | match | NONE |

**Headline and abstract trace (statement → map → artifact → field → my recomputation):**

| Manuscript statement | Source field | Verified value | Class |
|---|---|---|---|
| Development joint LL 0.09015 → 0.08959 | `/development/M{1,2}/scores/joint_log_loss` | 0.0901513, 0.0895928 | NONE |
| Seen temporal LL 0.089201 → 0.105323 | `/primary/M{1,2}/scores/joint_log_loss` | 0.0892013, 0.1053229 | NONE |
| 2020 contributes 89.34% | recomputed from `/stability/calendar/*` | 0.893375 | NONE (value). The *interpretation* "concentrated" is qualified in §H. |
| Pooled payoff AUC 0.56543 → 0.62587 | `/primary/M*/scores/payoff_auc` | 0.5654257, 0.6258680 | NONE |
| Between-year 98.19% | recomputed | 0.981915 | NONE |
| 55 supported months; within-month 0.56014 vs 0.55974 | closure `SA01_month` | exact | NONE (value). The estimand is support-restricted, which is disclosed. |
| Unseen LL 0.070631 → 0.069811 | `/unseen_vintage/M*/scores` | exact | NONE (value). "Reverses" is a SCOPE issue (point estimate only, §J/§N). |
| Payoff mean 1.536% / 1.138% / 2.830%; slopes 0.509 / 0.249 | `/primary/M*/calibration/payoff` | exact | NONE (value). **INTERPRETATION_ERROR (partial)**: the pooled "over-prediction" framing (§4.2, §5) hides a sign reversal by year (§K). |
| Payoff-Brier facility vs calendar intervals | `/paired_*/intervals/payoff_brier` | exact | NONE |
| CIF Table F (all 24 cells) | `/cif/horizons/*` | exact | NONE |

**Minor binding issues.**
- **Q0040/Q0041** ("a month needs at least 20 cases and 20 controls").
  - These bind to `/protocol/cell_suppression/minimum_cause_events` = 20. That protocol cell rule also carries `minimum_facilities = 100`, which the month closure did not apply.
  - The rule actually used is the closure script's hard-coded `SUPPORT = 20`, applied to cases and controls. It mirrors `scores()` / `bootstrap.auc_min_events`.
  - The value is correct; the bound field is a neighbouring rule. **SOURCE_ERROR (minor).**
- **Label-only bindings.** Q0005/Q0015/Q0046/Q0132/Q0140/Q0148/Q0216/Q0255–Q0257 bind the year label "2020" to `/per_year/1/year`. These verify a label, not the attached claim.
  - Example: Q0216's sentence "All landmark projection windows at the 24-month horizon include calendar 2020". I verified that claim separately from closure `SA06_entry.path_24m_endpoint_distribution`: every 24-month path ends between 2020-12 and 2021-12 and starts between 2019-01 and 2020-01, so every one includes 2020.
  - Class NONE, but these bindings are **not evidence** for their sentences.
- **Stale bindings.** The map's base commit 3098da7… differs from the manuscript commit 1a42d505…, but every bound source hash equals the package file. **No STALE_BINDING found.**

**Conclusion:**
- Of the 257 numeric bindings, 255 are verified (159 directly, 96 by recomputation from primary sources). 2 are UNVERIFIABLE-but-plausible.
- There is no value error, and no binding points to the wrong population.
- The remaining problems are interpretive, not numerical (§W).

---

## 4. Stage 1 arithmetic against frozen artifacts (exact source locations)

All sources are `reports/track_b/macro_competing_risk_validation.json` (abbreviated **R**) unless stated.

| Quantity | Stage 1 | Frozen value | Location |
|---|---|---|---|
| Seen intervals | 248,939 | 248,939 | R `/split_counts/evaluation_seen/intervals`; sum of `/stability/calendar/*/M1/intervals` |
| M1 joint LL | 0.089201 | 0.08920130868 | R `/primary/M1/scores/joint_log_loss` |
| M2 joint LL | 0.105323 | 0.10532294511 | R `/primary/M2/scores/joint_log_loss` |
| Δ | +0.016122 | +0.01612163643 | R `/paired_facility/intervals/joint_log_loss/delta`; also Σ year contributions = 0.0161216364278 |
| 2020 contribution / share | 89.34% | 0.0144026642 / 0.893375 | from R `/stability/calendar/2020` |
| Ex-2020 renormalised Δ | +0.00217 | 0.07977318 − 0.07760243 = +0.00217075 (2.797% relative) | from R `/stability/calendar/*` |
| Contribution residual | — | +0.00171897 | same |
| Pooled payoff AUC M1 / M2 / gain | 0.5654 / 0.6259 / +0.06044 | 0.5654257 / 0.6258680 / +0.0604423 | R `/primary/M*/scores/payoff_auc` |
| Year pair weight (within) | 17.195% | 0.1719461 = Σ_y payoffs_y(n_y − payoffs_y) / [3823 × (248939 − 3823)] | from R `/stability/calendar/*` |
| Within-year AUC M1 / M2 | 0.55555 / 0.56191 | 0.5555523 / 0.5619094 | pair-weighted from per-year AUCs |
| Between-year (solved) | 0.56748 / 0.63915 | 0.5674759 / 0.6391490 | identity |
| Contributions / shares | 0.001093 / 0.059349; 1.81% / 98.19% | 0.0010931 / 0.0593492; 1.808% / 98.192% | identity |
| Supported-month result | −0.00040, 55 months | −0.000398619, 55 months, 191,938 intervals | closure `/SA01_month/gains`, `/eligible_months` |
| Seen/unseen joint LL | 0.089201 / 0.105323 / 0.070631 / 0.069811 | exact | R `/primary`, `/unseen_vintage` |
| Calibration (seen payoff) | 1.536% / 1.138% / 2.830%; slopes 0.509 / 0.249 | 0.0153572 / 0.0113792 / 0.0283047; 0.50939 / 0.24934 | R `/primary/M*/calibration/payoff` |
| CIF horizon values | Table F | all 24 values exact | R `/cif/horizons/{12,24,36,60}/observed/[h−1]` and `/models/*/{default,payoff}/mean_predicted_cif` |
| Calendar 8-block CI | — | [+0.0000533, +0.0415021] | R `/paired_calendar`. **Reproduced exactly** by resampling the 8 annual (n_y, Δ_y) aggregates with seed 61036. The calendar bootstrap is therefore literally a percentile bootstrap over eight numbers. |

All Stage 1 arithmetic is **CONFIRMED**.

---

## D. Estimand reconstruction (from code and protocol, not prose)

Sources:
- `docs/track_b/mortgage_research_protocol.json` (`event`)
- `src/.../data/panel.py::event_category`
- `src/.../multivintage/core.py`
- `src/.../macro_support/eligibility.py`
- `src/.../macro_hazard/{protocol,data,models,metrics,cif,study}.py`
- `src/.../survival/math.py`
- `docs/track_b/macro_support_validation_design.json`
- `reports/track_b/multi_vintage_recovery_audit.json`

| Element | Reconstruction |
|---|---|
| **Target (per interval)** | An interval runs from reporting month t0 to t0+1. Its event is the `event_category` of month t0+1:<br>• **default** if `delinquency_state` is numeric in [3, 99], **or** equals `RA` (REO), **or** the termination code is in {02, 03, 09};<br>• **payoff** if the termination code is 01 and the month is not a default;<br>• **ambiguous** if both apply, or if the termination code and termination month disagree;<br>• **administrative** for codes 15/16/96;<br>• **unknown** for `XX` or blank;<br>• otherwise **none**.<br>Assistance, deferral, disaster and modification fields are **not** read by `event_category`. `histories()` projects only six fields (loan_id, month, event_category, analytical_prefix, loan_age, first-payment-proxy months). |
| **Competing events** | Default and payoff/maturity. The first event is absorbing (`Repeated first endpoint` raises an error). Same-month default plus payoff is ambiguous and censored. |
| **Censoring** | The risk set ends at a gap, duplicate, unknown, ambiguous or administrative month, at observation end, at macro-support end, or at the cutoff 2026-02. Intervals are kept only if the target lies in the first contiguous active prefix ("no re-entry"). |
| **Eligibility** | Current state is none and inside the analytical prefix; ≥ 6 months of prior history (`index+1 ≥ 6`); first-payment proxy ≥ 0; target category in {none, default, payoff}; macro features AVAILABLE at t0 (`macro_reasons`). |
| **Facility roles** | `validation_role`: SHA-256(salt:vintage:loan) mod 10 ≤ 6 means development (70%), otherwise temporal evaluation (30%). Development-role facilities contribute only target months 2010-09..2017-12 (primary). Evaluation-role facilities contribute only 2019-01..2026-02. 2018 is purged for everyone. Facility disjointness is asserted in code (`Facility role leakage`). |
| **Development population** | 42,609 facilities, 1,568,661 intervals, 1,904 defaults, 25,822 payoffs; vintages 2006/2008/2010/2014; 88 distinct macro months. |
| **Seen-vintage evaluation** | 5,619 evaluation-role facilities of 2006/2008/2010/2014 (300 / 361 / 1,554 / 3,404); 248,939 intervals; 280 defaults; 3,823 payoffs. |
| **Unseen-vintage evaluation** | 17,352 evaluation-role facilities of 2018/2020/2022 (5,768 / 5,683 / 5,901); 652,508 intervals (184,349 / 261,306 / 206,853); 588 defaults (320 / 74 / 194); 7,323 payoffs (4,477 / 1,879 / 967). Source: `macro_support_validation_design.json/by_vintage`. |
| **Landmark / CIF population** | First eligible evaluation interval per seen facility: 5,619 landmarks. The first target month is 2019-01 for 5,564 (99.02%); the latest is 2020-01. No horizon truncation at 12/24/36/60. Time is relative to the landmark (`entry_time = 0`). |
| **Evaluation windows** | Development 2010-09..2017-12; purge 2018; evaluation 2019-01..2026-02 (86 months, 8 calendar-year blocks, 2026 partial). |
| **Scoring weights** | Equal per interval for log loss and Brier (`losses().mean`). AUC is over intervals: case = payoff (or default) interval, controls = all other intervals. |
| **Model inputs** | Static origination covariates, duration band at t0, vintage indicator, and (M2) seven national PIT macro values joined at **t0 = end of the month preceding the target month** (`previous_month_end`). |
| **Prediction outputs** | Per interval, P(none), P(default), P(payoff) from a multinomial logistic (rows sum to 1; checked to 1e-10). CIF from the recursion S(k) = S(k−1)(1 − h_D − h_P), driven by the realized PIT macro path and deterministic duration increments. |

**Mismatches with what Stage 1 inferred:**
1. **Timing of the macro join.** Stage 1 feared that "reporting-month macro values" might be slightly ahead of availability. In fact macro is joined as of the end of t0 (the month before the target month), and only values whose ALFRED archive validity began on or before t0 are used. This is more conservative than Stage 1 assumed.
2. **Sampling.** Stage 1 called the extract "unspecified". It is a deterministic SHA-256 identifier ranking, 20,000 per vintage, from the Standard FRM origination universe, frozen before performance access (§N).
3. **Unseen population.** It is only the 30% evaluation-role subset of the unseen vintages. Stage 1 did not know this, but it does not change any conclusion.
4. **Duration-band fallback (new to Stage 2).** Duration bands with no development support (181–240, 241+) get zero coefficients, so they inherit the 0–12 reference effect. This applies to **11,167 seen-vintage intervals (4.5%)** (`error_partitions/duration:181-240`). Stage 1 knew only about the unseen *cohort* fallback.
5. **Unseen cohort fallback, made precise.** Vintage indicators for 2018/2020/2022 are all-zero design columns, so unseen loans are scored with the **2006 reference cohort effect**. No unknown loan_purpose/occupancy categories occurred (all counts 0).

---

## E. Target / default endpoint audit (priority item)

**Answers to the seven questions.**

1. **What triggers research default?** The first month in which the monthly delinquency state is numeric 3–99, or the state is `RA`, or the termination code is 02, 03 or 09.
2. **Is 90+ day contractual delinquency sufficient?**
   - In code terms: **yes. A delinquency state of 3 or more alone is sufficient.**
   - Whether state "3" means contractual 90+ days for every servicing regime is a provider-semantics question. The package contains no provider document that settles it. The package describes the field only as "Monthly categorical state; XX unknown, RA REO, numeric bands. Not exact daily DPD."
3. **Are forbearance, deferral or modification explicitly handled?**
   - **No.** `assistance_plan`, `payment_deferral_flag`, `disaster_flag` and `modification_flag` are parsed and audited for missingness, but they are not used in `event_category`, in eligibility, or as predictors.
   - The project's own design document (`docs/track_b/MACRO_IDENTIFICATION_DESIGN.md`, "Pandemic regime and measurement breaks") states: *"Payment assistance, forbearance … can change links between delinquency and loss/default proxies. A later protocol must harmonize disclosed assistance/deferral fields … prespecify regime indicators or separate regime analyses …"*
   - The Task 10 protocol does **not** implement this. Its `prohibited` list excludes period fixed effects and interactions, and the `pandemic_2020_2021` sensitivity is a calendar subset only.
4. **Are any such loans excluded?** No. No exclusion is conditioned on assistance or deferral status.
5. **Can frozen artifacts distinguish forborne from ordinary severe delinquency?**
   - **No.** The only frozen tabulations of these fields are row-level missingness counts per vintage (e.g. 2014: `assistance_plan` observed on 6,786 of 1,352,589 rows; `payment_deferral_flag` on 13,083).
   - There is no cross-tabulation against default events, months or roles.
6. **Economic default, delinquency proxy, or composite?** A **composite research endpoint dominated by a first-passage-to-severe-delinquency proxy**. The protocol names it "first observed composite adverse mortgage event" and prohibits "regulatory default/PD equivalence".
7. **Could COVID-era servicing policy change the meaning of the 2020 endpoint?** The frozen evidence shows an event-timing pattern that makes this a live, material possibility. I am describing the data, not making a causal claim:
   - **CIF observed table** (R `/cif/horizons/60/observed`): monthly seen-vintage default events at landmark-relative months 1–17 range from 0 to 8. At **month 18 there are 81 events**, then 29 at month 19, then 13, 10, 9, 7, 6. For the 99.0% of landmarks that enter in 2019-01, months 18–19 are **2020-06 and 2020-07**. About **110 of the 280 seen-vintage defaults (39%) fall in two calendar months.**
   - **Per-year counts** (R `/stability/calendar`): defaults 34 (2019), **167 (2020)**, 33 (2021), then 15, 13, 5, 10, 3. **60% of all seen-vintage defaults occur in 2020.** The 2020 default rate is 0.322% per interval, against 0.054% in 2019.
   - **Where the metrics move with it:**
     - M1's default calibration-in-the-large is reasonable outside 2020 (2019: 0.00064 predicted vs 0.00054 observed; 2021: 0.00066 vs 0.00091) but fails in 2020 (0.00064 vs 0.00322).
     - M2's 2020 default AUC is 0.508 (M1 0.665). M2's 2020 default calibration slope is 0.027.
     - The CIF default under-prediction by both models at 24–60 months arises entirely after month 17. At 12 months M1 is 0.0060 vs 0.0061 observed.

**What it does and does not affect.**
- The full-period realized-class decomposition of the joint-log-loss difference (frozen; §H) gives a **default-row contribution of −0.0000415** out of +0.01612.
- The payoff-based central findings (the pooled payoff-AUC decomposition, the payoff miscalibration) therefore do **not** depend materially on the default label.
- The following **do** depend on it:
  - default AUC and default Brier (seen and unseen; 320 of the 588 unseen defaults are in the 2018 vintage, which has 2020 exposure),
  - CIF default rows,
  - the "comparator calibration" and "default underprediction shared by baselines" statements,
  - the observed 2020 default level and the interpretation of the joint-loss contribution of 2020 as a "stress" period.

**Classification: MATERIAL_TARGET_RISK** for default-specific and 2020-default statements. **SENSITIVITY_NEEDED** is the remedy. For the payoff-centred headline results, the effect is DISCLOSURE_ONLY.

**External documentation that would be required (not consulted):**
- The Freddie Mac Single-Family Loan-Level Dataset General User Guide and file layout for the release used (R47, pinned July 2026). Specifically, the definitions of **Current Loan Delinquency Status** for loans in COVID-19 forbearance; whether status continued to age during forbearance; and whether it is reset on payment deferral.
- The definitions and population-coverage dates of **Borrower Assistance Status Code** (`assistance_plan`), **Payment Deferral**, and **Delinquency Due to Disaster** (`disaster_flag`).
- The release notes and FAQ for 2020–2021 reporting changes.

**Internal artifact needed:** a frozen-label tabulation of `assistance_plan` / `payment_deferral_flag` / `disaster_flag` at, and in the 3 months before, each seen- and unseen-vintage default event in 2020–2021. This would be descriptive and label-only, with no model involvement. It does not exist in the package.

---

## F. Model specification (reconstructed from code and frozen parameters)

Sources: `macro_hazard/models.py`, `protocol.py`, R `/development/*/parameters`.

| Item | M0 | M1 | M2 |
|---|---|---|---|
| Family | Monthly multinomial logistic, classes {0 none, 1 default, 2 payoff} (sklearn `LogisticRegression`, lbfgs, scikit-learn 1.9.1; `coef_` shape 3 × p) | same | same |
| Penalty | L2, **C = 1.0** (fixed in frozen protocol); tol 1e-8; max_iter 3000; random_state 61010; no class weights. Intercepts are unpenalised (sklearn convention). | same | same |
| Stored features p | 14 | 30 | **37** |
| Numeric origination | — | 6: credit score, LTV, DTI, log1p(UPB), note rate, original term (dev median-imputed, dev mean/SD standardised) | same 6 |
| Missing indicators | — | 6 declared (one per numeric). **3 identically zero in development** (UPB, rate, term never missing), so the active ones are credit score, LTV, DTI. | same; **no macro missing indicators** (non-finite macro raises an error) |
| Macro | — | — | **7 transforms of 6 series**: UNRATE level and 3-month change, DGS10 level, MORTGAGE30US level, USSTHPI YoY, CPIAUCSL YoY, GDPC1 QoQ (dev-standardised) |
| Duration | 8 band dummies (ref 0–12); **181–240 and 241+ are zero in development** | same | same |
| Cohort | 6 vintage dummies (ref 2006); **2018/2020/2022 zero in development** | same | same |
| Categorical | — | loan_purpose (2 dummies: N, P), occupancy (2: P, S); dev vocabulary, drop-first | same |
| Zero-energy design columns (frozen list) | 5 | 8 | 8 |
| Stored coefficients | 3 × 14 + 3 = 45 | 3 × 30 + 3 = 93 | **3 × 37 + 3 = 114** |
| Cause-vs-none contrasts (stored) | 2 × 14 | 2 × 30 | 2 × 37 = 74 (plus 2 intercept contrasts) |
| Contrasts on columns with development variation | 2 × 9 = 18 | 2 × 22 = 44 | **2 × 29 = 58** |
| Interactions | **Prohibited** | Prohibited | Prohibited |
| Period fixed effects | **Prohibited** | Prohibited | Prohibited |
| Current-state predictors | None (protocol "future loan states" prohibited; `static_origination_only = True`) | None | None |
| Unseen-category handling | All-zero reference encoding. Realised counts are 0 for loan_purpose/occupancy; unseen **cohort** is scored as **2006 reference**; unseen **duration bands** are scored as **0–12 reference** | same | same |
| Convergence | iterations: M2 965 (< 3000) | — | — |

**Notes.**
- I do not convert feature counts into an overfitting claim, and no effective degrees of freedom are identified.
- With about 1.57 M rows, standardised inputs and C = 1, the L2 term is small relative to the summed log-likelihood. I state this only as a scale observation.
- Frozen M2 cause-vs-none macro coefficients (per development SD) include **payoff: unemployment_level +0.372** and mortgage_30y_level −0.150. These are "conditional predictive associations, not causal" (artifact wording). I report them only because §H uses the macro ranges.

**Pre-specified sensitivity not reported in the manuscript (new):**
- The protocol pre-specified `RATE` (spread instead of mortgage level; numerically identical to M2: LL 0.105324) and `REDUCED` (5 macro terms without HPI and mortgage rate, developed on 2006-02..2017-12, which includes 2007–2009). Both are evaluated on the same seen population (R `/primary/REDUCED_M*`):

| Model | Joint LL | Default Brier | Payoff Brier | Default AUC | Payoff AUC |
|---|---|---|---|---|---|
| REDUCED_M1 | 0.088963 | 0.001380 | 0.015119 | 0.6916 | 0.5650 |
| REDUCED_M2 | 0.090945 (+0.00198, +2.2%) | 0.001415 | 0.015314 | **0.8136** | 0.6084 |

- Under REDUCED the macro block degrades joint LL roughly eight times less than M2 does (+0.0161), and default discrimination improves strongly.
- This does not overturn the primary result: REDUCED is still worse on proper scores. It does show that the size and pattern of the "macro increment" depend on specification and development window. The manuscript's general title and framing ("Macroeconomic Features") do not reflect this.

---

## G. Macro information contract

| Item | Finding (source) |
|---|---|
| Series | UNRATE (BLS, M, SA), DGS10 (FRB, D), MORTGAGE30US (Freddie PMMS, W), USSTHPI (FHFA, Q), CPIAUCSL (BLS, M, SA), GDPC1 (BEA, Q). Source: `pit_macro_series_registry.json`. |
| Transformations | level; 3-month difference (UNRATE); YoY growth (HPI, CPI); QoQ growth (GDP). Both operands are selected from the same t0 information set, and the measurement regime must match (`pit_macro/engine.py::engineer`). Source: `pit_macro_feature_registry.json`. |
| Vintage handling | ALFRED vintage records only (`representation = "vintage"`; `current_revised` raises an error in `macro_join` and `validate_versions`). Rule `LATEST_KNOWN_AS_OF_T0`. |
| Revision handling | A value is eligible only if `reference_period`, `publication_upper_bound` and `revision_upper_bound` are ≤ t0 and `archive_start ≤ t0 ≤ archive_end`. Overlapping versions raise an error. |
| Release-date handling | Exact agency release dates are **not** used. The ALFRED archive start acts as a *conservative upper bound* (`date_evidence = "conservative_archive_upper_bound"`). `certified_initial` requires independent evidence, and the initial-release rule is unused in Task 10. |
| Release-lag status | `macro_release_lag_probe.json`: a small 2014 probe (UNRATE / PAYEMS median 4 days, CPI 15, GDP 30). **DGS10, MORTGAGE30US and USSTHPI are unmeasured.** The scope is explicitly "NOT full-history estimate". |
| Availability rule | Risk month m uses information at end of month m−1 (`previous_month_end`). Freshness caps: UNRATE/CPI 62 days, DGS10 7, PMMS 14, HPI/GDP 183. |
| Missing values | Any unavailable or stale required feature makes the interval ineligible (excluded during support design). Non-finite macro in fitting raises an error. **No macro imputation.** |
| Development support | 88 months. Ranges 2010–2017: unemployment 4.1–9.8; 10-year 1.50–3.50; 30-year mortgage 3.32–4.95; HPI YoY −4.95–6.35; CPI −0.19–3.90; GDP QoQ −0.74–1.22. Source: `macro_signal…/macro_shift/regime_macro_summaries/development`. |
| Evaluation support | 77 of 86 months lie outside the development 95% squared-Mahalanobis reference (12.94). By year: 2019 4/12; **2020 11/12 (median 655); 2021 12/12 (105); 2022 12/12 (353); 2023 12/12 (247); 2024 12/12 (200); 2025 12/12 (167); 2026 2/2 (105).** |
| Measurement break inside the evaluation window | **The PMMS methodology change on 2022-11-17** is flagged in `pit_macro_series_registry.json` and `MACRO_DATA_PROVENANCE.md` ("a new study must document that break rather than silently pool measurement regimes"). `mortgage_30y_level` is a *level* feature, so no regime-matching check applies. The model pools the regimes. **The manuscript does not mention the break.** |

**Verdicts.**
- **"Vintage-aware" (and "revision-aware"): SUPPORTED.**
  - The code enforces ALFRED-vintage selection and the t0 availability bounds, and it rejects current-revised history.
  - The manuscript's own caveat (exact first-release timestamps not certified) is accurate.
- **"Point-in-time": SUPPORTED ONLY IN A QUALIFIED SENSE.**
  - Macro inputs are point-in-time with respect to ALFRED archive validity dates. These are used as conservative upper bounds, not certified release instants.
  - The full information set is **not** point-in-time. The mortgage performance panel is a retrospective current-release disclosure (`source_operational_knowledge_time: "UNVERIFIED; retrospective current-release disclosure"`).
  - Unqualified "point-in-time" would overstate the contract. "ALFRED-archive point-in-time macro inputs" would be accurate.
- **Stage 1 "possible mild look-ahead" from release lags: WEAKENED.**
  - By construction no value whose archive validity began after t0 is used.
  - The residual is only whether ALFRED dates could precede true public release. That is not certified, but the design is conservative.

---

## H. Calendar / 2020 analysis (cold verification)

**Direct verification.**
- The year table (Table C) reproduces exactly from R `/stability/calendar` (§4).
- Contributions sum to the primary Δ to 1e-16.

**Frozen per-year evidence beyond Table C** (all frozen; post-hoc diagnostics marked [D] are from `macro_signal_attribution_stability.json`):

| Year | Intervals | Macro distance median [D] | Δ joint LL | Relative Δ | Δ payoff Brier | Payoff mean obs / M1 / M2 [D annual_calibration] | Defaults |
|---|---:|---:|---:|---:|---:|---|---:|
| 2019 | 62,976 | 12 | +0.00018 | +0.3% | −0.000001 | 1.17% / 1.24% / 1.05% | 34 |
| **2020** | 51,809 | **655** | **+0.06920** | **+51.9%** | **+0.02261** | 2.21% / 1.28% / **10.05%** | **167** |
| 2021 | 36,397 | 105 | −0.00608 | −4.5% | −0.00014 | 2.61% / 1.17% / 2.49% | 33 |
| 2022 | 28,068 | 353 | +0.00581 | +7.6% | +0.00004 | 1.32% / 1.02% / **0.46%** | 15 |
| 2023 | 24,719 | 247 | +0.00750 | +13.4% | +0.00003 | 0.87% / 1.01% / **0.18%** | 13 |
| 2024 | 22,138 | 200 | +0.00619 | +11.8% | +0.00003 | 0.86% / 0.98% / **0.21%** | 5 |
| 2025 | 19,781 | 167 | +0.00694 | +11.7% | +0.00003 | 0.95% / 0.91% / **0.23%** | 10 |
| 2026 (partial) | 3,051 | 105 | +0.00503 | +8.5% | +0.00002 | 0.85% / 0.89% / 0.28% | 3 |

**Macro ranges by year** [D `regime_macro_summaries`]:

| Period | Unemployment | 10-year | 30-year mortgage | HPI YoY | CPI YoY | GDP QoQ |
|---|---|---|---|---|---|---|
| dev 2016–17 | 4.1–5.0 | 1.50–2.49 | 3.42–4.32 | 5.39–6.35 | 0.44–2.80 | 0.13–0.87 |
| 2019 | 3.5–4.0 | 1.50–2.72 | 3.58–4.55 | 4.63–6.64 | 1.50–2.21 | 0.48–0.83 |
| **2020** | **3.5–14.7** | 0.55–1.90 | 2.72–3.74 | 4.04–5.07 | 0.24–2.48 | **−9.49–7.41** |
| 2021 | 4.6–6.7 | 0.93–1.73 | 2.67–3.17 | 4.69–**16.41** | 1.16–6.24 | 0.50–7.48 |
| 2022 | 3.5–4.2 | 1.55–4.02 | 3.11–**7.08** | **16.41–20.88** | **6.88–9.00** | −0.40–1.70 |
| 2023 | 3.4–3.9 | 3.53–**4.88** | **6.13–7.79** | 4.46–16.57 | 3.09–7.12 | 0.26–1.26 |
| 2024 | 3.7–4.3 | 3.75–4.63 | 6.08–7.17 | 4.77–6.26 | 2.41–3.48 | 0.31–1.19 |
| 2025 | 4.0–4.4 | 4.00–4.55 | 6.17–6.95 | 3.25–5.37 | 2.33–3.02 | −0.13–0.95 |

Development maxima: mortgage 4.95; HPI 6.35; CPI 3.90; 10-year 3.50.

**Component decomposition question (Stage 1 request; prompt §7).**
- **Can it be computed from frozen predictions without refit or regeneration?** Yes, in principle. `evaluation.npy` (event, month) and `M1/M2_evaluation.npy` are frozen and hash-recorded, so a per-year realized-class log-loss split is pure arithmetic on fixed arrays.
- **But:**
  1. the arrays are not in this environment;
  2. a 2020-specific realized-class split is **not registered**, so computing it would be **new post-hoc analysis**.
  
  Not run.
- **What the frozen evidence already identifies (not inferred from calibration alone):**
  - **Full-period realized-class decomposition of Δ joint LL** [D `score_accounting/joint_loss_by_realized_event`]:
    - no-event intervals (244,836): **+0.020298**
    - default intervals (280): −0.000042
    - payoff intervals (3,823): −0.004135
    - Total: +0.016122.
    - M2's deterioration accrues entirely on intervals where **no event occurred**. M2 is *better* on realized payoff and default intervals.
  - **2020 by cause (Brier, frozen per year):** 2020 payoff Brier Δ = +0.022611, which is **100.2% of the full-period payoff-Brier Δ**. The 2020 default Brier Δ is +0.000049.
  - Together these show that the 2020 deterioration is associated with **payoff probability mass assigned to intervals that had no event**.
  - A strictly 2020-specific realized-class log-loss split is **not** frozen. To get it exactly, compute per calendar year the mean of −log p_y(M2) + log p_y(M1) by realized class on the frozen seen arrays, under a new registration.

**Answers to the prompt's questions.**
- **Is 2020 unique in predictor movement?**
  - **No.** It is the most extreme (median distance 655) and the only year with a large unemployment/GDP shock.
  - **2021–2025 are all fully outside development support**, especially 2022–2023: mortgage rates 2–3 points above the development maximum, HPI growth three times the development maximum, CPI twice it.
- **Is 2020 unique in model deterioration?**
  - **In absolute log-loss contribution, yes** (89%).
  - **In relative terms and in calibration, no.**
    - 2022–2026 show a consistent **+7.6% to +13.4%** relative deterioration (5 of 5 years).
    - M2's payoff calibration-in-the-large fails in every year from 2022, in the *opposite* direction to 2020: it predicts 0.18–0.46% against 0.85–1.32% observed, about 3–5 times too low.
    - The pooled 2.83% vs 1.54% "over-prediction" is a mixture of a 4.5-fold over-prediction in 2020 and roughly 4-fold under-predictions in 2022–2026.
  - Part of the absolute concentration therefore reflects **log-loss asymmetry**:
    - Over-predicting a rare event on the 98% of intervals that have no event is costly.
    - Under-predicting a rare event costs little in absolute log loss.
- **Does large macro movement reliably coincide with large deterioration?**
  - **No, not reliably.**
    - 2021 (far out of support, HPI to 16%, CPI to 6%) shows M2 *better* (−4.5%).
    - 2022 (HPI to 21%, CPI to 9%, rates to 7%) shows +7.6%, smaller than 2023 (+13.4%) at lower distance.
  - Across the eight years, rank association between distance and relative Δ is weak. I state this descriptively and do not test it.
- **Does the evidence support "regime-dependent"?**
  - It supports more than Stage 1 credited: the *direction* of M2's payoff calibration error differs systematically between the low-rate/shock period (2020) and the high-rate period (2022–2026). Those are distinguishable macro states within the one history.
  - It is still one realized trajectory, the regimes are not defined beforehand, and there is no replication.
  - **"Regime-dependent" remains unestablished as a general property.**
- **Safest descriptive wording.** **"Period-heterogeneous"** (sign- and magnitude-varying by calendar year), with **"absolute log-loss deterioration concentrated in 2020"**. Add an explicit statement that payoff calibration fails in both directions across periods with different rate levels.

---

## I. Pooled AUC decomposition (payoff)

**A. Stratum pair weights** (share of payoff-case × non-payoff-control pairs):

| Resolution | Population | Within | Between |
|---|---|---|---|
| Year | full primary (248,939) | **17.195%** | 82.805% |
| Month | eligible months only (55 months, 191,938 intervals) | **1.979%** | 98.021% |

**B. Stratum-specific AUC gains (M2 − M1):**

| Resolution | Within (pair-weighted) | Between (solved) |
|---|---|---|
| Year | **+0.006357** | +0.071673 |
| Month | **−0.000399** | +0.049299 |

Per-year *within-year* AUC gains (frozen `stability/calendar`):

| 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| +0.0070 | +0.0047 | −0.0044 | **+0.0646** | −0.0140 | +0.0024 | −0.0044 | −0.0097 |

Per-year pair-weighted *within-month* gains (closure per-month rows):

| 2019 | 2020 | 2021 | 2022 | 2023 (5 months) | 2024 (2) | 2025 (2) |
|---|---|---|---|---|---|---|
| −0.0018 | −0.0009 | +0.0031 | +0.0042 | −0.0303 | −0.0039 | +0.0130 |

**C. Contributions to the pooled gain:**

| Resolution | Within | Between | Sum |
|---|---|---|---|
| Year | +0.0010931 | +0.0593492 | +0.0604423 (full pooled) |
| Month | −0.0000079 | +0.0483228 | +0.0483149 (**eligible-month** pooled) |

**D. Contribution shares:**

| Resolution | Within | Between |
|---|---|---|
| Year | 1.81% | 98.19% |
| Month | −0.016% | 100.016% |

**Stability across resolutions:**
- **Pair weights are not stable.** The within share falls from 17.2% to 2.0%, as expected with finer strata.
- **Within-stratum gain changes sign**: +0.0064 at year level, −0.0004 at month level.
  - The positive within-year gain is driven mainly by 2022 (+0.065). In 2022 the within-month gains are small (+0.004), so the 2022 within-year gain is largely **between months inside 2022**, a year of rapidly changing rates. Year stratification still leaves calendar separation inside the "within" component.
- **Contributions are qualitatively stable** (between ≥ 98% at both resolutions). This is the robust conclusion.
- **Support exclusions change the estimand.**
  - All 31 excluded months are **2022-10 or later** (the payoff-sparse high-rate period).
  - The month-level decomposition describes mostly 2019-01..2022-09.
  - The eligible-population pooled gain (+0.0483) is 20% smaller than the full-primary gain (+0.0604).
  - The manuscript discloses the exclusion. It does not say *which* period is excluded.

Never conflated here: pair weights (A) are not contribution shares (D).

---

## J. Unseen-vintage analysis (most important Stage 2 issue)

**Verified facts:**

| Item | Value | Source |
|---|---|---|
| Definition | 30% evaluation-role facilities of vintages 2018/2020/2022, intervals in 2019-01..2026-02 | code (`block`, `validation_role`) |
| Facilities / intervals | 17,352 / 652,508 (2018: 5,768 / 184,349; 2020: 5,683 / 261,306; 2022: 5,901 / 206,853) | `macro_support_validation_design.json` |
| Events | defaults 588 (320 / 74 / 194); payoffs 7,323 (4,477 / 1,879 / 967) | same |
| Event rates | default 0.090%, payoff 1.122% (seen: 0.112%, 1.536%) | R |
| Overlap with seen | No facility overlap (different vintages); same calendar window | code |
| Entry / age | First-row age median 4 months (min 4, max 23); interval age median 26 (seen: 104) | [D composition] |
| Rate gap (note rate − current 30-year) | Interval median **−1.92** (seen +0.40; dev +0.90) | [D composition] |
| Approximate calendar exposure | 2019 8.8%, **2020 9.5%**, 2021 11.9%, 2022 14.9%, 2023 18.1%, 2024 17.1%, 2025 16.0%, 2026 3.8% (seen: 2020 20.8%; 2022–26 39%). Derived from `multi_vintage_recovery_audit.json/age_period_support` (all sampled rows, both roles). Validation: the same method reproduces seen evaluation shares to ±0.001, except partial 2026. **Approximate**; this is not a frozen unseen per-year decomposition. | audit |
| Encoding | Cohort scored with the **2006 reference effect**; young-age duration bands are well supported | §F |
| Joint LL | M1 0.070631, M2 0.069811 (Δ −0.000819, −1.16%) | R |
| Brier | default M1 0.00089946 / M2 0.00090017 (Δ +7e-7); payoff 0.011217 / 0.011682 (Δ +0.000465) | R |
| AUC | default 0.7779 → 0.7365; payoff 0.5522 → 0.7373 | R |
| Calibration | See §K. M1 default is near-perfect (slope 1.01); M1 **over-predicts payoff** by 70% (1.91% vs 1.12%). M2 **under-predicts default** by 71% (0.026% vs 0.090%) and under-predicts payoff by 25%. | R |
| Per-vintage or per-year unseen scores | **Not frozen** | — |
| Uncertainty for unseen Δ | **None frozen** | — |

**Statements A–E:**

| Statement | Supported? | Reason |
|---|---|---|
| **A.** M2 performs better on unseen vintages | **No** | Only joint LL improves (−1.2%). Payoff Brier, default Brier, default AUC and default calibration are worse. Payoff AUC and payoff calibration are better. This is mixed, not "better". |
| **B.** The primary joint-loss point estimate reverses on the unseen set | **Yes, as a point-estimate statement** | Δ = −0.000819 exactly, with no interval. |
| **C.** Performance differs across evaluation populations | **Yes (descriptive)** | All metric vectors differ, several by large amounts (payoff AUC Δ +0.185 unseen vs +0.060 seen). |
| **D.** The difference is caused by vintage/population | **No** | Vintage, age (median 26 vs 104 months), calendar exposure, refinance moneyness and encoding (2006-reference cohort) all differ together. |
| **E.** The difference represents formal statistical transportability | **No** | No transport assumptions, no selection diagram, no invariance test. |

**Can calendar exposure plausibly account for the reversal?**
- **Not by calendar re-weighting alone.** As reviewer plausibility arithmetic only (not registered, not evidence for the manuscript), I re-weighted the *seen* population's frozen per-year Δ joint LL with the approximate unseen calendar weights. The result is **+0.0104**: M2 still worse, and about 12% relative.
- A pure change in calendar mix, holding the seen population's within-year behaviour fixed, therefore does not produce a reversal.
- The reversal must involve **within-year differences between populations**. The frozen calibration suggests a candidate, stated descriptively:
  - M1, scoring young loans with the 2006-reference cohort effect and development-era payoff hazards, over-predicts unseen payoff by 70%.
  - M2's high-rate macro terms (2022+) push payoff probabilities down. This offsets M1's error on payoff while degrading default calibration.
  - This is a population × encoding × calendar **interaction**.
- **Do the frozen artifacts distinguish population composition from calendar composition? No.** No per-year or per-vintage unseen scores exist. The arithmetic above rules out only the simplest calendar-only account. Stage 1's specific worry ("calendar exposure could plausibly account for the reversal") is **WEAKENED**. Stage 1's broader point (population and calendar are confounded) is **CONFIRMED**.

---

## K. Calibration audit

The calibration "intercept" in the artifacts is the intercept of the joint recalibration fit logit P(y) = a + b·logit(p) (`metrics.calibration`). **It is not calibration-in-the-large.** CITL below is the mean predicted minus observed.

**Seen population (R `/primary/*/calibration`):**

| Cause | Model | Observed | Mean pred. | CITL (pred − obs) | Slope | Recal. intercept | Top-decile pred / obs |
|---|---|---|---|---|---|---|---|
| Default | M0 | 0.001125 | 0.000484 | −0.000641 | 0.567 | −2.404 | — |
| Default | M1 | 0.001125 | 0.000649 | **−0.000475** | **0.482** | −2.935 | 0.00406 / 0.00269 |
| Default | M2 | 0.001125 | 0.000886 | **−0.000239** (better) | **0.350** (worse) | −4.049 | 0.00572 / 0.00269 |
| Payoff | M0 | 0.015357 | 0.010694 | −0.004664 | 0.569 | −1.581 | — |
| Payoff | M1 | 0.015357 | 0.011379 | −0.003978 | 0.509 | −1.863 | 0.0222 / 0.0220 |
| Payoff | M2 | 0.015357 | 0.028305 | **+0.012948** (worse) | **0.249** (worse) | −3.043 | **0.1929 / 0.0295** |

**Unseen population (R `/unseen_vintage/*/calibration`):**

| Cause | Model | Observed | Mean pred. | CITL | Slope | Recal. intercept |
|---|---|---|---|---|---|---|
| Default | M1 | 0.000901 | 0.000858 | −0.000043 | **1.009** | 0.109 |
| Default | M2 | 0.000901 | 0.000258 | **−0.000643** (worse) | **0.764** (worse) | −0.580 |
| Payoff | M1 | 0.011223 | 0.019067 | **+0.007844** | 0.346 | −3.089 |
| Payoff | M2 | 0.011223 | 0.008423 | **−0.002800** (better) | **0.529** (better) | −1.599 |

Reliability deciles (10 equal-count bins) are frozen for all of these. Key pattern: M2's seen-payoff miscalibration is concentrated in the **top decile**. Per-year calibration for the seen population is in §H.

**Where M2 improves one dimension and worsens another:**
- **Seen default:** CITL improves (|error| halves), slope worsens (0.48 → 0.35).
- **Seen payoff:** both worsen. Per year, the sign of the CITL error flips (2020 over by 4.5-fold, 2022–26 under by about 4-fold).
- **Unseen:** payoff improves on both CITL and slope; default worsens on both.

**Blanket statements to reject:**
- "M2 is worse calibrated".
- "M2 over-predicts payoff", stated generally. That is true pooled over the seen population, false for seen 2022–2026, and false for the unseen population.
- "M1 is poorly calibrated". M1's default CITL failure is a 2020 phenomenon (§E). Unseen M1 default calibration is excellent. Unseen M1 payoff over-predicts.

---

## L. CIF verification

| Item | Finding |
|---|---|
| Conditional entry | One landmark per seen facility, at its first eligible evaluation interval. Window-relative time; `entry_time = 0` for all. Contiguity is asserted. |
| Entry distribution | 2019-01: 5,564 (99.02%); 2019-05: 23; 2019-09: 19; 2019-10: 5; 2020-01: 4; four singletons. Latest 2020-01. |
| Observed reference | Aalen–Johansen with payoff as competing event (`survival/math.py::aj`). Default and payoff CIFs come from the same recursion; conservation is checked to 1e-10. |
| Censoring weights | KM of censoring estimated **within the same landmark cohort** (pooled across facilities); events precede tied censoring. **Censoring is negligible:** censor survival is 0.9994 at 12 months, 0.9992 at 24 and 36, 0.9988 at 60. Total censored by month 60 is 5 facilities (5,614 of 5,619 have known status). |
| IPCW metrics | `horizon_metrics`: Brier and cumulative/dynamic AUC with controls = "all non-defaults including prior competing payoff". The payoff block reuses the function with swapped codes, so its field is still named `default_facilities_by_horizon` (naming artefact, values correct). |
| Model recursion | `curves()`: S_k = S_{k−1}(1 − d_k − p_k); F_D += S_{k−1} d_k; F_P += S_{k−1} p_k. This is correct. It is driven by the realized PIT macro path (`path_forecast`, the same `macro_join`) with deterministic duration increments. |
| Horizons / values | 12 / 24 / 36 / 60, all SUPPORTED. All Table F values verified (§4). Unreported but frozen: IPCW payoff Brier at 24 months, M1 0.2255 vs **M2 0.3989**; cumulative payoff AUC 0.591 vs 0.591. |
| Calendar-path dependence | 99% of 24-month paths span 2019-01..2020-12; all 24-month paths include 2020. It is the same facilities, the same predictions (CIF arrays generated from the same frozen models) and the same calendar as the score analysis. |

**CG03 (compatibility of conditional entry and the observed estimator with the censoring weights):**
- **Resolvable from the implementation for practical purposes.**
- Window-relative entry with a common origin makes the AJ estimator standard (no left-truncation in analysis time).
- With censor survival ≥ 0.9988, IPCW and unweighted proportions differ negligibly.
- The only remaining assumption is conditional independent censoring of 5 censored facilities, which is immaterial at this scale.
- **CG03: RESOLVED (practically immaterial)**, with one caveat: the observed default reference inherits the target concern in §E.

**Classification: SUPPORTING_ILLUSTRATION.**
- It is not independent corroboration.
- Intervals are **not necessary** for its descriptive role. The manuscript claims no test, and the M2 payoff gaps of +0.42 / +0.30 / +0.21 are far beyond any sampling scale relevant to 5,619 facilities conditional on the path.
- Intervals would also not address the dominant uncertainty, which is the single realized path.

---

## M. Uncertainty

**Frozen intervals that exist** (R `/paired_facility`, `/paired_calendar`; fixed models; percentile 95%; 1000 draws):

| Δ (M2 − M1), seen | Point | Facility CI | Calendar-year CI |
|---|---|---|---|
| Joint LL | +0.016122 | [+0.01541, +0.01678] | [+0.0000533, +0.04150] |
| Default Brier | +0.0000491 | [+0.0000012, +0.0001463] | **[+0.0000262, +0.0000776]** (excludes 0) |
| Payoff Brier | +0.004698 | [+0.004562, +0.004822] | [−0.0000474, +0.013215] |
| **Default AUC** | **−0.07068** | **[−0.0939, −0.0453]** | not computed (`ranking = False`) |
| **Payoff AUC** | **+0.06044** | **[+0.0513, +0.0700]** | not computed |

**What this changes from Stage 1.**
- Stage 1 said there were "no intervals for any AUC difference". That is **partially REFUTED**: facility intervals exist for the pooled default and payoff AUC differences. The manuscript does not report them.
- The default-Brier deterioration excludes zero under **both** schemes. The manuscript does not report this either.
- Still missing:
  - intervals for within-stratum AUC differences (year or month),
  - any unseen difference,
  - calibration statistics,
  - CIF quantities,
  - training/refit uncertainty.

**Calendar bootstrap.**
- It is reproduced exactly by resampling the 8 annual aggregates.
- Its lower bound is set by draws that omit 2020; P(2020 absent) = (7/8)^8 = 0.344.
- It is a sensitivity display, not a calibrated interval.
- The ex-2020 interval [−0.00151, +0.00671] is plausible (my re-simulation gives [−0.00150, +0.00671]) but not exactly verifiable.

**§12 — Does the lack of AUC intervals invalidate the decomposition?**
1. **Is the decomposition arithmetically valid?** Yes. It is an exact identity on fixed arrays. It reconciles at year level from frozen aggregates and at month level in the closure, to below 1e-9.
2. **Is inferential uncertainty required?**
   - Not for descriptive statements about *this* evaluation set ("98% of the pooled gain comes from between-year pairs").
   - Yes, for statements generalising to "macro inputs do not improve within-period ranking", or for treating "no within-month gain" (−0.0004) as an established zero.
   - The missing interval **limits generalisation only**. It does not invalidate the decomposition and is **not preprint-blocking**, provided the wording stays descriptive. The manuscript largely does this ("support-qualified", "no formal equivalence bound").
   - An interval for the within-month difference would strengthen the inference but is not necessary for the identity.

**§14 — Unseen uncertainty.**
- Feasible without refitting: `evaluation.npy` and `M1/M2_evaluation.npy` cover all 901,447 rows, including the unseen ones.
- It is **not frozen and not registered**. The protocol treats unseen as a "separate extrapolation sensitivity" with no paired intervals, so computing one would be **new post-hoc analysis**. Not computed.
- **Wording.**
  - "The joint-loss **point estimate** is lower for M2 on the unseen set (−0.0008, about 1%)" needs no interval.
  - "The primary metric **reverses direction**" as a finding needs one. So would any use of the reversal to support "population-dependent".
  - A sign reversal in a point estimate is not a statistically established difference.

---

## N. Sampling and training procedure

**§19 Sampling design.**

| Item | Finding (sources: `multi_vintage_recovery_audit.json/protocol` and `vintages`; `SAMPLE_EXPANSION_AMENDMENT.md`; `data/sampling.py`) |
|---|---|
| Source population | Freddie Mac single-family loan-level "Standard" origination universe, **FRM only**, release R47 (pinned July 2026). Universe sizes: 2006 1,193,543; 2008 1,233,145; 2014 1,142,999; 2018 1,285,434; **2020 3,913,740**; 2022 1,580,752; 2010 from an earlier frozen universe. |
| Method | **Deterministic.** SHA-256 of the namespace plus loan ID, ranked; the first N are taken. Selection is identifier-only ("No outcomes/credit/geography/history in selection"). 2010 reuses an earlier frozen ranking with a different salt. |
| Sample size | **20,000 per vintage** (7 × 20,000), frozen before performance access. N was chosen in a Task 4 amendment using the *event count of an earlier 1,000-loan 2010 pilot* for precision planning. This is not outcome-adaptive selection. |
| Eligibility | Then Task 9A eligibility and the 70/30 role hash (§D). |
| Reproducible | Yes, given the source files: sample hashes, universe hashes and counterfactual-inclusion hashes are recorded and match. |
| Sampling weights | **None.** Equal per-vintage samples, so sampling fractions range from 0.5% (2020) to 1.75%. Pooled metrics weight vintages by sample composition, not by portfolio share. |

Verdict on the Stage 1 concern: **PARTIALLY_RESOLVED**. It is resolved in the evidence. The manuscript still does not describe the design, and the lack of weights limits population inference.

**§20 L2 / training procedure.**
- C = 1.0 is **fixed in advance** in the frozen protocol (`freeze()`), registered before fitting (`ledger.check("REGISTERED_BEFORE_FIT")`). It equals the sklearn default.
- No grid, no CV and no validation tuning exist in the code.
- Evaluation data could not influence the fit:
  - preprocessing raises an error if any row has role ≠ 0;
  - the temporal ledger was consumed once (`prediction_generation_count = 1`; replay max error 0.0; `post_evaluation_retuning = false`).
- Caveat: Task 9A support design inspected evaluation-role **event counts** before the Task 10 protocol was frozen (`virgin_holdout = false`; "prior_support_outcomes_inspected": true). No evidence shows that model choices depended on evaluation performance.
- **Stage 1 "L2 selection unspecified": RESOLVED** (fixed default, no tuning, no evaluation influence).

---

## O. Bibliography and internal literature findings (no web)

| Stage 1 statement | Finding | Classification |
|---|---|---|
| 8 .bib entries uncited (Sadhwani2021, Wang2024, Schwartz1989, Fine1999, Roschewitz2025, OPSurv2024, Graf1999, Brier1950) | Exactly these 8 are uncited; all 23 cited keys are present | CONFIRMED |
| Sadhwani et al. 2021 relevant but not engaged | In .bib, 0 mentions in the manuscript | CONFIRMED (relevance judgement: NEEDS_EXTERNAL_LITERATURE_CHECK) |
| No transportability literature | 16 uses of "transport" in the manuscript; no transportability or external-validity citation in the .bib or the text | CONFIRMED; NEEDS_EXTERNAL_LITERATURE_CHECK for what to cite |
| No COVID/forbearance literature | "forbear" appears 0 times in the manuscript and .bib; "pandemic"/"COVID" once each (to disclaim a COVID effect) | CONFIRMED; NEEDS_EXTERNAL_LITERATURE_CHECK |
| Metadata/status tags | "PEER_REVIEWED" appears 23 times in the manuscript reference list (not in the .bib). The Bhattacharya2019 URL points to a tcd.ie profile page in both files (DOI present). | CONFIRMED |
| 2026 references unverifiable | Bu2026, Peng2026, Bianchi2026 | NEEDS_EXTERNAL_LITERATURE_CHECK |

---

## P. Stage 1 concern-by-concern reassessment

| # | Stage 1 concern (original statement, abbreviated) | Original severity | Evidence inspected | Stage 2 finding | Verdict | Revised severity | Reason |
|---|---|---|---|---|---|---|---|
| 1 | Title / "population-dependent transport" rests on an interval-free, confounded point reversal | High | R unseen block; support design per-vintage counts; recovery-audit calendar support; diagnostics composition | Unseen differs simultaneously in age (26 vs 104 months), moneyness (−1.9 vs +0.4), 2020 exposure (~9.5% vs 20.8%) and cohort encoding (2006 reference). There is no interval and no per-year or per-vintage unseen scores. | **CONFIRMED** | High | The evidence cannot attribute the difference to population. |
| 2 | "Regime-dependent" rests on one year | Medium–High | Per-year scores, calibration, macro ranges and distances | Deterioration and miscalibration also appear in 2022–2026 (relative +8–13%; payoff under-prediction about 4-fold), and in the opposite direction to 2020. There are two distinguishable error patterns across macro states, but one history and no a-priori regimes. | **WEAKENED** | Medium | More descriptive support than Stage 1 credited; still not a general property. |
| 3 | Uncertainty missing for AUC differences, within-month result, unseen, calibration, CIF | High | `paired_facility`, `paired_calendar`; closure | Facility CIs **exist** for pooled payoff and default AUC differences, and default Brier excludes 0 under both schemes. All are unreported. Within-stratum, unseen, calibration and CIF intervals are still absent. | **WEAKENED** (partially REFUTED for pooled AUC) | Medium | Some gaps are reporting gaps, not evidence gaps. |
| 4 | One realized macro history | High | Diagnostics (88 / 86 months, distances); calendar CI reproduction | Confirmed. The calendar bootstrap is a resample of 8 numbers. Stage 1's phrase "effective sample of about one episode" should not be read as a derived effective sample size (§17 below). | **CONFIRMED** | High (fundamental scope limit; disclosure-level) | — |
| 5 | Unseen-vintage reversal: weakest evidence, carries the headline | High | R unseen; calibration | Δ = −0.000819 verified. The joint-LL gain coincides with M2 offsetting M1's 70% payoff over-prediction, while default calibration worsens (71% under). | **CONFIRMED** | High | "Reversal" is a point-estimate statement only. |
| 6 | Calendar exposure could plausibly account for the reversal | High | Approximate unseen calendar weights; seen per-year Δ | Re-weighting the seen per-year Δ to the unseen calendar mix gives +0.0104 (no reversal). Calendar-only is implausible, but population and calendar × population remain inseparable. | **WEAKENED** | Medium | The simple account fails; the confounding persists. |
| 7 | Forbearance may contaminate the 2020 default endpoint | Medium (highest-priority open question) | `event_category`; protocol; field audits; CIF monthly table; per-year counts; design doc | No forbearance handling. 81 defaults in one month (2020-06), and 60% of seen defaults in 2020. Default AUC, Brier and calibration failures are concentrated in 2020. The project's own design required harmonisation, and it was not implemented. | **STRENGTHENED** | **High for default-specific claims**; low for payoff-centred claims | MATERIAL_TARGET_RISK |
| 8 | Results specific to additive spec, national rate level, no refi incentive | Medium | `models.py`; protocol; REDUCED sensitivity | Confirmed (interactions and period FE prohibited). In addition, the pre-specified REDUCED macro sensitivity behaves very differently (LL +0.0020; default AUC 0.81) and is unreported. Duration-band fallback affects 4.5% of seen intervals. | **STRENGTHENED** | Medium–High | Specification dependence is demonstrable from frozen evidence. |
| 9 | CIF: implementation compatibility open (CG03), not independent, no intervals | Medium | `cif.py`, `survival/math.py`; CIF tables | CG03 is practically resolved (common origin; censor survival ≥ 0.9988). Non-independence confirmed. Intervals are unnecessary for the illustrative role. | **WEAKENED** (CG03 RESOLVED; dependence CONFIRMED) | Low | Supporting illustration only. |
| 10 | Non-virgin, partly post-hoc evaluation | Medium | Ledger; closure registration; support design | Task 10 ledger consumed once, no retuning, code hashes match. Evaluation event counts were inspected beforehand. Month closure and year decomposition are post-hoc (registered 2026-10-08, after a review audit). | **CONFIRMED** | Medium–Low | Disclosed; the primary experiment is well controlled. |
| 11 | Payoff includes maturity; not quantified | Low–Medium | Protocol; term composition | Code 01 combines payoff and maturity. 15-year and 10-year loans in the 2006/2008/2010 seen vintages could reach scheduled maturity inside the window. There is no frozen count of maturity events. | **CONFIRMED** | Low–Medium | Limitation. Quantifiable only from private data via `maturity_month`. |
| 12 | Sampling design unspecified | Low | Recovery audit; sampling code; amendment | Deterministic identifier hash, 20k per vintage, FRM Standard, frozen pre-access, no weights. | **RESOLVED** in evidence (PARTIALLY_RESOLVED for the manuscript) | Low | Manuscript-only fix, plus a note on unweighted pooling. |
| 13 | L2 selection unspecified | Low | Protocol; models; ledger | C = 1 fixed a priori; no tuning; no evaluation influence. | **RESOLVED** | None | — |
| 14 | Comparator (M1) poorly calibrated out of time | Medium | Per-year and unseen calibration | M1's seen default under-prediction is a 2020 phenomenon (linked to #7). Outside 2020, M1 is reasonable in-the-large. Unseen M1 default is excellent, but unseen M1 payoff over-predicts 70%. | **WEAKENED** (refined) | Medium (unseen payoff) / Low (seen) | The comparator's weakness is population- and period-specific. |
| 15 | Release lags: possible mild look-ahead | Low | `macro_join`; PIT engine; lag probe | Join at t0 = end of the prior month with ALFRED archive-start bounds. | **WEAKENED** | Low | Conservative by construction; residual is uncertified, not demonstrated. |
| 16 | Month-support rule probably excludes later payoff-sparse months | Low | Closure excluded list | All 31 excluded months are ≥ 2022-10. | **CONFIRMED** | Low–Medium | The estimand is mostly 2019–2022. |
| 17 | CIF not independent of the calendar-score evidence | Medium | CIF code and entry distribution | Same facilities, models, path. | **CONFIRMED** | Low (disclosed) | — |

**Tally of the 14 required concerns:**
- CONFIRMED or STRENGTHENED: **7** (#1, 4, 5, 7, 8, 10, 11).
- WEAKENED or RESOLVED: **7** (#2, 3, 6, 9, 12, 13, 14).
- Extras #15–17: one weakened, two confirmed.

**§17 — Sample-size language.** Safest wording, which corrects Stage 1's "effective sample of about one episode":
- **88** distinct development calendar months and **86** evaluation months. These are serially dependent, so they are not independent observations.
- **8** calendar-year blocks (one partial).
- **1,568,661 / 248,939 / 652,508** interval observations.
- **42,609 / 5,619 / 17,352** facilities. There is no borrower identifier, so borrower independence is not established.
- **One** realized national macro trajectory.

What can be said: macro-coefficient information comes only from co-variation over 88 dependent months within one trajectory, and out-of-time assessment covers one subsequent trajectory. What cannot be said is any number labelled "effective sample size". None is derived, and Stage 1's phrase should be read as a qualitative scope statement only.

---

## Q. New evidence-level flaws and concerns discovered in Stage 2

| ID | Concern | Evidence | Severity |
|---|---|---|---|
| N1 | **Pre-specified REDUCED macro sensitivity omitted from the manuscript.** It shows much smaller LL deterioration (+0.0020) and large default-AUC gain (0.69 → 0.81). | R `/primary/REDUCED_*`; protocol `sensitivities.reduced = true` | **Material** (selective reporting of the pre-specified design; affects generality) |
| N2 | **Pooled payoff "over-prediction" masks a sign reversal by year.** M2 over-predicts 4.5-fold in 2020 and under-predicts about 4-fold in every year 2022–2026. Relative LL deterioration is +8–13% per year in 2022–2026. The manuscript's §4.2 and §5 framing is incomplete. | Diagnostics `annual_calibration`; R `stability` | **Material** (interpretation of concentration and mechanism) |
| N3 | **Unseen joint-LL gain arises with offsetting calibration errors.** M1's unseen payoff over-prediction comes with the 2006-reference cohort fallback; M2 corrects payoff but worsens default (71% under). The manuscript does not state what the "frozen encoding fallback" assigns. | R unseen calibration; `models.py`; frozen zero columns | **Material** (central to "population" framing) |
| N4 | **Duration-band fallback for seasoned seen loans.** 11,167 intervals (4.5%) aged 181–240 months are scored with the 0–12-month reference effect (zero coefficients). It is disclosed in artifact limitations but not in the manuscript. | `error_partitions/duration:181-240`; zero-column list | Moderate |
| N5 | **PMMS methodology break (2022-11-17) inside the evaluation window** is pooled without disclosure, contrary to the project's own provenance rule. | Series registry; `MACRO_DATA_PROVENANCE.md` | Minor–Moderate |
| N6 | **Unreported frozen results** that bear on claims: facility CIs for pooled AUC differences; default-Brier Δ excluding 0 under both schemes; full-period realized-class LL decomposition (deterioration entirely on no-event intervals); CIF IPCW payoff Brier. | R; diagnostics | Moderate (reporting) |
| N7 | Evidence map embeds Task 17/18 review verdicts (process contamination; not a scientific flaw) | Map fields | Process only |

No new problems were found in leakage, target coding versus protocol, cohort overlap, censoring logic, competing-risk recursion, prediction/label alignment, calendar join, macro availability, hash/provenance claims, or arithmetic consistency. Details are in §R.

---

## R. Fatal-flaw reassessment (evidence level)

| Candidate | Finding |
|---|---|
| Leakage not visible from the manuscript | **None.** Preprocessing fits on role 0 only (an error otherwise). Facility-role disjointness is asserted. Macro is joined at the end of the prior month with vintage bounds. Models are frozen before a single ledger consumption. |
| Target coding error | **None relative to the protocol.** The code implements the protocol exactly. The *semantic* problem (forbearance) is a target-validity risk, not a coding error (§E). |
| Cohort overlap | **None.** Roles are disjoint, and unseen vintages are distinct. |
| Invalid censoring | **None found.** First-prefix rules hold; administrative, unknown and ambiguous months are censored; CIF censoring is negligible. |
| Incorrect CR recursion | **None.** Both `curves` and `aj` are correct and conservation-checked. |
| Prediction/label misalignment | **None.** Predictions come row-for-row from the same sealed array. The closure reconciles full AUCs to 1e-9. |
| Calendar join errors | **None.** `month` is the target-month ordinal; year blocks use `month // 12`; the CIF path uses the same join. |
| Macro availability violations | **None demonstrated.** The residual is uncertified release dates under a conservative bound. |
| Unsupported hash/provenance claims | **None.** All 369 manifest hashes, the 7 code hashes against the ledger, prediction hashes across artifacts, and the closure chain (CRLF/LF reconciled) are consistent. |
| Specification differs from the manuscript | **No contradiction.** The omissions are N1, N4 and N5. |
| Arithmetic inconsistent with frozen aggregates | **None.** 255 of 257 bindings verified; 2 are plausible but unverifiable. |
| Evidence-map bindings to wrong artifacts | Only minor SOURCE_ERROR (Q0040/41) and label-only bindings. |

**Stage 2 fatal-flaw judgement: NO.** There is no flaw that invalidates the central payoff-based findings. The forbearance issue is a **material target risk for default-specific claims**, not a fatal flaw for the paper's central claims: the frozen realized-class decomposition shows the joint-LL deterioration does not run through default intervals.

---

## S. Updated preprint decision

**MAJOR REVISION** (unchanged).
- What newly inspected evidence did:
  - **Raised confidence** in the integrity and reproducibility of the core numbers (exact provenance, code bound to the ledger, arithmetic reproduced).
  - **Added material interpretive problems:** N1–N3, and the strengthened forbearance risk (#7).
  - These offset each other, so the decision category does not change.
- Why not "NOT READY — ANALYSIS REQUIRED": every blocking item can be cleared either by the manuscript-only corrections in §W or by small descriptive analyses on already-frozen arrays (§U).

## T. Updated peer-review decision

**WEAK REJECT** (unchanged).
- The contribution is a careful but modest validation case study on one specification and one history.
- New evidence shows its headline framing (population/regime dependence) is less supported than its central descriptive findings.
- Material pre-specified results (REDUCED) and calibration heterogeneity are absent from the paper.
- Better provenance does not offset these at a peer-review bar.

## U. Analyses genuinely required before preprint

Each item is required **only if the corresponding claim is retained**. Otherwise the §W wording changes suffice.
1. **To retain any default-specific interpretation** (default AUC/Brier/calibration, CIF default rows, "baseline default under-prediction"): a frozen-label descriptive tabulation of `assistance_plan` / `payment_deferral_flag` / `disaster_flag` around seen and unseen default events in 2020–2021, plus the provider documentation listed in §E. No model involvement.
2. **To retain "population-dependent", "transport" or "reversal" as a finding:** a registered per-calendar-year (and per-vintage) score breakdown of the unseen population on the frozen arrays, and a facility (and, where feasible, calendar) paired interval for the unseen Δ. Both are new post-hoc analyses that need no refit.

## V. Analyses useful only for peer review

- Per-year realized-class log-loss decomposition (seen) on frozen arrays, to show 2020 and 2022–2026 mechanisms separately.
- Intervals for within-year and within-month AUC differences (facility bootstrap on frozen arrays).
- Maturity-event count inside payoff code 01, using `maturity_month`.
- Facility-weighted scoring sensitivity.
- Calendar-standardised seen vs unseen comparison.
- Default-endpoint sensitivity excluding assistance/deferral-flagged episodes (would require a new protocol).
- Reporting and discussion of the pre-specified REDUCED and RATE sensitivities alongside M2.
- A PMMS-break sensitivity (pre- vs post-2022-11 split of 2022–2023 scores).

## W. Claims requiring manuscript revision (manuscript-only)

1. **Title.** Drop "Population- and Regime-Dependent Transport". Replace it with descriptive language (e.g. "period-heterogeneous out-of-time performance … across two evaluation populations").
2. **Abstract and §4.5.** State "the joint-loss point estimate is about 1% lower for M2 in the unseen population (no interval)". Do not say "reverses". Report that unseen M2 default calibration worsens (−71% CITL) while payoff calibration improves, and that M1 over-predicts unseen payoff by 70%.
3. **§3.2 and §4.5.** Say what the encoding fallback does: unseen cohorts are scored as the 2006 reference, and unsupported duration bands (181–240 months; 4.5% of seen intervals) as the 0–12 reference.
4. **§4.2 and §5.** Replace "payoff over-prediction" with the per-year pattern: over in 2020 and under in 2022–2026. Report the 2022–2026 relative deterioration (+8–13%). Explain that absolute 2020 concentration partly reflects log-loss asymmetry.
5. **§4.3.** Report that 2021–2025 are also entirely outside development macro support. 2020 is not unique in extrapolation.
6. **§3.1 and §6.** Disclose that the default endpoint does not account for forbearance or deferral, and that 60% of seen defaults (and 81 in 2020-06 alone) fall in 2020. Qualify all default-specific statements accordingly.
7. **§4.4.** State that excluded months are all 2022-10 or later. State that the within-stratum gain changes sign between year and month resolution (+0.0064 → −0.0004), and that 2022 drives the within-year gain.
8. **§4.7.** Report the existing facility CIs for pooled payoff AUC (+0.051, +0.070) and default AUC (−0.094, −0.045), and the default-Brier deterioration excluding zero under both schemes.
9. **§3.3 and §4.** Report the pre-specified REDUCED (and RATE) sensitivities.
10. **§3.4.** Qualify "point-in-time" as ALFRED-archive point-in-time. Disclose the PMMS 2022-11-17 methodology break.
11. **§3.2.** Describe the sampling design (deterministic identifier hash, 20,000 per vintage, FRM Standard universe, no weights) and the 70/30 role split. State that C = 1 was fixed a priori.
12. **Report the full-period realized-class decomposition** (deterioration entirely on no-event intervals).
13. **Presentation.** Remove internal task, review and status tokens and `Q19` tags. Fix reference metadata (Bhattacharya2019 URL; "PEER_REVIEWED" tags). Engage Sadhwani et al. or remove it from the .bib. Add transportability and forbearance references if those topics are retained.

---

## Final verdict

STAGE 1 FATAL FLAW:
NO

STAGE 2 FATAL FLAW:
NO

NEW FATAL FLAW DISCOVERED:
NO

STAGE 1 PREPRINT DECISION:
MAJOR REVISION

STAGE 2 PREPRINT DECISION:
MAJOR REVISION

STAGE 1 PEER-REVIEW DECISION:
WEAK REJECT

STAGE 2 PEER-REVIEW DECISION:
WEAK REJECT

STAGE 1 CONCERNS CONFIRMED:
7 (of the 14 required concerns; includes 2 STRENGTHENED. Plus 2 of 3 additional items confirmed.)

STAGE 1 CONCERNS WEAKENED/RESOLVED:
7 (5 WEAKENED, 2 RESOLVED. Plus 1 additional item weakened.)

NEW MATERIAL CONCERNS:
3 material (N1 omitted REDUCED sensitivity; N2 payoff-calibration sign reversal and 2022–2026 deterioration; N3 unseen gain via offsetting calibration errors with 2006-reference cohort fallback). Plus 1 moderate (N4 duration-band fallback). Plus the escalation of Stage 1 #7 to MATERIAL_TARGET_RISK.

ARXIV-BLOCKING ANALYSES:
1. Only if default-specific claims are retained: a frozen-label tabulation of assistance, deferral and disaster status around 2020–2021 default events (plus provider documentation of how delinquency status is reported under forbearance).
2. Only if "population-dependent", "transport" or "reversal" is retained as a finding: a registered per-year/per-vintage unseen breakdown and a paired interval for the unseen Δ on frozen arrays.

Otherwise NONE, provided the manuscript-only corrections are made.

PEER-REVIEW-STRENGTHENING ANALYSES:
- Per-year realized-class log-loss decomposition (seen).
- Intervals for within-year and within-month AUC differences.
- Maturity-event count within code 01.
- Facility-weighted scoring.
- Calendar-standardised seen vs unseen comparison.
- Forbearance-excluded default-endpoint sensitivity (new protocol).
- Full reporting of the pre-specified REDUCED/RATE sensitivities.
- PMMS-break split.

MANUSCRIPT-ONLY CORRECTIONS:
- Retitle without "population-/regime-dependent transport".
- Describe the unseen result as a ~1% point-estimate difference with mixed calibration, not a reversal.
- Disclose the encoding fallbacks (2006-reference cohort; 0–12 duration for 181–240 months).
- Replace pooled "payoff over-prediction" with the per-year sign-reversing calibration, and report the 2022–2026 relative deterioration.
- State that 2021–2025 are also outside macro support.
- Disclose the absence of forbearance handling and the 2020 default concentration (81 events in 2020-06), and qualify default claims.
- Name the excluded months (≥ 2022-10) and the within-stratum sign change across resolutions.
- Report the existing facility CIs for pooled AUC differences and the default-Brier result.
- Report the pre-specified REDUCED/RATE sensitivities.
- Qualify "point-in-time" as ALFRED-archive PIT, and disclose the PMMS 2022-11 break.
- Describe the sampling design, role split, absence of weights, and the a-priori C = 1.
- Report the realized-class LL decomposition.
- Remove internal process tokens and Q19 tags; fix reference metadata; engage or remove uncited references.

STOP.

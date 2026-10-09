# Task 18 — Frozen-Array Sensitivity Closure

**Base commit:** `b64c20f2ab7f8779e2b1eb233deebd35da2c2645`
**Registration:** `reports/paper/task18_analysis_registration.json`, sha256_lf `b78aa2b0f63c620d6890efe203d6da811a887bd8a2697dd5a64d2a7bcfcefbed`, written and hashed before any metric was produced.
**Task 17 decision:** MANUSCRIPT CORE SURVIVES WITH MAJOR REVISION
**Scope:** scoring and disclosure only. Nothing fitted, no prediction regenerated, no frozen metric altered, no Fannie outcome inspected, no manuscript version created or edited.

**Executed:** SA01, SA02, SA03, SA06.
**Not executed:** SA04, SA05, SA07, SA08, SA09, SA10, SA11. SA12 documentary only.

---

## Environment constraint and method substitution

The frozen prediction arrays are **absent from this checkout**: no `.npy` files exist, there is no `artifacts/` directory, and `data/` contains only `README.md`. This matches the repository's documented policy that original CSVs, fitted models and artifacts are ignored and not distributed. Interval-level rescoring was therefore impossible here, and two substitutions were registered before scoring:

- **SA01** executed as an **exact Mann–Whitney decomposition** of the frozen pooled and per-calendar-year payoff AUCs, rather than as a month-stratified rescoring. The decomposition is an identity, not an approximation: pooled AUC is the probability a random payoff interval outranks a random non-payoff interval, so `pooled × total_pairs = within_pairs × within_AUC + between_pairs × between_AUC`. The eight frozen per-year AUCs supply the within component; the between component is solved.
- **SA06** executed as a **derivation of entry bounds** from the frozen per-horizon truncation counts and the protocol exclusion rule, independently corroborated by frozen per-year facility counts, rather than as an entry-timestamp histogram.

Residual gaps: calendar-**month** stratified payoff AUC, and the literal entry histogram with median and modal shares. `scripts/task18_local_closure.py` closes both on the machine holding the private artifacts. It verifies array hashes against the frozen report before use, fits nothing, regenerates nothing, and emits aggregates only — no loan-level value or identifier.

---

## A. Decision

**SENSITIVITY CLOSURE ACHIEVED FOR SA02 AND SA03; SA01 AND SA06 ANSWERED AT REDUCED GRANULARITY.**

Every quantitative question Task 17 raised as submission-blocking now has a verified answer. Two answers come from substituted methods that address the scientific question but not the literal computation specified.

## B. Preprint analysis gate

**PREPRINT_ANALYSIS_GAPS_PARTIALLY_CLOSED**

SA02 and SA03 are closed outright. SA01's question is answered decisively at year granularity and the interpretation Task 17 attacked is confirmed indefensible. SA06's inference is upgraded to a derivation with corroboration. Both residual gaps need the local arrays, and **neither blocks any manuscript change the four analyses imply** — all eleven consequences in `task18_manuscript_consequences.json` are marked resolved.

This gate concerns Task 17's quantitative blockers only. It does not mean the manuscript is ready.

## C. SA01 result — QUALIFIED, `WITHIN_PERIOD_GAIN_NEGLIGIBLE`

| Component | M1 | M2 | Gain |
|---|---|---|---|
| Pooled | 0.5654 | 0.6259 | **+0.0604** |
| Within-year, pair-weighted | 0.5556 | 0.5619 | **+0.0064** |
| Within-year, equal-weighted | 0.5171 | 0.5229 | +0.0058 |
| Between-year, solved | 0.5675 | 0.6391 | **+0.0717** |

**82.81% of the case–control pairs entering the pooled AUC cross year boundaries**, so the pooled statistic is predominantly a between-period quantity by construction.

Pair-weighted contributions to the pooled gain, which is the decomposition that matters:

| Stratum | Pair weight | Stratum gain | Contribution | Share |
|---|---|---|---|---|
| Within-year | 0.17195 | +0.006357 | +0.001093 | **1.81%** |
| Between-year | 0.82805 | +0.071673 | +0.059349 | **98.19%** |
| | | | **+0.060442** | 100% |

Identity verified: `0.17195 × 0.006357 + 0.82805 × 0.071673 = 0.060442`, the pooled gain exactly.

> *Correction (T18-C01).* An earlier version of this report stated that "within-period movement accounts for 10.5% of the pooled gain". That figure is the ratio of stratum-gain *magnitudes* (`0.006357 / 0.060442`), not a contribution share. The contribution is pair-weighted and is **1.81%**. Raised by user-side audit; the error was mine. It strengthens the conclusion.

| Year | Intervals | Payoffs | Within pairs | M1 AUC | M2 AUC | Gap | M1 mean pred | M2 mean pred |
|---|---|---|---|---|---|---|---|---|
| 2019 | 62,976 | 738 | 28.51% | 0.5393 | 0.5462 | +0.0070 | 0.01244 | 0.01046 |
| 2020 | 51,809 | 1,146 | 36.03% | 0.5887 | 0.5935 | +0.0047 | 0.01282 | 0.10048 |
| 2021 | 36,397 | 951 | 20.92% | 0.5651 | 0.5606 | −0.0044 | 0.01171 | 0.02494 |
| 2022 | 28,068 | 369 | 6.34% | 0.4933 | 0.5579 | +0.0646 | 0.01020 | 0.00464 |
| 2023 | 24,719 | 216 | 3.28% | 0.4939 | 0.4799 | −0.0140 | 0.01006 | 0.00179 |
| 2024 | 22,138 | 190 | 2.59% | 0.4932 | 0.4955 | +0.0024 | 0.00977 | 0.00206 |
| 2025 | 19,781 | 187 | 2.27% | 0.4814 | 0.4771 | −0.0044 | 0.00912 | 0.00229 |
| 2026 | 3,051 | 26 | 0.05% | 0.4823 | 0.4725 | −0.0097 | 0.00893 | 0.00280 |

All eight years are eligible under the frozen support rule; none excluded, none imputed. Per-year gaps are **four positive, four negative**. The within-year gain is stable under leave-one-year-out, ranging +0.0024 (excluding 2022) to +0.0092 (excluding 2021).

Facility-level horizon AUCs corroborate: gaps of **−0.0006, +0.0002, +0.0017, +0.0031** at 12/24/36/60 months.

Classified `WITHIN_PERIOD_GAIN_NEGLIGIBLE` rather than reversed or absent: the within-year gain is small, positive and consistently signed.

**Month-level result, reported by user-side audit** (executed against the frozen arrays with a corrected script; *not* independently verified in this environment): within-month payoff AUC M1 **0.56014**, M2 **0.55974**, gain **−0.00040**, over 55 supported months with 31 excluded as sparse. Consistent in direction and magnitude with the year-level finding, and at month granularity the within-period gain is marginally *negative* rather than marginally positive. The conclusion that there is no material within-period discrimination gain is strengthened. Caveat: 31 of 86 months fall below the frozen 20-event support rule, so the estimate covers 55 months, and excluded months' within-pairs must be held in a separate bucket rather than folded into the between-month residual.

**Status QUALIFIED**, because the brief's month-stratified computation was not possible and Task 17's supporting structural argument is refuted rather than verified.

### A finding that sharpens the paper

M2's 2020 mean predicted monthly payoff probability is **0.10048 against an observed 0.02212** — a 4.54× over-prediction — and its across-year predicted spread ratio is **56×** against M1's **1.44×**. Because AUC is scale-free, that exaggerated spread raises between-year concordance while destroying calibration. M2's across-year *ordering* of yearly rates is in fact slightly **worse** than M1's: 0.786 against 0.821 of year-pairs ordered correctly.

On this reading the pooled AUC gain and the probability deterioration would be two views of one 2020 over-prediction rather than two independent properties, and selecting on AUC would reward precisely what proper scores penalise.

> *Correction (T18-C05).* An earlier version stated this as established. It is a **diagnostic hypothesis**: well supported by the annual means, the scale-free property of AUC and the year-level decomposition, but not established by them. Attributing the between-period concordance gain to the 2020 cell specifically would need evidence the available aggregates do not provide. The decomposition result in the table above stands independently of this interpretation.

## D. Structural within-month ranking result — **TASK 17 CLAIM REFUTED**

Task 17 asserted that because M2 adds only month-constant national terms and interactions are prohibited, a facility-level ranking gain was "structurally unavailable by construction". The specification half of that is verified: the seven added terms are month-constant, and `prohibited` includes interactions and period fixed effects.

**The inference is wrong.** The model is a multinomial softmax with no-event reference, `h_P = exp(η_P)/(1+exp(η_D)+exp(η_P))`. A month-constant shift adds `g_P` to `η_P` and `g_D` to `η_D` for every facility, but each facility's own normalizer contains `exp(η_D,i + g_D)`, so the shift is **not** a monotone transform of `h_P` across facilities.

Counterexample, facilities `(a,b) = (0,0)` and `(−1,−10)`:

| Shift | h_P(f1) | h_P(f2) | Order |
|---|---|---|---|
| `g=(0,0)` | 0.333333 | 0.268932 | f1 > f2 |
| `g=(0,+10)` | 0.000045 | 0.155362 | **f2 > f1** |

Rank invariance under a shared additive shift holds for a single binary logit, not for the multinomial used here. Independently, M2 is a joint refit, so its mortgage-characteristic coefficients differ from M1's; within-month reordering is available through that channel alone.

**Consequence:** the between-period conclusion stands on the SA01 decomposition and must rest on that evidence alone. The structural argument must be withdrawn from Task 17 and must never appear in the manuscript. Calendar-period ranking can still be useful predictive information; the issue is the phrase "payoff ranking improved".

## E. SA02 result — CONFIRMED

Recomputed independently from `stability.calendar`; Task 17 arithmetic not relied upon.

| Year | Intervals | Weight | M1 | M2 | Delta | Contribution | Share |
|---|---|---|---|---|---|---|---|
| 2019 | 62,976 | 0.2530 | 0.06867 | 0.06885 | +0.00018 | +0.00005 | 0.3% |
| 2020 | 51,809 | 0.2081 | 0.13333 | 0.20254 | **+0.06920** | **+0.01440** | **89.3%** |
| 2021 | 36,397 | 0.1462 | 0.13513 | 0.12905 | **−0.00608** | −0.00089 | −5.5% |
| 2022 | 28,068 | 0.1128 | 0.07659 | 0.08239 | +0.00581 | +0.00065 | 4.1% |
| 2023 | 24,719 | 0.0993 | 0.05594 | 0.06344 | +0.00750 | +0.00074 | 4.6% |
| 2024 | 22,138 | 0.0889 | 0.05269 | 0.05888 | +0.00619 | +0.00055 | 3.4% |
| 2025 | 19,781 | 0.0795 | 0.05938 | 0.06632 | +0.00694 | +0.00055 | 3.4% |
| 2026 | 3,051 | 0.0123 | 0.05956 | 0.06460 | +0.00503 | +0.00006 | 0.4% |
| **Total** | **248,939** | 1.0000 | | | | **+0.01612** | 100% |

Reconstructed delta `0.01612163642779936` against frozen `0.01612163642779664`, **difference 2.7×10⁻¹⁵** — the per-year aggregates are the exact components of the headline delta.

## F. 2020 contribution

**+0.01440 of +0.01612, i.e. 89.34%**, on 20.81% of evaluation intervals. 2021 is the one evaluation year in which M2 scores better. Six of the seven non-2020 years retain positive deltas.

## G. Ex-2020 result — and a correction

**Two distinct quantities have both been called "ex-2020" and they are not interchangeable.**

| Quantity | Value | Definition |
|---|---|---|
| Contribution residual | **+0.00172** | Full-period weighted mean minus 2020's contribution, still divided by the full interval total |
| Renormalised ex-2020 evaluation delta | **+0.00217** | Interval-weighted delta over the seven non-2020 years, weights renormalised to that subpopulation |

Task 17 reported the first and described it as "the ex-2020 residual delta", which reads as the second. The Task 18 brief asks for the second. **The ex-2020 evaluation delta is +0.00217, not +0.00172.** The manuscript must state which quantity it reports.

Relative deterioration: **18.1% full period against 2.8% excluding calendar 2020** (197,130 intervals, 79.2% of the evaluation). Direction positive, M2 worse.

Year-block sensitivity, performed on frozen aggregates and validated first against the frozen interval-level computation:

| | Interval | Blocks |
|---|---|---|
| Full period, aggregate method | [−0.00030, +0.04148] | 8 |
| Full period, frozen interval-level | [+0.00005, +0.04150] | 8 |
| **Ex-2020, aggregate method** | **[−0.00151, +0.00671]** | **7** |

Upper bounds agree to 2×10⁻⁵ and both full-period lower bounds sit at zero to within 3×10⁻⁴, so the aggregate method is adequate for sensitivity use. **The ex-2020 interval includes zero.** With seven blocks, one of them a partial 2026, coverage is weak; this is sensitivity information and is explicitly not offered as frequentist inference.

Per the registration, the useful statement is not whether the result "survives" exclusion of 2020 but **how much remains: the deterioration collapses from 18.1% to 2.8% relative.**

**Interpretation, per the registered rule.** Supported: deterioration is highly concentrated in 2020; it is not exclusive to 2020; the full-period magnitude is unrepresentative of a typical evaluation year; 2021 reverses sign. Not asserted: that COVID caused the failure. Task 11's H7 "pandemic alone explains failure" remains NOT_SUPPORTED and correct — direction persists outside 2020. Task 18 adds the magnitude statement Task 11 did not make: the deterioration is **pandemic-dominated**. Pandemic-only explanation and pandemic-dominated magnitude are different claims and both findings stand together.

## H. SA03 result — CONFIRMED

The sign reversal exists. It applies to the unseen-vintage supplementary population: origination vintages 2018, 2020 and 2022, **17,352 facilities and 652,508 risk intervals** over the same 2019-01 to 2026-02 window — 3.09× the facilities and 2.62× the intervals of the primary evaluation.

## I. Unseen-vintage comparison

| Metric | Seen M1 | Seen M2 | Δ seen | Unseen M1 | Unseen M2 | Δ unseen | Reverses |
|---|---|---|---|---|---|---|---|
| joint_log_loss | 0.089201 | 0.105323 | +0.016122 | 0.070631 | 0.069811 | **−0.000819** | **yes** |
| default_brier | 0.001223 | 0.001272 | +0.000049 | 0.000899 | 0.000900 | +0.000001 | no |
| payoff_brier | 0.015127 | 0.019825 | +0.004698 | 0.011217 | 0.011682 | +0.000465 | no |
| default_auc | 0.693508 | 0.622831 | −0.070676 | 0.777935 | 0.736496 | −0.041439 | no |
| payoff_auc | 0.565426 | 0.625868 | +0.060442 | 0.552197 | **0.737342** | +0.185145 | no |

Payoff calibration **inverts direction** between populations. On seen vintages M1 under-predicts (0.01138 vs 0.01536) and M2 over-predicts heavily (0.02831), slopes 0.509 and 0.249. On unseen vintages M1 **over**-predicts (0.01907 vs 0.01122) and M2 under-predicts mildly (0.00842), slopes 0.347 and **0.529** — M2's slope is better than M1's there and worse on the seen population.

Favouring M2 on unseen: joint log loss, payoff AUC, payoff calibration slope. Still favouring M1: payoff Brier (an order of magnitude smaller than on seen), default AUC, default calibration slope. Effectively tied: default Brier.

Per-year decomposition of this population is **not frozen** and cannot be produced without the arrays.

All candidate explanations — younger loans, calendar exposure, zero-reference cohort encoding, duration support, absent burnout, the between-period AUC mechanism recurring — are recorded as **HYPOTHESIS**. One has a structural component: a 2022-vintage facility cannot contribute intervals during 2020, so the unseen population is demonstrably less exposed to the 2020 cell as a seasoned cohort; the magnitude of that effect is not established.

## J. Study-level macro conclusion classification

**C. MIXED_TEMPORAL_TRANSPORT_ACROSS_POPULATIONS**

A fails: the primary metric reverses. B overstates: deterioration is not wholly confined to the seen-vintage population, since payoff Brier and default AUC still favour M1 on unseen vintages. D is wrong: both results are frozen, internally consistent and reportable.

The evidence shows transport behaviour that **differs by population**: severe joint-loss deterioration in the seasoned seen-vintage cohort, slight joint-loss improvement with persistent but much smaller cause-Brier deterioration in the younger unseen cohort.

## K. SA06 result — QUALIFIED

Task 17's inference F6 is **confirmed and upgraded from an inference to a derivation with independent corroboration.**

**Upper bound, derived.** The frozen exclusion mask is `entries["month"] + h − 1 <= ordinal("2026-02")`, verified in `cif.py`. Zero landmarks are excluded at any horizon including 60 months, so for `h=60` every entry satisfies **entry ≤ 2021-03** — 27 of the 86 window months, 31.4%.

**Measured distribution, reported by user-side audit** (not independently verified here): **99.02% of landmarks enter in the single month 2019-01**, with the latest entry in **2020-01**. Far tighter than either bound derived here, and it upgrades the classification from near-single to effectively single: the dominant 24-month window is 2019-01 to 2020-12.

> *Correction:* an earlier draft of this derivation used `entry + h <= cutoff` and reported 2021-02 over 26 months. The frozen mask uses `entry + h − 1`, giving 2021-03 over 27 months. Corrected against source before publication.

**Lower bound, corroborated.** Landmark entry is the first row of each contiguous facility history (`first_month = data["month"][starts]`), so presence in a calendar year implies entry on or before the end of it. The frozen 2019 cell contains **5,615 facilities of the 5,619** in the CIF cohort. Therefore **at least 99.93% of entries occur in calendar 2019**, and at most 4 facilities entered later.

Frozen per-year facility counts: 2019 → 5,615; 2020 → 4,844; 2021 → 3,530; 2022 → 2,546; 2023 → 2,161; 2024 → 1,932; 2025 → 1,737; 2026 → 1,532. At-risk decay from 5,619 at month 1 to 4,906 / 3,647 / 2,611 / 1,949 at months 12/24/36/60 is consistent with one synchronised cohort.

Unquantified without the arrays: entry histogram, earliest and latest entry, median entry, modal month and year shares.

## L. CIF entry distribution

At least 99.93% in calendar 2019; no entry later than 2021-03. **Effectively synchronised at the start of the evaluation window.**

## M. CIF path classification

**SINGLE_OR_NEAR_SINGLE_HISTORICAL_PATH**

With ≥99.93% of entries inside a single twelve-month span, the 24-month paths are overlapping windows separated by at most twelve months of phase shift. This is one macro trajectory observed with small offsets, not an average over diverse historical paths. With the audit's measured concentration — 99.02% entering 2019-01 — it is effectively a single path.

> *Correction (T18-C04).* An earlier version said "every 24-month path spans approximately 2019-01 to 2021-02". That 26-month range is the *union* of possible windows, not the span of any single 24-month path.

**Manuscript language "historical rolling macro paths" is NOT supported.** The plural, with "rolling", implies averaging over many entry dates and therefore many trajectories. Accurate alternatives: *"a single historical macro path, observed from a cohort entering within one twelve-month window"* or *"the realised 2019–2021 macro trajectory"*. The existing and correct statement that these are retrospective evaluations rather than prospective forecasts should be **retained**; only the plurality claim must change.

**Calendar alignment.** The 24-month paths are overlapping windows whose union spans roughly 2019-01 to 2021-11, and every such window contains most or all of calendar 2020. Under the audit's measured entry concentration the dominant window is 2019-01 to 2020-12, containing calendar 2020 in its entirety. **The 24-month CIF comparison and the 2020 calendar deterioration are not statistically independent pieces of evidence** — they are two views of the same period in the same cohort. This is a calendar-alignment statement only; no causal dependence is claimed. The abstract currently presents the CIF discrepancy as a third finding alongside joint log loss and payoff AUC; it cannot stand as independent corroboration.

## N. CIF table, all four frozen horizons

| Horizon | At risk | Obs payoff | M1 payoff | M2 payoff | Obs default | M1 default | M2 default |
|---|---|---|---|---|---|---|---|
| 12m | 4,906 | 0.1330 | 0.1385 | **0.1222** | 0.0061 | 0.0060 | 0.0064 |
| 24m | 3,647 | 0.3367 | 0.2587 | **0.7581** | 0.0360 | 0.0107 | 0.0127 |
| 36m | 2,611 | 0.5056 | 0.3536 | **0.8079** | 0.0417 | 0.0144 | 0.0149 |
| 60m | 1,949 | 0.6094 | 0.4860 | **0.8187** | 0.0467 | 0.0193 | 0.0163 |

No horizon is selected as headline. Observations:

- At 12 months **no qualitative failure appears**, in contrast to the 2.25× over-prediction at 24 months — but M1 is the more accurate of the two there: absolute errors are M0 0.00234, M1 0.00547, M2 0.01081, so **M1 is closer by a factor of 1.98**.
- The dramatic over-prediction emerges from 24 months onward as the monthly error compounds.
- **All three models under-predict observed default CIF beyond 12 months**, by roughly 3.4× at 24 months — a baseline limitation, not an M2-specific one.

> *Correction (T18-C02).* An earlier version claimed M2 was "in fact closer to observed than M1" at 12 months. That is false, as the absolute errors above show. Task 17 §F5 had this right; the Task 18 report overstated it.

## O. Payoff Brier uncertainty result

| Resampling unit | Delta | Interval | Verdict |
|---|---|---|---|
| Facility (5,619 clusters) | +0.00469766 | [+0.00456248, +0.00482166] | excludes zero |
| Calendar year (8 blocks) | +0.00469766 | **[−0.00004738, +0.01321485]** | **includes zero** |

Task 17's quoted values verified exactly. The frozen decision rule requires *"Both facility and calendar 95 CI upper<0"*, treating the two as co-primary.

**Classification: `ROBUST_FACILITY_ONLY`.** A difference robust under only one resampling unit cannot be reported as robust. For comparison, the joint log-loss calendar interval has lower bound +5.33×10⁻⁵ — direction preserved by a margin numerically indistinguishable from zero on eight blocks.

## P. Facility-vs-calendar uncertainty interpretation

Wording checked against `statistical_unit_audit.json` and the frozen artifacts:

1. **Facility resampling** estimates uncertainty conditional on the realised calendar path and on facility composition. The artifact records `conditional_on_realized_calendar: true`. It does **not** estimate macroeconomic or calendar uncertainty.
2. **Calendar-block resampling** probes sensitivity to calendar-period variation; `conditional_on_realized_calendar: false`. Eight blocks full period, seven ex-2020, one partial. Percentile coverage is weak.
3. **Macroeconomic sampling uncertainty** is estimated by no analysis in this study. The study observes one realised national trajectory; effective independent macro support is the number of distinct calendar months — 88 in development — not the interval count.

**Verified wording:** *Facility resampling estimates uncertainty conditional on the realised calendar path and facility composition. Calendar-block resampling probes sensitivity to calendar-period variation but has very few independent blocks. Neither establishes broad macroeconomic sampling uncertainty.*

## Q. PIT terminology verdict

**`VINTAGE_AND_REVISION_AWARE`**

Established: ALFRED `vintagedates` retrieved for all six series; real-time observation pulls used `realtime_start`/`realtime_end` with `output_type=1`, so retrieved values are vintage-specific rather than latest-revision; reference period, availability bounds, revisions and retrieval time are separated in the information contract.

Not established: `release_lags` records `count: 0` and `evidence_quality: UNMEASURED_EXACT_PROVIDER_DATES` for **all six** of UNRATE, DGS10, MORTGAGE30US, USSTHPI, GDPC1 and CPIAUCSL, with the explicit reason that no certified initial provider release dates were obtained.

Availability at the assessment date is therefore governed by ALFRED vintage dates, not verified first-release dates. **Unqualified "point-in-time" implies the latter and is not supported.** POINT_IN_TIME may be retained only if defined explicitly at first use as ALFRED vintage-date availability, with a statement that exact provider release lags were not measured.

No macro data was acquired. CG06 remains PARTIALLY_RESOLVED.

## R. Claims strengthened

- `MACRO_DELTA_FACILITY_JOINT_LOG_LOSS` — the per-year decomposition reconciles to 2.7×10⁻¹⁵, confirming the frozen delta's internal consistency.
- `MACRO_CIF_24_SUPPORT` — entry concentration is now derived rather than inferred.
- `MACRO_PRIMARY_M2_PAYOFF_BRIER` / `MACRO_CAL_M2_PAYOFF` — SA01 verifies the decomposition and supports a diagnostic hypothesis involving 2020 over-prediction and the 56× spread; it does not establish the mechanism.
- The study's **diagnostic** interpretation is sharpened: the verified decomposition and calendar overprediction suggest a candidate account of the AUC-versus-proper-score contrast, without identifying a unique numerical cause.

## S. Claims narrowed

- `MACRO_PRIMARY_M2_JOINT_LOG_LOSS` — narrowed twice: to a pandemic-dominated magnitude, and to the seen-vintage population.
- `MACRO_PRIMARY_M2_PAYOFF_AUC` — narrowed to across-period ranking.
- `MACRO_PRIMARY_M2_PAYOFF_BRIER` — narrowed to robust under facility resampling only.
- `MACRO_CIF_24_M2_PAYOFF` — narrowed to a consequence of the monthly result on a single macro path, not independent evidence.
- `PIT_FEATURES`, `PIT_SERIES`, `PIT_PROVENANCE`, `PIT_RELEASES`, `PIT_GAPS` — narrowed to vintage-and-revision-aware.
- `MACRO_UNSEEN` — promoted from supplement to a result bearing on the study-level conclusion.

## T. Claims no longer defensible

**No frozen claim value is undefensible.** Every recorded number verified. Two *interpretations* fall:

1. "Payoff ranking improved" as facility-level discrimination — refuted by the decomposition.
2. Task 17's own structural argument that a within-month gain was unavailable by construction — refuted mathematically. **This is a Task 17 error, not a manuscript error**, and Task 17 requires an erratum annotation.

## U. Manuscript consequences

Eleven findings in `reports/paper/task18_manuscript_consequences.json`, each with affected Task 17 change IDs, affected claim IDs, required change, allowed language and prohibited language. All eleven marked `preprint_blocking_resolved: true`.

Highest-value reframing available: the ranking-versus-probability contrast is **one phenomenon, not two**. That is more precise, harder to oversell, and more useful to a model-risk reader than the current framing.

## V. Remaining preprint blockers

**No analysis blockers remain.** Residual, non-blocking:

- Month-level SA01 and the CIF entry histogram — `scripts/task18_local_closure.py`.
- Per-year decomposition of the unseen-vintage population — not frozen.
- Governance, unchanged from Task 17: `[PUBLICATION TERMS REVIEW REQUIRED]`, `[ETHICS AND AUTHOR REVIEW REQUIRED]`, author names and affiliations.

## W. Remaining peer-review analyses

SA04 complexity adjustment · SA05 facility-weighted rescoring · SA08 support-restricted evaluation · SA12 CG03 implementation review (documentary). Then optional: SA07 current-state conditioning, SA09 nonlinear baseline, SA10 maturity separation, SA11 Fannie. A reviewer will also now ask for a per-year decomposition of the unseen-vintage population, which requires the arrays.

## X. Tests

`tests/test_frozen_array_sensitivity.py`. Gates: registration predates and scopes outputs; only SA01/02/03/06 executed; no model-fit, prediction-generation or calibration-fit call in any Task 18 artifact; no Fannie outcome access; Task 10 predictions, Task 14, 15, 16 and 17 unchanged; AUC eligibility rule frozen and no AUC imputed; calendar decomposition reconstructs the headline delta to 1×10⁻¹²; unseen-vintage metrics match the frozen source exactly; CIF entry counts reconcile; all four horizons reported; facility and calendar uncertainty distinguished; PIT terminology respects the release-lag limitation; no `main_v0.3.md`; the local closure script fits nothing and emits no loan-level field.

## Y. Preservation

No tracked file modified. Verified unchanged: Track A, all Track B empirical artifacts, Task 10 predictions and models (hashes unverifiable here — arrays absent — and untouched by construction), Tasks 11, 12, 13/T, 14, 15, 16, 17, Freddie and Fannie source archives, macro source evidence, all frozen experiment hashes, all 123 manuscript numeric bindings. `paper/main_v0.2.md` unchanged and no new manuscript version created.

## Z. Files

Created: `task18_analysis_registration.json`, `task18_results.json`, `task18_sa01_within_period_auc.json`, `task18_sa02_calendar_decomposition.json`, `task18_sa03_unseen_vintage.json`, `task18_sa06_cif_entry_distribution.json`, `task18_manuscript_consequences.json`, `TASK18_FROZEN_ARRAY_SENSITIVITY_CLOSURE.md`, `task18_verification.json` (all under `reports/paper/`), plus `scripts/task18_local_closure.py` and `tests/test_frozen_array_sensitivity.py`.

## AA. Commit

`paper: close frozen-array sensitivity gaps` — Task 18 files only, not pushed.

## AB. Recommended Task 19

**Task 19 should now be the v0.3 rewrite**, and the evidence says what the paper is.

The current title and abstract claim a study-level finding that the evidence does not support. The defensible paper is about **population- and regime-dependent transport**: a near-collinear national macro block, estimated over a monotone post-GFC recovery window, failed badly in a seasoned survivor cohort during one unprecedented macro excursion, did not reproduce that failure in a younger cohort, and produced a scale-free ranking gain from the very over-prediction that destroyed its probabilities.

Suggested title: *Population- and regime-dependent transport of vintage-aware macroeconomic features in mortgage competing-risk models*.

Two sequencing notes. First, run `scripts/task18_local_closure.py` before Task 19 if convenient — it is a ten-minute job and removes the two residual gaps a referee would notice. Second, Task 19 should open with a short **Task 17 erratum** recording the refuted structural claim, so the audit trail stays honest in both directions.

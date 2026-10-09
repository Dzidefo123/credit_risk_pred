# STAGE 3 — INDEPENDENT REVIEW RECONCILIATION

**Target:** manuscript v0.3 (`paper/main_v0.3.md`, sha256 `dd7738c5…5f17a`) at commit 1a42d505.

**Scope.**
- I read only files inside /home/claude/cold_review_stage3/ and /home/claude/cold_review_stage2/.
- I used no web and no git, and ran no new empirical analysis.
- The only arithmetic was on frozen aggregates already used in Stage 2. Two lookups were added: the frozen VIFs and the per-year default AUC from `stability.calendar`.
- I did not edit the manuscript, Stage 1 or Stage 2.

**Reconciler disclosure.**
- The Stage 3 reconciler is the same reviewer who wrote Stage 1 and Stage 2. That is a self-favouring risk when adjudicating Stage 1/2 against Task 20.
- To control for it, every adjudication below is tied to a frozen artifact or code line, not to which review said it.
- Where Stage 1/2 was wrong or overreached, I say so (§22).

---

## 1. Freeze verification

| File | SHA-256 | Status |
|---|---|---|
| `stage1/STAGE1_PROVISIONAL_ASSESSMENT.md` | `63aa6f5a6801474249ab934079d1ba333a3ab55cc1a2cf46b74165659362ff58` | Identical to the original Stage 1 output and the Stage 2 copy. Unchanged. |
| `stage2/STAGE2_EVIDENCE_VERIFICATION.md` | `130836694c63c71e39c9164a85cf1e3c8f89fb753c020b46579529eb92584be9` | Identical to `/home/claude/cold_review_stage2/output/`. Unchanged. |

**Package integrity.**
- `PACKAGE_MANIFEST.sha256` covers 61 entries. 60 files verify.
- The single "failure" is the manifest's entry for itself. It carries the empty-file hash (`e3b0c442…`), which means it was hashed before it was written. This is a self-reference artefact, not tampering.
- The Stage 2 primary package was not modified.

---

## 2. Task 20 inventory and chronology

| Artifact | ee5049a (original) | 3de6cdf (round 1) | 0439344 (round 2) | Role |
|---|---|---|---|---|
| `TASK20_HOSTILE_MANUSCRIPT_REVIEW.md` | ✓ | ✓ (§0a added) | ✓ (§0a round 2 added) | Original hostile review and its in-place corrections |
| `task20_abstract_audit.json` | ✓ | ✓ | ✓ | Sentence-level abstract audit |
| `task20_analysis_triage.json` | ✓ | ✓ | ✓ | Analysis triage |
| `task20_claim_sample_audit.json` | ✓ | ✓ | ✓ | Claim / binding audit (257 bindings, 143-statement pass) |
| `task20_estimand_audit.json` | ✓ | ✓ | ✓ | Estimand table E1–E8 |
| `task20_review_verification.json` | ✓ | ✓ | ✓ | Review verification record |
| `task20_corrections.json` | — | ✓ (C01–C04) | ✓ (+ round_two C05–C14) | Correction ledger, written by the Task 20 reviewer |
| `tests/test_hostile_manuscript_review.py` | ✓ | ✓ | ✓ | Tests |

**External, independent sources:**
- `external_audits/zip/task20_review_audit/AUDIT_TASK20.md`, `audit_task20.py`, `audit_task20_results.json` (file times 2026-10-08 23:32–23:34);
- `FOLLOWUP_CORRECTIONS.md` (2026-10-09 09:58);
- `task18_audit_summary_from_author.md` (a pasted summary only).

**Reconstructed sequence:**

1. **2026-10-08 22:38 (ee5049a).** Original Task 20.
   - 7 major and 7 minor concerns.
   - 1 arXiv blocker (calendar AUC intervals).
   - Preprint: REQUIRES_MAJOR_REVISION. Peer review: BORDERLINE.
   - The reviewer discloses that it wrote Tasks 17/18 and cannot do a cold read.
2. **2026-10-08 23:32–23:34.** External audit `AUDIT_TASK20.md` (by the author). It already contained **every** point later used in both correction rounds:
   - M2: qualified, not a contradiction.
   - M3: "resolution-free" false; pair weights versus contributions conflated.
   - M4: false sole-macro-movement premise; no materiality threshold.
   - M6: seen calibration mixed; "near-ideal" too broad.
   - Transport is a terminology dispute.
   - Table F caption exists.
   - The closure-reproducibility limit is already disclosed.
   - **"21 stored macro coefficients and 14 cause-versus-no-event contrasts"**.
   - Economic negligibility unsupported.
   - Abstract sentence 1 is modal.
   - Certainty statements need qualification.
3. **2026-10-09 09:28 (3de6cdf), round 1.** Only **4** items applied (C01–C04).
   - **C03 introduced a new error that contradicted the audit text it cited.** It said "at least 14" coefficients, possibly more via macro missing indicators "not verifiable", and that the correction "strengthens the effective-sample-size concern".
   - The remaining audit points were not applied, even though they were already in the audit.
4. **2026-10-09 09:58.** `FOLLOWUP_CORRECTIONS.md` re-lists the unapplied points, explicitly "beyond the four reported corrections", and refutes the C03 premise using the public code.
5. **2026-10-09 10:08 (0439344), round 2.** C05–C14 applied.
   - M2 and M3 downgraded to MODERATE.
   - arXiv blockers 1 → 0.
   - Decisions unchanged.

**Interpretive consequence.** The corrected Task 20 was not self-correcting. Every correction originated in the external audit, and round 1 was incomplete relative to an audit that already existed. Reading the latest Task 20 as if it had always held the corrected interpretation would overstate its reliability.

---

## 3. Provenance of Task 20 conclusions

| Task 20 conclusion (final form) | Provenance |
|---|---|
| Independence compromised and disclosed | UNCHANGED_ACROSS_ROUNDS |
| No fatal flaw | UNCHANGED (the "no target leakage" certainty was qualified in ROUND_2, C14) |
| 257/257 bindings, 143/143 re-derivations | UNCHANGED (the 143 pass relabelled as attestation in ROUND_2, C14) |
| M1 default discrimination buried (MAJOR; "robust", "event that generates loss") | UNCHANGED. **Not corrected; see §6 residual errors.** |
| M2 no calendar AUC uncertainty | ORIGINAL (MAJOR, arXiv blocker) → CORRECTION_ROUND_2 (MODERATE, disclosure only) |
| M3 resolution dependence | ORIGINAL ("resolution-free within gains"; conflated weights and shares) → CORRECTION_ROUND_2 (MODERATE; weights and contributions separated) |
| M4 2020 "strongest argument against it" | ORIGINAL (false premises) → CORRECTION_ROUND_2 (premises withdrawn; both-readings recommendation kept) |
| M5 internal vocabulary | UNCHANGED |
| M6 default calibration unreported | ORIGINAL → ROUND_1 ("near-ideal" → weak calibration) → ROUND_2 (seen vector mixed; no blanket claim) |
| M7 no figures | UNCHANGED |
| Minor 1 title (regime, transport) | ORIGINAL ("transport is the wrong word") → ROUND_2 (terminology preference) |
| Minor 2 VIFs | UNCHANGED (values verified, §9) |
| Minor 3 parameter count / effective sample | ORIGINAL ("seven coefficients", inherited from Task 17) → ROUND_1 ("at least 14", new error) → ROUND_2 (exactly 14 contrasts / 21 stored) |
| Minor 4 "interval" for 8 blocks | ORIGINAL → ROUND_2 (presentation choice) |
| Minor 5 closure irreproducibility | ORIGINAL → ROUND_1 withdrawn |
| Minor 6 Table F functional mismatch | ORIGINAL → ROUND_1 withdrawn |
| Minor 7 Bu2026 incomplete | UNCHANGED |
| §7 outcome-correlated interval weighting | UNCHANGED |
| §10 "cross-period concordance" terminology; economic negligibility | Terminology UNCHANGED; negligibility ROUND_2 withdrawn (C11) |
| §12 CIF is a supporting result, not independent | UNCHANGED |
| §8 CG03 "remains open" / triage HIGH_VALUE | UNCHANGED |
| Abstract sentence 1 TOO_BROAD | ORIGINAL → ROUND_2 SUPPORTED (C13) |
| Triage: SA04 complexity, SA05 facility-weighted, CG03, CG06 as HIGH_VALUE | UNCHANGED (SA04 rationale rewritten in ROUND_1 and ROUND_2) |

---

## 4. Central claim — were the reviews evaluating the same claim?

| Source | Central claim as understood |
|---|---|
| Stage 1 | M2 improves development fit and pooled payoff AUC, but the AUC gain is almost entirely between-period. Out-of-time probability quality worsens, concentrated in 2020. A small unseen point reversal makes the picture mixed. |
| Stage 2 | Same, plus: the period pattern is sign-varying in payoff calibration (over in 2020, under in 2022–26); 2022–26 remain worse by 8–13% relative; the default endpoint is suspect in 2020; the "reversal" is a point estimate produced by offsetting calibration errors. |
| Original Task 20 | In one specification and cohort, macro raises pooled payoff concordance through cross-period pairs while degrading probability quality, and "the direction of that degradation does not hold across origination-vintage populations". Main emphasis: **default discrimination loss** as the buried headline. 2020 is "the indictment". |
| Corrected Task 20 | Same contribution sentence (unchanged). 2020 is presented with two bounded readings. Transport is a terminology choice. |

**Verdict: largely the same scientific claim, with three framing differences.**

1. **Emphasis.**
   - Task 20 promotes *default discrimination* to the headline.
   - Stage 2 shows default AUC loss is a 2020-only phenomenon: per-year change +0.0024 in 2019, **−0.1568 in 2020**, +0.0112 in 2021, and suppressed thereafter. It also falls in the year where the default target is least trustworthy.
   - The reviews therefore weight the default arm in opposite directions.
2. **Population.** Task 20 states the unseen comparison as "the direction of that degradation does not hold across … populations". That is accepted at point-estimate level. Stage 2 adds that the population and calendar contrasts cannot be separated, and that the joint-loss point gain coincides with M2 offsetting M1's 70% payoff over-prediction.
3. **2020.**
   - Original Task 20 framed 2020 as the only macro-moving period and an indictment, which was wrong.
   - Corrected Task 20 keeps two readings and notes, in absolute terms, that 2022–23 did not recur.
   - Stage 2 treats 2020 as absolute concentration within a *sign-varying* error pattern that persists relatively after 2021.

---

## 5. Concern crosswalk

Key to "Independent?":
- ✓ = found by both review paths before either saw the other.
- T = Task 20 only.
- C = cold review only.

Severity is given as original → corrected.

| ID | Concern | Origin | Independent? | Evidence basis | Severity (orig → corr) | Stage 3 judgement | Manuscript consequence | Analysis consequence | Agreement class |
|---|---|---|---|---|---|---|---|---|---|
| X01 | Title: "regime-dependent" unestablished | T20 minor 1; S1 Q13 | ✓ | No regime model; one year dominates | T20 minor; S1 med-high → S2 med | Valid; remove from title | Retitle | None | INDEPENDENT_AGREEMENT |
| X02 | "Transport" / "population-dependent" over-claims | T20 §9; S1 Q12 | ✓ (T20 softened in R2) | Six simultaneous population differences | T20 major-ish → minor (terminology); S1 high | "Transport" is usable for predictive performance in another evaluation population, but **not** as a headline paired with "population-dependent" (attribution) and with no interval | Retitle; body wording | Optional unseen breakdown | PARTIAL_AGREEMENT |
| X03 | Internal process vocabulary and tokens | T20 M5; S1 obs | ✓ | Text | Major / presentation | Valid, P1 | Remove | None | INDEPENDENT_AGREEMENT |
| X04 | One realized macro history; 88 months is not an effective sample | T20 minor 3; S1 Q5/threat 2 | ✓ (both phrased imprecisely, both corrected) | 88 / 86 months, 8 blocks | Minor; S1 high → S2 high (disclosure) | Valid scope limit; already disclosed in v0.3 §3.4/§6 | Keep; avoid "effective sample" | None | INDEPENDENT_AGREEMENT |
| X05 | Default AUC fall under-reported | T20 M1; S1 obs ("little discussed") | ✓ | R `/primary`, facility CI [−0.094, −0.045] | T20 MAJOR (unchanged) | Valid to report. T20's "robust" and "event that generates loss" framing is **not** supported: the drop is 2020-only and target-exposed | Report with 2020 and target caveat; not a headline | Label tabulation (journal) | PARTIAL_AGREEMENT |
| X06 | Default calibration unreported | T20 M6; S1 Q11 | ✓ | R calibration | Major (→ mixed vector, R2) | Valid, P1 (full vector) | Report both populations | None | INDEPENDENT_AGREEMENT |
| X07 | Calendar AUC uncertainty absent | T20 M2; S1 Q17 (as "no AUC intervals") | ✓ (S1 partly wrong: facility AUC CIs exist) | `paired_calendar` has no AUC | T20 major/blocker → moderate; S1 high → S2 medium | Disclosure only; descriptive identity needs no interval | State absence; report facility CIs | Optional | INDEPENDENT_AGREEMENT (after corrections on both sides) |
| X08 | Existing facility AUC CIs unreported | T20 M2 (orig) | T then C (Stage 2 found independently) | R `/paired_facility` | moderate | Valid, P2 | Report, labelled calendar-conditional | None | INDEPENDENT_AGREEMENT (Stage 1 missed it) |
| X09 | Decomposition: resolution/population; pair weights ≠ contributions | T20 M3; S2 §I | ✓ (T20 original wrong; S2 correct before seeing T20) | Closure output; R per-year | T20 major → moderate | Valid, P2 | Name resolution and population; separate quantities | None | INDEPENDENT_AGREEMENT (with T20-R2) |
| X10 | Two readings of 2020 not discussed | T20 M4; S1 Q14/Q15 | ✓ | Table C | Major (premises withdrawn) | Valid, P2, with corrected premises (§10) | Add bounded discussion | None | INDEPENDENT_AGREEMENT |
| X11 | Only 2020 had macro movement | T20 M4 (orig) | — | Refuted by frozen ranges | — | Error; corrected (C09) | — | — | OBSOLETE_AFTER_CORRECTION |
| X12 | CIF is not independent evidence | T20 §12; S1 Q16 | ✓ | Entry distribution 99.02% | supporting | Valid | Keep as illustration | None | INDEPENDENT_AGREEMENT |
| X13 | CG03 open / high value | T20 §8, triage; S1 | ✓ raised; **CONTRADICTION** on status | Censor survival ≥ 0.9988 (R CIF) | T20 HIGH_VALUE; S2 resolved-practically | Practically resolved; state the evidence | One sentence | None | CONTRADICTION → resolved for S2 |
| X14 | Table F functional mismatch | T20 minor 6 | — | Caption exists | — | Withdrawn correctly | — | — | OBSOLETE_AFTER_CORRECTION |
| X15 | Closure irreproducibility | T20 minor 5 | — | v0.3 §3.6/§8 | — | Withdrawn correctly | — | — | OBSOLETE_AFTER_CORRECTION |
| X16 | Unseen result has no uncertainty | T20 §13; S1 Q17 | ✓ | None frozen | moderate | Valid; disclose; interval needed only to keep "reversal" | Wording | Optional | INDEPENDENT_AGREEMENT |
| X17 | Unseen populations differ simultaneously (vintage, seasoning, calendar, encoding, duration) | T20 §9/E6; S1 Q12 | ✓ | Composition, counts | — | Valid | Disclose | — | INDEPENDENT_AGREEMENT |
| X18 | Interval weighting is outcome-correlated | T20 §7; S1 Q9 (secondary) | ✓ | Code (equal interval weights) | Minor | Valid, P3 analysis | Already disclosed | Facility-weighted (P3) | INDEPENDENT_AGREEMENT |
| X19 | Non-virgin / post-hoc evaluation | T20 §15; S1 threat 5 | ✓ | Ledger, closure registration | — | Valid, disclosed | Keep | — | INDEPENDENT_AGREEMENT |
| X20 | Bibliography/novelty: Bu2026 incomplete; Sadhwani | T20 minor 7, §14; S1 bib | ✓ (partial) | .bib | Minor | Valid; journal | Engage or remove; fix tags | Literature | PARTIAL_AGREEMENT |
| X21 | Abstract sentence 1 too broad | T20 (orig) | S1 said "existence claim; supported" | Text | → withdrawn (C13) | Supported as modal | Optional tightening | — | OBSOLETE_AFTER_CORRECTION |
| X22 | No figures | T20 M7 | T | Text | Major presentation | Valid, P2 | Add 3 figures from frozen values | Rendering only | TASK20_ONLY |
| X23 | VIFs omitted | T20 minor 2 | T | Diagnostics VIFs 10.21/11.56/11.57/11.22 (verified) | Minor | Valid but minor, P3 | Optional | None | TASK20_ONLY |
| X24 | Base-rate reference Brier absent | T20 §11 | T | — | Minor | Valid but minor, P3 | Optional | None | TASK20_ONLY |
| X25 | 8-block output called "interval" | T20 minor 4; S1 Q17 (poor coverage) | ✓ (partial) | Calendar CI reproduced from 8 numbers (S2) | → presentation | Valid, P2 wording | Rename or qualify | None | PARTIAL_AGREEMENT |
| X26 | Joint LL pre-specified as primary | T20 §11 | T (also visible in S2 protocol) | Protocol `decision.primary` | Strength | Valid | — | — | INDEPENDENT_AGREEMENT (S2 §D) |
| X27 | Peng2026 as nearest neighbour | T20 §14 | T | .bib | — | Valid but minor (positioning) | Optional | — | TASK20_ONLY |
| X28 | **Forbearance / default target validity** | S1 threat 3 → S2 §E | C | `event_category` ignores assistance fields; 167/280 seen defaults in 2020; 81 in 2020-06 | S1 med → S2 high (default arm) | Valid, **P1 disclosure**; journal needs label tabulation | Qualify every default statement | Label-only tabulation (P3, journal-required) | COLD_REVIEW_ONLY |
| X29 | **REDUCED / RATE pre-specified sensitivity omitted** | S2 N1 | C (Task 17 knew it existed; no Task flagged the omission) | R `/primary/REDUCED_*`; protocol | S2 material | Valid, P1 (INCOMPLETE_REPORTING, §14) | Report | Optional intervals | COLD_REVIEW_ONLY |
| X30 | **Per-year payoff calibration sign reversal; 2022–26 relative deterioration** | S2 N2 | C (T20-R2 notes absolute 2022–23 deltas only) | Diagnostics `annual_calibration`; R stability | S2 material | Valid, P1 | Replace "over-prediction" framing | None | COLD_REVIEW_ONLY |
| X31 | **Unseen gain via offsetting errors; 2006-reference cohort fallback** | S2 N3 | C (T20 noted "zero-reference fallback" only generically) | R unseen calibration; `models.py` | S2 material | Valid, P1 (part of unseen presentation) | Disclose | — | COLD_REVIEW_ONLY (partial T20 awareness) |
| X32 | Duration 181–240 fallback in seen population (4.5%) | S2 N4 | C | `error_partitions`; zero columns | moderate | Valid, P2 | Disclose | — | COLD_REVIEW_ONLY |
| X33 | PMMS 2022-11 methodology break | S2 N5 | C | Series registry | minor–moderate | Valid, P2 | Disclose | Optional split | COLD_REVIEW_ONLY |
| X34 | Sampling design, no weights; role split; C = 1 fixed | S1 (as unspecified) → S2 resolved | C | Recovery audit; protocol | low | Valid, P2 methods text | Describe | — | COLD_REVIEW_ONLY |
| X35 | Excluded months all ≥ 2022-10; within sign changes year → month | S2 §I | C | Closure | low–moderate | Valid, P2 | State | — | COLD_REVIEW_ONLY |
| X36 | Realized-class LL decomposition (deterioration entirely on no-event rows) unreported | S2 N6 | C | Diagnostics `score_accounting` | moderate | Valid, P2 | Report | — | COLD_REVIEW_ONLY |
| X37 | Default-Brier deterioration excludes 0 under both schemes (unreported) | S2 §M | C | R paired | low | Valid, P2 | Report | — | COLD_REVIEW_ONLY |
| X38 | Comparator M1 calibration | S1 obs → S2 refined | C | Per-year calibration | medium | M1 default failure is 2020-specific; unseen M1 payoff +70% | Report | — | COLD_REVIEW_ONLY |
| X39 | Development fit lacks complexity adjustment | T20 triage SA04 (HIGH_VALUE) | T (S1 noted "not complexity-adjusted", which v0.3 states) | v0.3 §4.1 | HIGH_VALUE | Valid but minor. Development fit carries no weight in the conclusions. | None | P4 | PARTIAL_AGREEMENT (severity contradiction) |
| X40 | Maturity inside payoff | T20 triage; S1/S2 | ✓ | Protocol code 01 | Minor | Disclosed in v0.3 §6; a count is P3 | — | P3 | INDEPENDENT_AGREEMENT |
| X41 | Point-in-time vs vintage-aware wording | T20 §3 strengths; S2 §G | ✓ | `macro_join`, PIT engine | — | v0.3 wording is accurate; qualify any "point-in-time" | — | — | INDEPENDENT_AGREEMENT |

---

## 6. Task 20 error audit

**Count.**
- The final ledger holds **14 entries (C01–C14)**. All are marked `audit_was_correct: true`, and every one is traceable to `AUDIT_TASK20.md` or `FOLLOWUP_CORRECTIONS.md`.
- Unbundled, they cover:
  - 13 entries on original-review content;
  - 1 entry (C05) correcting an error **introduced by round 1** (C03's correction);
  - C09 holds two premises and C14 holds about five certainty statements.
- So there are about **19 distinct erroneous assertions** acknowledged.
- I also find **unacknowledged residual errors**, listed after the table. "14" is therefore a ledger count, not an error count.

| ID | Original claim | Why wrong | Correcting evidence | Correction adequate? | Affected | Class |
|---|---|---|---|---|---|---|
| C01 | Table F compares different functionals "without a note" | Caption and §3.6 label both | v0.3 text | Yes | interpretation, recommendation | MINOR |
| C02 | Closure unreproducible; §8 should say so | Already disclosed | v0.3 §3.6/§8 | Yes | recommendation | MINOR |
| C03 | "Seven macro coefficients / parameters" (from Task 17) | Seven predictors × 2 contrasts | Protocol | **No.** It introduced "at least 14", "not verifiable" and "strengthens the concern" | arithmetic (count), interpretation | MODERATE |
| C04 | Unseen M1 default calibration "near-ideal" | Mean and slope give weak calibration only | Calibration hierarchy | Yes | interpretation | MINOR |
| C05 | (round-1 error) "at least 14 … not verifiable" | Code builds `:missing` only for mortgage numerics; 37-feature order | `models.py`; R parameters | Yes | arithmetic, interpretation | MODERATE |
| C06 | Within-stratum gains "resolution-free" | +0.00636 (year) vs −0.00040 (month), different populations | Closure, R | Yes | interpretation, recommendation | MODERATE |
| C07 | Month pair weights shown as contribution shares | 1.979% is a weight; the contribution is −0.016% | Closure | Yes | arithmetic labelling, interpretation | MODERATE |
| C08 | Missing calendar AUC CI = internal inconsistency, arXiv blocker | A descriptive identity needs no CI | Logic and v0.3 text | Yes | severity, recommendation, decision gate | **MATERIAL** (removed the only arXiv blocker) |
| C09 | Only 2020 had macro movement; M2 "never materially better" | 2022–23 rates far out of range; 2021 −4.5%; no threshold | Diagnostics ranges; R stability | Partly. Premises withdrawn, but the 2022–26 relative deterioration and calibration sign reversal are still not recognised. | interpretation, severity | MATERIAL |
| C10 | Default calibration uniformly worse under M2 | Seen CITL improves, slope worsens | R calibration | Yes for default. Payoff-side sign reversal still absent. | interpretation | MINOR |
| C11 | Period ranking economically negligible | Not evaluated | — | Yes | interpretation | MINOR |
| C12 | "Transport" is the wrong word | Predictive transportability usage exists | Audit citation | Adequate as terminology. **R8 still says "replace Transport"** (internal inconsistency) | interpretation, recommendation | MINOR |
| C13 | Abstract sentence 1 too broad | Modal | Text | Yes (agrees with Stage 1) | interpretation | MINOR |
| C14 | Five certainty statements (leakage, 143-pass, novelty label, "interval", objective gates) | Overstated | Various | Yes | interpretation | MINOR |

**Unacknowledged residual errors in final Task 20 (0439344):**

| # | Statement | Problem | Evidence | Class |
|---|---|---|---|---|
| U1 | M1: "a **robust** 7-point fall in default discrimination"; "concerns the event that generates loss" | The robustness is facility-only, which Task 20's own M2 logic says cannot speak to calendar questions. The fall is a single-year event (2020: 0.665 → 0.508; 2019 and 2021 slightly positive). The "default" is a delinquency-composite proxy that does not handle forbearance. | R `stability.calendar`; `event_category`; CIF month 18 = 81 events | **MATERIAL**. It is Task 20's top concern and its R1 asks for abstract placement. |
| U2 | M4 (corrected): "the large 2020 deterioration did not recur in the 2022–23 rate rise" | True only in absolute terms. 2022–26 are +7.6% to +13.4% relative, and M2 under-predicts payoff 3–5-fold. | Diagnostics `annual_calibration` | MODERATE |
| U3 | §8 / triage: CG03 "remains open", HIGH_VALUE | Censoring is negligible (censor survival ≥ 0.9988; 5 censored by 60 months) | R CIF observed | MINOR |
| U4 | R8 still recommends replacing "Transport" after C12 softened it to a preference | Internal inconsistency | Task 20 text | MINOR |
| U5 | §9 "burnout-selected population" stated as fact | v0.3 treats burnout as a hypothesis | v0.3 §3.1 | MINOR |
| U6 | Triage SA04 complexity adjustment as HIGH_VALUE | Development fit carries no weight in the conclusions (Task 20 says so in the same sentence) | — | MINOR |
| U7 | M6 is one-sided on the unseen population | Omits that M2 improves unseen payoff calibration (CITL 0.0078 → 0.0028; slope 0.35 → 0.53) and that M1 over-predicts unseen payoff by 70% | R unseen calibration | MODERATE |

**Lingering contamination.**
- **U1 carries the uncorrected framing of the original M1 into the final recommendation set** (R1 abstract placement). Its "robust" label never received the calendar-conditioning qualification that the round-2 audit applied to AUC uncertainty generally. Task 20's own correction C08 ("facility intervals condition on the calendar") was never propagated to M1.
- U2 is a partial carry-over of the C09 premise error. The premises were withdrawn, but the replacement description is still built on absolute deltas.

---

## 7. Findings with the highest independence value

These were found by both paths before either saw the other. Stage 1 was blind. Task 20 was not blind to Tasks 17/18, but neither path saw the other.

1. Title over-reach: "regime-dependent" (X01) and "transport/population-dependent" (X02).
2. Internal process vocabulary and tokens (X03).
3. One realized macro history; 88 dependent months; 8 blocks (X04). Both phrased it imperfectly and both were later corrected.
4. Default discrimination and default calibration under-reported (X05, X06). There is agreement on *reporting*, not on *weight*.
5. Calendar-unit uncertainty for AUC absent (X07).
6. The two readings of the 2020 concentration (X10): Stage 1 Q14/Q15 and Task 20 M4.
7. CIF dependence on the same cohort and calendar path (X12).
8. No uncertainty for the unseen result, and population differences that cannot be separated (X16, X17).
9. Interval weighting and non-virgin evaluation are disclosed limitations (X18, X19).
10. Decomposition resolution/population discipline (X09). Agreement between Stage 2 and *corrected* Task 20. The original Task 20 was wrong.

The 2020 concentration's *existence* is agreed by all paths and by Task 17/18. Its *interpretation* is not (§10).

---

## 8. Cold-review-only findings (not in corrected Task 20)

| Finding | Evidence | Classification |
|---|---|---|
| Forbearance / default target validity (X28) | `event_category` ignores assistance and deferral; 60% of seen defaults in 2020, 81 in 2020-06; the project's own design doc required harmonisation | **MUST_FIX_PREPRINT** (disclosure and qualification); label tabulation PEER_REVIEW_STRENGTHENING, required before journal (§13) |
| Omitted pre-specified REDUCED (and RATE) sensitivity (X29) | R `/primary/REDUCED_*` | **MUST_FIX_PREPRINT** (report frozen values) |
| Year-specific payoff calibration sign reversal; 2022–26 relative deterioration (X30) | `annual_calibration`; R stability | **MUST_FIX_PREPRINT** (it changes the meaning of "over-prediction" and "concentrated") |
| Unseen gain via offsetting calibration; 2006-reference cohort fallback (X31) | R unseen calibration; `models.py` | **MUST_FIX_PREPRINT** (part of the unseen presentation) |
| Unsupported duration fallback 181–240 months, 4.5% of seen intervals (X32) | `error_partitions`; zero-energy columns | SHOULD_FIX_PREPRINT |
| PMMS 2022-11-17 methodology break (X33) | Series registry; provenance doc | SHOULD_FIX_PREPRINT |
| Sampling design (deterministic hash, 20k per vintage, no weights), 70/30 roles (X34) | Recovery audit | SHOULD_FIX_PREPRINT |
| Fixed C = 1 a priori (X34) | Protocol; ledger | SHOULD_FIX_PREPRINT (one sentence) |
| Existing but unreported results: facility AUC CIs (shared with T20), default Brier excluding 0 under both schemes, realized-class LL decomposition (X36, X37) | R; diagnostics | SHOULD_FIX_PREPRINT |
| Excluded months ≥ 2022-10; within-gain sign change across resolutions; 2022 drives within-year gain (X35) | Closure; R | SHOULD_FIX_PREPRINT |
| Calendar bootstrap is a resample of 8 aggregates (reproduced exactly) | R `paired_calendar` | OPTIONAL (supports X25 wording) |
| Macro joined at end of month t−1 with ALFRED bounds | `macro_join` | OPTIONAL (methods clarity) |

---

## 9. Task 20-only findings (survived all corrections)

| Finding | Evidence check | Classification |
|---|---|---|
| No figures (M7) | Objective: v0.3 has 7 tables and 0 figures | VALID_AND_IMPORTANT (presentation, P2) |
| VIFs omitted (minor 2) | Verified: dev VIF 10.21 / 1.08 / 11.56 / 11.57 / 11.22 / 1.76 / 1.11; evaluation 10-year/mortgage VIFs 26.0/24.5 | VALID_BUT_MINOR |
| Base-rate reference Brier absent | Plausible presentation aid | VALID_BUT_MINOR |
| "Cross-period concordance" / "calendar-level separation" terminology | Sound; avoids "regime" | VALID_BUT_MINOR (useful wording) |
| Peng2026 as nearest neighbour and stated open question | Bibliographic; not verified here | VALID_BUT_MINOR (needs external literature check) |
| Joint LL was pre-specified primary | Verified in protocol | VALID_BUT_MINOR (strength; Stage 2 also saw it) |
| Default AUC as the loss-relevant headline (M1 framing) | Contradicted by per-year concentration and target concern | **NOT_SUPPORTED** as framing; SUPERSEDED as a reporting item by X05 |
| SA04 complexity adjustment as HIGH_VALUE | Development fit not used in conclusions | NOT_SUPPORTED at that priority |
| CG03 open / HIGH_VALUE | Censoring negligible | SUPERSEDED |
| SA05 facility-weighted rescoring | Valid (also S1) | VALID_BUT_MINOR (P3) |

---

## 10. 2020 reconciliation

| Aspect | Task 17 (inherited) | Original Task 20 | Corrected Task 20 | Stage 1 | Stage 2 | Frozen evidence |
|---|---|---|---|---|---|---|
| Concentration | 89.3%, ex-2020 "+0.00172" (residual error, later corrected) | 89.34%; "the indictment" | 89.34%; two readings | 89%; strengthens method, weakens substantive claim | 89% absolute; partly log-loss asymmetry | 89.34% of Δ; 2020 Δ +0.0692 (+51.9% relative) |
| Macro movement | — | "only 2020 moved" (false) | 2022–23 rates moved | 2020 off-support (correct) but treated as unique | 2021–2025 all fully off-support; 2020 the most extreme | Distance medians 655 / 105 / 353 / 247 / 200 / 167 |
| Stress relevance | — | "whole point" of macro | Bounded reading | Stress test plus limited failure | Same, without mechanism | — |
| Ex-2020 | +0.00172 (wrong) | +0.00217, CI includes 0 | same | +0.00217 | +0.00217; CI is a resample of 7 numbers | +0.00217 (2.8%) |
| Later high-rate years | "MINOR, small magnitude" | "mildly worse" | "did not recur" (absolute) | "small positive differences" | **+7.6% to +13.4% relative, 5 of 5 years** | Confirmed |
| Calibration | — | not per-year | not per-year | pooled over-prediction | **2020 over ×4.5; 2022–26 under ×3–5; 2021 near-calibrated** | Confirmed |
| AUC | — | — | — | — | Within-year gain driven by 2022; 2020 within-month −0.0009 | Confirmed |

**Narrowest defensible interpretation (no mechanism).**
- On the one 2019-01 to 2026-02 evaluation path of the seasoned seen population, M2's joint log loss exceeds M1's in 7 of 8 calendar years.
- The absolute excess is dominated by 2020 (89% of the total). In 2020, M2's mean payoff probability is about 4.5 times the observed rate.
- In 2022–2026 the excess is smaller in absolute terms but persistent (about 8–13% relative per year). In those years M2's mean payoff probability is about 3–5 times *below* the observed rate.
- 2021 favours M2 (−4.5%).
- The macro inputs are outside development support in every year from 2020 onward. 2020 is the most extreme year but not the only one.
- Therefore:
  - M2's payoff calibration error is **sign-varying across periods**, not a single over-prediction;
  - "concentrated in 2020" is accurate for absolute loss only;
  - neither "regime-dependent" nor "2020-specific failure" is established by one realized path.

---

## 11. AUC reconciliation

| Statement | Year (full primary) | Month (eligible 55 of 86 months, 191,938 intervals) |
|---|---|---|
| Pooled AUC M1 → M2 | 0.5654 → 0.6259 (+0.0604) | 0.5612 → 0.6095 (+0.0483) |
| **Pair weights** within / between | 17.195% / 82.805% | 1.979% / 98.021% |
| Within-stratum gain | +0.00636 | −0.00040 |
| Between-stratum gain | +0.07167 | +0.04930 |
| **Contributions** within / between | +0.00109 / +0.05935 | −0.0000079 / +0.04832 |
| **Contribution shares** within / between | 1.81% / 98.19% | −0.016% / 100.016% |
| Uncertainty | Facility CI for the pooled gain only, [+0.051, +0.070] (exists, unreported). No within-component or calendar CI. | None |

**Review statements.**
- Original Task 20 conflated pair weights and contribution shares and called within gains resolution-free. Both are wrong and corrected in round 2.
- Stage 1 said "no AUC intervals". Wrong for the pooled AUC; corrected in Stage 2.
- Stage 2 and corrected Task 20 agree on every number above.

**What survives.**
- *Year:* the between-year share is 98.2%; within-year gain is small and positive, driven mostly by 2022.
- *Month:* the between-month share is about 100%; within-month gain is about zero and slightly negative. The population is support-restricted (excludes every month from 2022-10).
- *Both resolutions:* the pooled payoff-AUC gain is ≥ 98% attributable to between-stratum pairs.

**Strongest resolution-invariant conclusion.**
- At both year and month resolution, in their respective populations, at least 98% of M2's pooled payoff-AUC gain comes from cross-period case–control pairs.
- The within-period gain is at most +0.006 against a pooled gain of +0.05–0.06.
- The *sign* of the within-period gain is **not** invariant.
- This is a descriptive identity of the observed data. It does not need an interval. Any inferential reading ("macro inputs do not improve within-period ranking") does.

---

## 12. Population / transport reconciliation

**Agreed facts (Stage 2, verified):**
- point joint-loss Δ −0.000819 (−1.2%), with no interval;
- mixed metric vector (payoff AUC and payoff calibration better; default AUC, default calibration and both Brier scores worse);
- interval age median 26 months vs 104;
- 2020 calendar share about 9.5% vs 20.8%;
- unseen cohort scored as the 2006 reference;
- M1 over-predicts unseen payoff by 70%, and M2 offsets it while under-predicting default by 71%;
- seen per-year Δ re-weighted to the unseen calendar mix gives +0.0104, so calendar mix alone does not produce the reversal.

**Positions.**
- Original Task 20: "transport" is wrong.
- Corrected Task 20: a terminology preference; not external validation.
- Stage 2: formal transportability not supported; avoid as headline.
- External audit: predictive transportability usage is legitimate in principle.

**Stage 3 terminology decision:**

| Term | Headline? | Reason |
|---|---|---|
| transport | **No** | Defensible in the body only if defined as predictive performance in a second evaluation population. In a headline it implies a tested invariance or generalisation claim, and it is not independent external validation. |
| population-dependent | **No** | Implies attribution to population. Population, age, calendar and encoding are confounded. |
| population heterogeneity | No | Implies a characterised source of heterogeneity. |
| evaluation-population sensitivity | Acceptable in the body | Describes that results differ by evaluation set without attribution. |
| **none in headline** | **Recommended** | The unseen result is a secondary, mixed, interval-free comparison. It belongs in results/discussion as "a second evaluation population (vintages absent from development)". |

---

## 13. Default / forbearance reconciliation

**Did Task 20 notice it?**
- No. There is no mention of forbearance, deferral, assistance or default-endpoint validity in any Task 20, Task 17 or Task 18 file (searched).
- Task 20 *elevated* the default arm to its top concern. Stage 1 raised the issue blind (threat 3); Stage 2 confirmed it from code and frozen counts.

**Consequences:**

| Quantity | Exposure | Judgement |
|---|---|---|
| Default AUC (seen) | Fall is 2020-only (−0.157 in 2020; +0.002 / +0.011 in adjacent years) | Report with caveat; do not headline |
| Default Brier (seen) | Δ +0.00005, both CIs exclude 0; 2020 rows dominate events | Report; tiny magnitude; caveat |
| Default calibration | M1's seen CITL failure is 2020-specific (0.00064 predicted vs 0.00322 observed) | Report the full vector; caveat |
| Unseen default | 320 of 588 unseen defaults are in the 2018 vintage, which has 2020 exposure | Caveat |
| CIF default | Under-prediction by all models arises after month 17 (2020-06 onward) | Caveat; the illustration stays valid as a description |
| 2020 interpretation | The observed "stress" partly includes a possibly policy-driven reporting state | Discuss as an open endpoint question |
| Joint log loss | Default rows contribute −0.00004 of +0.0161 | Essentially unaffected |
| Payoff findings | A forborne loan labelled "default" leaves the payoff risk set, a minor effect | Unaffected for the headline |

**Decisions:**
- **A. arXiv.** Neither provider documentation nor a label tabulation is required, **provided** every default-specific statement is explicitly qualified (endpoint does not handle forbearance or deferral; 60% of seen defaults in 2020; 81 in 2020-06) and the default arm is not used as a headline result. Obtaining the Freddie user-guide field definitions is strongly recommended because it is cheap.
- **B. Journal/conference.** **Both are required** before interpreting any default-specific result:
  - provider documentation of Current Loan Delinquency Status under COVID-19 forbearance and deferral, and of the Borrower Assistance, Payment Deferral and Disaster fields;
  - a label-only tabulation of those fields around 2020–2021 default events.

---

## 14. REDUCED / RATE sensitivity

**Did Task 20 know?**
- Task 17 records that "RATE and REDUCED sensitivities" were pre-specified and frozen.
- Neither Task 17, Task 18 nor Task 20 (any round) flags their **omission from v0.3**. Task 20's "the freeze … did not prevent reporting selection" remark was applied only to the default results.

**Stage 2 numbers re-checked** against R `/primary`:

| Model | Joint LL | Default AUC | Payoff AUC |
|---|---|---|---|
| REDUCED_M1 | 0.088963 | 0.6916 | 0.5650 |
| REDUCED_M2 | 0.090945 (Δ +0.00198, +2.2%) | 0.8136 | 0.6084 |
| RATE | 0.105324 (≈ M2 to 1e-6) | — | — |

All are correct.

**Classification: INCOMPLETE_REPORTING.**
- There is no evidence of intent, so I withdraw Stage 2's wording "selective reporting of the pre-specified design".
- It is **material**:
  - the proper-score direction is unchanged (worse with macro);
  - the magnitude is about 8 times smaller;
  - default discrimination moves in the *opposite* direction (+0.12 vs −0.07).
- REDUCED differs in feature set (no HPI or mortgage rate) and in development window (2006-02 onward, including the GFC). It is not a like-for-like replicate, but it was pre-specified for exactly this purpose.

**Manuscript consequence (P1):**
- Report REDUCED and RATE as pre-specified sensitivities in the results.
- State that the direction of the proper-score result holds, but that magnitude and default discrimination are specification- and window-dependent.
- Scope the title and claims to "a national macro specification" rather than to macroeconomic features generally.

---

## 15. Calibration reconciliation

**Blanket statements found and rejected:**

| Source | Statement | Status |
|---|---|---|
| v0.3 §4.2/§5 | Payoff "over-prediction" (pooled) | Incomplete: sign reverses by year |
| Task 20 original M6 | M1 unseen default "near-ideal" | Corrected (C04) |
| Task 20 original M6 | M2 default calibration uniformly worse | Corrected (C10) |
| Task 20 final M6 | Unseen: "omits the most damaging item" | One-sided (U7) |
| Stage 1 | "comparator poorly calibrated out of time" | Refined in Stage 2 (2020-specific for default) |

**Final interpretation (full vector, verified):**
- **Seen default:** CITL improves (|error| 0.000475 → 0.000239); slope worsens (0.48 → 0.35).
- **Seen payoff:** pooled CITL and slope worsen (+0.0129; 0.51 → 0.25). The error is concentrated in the top decile (19.3% predicted vs 3.0% observed). By year it is over in 2020 (10.0% vs 2.2%) and under in 2022–26 (0.18–0.46% vs 0.85–1.32%).
- **Unseen payoff:** M2 improves (CITL +0.0078 → −0.0028; slope 0.35 → 0.53).
- **Unseen default:** M2 worsens (CITL −0.00004 → −0.00064; slope 1.01 → 0.76).

**No global "M2 calibration is better/worse" statement is admissible.**

---

## 16. Uncertainty reconciliation

| Quantity | Status |
|---|---|
| Joint LL Δ (seen), facility and calendar | EXISTS_AND_REPORTED |
| Payoff Brier Δ, facility and calendar | EXISTS_AND_REPORTED |
| Default Brier Δ, facility and calendar (both exclude 0) | EXISTS_BUT_UNREPORTED |
| Pooled payoff AUC Δ, facility [+0.051, +0.070] | EXISTS_BUT_UNREPORTED |
| Pooled default AUC Δ, facility [−0.094, −0.045] | EXISTS_BUT_UNREPORTED |
| AUC Δ, calendar blocks | DOES_NOT_EXIST. NOT_NEEDED_FOR_DESCRIPTIVE_IDENTITY |
| Within-year / within-month AUC Δ | DOES_NOT_EXIST. NOT_NEEDED_FOR_DESCRIPTIVE_IDENTITY; NEEDED_IF_INFERENTIAL_CLAIM_RETAINED ("no within-period gain" as a general statement) |
| Ex-2020 Δ year-block range | EXISTS_AND_REPORTED (7-block resample; plausible, not exactly re-verifiable) |
| Unseen joint LL Δ | DOES_NOT_EXIST. NEEDED_IF_INFERENTIAL_CLAIM_RETAINED ("reversal" as an established difference) |
| Calibration statistics | DOES_NOT_EXIST. Not needed for descriptive reporting. |
| CIF horizons | DOES_NOT_EXIST. NOT_NEEDED for the illustrative role. |
| Refit / training uncertainty | DOES_NOT_EXIST. Disclose. |

**None of the missing intervals is an arXiv blocker** if the wording stays descriptive.

---

## 17. Manuscript v0.3 defect register

**P0 — invalidates central result:** none.

| ID | Defect | Evidence | Sections | Manuscript-only | Analysis | External docs |
|---|---|---|---|---|---|---|
| **P1-1** | Title and headline: "Population- and Regime-Dependent Transport"; `MIXED_TEMPORAL_TRANSPORT_ACROSS_POPULATIONS` | X01, X02 | Title, abstract, §1, §4.5, §7 | Yes | No | No |
| **P1-2** | Unseen result framed as reversal/transport. Should be a point estimate with no interval, a mixed metric vector, offsetting calibration, the 2006-reference cohort fallback, and confounded age/calendar/encoding. | X16, X17, X31; §12 | Abstract, §3.2, §4.5, §5, §7 | Yes | No (optional P3) | No |
| **P1-3** | Default endpoint validity is undisclosed: no forbearance/deferral handling; 60% of seen defaults in 2020, 81 in 2020-06. Default AUC fall is 2020-only. | X28; §13 | §3.1, §4.2, §4.5, §4.6, §6 | Yes (for arXiv) | Label tabulation for journal | Provider docs for journal; recommended for arXiv |
| **P1-4** | Pre-specified REDUCED / RATE sensitivities omitted | X29; §14 | §3.3, §4, §6, title scope | Yes | No | No |
| **P1-5** | Payoff calibration "over-prediction" and "concentrated in 2020" framing omits the per-year sign reversal and the 2022–26 relative deterioration | X30; §10 | Abstract, §4.2, §4.3, §5 | Yes | No | No |
| **P1-6** | Default discrimination and calibration vectors not reported for either population | X05, X06; §15 | Abstract, §4.2, §4.5, Table G | Yes | No | No |
| **P1-7** | Internal workflow vocabulary, status tokens, `Q19` tags and placeholders in the body | X03 | Throughout | Yes | No | No |
| P2-1 | Uncertainty reporting: add the existing facility AUC CIs and both default-Brier CIs; state absent calendar-AUC, within-component, unseen, calibration and CIF uncertainty; qualify the 8-block "interval" | §16; X07, X08, X25, X37 | §3.5, §4.4, §4.7, Table G | Yes | No | No |
| P2-2 | AUC decomposition presentation: pair weights vs contributions; resolution and population named; excluded months ≥ 2022-10; within-gain sign change | §11; X09, X35 | Abstract, §4.4 | Yes | No | No |
| P2-3 | 2020 discussion: two bounded readings without false premises; all years 2020–26 outside macro support; log-loss asymmetry | §10; X10 | §4.3, §5 | Yes | No | No |
| P2-4 | Duration-band fallback (181–240 months, 4.5% of seen intervals) undisclosed | X32 | §3.3, §6 | Yes | No | No |
| P2-5 | Methods: sampling design (hash, 20k per vintage, no weights), 70/30 roles, C = 1 fixed a priori, macro join at end of t−1 | X34 | §3.1–§3.4 | Yes | No | No |
| P2-6 | PMMS 2022-11-17 methodology break pooled without disclosure | X33 | §3.4, §6 | Yes | No | No |
| P2-7 | CIF: state that censoring is negligible (CG03 practically resolved); keep as supporting illustration | X13 | §3.6, §4.6, §6 | Yes | No | No |
| P2-8 | Realized-class LL decomposition (deterioration entirely on no-event intervals) unreported | X36 | §4.2 | Yes | No | No |
| P2-9 | No figures (3 from frozen values) | X22 | §4 | Yes (rendering) | No | No |
| P2-10 | Bibliography: "PEER_REVIEWED" tags, Bhattacharya URL, uncited entries (Sadhwani); add transportability/forbearance references if those terms are kept | X20 | §2, References | Yes | No | No |
| P3 | Label-only forbearance tabulation; unseen per-year/per-vintage breakdown with paired interval; per-year realized-class decomposition; within-AUC facility intervals; REDUCED intervals and per-year; facility-weighted rescoring; maturity count; PMMS split; VIFs; reference Brier; Bu2026 full text | §19 | — | — | Yes | Partly |
| P4 | Nonlinear/interaction challenger; current-state conditioning; regime modelling; Fannie replication; complexity-adjusted development fit | — | — | — | Yes | — |

**Counts: P1 = 7; P2 = 10.**

---

## 18. arXiv minimum

**No new analysis is required.** The minimum defensible path is reframing, removing unsupported claims and reporting frozen values:

1. Retitle and re-headline around the strongest surviving result (§20/§21). Remove "regime-dependent", "population-dependent", "transport" and the MIXED token from the headline (P1-1).
2. Restate the unseen comparison as a secondary, mixed, point-estimate result with its confounds and encoding fallback (P1-2).
3. Add a target-validity paragraph and qualify every default statement. Do not headline default AUC (P1-3, P1-6).
4. Report the pre-specified REDUCED and RATE sensitivities from frozen values (P1-4).
5. Replace pooled payoff "over-prediction" with the per-year calibration pattern from frozen `annual_calibration` (P1-5).
6. Strip workflow vocabulary (P1-7).
7. Apply the P2 disclosures. Most are one sentence to one paragraph, using frozen values.
8. Non-scientific: author identity, publication terms, ethics review.

What is cut rather than computed:
- the "transport/population" headline;
- any inferential reading of within-period AUC or of the unseen reversal;
- default-arm interpretation beyond description.

---

## 19. Peer-review path (ranked by expected scientific value)

1. **Default endpoint validation.** Provider documentation plus a label-only tabulation of assistance, deferral and disaster status around 2020–21 defaults, and a sensitivity excluding flagged episodes (new protocol). This decides whether the default arm means credit default.
2. **Unseen per-year and per-vintage scores, with a paired facility interval for the unseen Δ** (frozen arrays, no refit, registered). This decides whether any population result exists beyond calendar mix.
3. **REDUCED full reporting**: paired intervals and per-year breakdown from the frozen REDUCED predictions. It quantifies specification dependence, which is currently the largest unexplored source of variation.
4. **Per-year realized-class log-loss decomposition and per-year calibration for both populations.** It locates the error without a mechanism.
5. **Facility-bootstrap intervals for within-year and within-month AUC differences.** Needed only if "no within-period ranking gain" is to be stated inferentially.
6. **Facility-weighted rescoring.** Retires the outcome-correlated weighting objection.
7. **Maturity-event count within code 01** (descriptive).
8. **PMMS-break split** for 2022–23.
9. **Bu2026 / Peng2026 full-text positioning** (literature).

Not recommended at priority: SA04 complexity adjustment (development fit is not used); support-restricted 9-month evaluation (too small); calendar-block AUC intervals (8 blocks give little information beyond what is already reproduced).

---

## 20. Proposed central claim

In a frozen Freddie Mac monthly default/payoff model evaluated out of time on 2019–2026, adding seven national macroeconomic inputs raised pooled payoff AUC almost entirely through comparisons between calendar periods (at least 98% of the gain at both year and month resolution), while worsening joint log loss in 7 of 8 calendar years, with payoff probabilities over-predicted in 2020 and under-predicted in 2022–2026. The pooled discrimination gain therefore did not reflect better within-period ranking or better-calibrated probabilities.

---

## 21. Proposed title family (ranked)

1. **Pooled Concordance Versus Probability Quality: Calendar-Stratified Out-of-Time Validation of Macroeconomic Inputs in a Mortgage Competing-Risk Model**
2. Between-Period Concordance and Period-Varying Calibration: Out-of-Time Evaluation of National Macroeconomic Inputs in a Freddie Mac Default–Payoff Model
3. When Pooled AUC Rises but Probability Quality Falls: Calendar-Stratified Validation of Macroeconomic Inputs for Mortgage Default and Payoff
4. Calendar-Stratified Validation of Vintage-Aware Macroeconomic Features in a Mortgage Competing-Risk Model
5. Period-Heterogeneous Out-of-Time Performance of National Macroeconomic Inputs in a Monthly Mortgage Default and Payoff Model

Rationale:
- Titles 1 and 2 name the robust, resolution-invariant result (cross-period concordance) and the probability-quality contrast without attribution.
- Title 3 is accurate for the seen population but slightly rhetorical.
- Title 4 is safe but does not convey the finding.
- Title 5 is accurate but foregrounds heterogeneity over the decomposition lesson.
- None uses "population-dependent", "regime-dependent", "transport", "causal" or "robust".

---

## 22. Review reliability assessment

| Dimension | Task 20 | Stage 1 | Stage 2 |
|---|---|---|---|
| Blindness | None; authored Tasks 17/18 (disclosed) | Blind (manuscript and bib only) | Post-Stage-1; evidence package; Task 17–20 narrative withheld |
| Contamination | Carried a Task 17 count error into Task 20 | None | Minimal: saw one Q0001 Task 17 verdict before excluding those fields |
| Evidence access | Full repository incl. Task 18 outputs | Manuscript only | Code, protocols, frozen aggregates; not arrays |
| Arithmetic accuracy | High: 257/257 confirmed by audit and Stage 2. One count error (seven coefficients) plus its flawed round-1 correction. | Correct (all recomputations confirmed) | Correct; one self-caught slip (censored 6 → 5) |
| Interpretive accuracy | Low to moderate: 14 ledger corrections (≈19 assertions), all externally originated; ≥ 7 residual (§6); round 1 incomplete and introduced a new error | Moderate: 7 of 14 major concerns weakened or resolved by evidence; wrong on AUC-CI existence; imprecise "effective sample" | Moderate to high: no refuted finding found here. Overreach: "selective reporting" (withdrawn §14). Approximate calendar weights (flagged). Forbearance inference rests on timing plus code, not provider docs. |
| Self-correction | Dependent on external audit | n/a (frozen) | Partly self-correcting (corrected Stage 1 items explicitly) |
| Reproducibility | Ledger and diffs, but 143-pass file not committed | Text only | Scripts and pointers stated; arithmetic reproducible from the package |
| Independence of Stage 3 | — | — | The same reviewer reconciles Stage 1/2 (bias risk, disclosed) |

**Ranking by finding, not by reviewer.**

Highest-confidence findings (multi-path, evidence-anchored):
- X01–X04, X06, X09 (corrected form), X10 (corrected premises), X12, X16, X17;
- plus cold-only X29 (REDUCED numbers) and X30 (annual calibration), which are direct frozen values.

Medium confidence:
- X28 (forbearance): strong timing evidence plus code, but provider semantics are unverified;
- X31 (offsetting-errors account of the unseen gain): descriptive coincidence, not decomposed;
- X05 framing.

Low confidence or rejected:
- Task 20's M1 "robust, loss-relevant headline";
- SA04 HIGH_VALUE;
- CG03 HIGH_VALUE;
- Stage 1's "no AUC intervals";
- Stage 2's "selective" label.

---

## 23. Final synthesis

All three review paths agree that v0.3 has:
- no fatal flaw;
- exact, reproducible arithmetic;
- an over-reaching title and headline vocabulary;
- a valid central methodological lesson (pooled concordance gains can be almost entirely between-period).

They disagree on what should replace the headline:
- Task 20 would elevate default discrimination.
- The evidence shows that arm is the most fragile part of the study: it is 2020-only and target-exposed.

The cold path adds four material, frozen-evidence findings that no Task 17–20 round identified:
- forbearance/target validity;
- omitted REDUCED sensitivity;
- sign-reversing payoff calibration with persistent 2022–26 deterioration;
- the offsetting-error structure of the unseen gain.

The corrected Task 20 remains useful for:
- presentation (figures, tokens);
- the default-calibration reporting gap;
- disciplined AUC terminology.

Its interpretive judgements should not control the revision without checking against frozen evidence, as §6 shows.

The next revision should be controlled by:
1. the frozen Task 10 report (`macro_competing_risk_validation.json`);
2. the frozen Task 11 diagnostics (`annual_calibration`, `score_accounting`, macro ranges);
3. the Task 19 closure output;
4. the target-construction code;

and not by any review's narrative.

CENTRAL RESULT VALID:
QUALIFIED

FATAL FLAW:
NO

V0.3 PUBLICATION STATUS:
MAJOR REVISION

MINIMUM NEW ANALYSIS REQUIRED FOR ARXIV:
NONE. All required content is frozen values plus reframing. Default-specific claims must be qualified rather than analysed.

EXTERNAL DOCUMENTATION REQUIRED FOR ARXIV:
NONE, provided default-specific results are explicitly qualified and not headlined. Freddie Mac loan-level dataset documentation of delinquency status under COVID-19 forbearance/deferral is strongly recommended, and is REQUIRED before journal submission.

P1 MANUSCRIPT DEFECTS:
7

P2 MANUSCRIPT DEFECTS:
10

PEER-REVIEW-STRENGTHENING ANALYSES:
1. Default-endpoint validation: provider documentation plus a label-only assistance/deferral tabulation around 2020–21 defaults, then an exclusion sensitivity.
2. Unseen per-year and per-vintage scores with a paired facility interval for the unseen Δ.
3. REDUCED sensitivity with paired intervals and per-year breakdown.
4. Per-year realized-class log-loss decomposition and per-year calibration for both populations.
5. Facility-bootstrap intervals for within-year and within-month AUC differences.
6. Facility-weighted rescoring.
7. Maturity-event count within payoff code 01.
8. PMMS-break split for 2022–23.
9. Bu2026 / Peng2026 full-text positioning.

RECOMMENDED CENTRAL CLAIM:
In a frozen Freddie Mac monthly default/payoff model evaluated out of time on 2019–2026, adding seven national macroeconomic inputs raised pooled payoff AUC almost entirely through comparisons between calendar periods (at least 98% of the gain at both year and month resolution) while worsening joint log loss in 7 of 8 calendar years, with payoff probabilities over-predicted in 2020 and under-predicted in 2022–2026, so the pooled discrimination gain did not reflect better within-period ranking or better-calibrated probabilities.

RECOMMENDED TITLE:
Pooled Concordance Versus Probability Quality: Calendar-Stratified Out-of-Time Validation of Macroeconomic Inputs in a Mortgage Competing-Risk Model

NEXT ACTION:
Draft a v0.4 revision plan (not the manuscript) that maps each P1 defect (P1-1 to P1-7) to the exact frozen artifact field that will supply the new text, starting with `annual_calibration` and `primary/REDUCED_*`, and have the author approve it before any manuscript editing begins.

STOP.

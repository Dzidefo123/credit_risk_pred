# Task 12 findings — exploratory evidence only

**Decision: REFINANCING INCENTIVE HYPOTHESIS MIXED EXPLORATORILY.** The borrower-relative representation does not meet the prespecified success criteria. No model is promoted, and Task 10's negative conclusion remains unchanged.

P2 improves payoff Brier by only 0.00001136 and increases payoff AUC from 0.565426 to 0.624555. Joint log loss nevertheless worsens from 0.08920131 to 0.09578293. Both facility and coarse calendar-year uncertainty intervals for this log-loss difference are above zero. The payoff Brier improvement has a facility interval below zero, but its calendar-year interval crosses zero. These are exploratory fixed-model comparisons, not independent confirmation.

Observed monthly payoff is 1.5357%. P1 predicts 1.1379%; P2 predicts 0.9294%. The payoff calibration slope falls from 0.5094 to 0.2209. Better ranking and a small squared-error improvement do not establish reliable probabilities.

The calendar pattern is especially informative. P2 reduces payoff underprediction in 2020–2021, but worsens it after 2022. In 2023, observed payoff is 0.8738% while P2 predicts 0.0491%. Nearly 24.7% of exploratory gaps lie outside the development range, predominantly in the stronger negative-incentive regime. The fitted payoff log-odds slopes are positive on both sides of parity, consistent with the incentive hypothesis; this conditional direction does not establish temporal transport or identify a causal refinancing effect. Actual payoff also includes mechanisms beyond rate-driven refinancing.

Payoff CIF results are uneven:

| Horizon | Aalen–Johansen | P1 | P2 |
|---|---:|---:|---:|
| 12 months | 13.30% | 13.85% | 11.50% |
| 24 months | 33.67% | 25.87% | 27.55% |
| 36 months | 50.56% | 35.36% | 40.85% |
| 60 months | 60.94% | 48.60% | 43.72% |

The default Brier difference is +0.00002084. Default CIF remains underpredicted: at 60 months, P2 predicts 2.11% against the 4.67% observed reference. Joint hazard changes prevent attributing this difference solely to payoff competition. The default and stability gates pass their prespecified tolerances; that does not establish good absolute calibration or stable behavior across regimes.

The linear-gap sensitivity also worsens joint log loss and pooled payoff calibration despite a small payoff Brier improvement. Adding non-rate macro context worsens both joint log loss and payoff Brier. Neither sensitivity replaces P2. Burnout was excluded before fitting; no threshold, interaction, window or recalibration search followed these results.

**Evidence boundary:** all modeled outcomes were previously inspected. March 2026 records exist in the source releases and frozen macro table, but do not establish a new independent validation population. [Availability clarification](../../docs/track_b/refinancing_evidence_availability_clarification.json) records the distinction between physical availability and admissible untouched evidence. Original-coupon proxies and unverified historical loan-data knowledge time remain limitations.

Exactly one next task is recommended: an external Fannie Mae replication feasibility and harmonization contract. Its purpose is to establish whether fresh evidence with compatible semantics can be sealed, rather than fitting another model. [Feasibility note](../../docs/track_b/REFINANCING_EXTERNAL_REPLICATION_FEASIBILITY.md). No external acquisition or next-task implementation was performed.

The complete [23-section report](REFINANCING_INCENTIVE_PAYOFF_RESEARCH.md) and [machine-readable evidence](refinancing_incentive_payoff_research.json) contain population counts, frozen specifications, sensitivity results, subgroup diagnostics, uncertainty and preservation evidence.

# Track B Task 4 sample-expansion amendment

50 evaluation defaults gives illustrative rare-event CITL halfwidth ~0.28 rather than ~0.88 at five; AUC halfwidth at assumed 0.8 ~0.07 rather than ~0.23. Development 150 supplies materially more support for later modest fixed models. Neither threshold is a validity/regulatory minimum; dependence and calibration slope still require empirical validation.

Chosen N: **20000**. Frozen before new performance access.

| N | Overall expected (95% predictive) | Development expected (95% predictive) | Evaluation expected (95% predictive) | Joint lower probability |
|---|---|---|---|---|
| 5000 | 156.9 [113, 209] | 66.9 [39, 103] | 27.0 [11, 51] | 0.000 |
| 10000 | 314.2 [220, 425] | 134.4 [76, 210] | 54.5 [20, 106] | 0.000 |
| 20000 | 628.9 [436, 856] | 269.2 [149, 425] | 109.4 [40, 215] | 0.913 |

Nested prediction: observed k stays fixed; new events ~ BetaBinomial(N-1000,k+0.5,1000-k+0.5); Jeffreys prior. The nested original event counts remain fixed. A joint union bound avoids assuming independence between development and evaluation events. Uniform-prior sensitivity is in JSON.

- Identifier hash ranking approximates outcome-independent exchangeable sampling within the fixed 2010 universe
- Facility events treated as independent for planning; unknown borrower links and finite-universe heterogeneity not captured
- Per-selected-ID temporal yield (5/1000) integrates eligibility and fixed split; not 5/130 applied to all selected loans
- 95% posterior predictive ranges, not frequentist confidence guarantees; finite population correction omitted (<1.1% sampled)
- No change in calendar windows or original split hash; approximate planning cannot guarantee calibration or future performance

50 evaluation / 150 development default-loan targets are research planning objectives, not guarantees of model validity. Marginal support is 30 evaluation / 100 development; lower support is inadequate. No second expansion is permitted here. Historical knowledge time remains UNVERIFIED. AP will be the primary future PR summary; no model metrics are computed in Task 4.

The original 1,000 must equal the first 1,000 of the same annual hash ranking and remain a subset of first N. New IDs are frozen privately before any performance stream opens. The 2010 universe, default proxy, t0, months 1-12 horizon, payoff, censoring, ambiguity, minimum history, firewall and original Task 3 hash/calendar split remain unchanged. This amendment supersedes only the cap and operational retention budgets for its own versioned output; original protocols and evidence remain immutable.

Phase A is aggregate-only and must pass tests with amendment/code hashes attested before Phase B. Phase B scans each performance member once, without layout probes that reopen it; selected records are validated and spooled locally, then one loan at a time is passed through the existing panel builder. No raw corpus extraction or adaptive replacements.

Bounded SQLite selected-row spool; one facility at a time through unchanged panel builder; bounded per-loan accounting. Universe IDs are retained during source scan for linkage, then released. Memory need not scale with retained history count. Estimated retained rows: 1,444,640; panel CSV about 296 MB, with selected-row cache/index and disk workspace expected 0.5-1.5 GiB. Scan runtime planning range 465-1200 seconds is heuristic; retained parsing adds work, so source scan time alone is not constant. 1.5 GiB process guard; 2 GiB spool and 1 GiB panel caps; no truncation to fit limits.

Brier precision cannot be predicted without score distributions. AUC approximations use Hanley-McNeil with assumed AUC scenarios, not expanded measurements; CITL/prevalence approximations use rare-event Fisher/Poisson information, not calibration fits. Unknown borrower dependence, overlapping monthly windows and score heterogeneity can worsen precision.

Source SHA256: `a82bc0f1efffccfb494b2b33f61877428bb4a6443c1a73d44bc7ee24d77aa45d`. Original sample SHA256: `b40de7596b0d4fbb189c1ceec5176f4461f7d290117250c1a7ec15b605ce7b11`. Original protocol SHA256: `5fccc658726e339d3fca9c76d300b5e45c0b9fce3189b595a19e12cb425f47c5`. Input report hashes and complete calculations are in the machine amendment.

User explicitly approved Task 4 Phase A and B; Phase B only after Phase A artifacts pass tests. No further expansion, outcome-adaptive resizing, changed cohort or model refitting authorized.

Planning references: [SciPy beta-binomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html), [Hanley and McNeil (1982)](https://doi.org/10.1148/radiology.143.1.7063747).

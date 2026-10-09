# Task 17 erratum — structural ranking argument withdrawn

Task 17's report and CH03 argued that adding month-constant national macro terms,
without interactions, made a within-month facility ranking improvement structurally
impossible. That argument is withdrawn. It was an audit error, not an error in the
frozen experiment. The original Task 17 review report and registers remain unchanged in history and
on disk; this additive erratum records the correction.

For the fitted multinomial model, payoff probability is

$$h_P=\frac{\exp(\eta_P)}{1+\exp(\eta_D)+\exp(\eta_P)}.$$

A shared shift to either cause logit does not generally induce a common monotone
transformation of payoff probability across facilities. The competing default logit
enters each facility's denominator at a different level. Consequently, ordering can
change even when the added covariates are constant within a month. In addition, M2
was jointly refitted: its mortgage-characteristic coefficients can differ from M1.

The empirical interpretation does not depend on this incorrect argument. Task 18's
corrected year-level decomposition assigns almost all pooled AUC gain to
between-year comparisons. The separately registered and already executed local
closure provides month-level summaries for supported months. Those findings are
empirical, population- and support-qualified observations, not structural
impossibility or causal results.

Task 18 also distinguishes the ex-calendar-2020 contribution residual from the
renormalized subgroup mean and records its own correction ledger. Manuscript v0.3
uses these corrected definitions without rewriting the original Task 17 review.

Evidence: `reports/paper/task18_sa01_within_period_auc.json` (structural check and
weighted decomposition), `reports/paper/task18_corrections.json`,
`src/credit_risk/track_b/macro_hazard/models.py`, and the additive Task 19 claim map.

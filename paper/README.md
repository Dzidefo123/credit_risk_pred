# Track B manuscript v0.1 — internal author-review draft

Read [main.md](main.md). It is not an arXiv submission or publication-ready manuscript.
Source: [main.source.md](main.source.md), with frozen claim/value tokens and comments.
No experiments are run by the Markdown rendering process.

Task 14's registry and evidence hierarchy govern every result. Task 15 adds narrative,
display formatting, claim bindings, citation gaps and author-review flags; it does not
change the evidence freeze. Task 13T remains immutable and Fannie remains proposed,
unauthorized and unexecuted. The detailed source investigation is left in the audit trail.

`{{CLAIM_ID|/pointer|format}}` copies a numeric value from the frozen claim point estimate.
`@uncertainty` selects the already recorded interval. Formatting is solely for presentation;
no raw data, fitted models, predictions or statistical estimators are used. Rendered numeric
markers resolve through `docs/paper/manuscript_numeric_bindings.json`. Claim comments and
usage resolve to the unchanged Task 14 registry.

Tables use those bindings. Eight figure placeholders and detailed captions correspond to
the frozen figure plan; no new plots are generated. Appendix placeholders define later
supplement work. Missing literature is explicitly marked and listed in citation_gaps.json;
no fabricated bibliography or DOI is inserted. Publication terms and ethics/author review
remain explicit flags, without invented permissions or approvals.

The one-shot builder preserves this draft and rejects overwriting outputs. Author edits
should use a separately authorized version/amendment and refreshed traceability checks.
Validation: `.venv/Scripts/python.exe scripts/check_manuscript.py` and the manuscript tests.
No LaTeX installation, compilation or PDF submission workflow is introduced in Task 15.

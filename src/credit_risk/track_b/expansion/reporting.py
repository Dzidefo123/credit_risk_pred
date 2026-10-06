"""Finalize completed aggregate evidence; never opens the source or fits models."""

import importlib.metadata
import json
from pathlib import Path

from .planning import lf_hash


def finalize(root):
    root = Path(root)
    path = root / "reports/track_b/sample_expansion_feasibility.json"
    r = json.loads(path.read_text(encoding="utf-8"))
    raw = r["any_qualifying_record_default_loans"]
    eligible = r["event_support"]["primary"]["default_loans"]
    r["planning_vs_observed"]["overall"].update(
        observed=raw,
        definition=(
            "Any qualifying raw default record, comparable to Task 2 planning count "
            "31; eligible-landmark default support reported separately"
        ),
    )
    lo, hi = r["planning_vs_observed"]["overall"]["predictive_95"]
    r["planning_vs_observed"]["overall"]["position"] = (
        "below" if raw < lo else "above" if raw > hi else "within"
    )
    r["incident_event_support_distinction"] = dict(
        raw_qualifying_default_loans=raw,
        eligible_positive_default_loans=eligible,
        raw_default_loans_without_eligible_positive_landmark=raw - eligible,
        interpretation=(
            "Five such loans have zero eligible landmarks: 17 insufficient-lookback "
            "rows and 217 prior/current event/unknown/gap/terminal rows. No "
            "eligibility exception or label imputation applied."
        ),
    )
    r["execution_provenance"] = json.loads(
        (root / "data/track_b/manifests/expansion_v1/phase_a_test_attestation.json").read_text(
            encoding="utf-8"
        )
    )
    r["package_versions"] = {
        n: importlib.metadata.version(n) for n in ["pandas", "numpy", "scipy", "credit-risk-lab"]
    }
    r["algorithm_version"] = "nested_2010_expansion_v1 / unchanged Task2 parser-1.0.0"
    r["resources"]["temporary_disk_measurement_scope"] = (
        "ZIP materialization measured at zero. SQLite journal/index-sort transient "
        "disk peak was not instrumented; panel and persistent cache sizes are "
        "measured separately."
    )
    r["limitations"].append(
        "SQLite transient scratch peak was not measured; zero "
        "temporary_source_bytes refers only to ZIP materialization"
    ) if (
        "SQLite transient scratch peak was not measured; zero "
        "temporary_source_bytes refers only to ZIP materialization"
    ) not in r["limitations"] else None
    r["reporting_source_sha256_lf"] = lf_hash(Path(__file__))
    path.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    p = r["phase_a"]
    parts = [
        "# Track B sample-expansion feasibility",
        "## Executive Summary",
        "**"
        + r["decision"]
        + (
            "**. The 20,000-loan frozen nested sample supports 246 development and 95 "
            "evaluation defaulting loans. No model fitted or expanded "
            "predictive/calibration metrics computed."
        ),
        "## Motivation",
        (
            "Task 3 had 13 development and five temporal evaluation event loans. This "
            "study expanded by identifier ranking, not by outcomes, while preserving "
            "the original baseline."
        ),
        "## Phase A Planning",
        p["probability_model"]
        + (
            ". Original events remain fixed. Uncertainty and conditional evaluation "
            "eligibility are included; per-selected-ID evaluation yield uses 5/1,000, "
            "not 5/130 applied to every selected loan."
        ),
        (
            "| N | Overall expected [95% predictive] | Development expected [95% "
            "predictive] | Evaluation expected [95% predictive] | Joint lower "
            "probability |"
        ),
        "|---|---|---|---|---|",
    ]
    for row in p["candidates"]:
        cells = [
            f"{row['support'][k]['expected']:.1f} {row['support'][k]['predictive_95']}"
            for k in ["overall", "development", "evaluation"]
        ]
        parts.append(
            "| "
            + " | ".join([str(row["n"]), *cells, f"{row['joint_probability_lower_bound']:.3f}"])
            + " |"
        )
    parts.extend(
        [
            "## Event-Support Objective",
            p["objective_rationale"],
            "## Selected Expansion Size",
            (
                "Exactly 20,000: the smallest considered candidate with a conservative "
                ">=90% joint probability lower bound for >=150 development and >=50 "
                "temporal evaluation event loans. Jeffreys and uniform-prior calculations "
                "were frozen before new outcomes; no resizing followed."
            ),
            "## Nested Sampling",
            (
                "Same 1,820,190 annual IDs and SHA256(salt:ID) order. First 1,000 match "
                "cryptographically and remain in first 20,000. Quarters Q1/Q2/Q3/Q4: 3,942 "
                "/ 3,947 / 5,613 / 6,498. Expanded sample SHA256: `"
            )
            + r["sample"]["sample_set_sha256"]
            + "`. Amendment SHA256: `"
            + r["amendment_sha256_lf"]
            + (
                "`. Full algorithm, original/source hashes and Phase A test attestation "
                "are in JSON."
            ),
            "## Phase B Empirical Results",
            (
                "All 20,000 selected histories found; zero replacements, missing "
                "histories, gaps reported or conflicting duplicates. Qualifying raw "
                "default records occur on 623 facilities. Eligible positive landmarks "
                "represent 618 facilities. Five raw-default facilities never have an "
                "eligible landmark (lookback/prefix rules); they are not silently added to "
                "development."
            ),
            "| Cohort | Landmarks | Loans | Positive landmarks | Default loans |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for k, v in r["event_support"].items():
        parts.append(
            "| "
            + " | ".join(
                [
                    k,
                    *[
                        str(v[x])
                        for x in ["landmarks", "loans", "positive_landmarks", "default_loans"]
                    ],
                ]
            )
            + " |"
        )
    parts.extend(
        [
            (
                "Eligible outcome status counts: 7,153 default-positive; 1,029,959 "
                "complete event-free; 203,933 payoff/maturity; 14,238 right-censored; 370 "
                "ambiguous. Unknown status remains unknown."
            ),
            "## Temporal Event Support",
            (
                "Task 3 hash groups and monthly landmarks are unchanged. Development ends "
                "2014-12, evaluation starts 2016-01; 2015 is purged. "
                "Development/evaluation loans do not overlap. Purged and unused row sets "
                "are disjoint, but their loan/event-loan counts overlap and must not be "
                "added. Loan-level fractions: development 246/13,801=1.78%; evaluation "
                "95/2,423=3.92%. These are descriptive event-support fractions across "
                "landmark horizons, not new calibrated twelve-month probabilities."
            ),
            "## Planning vs Observed",
            "| Quantity | Expected | 95% predictive range | Observed | Position |",
            "|---|---:|---|---:|---|",
        ]
    )
    for k, v in r["planning_vs_observed"].items():
        parts.append(
            f"| {k} | {v['expected']:.1f} | {v['predictive_95']} "
            f"| {v['observed']} | {v['position']} |"
        )
    parts.extend(
        [
            (
                "Overall comparison uses raw qualifying-record loans (623), matching the "
                "planning proxy. Usable incident-landmark default loans (618) are reported "
                "separately. All comparisons fall within the frozen ranges; no adaptive "
                "follow-up sample was selected."
            ),
            "## Resource Impact",
            (
                "One source-performance pass: 127,232,321 rows scanned, 1,400,360 "
                "retained. Source scan 645.0 seconds; total panel/audit run 1,180.9 "
                "seconds. Peak process working set 482.4 MiB (versus Task 2 675.9 MiB). "
                "Panel 285,888,374 bytes; retained cache 222,453,760 bytes; combined "
                "~484.8 MiB. No full corpus extraction. ZIP materialization peak zero; "
                "SQLite journal/index-sort transient disk peak not instrumented."
            ),
            (
                "Cached Git-ignored panel and selected-row spool have immutable "
                "manifests/hashes. Future research can avoid rescanning the population. "
                "Cached loading/model-matrix memory and fit runtime have not been "
                "benchmarked; do not infer them from streaming memory. Fresh full "
                "rebuilding cost was ~19.7 minutes in this environment."
            ),
            "## Preservation",
            (
                "Original source, sample, Task 2/3 evidence and Track A unchanged. All "
                "1,000 original panel histories reproduce after type normalization; "
                "original frozen bytes remain unchanged. The machine report records hashes "
                "for all previous protected files. No loan IDs or raw rows are published. "
                "Locked holdout untouched and historical model performance not re-evaluated."
            ),
            "## Limitations",
            "\n".join("- " + s for s in r["limitations"]),
            "## Decision",
            r["decision"]
            + (
                ": research event support clears the fixed 150/50 gate. This is potential "
                "for stronger validation, not established model accuracy/calibration or a "
                "regulatory minimum."
            ),
            "Next task: " + r["next_task"] + ". Not implemented.",
            (
                "Planning references: [SciPy "
                "beta-binomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.s"
                "tats.betabinom.html), [Hanley and McNeil "
                "(1982)](https://doi.org/10.1148/radiology.143.1.7063747)."
            ),
        ]
    )
    text = "\n\n".join(parts).replace("|\n\n|", "|\n|")
    (root / "reports/track_b/SAMPLE_EXPANSION_FEASIBILITY.md").write_text(
        text + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    finalize(Path(__file__).resolve().parents[4])

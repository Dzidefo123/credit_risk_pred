"""Finalize support-based validation, explicitly restricting unseen fixed effects."""

import json
from collections import Counter

from credit_risk.track_b.data.schemas import digest

from .eligibility import ordinal
from .study import immutable_json, read_json


def finalize(root):
    private = root / "data/track_b/macro_support"
    spec = read_json(root / "docs/track_b/macro_support_eligibility_spec.json")
    counts = read_json(private / "counts.json")
    candidate = spec["candidate_validation"]
    blocks = {
        "development": candidate["development"],
        "temporal_evaluation": candidate["temporal_evaluation"],
    }
    vintage_counts = {}
    hashes = {}
    for path in sorted((private / "v1").glob("eligibility_primary_*.jsonl")):
        vintage = path.stem.rsplit("_", 1)[1]
        roles = {r: Counter() for r in blocks}
        hashes[path.name] = digest(path)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                subject = json.loads(line)
                if not subject["contributes"]:
                    continue
                # Each retained prefix intersects one contiguous primary support window.
                if (
                    ordinal(subject["last_target"]) - ordinal(subject["first_target"]) + 1
                    != (subject["intervals"])
                ):
                    raise ValueError("Noncontiguous risk spans need exact interval replay")
                role = subject["role"]
                lower, upper = blocks[role]
                first = max(subject["first_target"], lower)
                last = min(subject["last_target"], upper)
                if first <= last:
                    roles[role]["facilities"] += 1
                    roles[role]["intervals"] += ordinal(last) - ordinal(first) + 1
                    for event in ["default", "payoff"]:
                        roles[role][event + "s"] += (
                            subject["exit_reason"] == event
                            and lower <= subject["last_target"] <= upper
                        )
        vintage_counts[vintage] = {r: dict(c) for r, c in roles.items()}
    pooled = {r: Counter() for r in blocks}
    for roles in vintage_counts.values():
        for r, c in roles.items():
            pooled[r].update(c)
    if {r: dict(c) for r, c in pooled.items()} != counts["validation_counts"]["PRIMARY"]:
        raise ValueError("Private endpoint/span validation counts did not reproduce")
    seen = [y for y, roles in vintage_counts.items() if roles["development"].get("intervals", 0)]
    unseen = [y for y in vintage_counts if y not in seen]
    seen_counts = Counter()
    unseen_counts = Counter()
    for y, roles in vintage_counts.items():
        (seen_counts if y in seen else unseen_counts).update(roles["temporal_evaluation"])
    feasible = (
        seen_counts["defaults"] >= candidate["minimum_unique_defaults"]
        and seen_counts["payoffs"] >= candidate["minimum_unique_payoffs"]
    )
    value = dict(
        version="macro-support-validation-v1.0.0",
        basis="Structural fixed-effect estimability and frozen support, no performance search",
        candidate_dates_and_facility_roles_unchanged=True,
        finalized_after_support_counts=True,
        seen_development_vintages=seen,
        unseen_temporal_vintages=unseen,
        primary_temporal_seen_vintage_counts=dict(seen_counts),
        external_unseen_vintage_counts=dict(unseen_counts),
        by_vintage=vintage_counts,
        eligibility_subject_files_sha256=hashes,
        primary_temporal_metric_population="Seen development vintages in evaluation hash role",
        unseen_temporal_handling=(
            "Separate extrapolation sensitivity; unseen vintage effect fixed to zero relative "
            "to 2006 reference, flagged UNSEEN_VINTAGE. Never pool with primary temporal metrics."
        ),
        leave_vintage_out_handling=(
            "Held-out vintage coefficient is unestimated, explicitly fixed to zero relative "
            "to training 2006 reference. If 2006 is held out, use 2008 training reference. "
            "Report restriction and cause/event feasibility for each fold; no learned "
            "held-out fixed-effect claim. This tests conditional extrapolation, not an "
            "estimated unseen-cohort effect."
        ),
        full_population_primary_cohort_design="Six vintage indicators, reference 2006",
        temporal_seen_vintage_feasibility_passed=feasible,
        uncertainty="Facility and calendar-block dependence; borrower clustering unavailable",
        task10_new_sealed_ledger_required=True,
        virgin_holdout_claim=False,
        model_fitting_performed=False,
    )
    immutable_json(root / "docs/track_b/macro_support_validation_design.json", value)
    return value

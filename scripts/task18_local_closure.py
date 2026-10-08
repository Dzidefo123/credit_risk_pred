"""Task 18 local closure: compute the two residual aggregates from frozen arrays.

Run this on the machine that holds data/track_b/models/macro_hazard_v1/. It closes the
two gaps Task 18 could not close in a fresh checkout:

  SA01-M  calendar-MONTH stratified payoff AUC (Task 18 reached year granularity only)
  SA06-H  CIF landmark entry month/year histogram (Task 18 derived bounds only)

What it does NOT do, by construction:
  - fit, refit, recalibrate or tune anything
  - regenerate, modify or re-save any prediction array
  - read or emit any loan-level value, identifier or raw row
  - touch Fannie anything
  - write to any frozen artifact

It opens the frozen arrays read-only, verifies their hashes against the frozen
Task 10 report before using them, and emits ONLY aggregate counts and metrics.
Inspect the output before sending it anywhere.

Usage, from the repository root:

    uv run --no-sync python scripts/task18_local_closure.py
    # or: python scripts/task18_local_closure.py

Writes: reports/paper/task18_local_closure_output.json
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data/track_b/models/macro_hazard_v1"
FROZEN_REPORT = ROOT / "reports/track_b/macro_competing_risk_validation.json"
PROTOCOL = ROOT / "docs/track_b/macro_competing_risk_protocol.json"
OUT = ROOT / "reports/paper/task18_local_closure_output.json"

REGISTRATION_SHA256_LF = "b78aa2b0f63c620d6890efe203d6da811a887bd8a2697dd5a64d2a7bcfcefbed"
MIN_CAUSE_EVENTS = 20  # frozen auc support rule, metrics.scores()
PAYOFF = 2


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordinal(month: str) -> int:
    text = str(month).strip().replace("-", "")
    return int(text[:4]) * 12 + int(text[4:6]) - 1


def label(n: int) -> str:
    return f"{n // 12:04d}-{n % 12 + 1:02d}"


def load_frozen():
    if not PRIVATE.is_dir():
        sys.exit(
            f"Frozen model directory not found: {PRIVATE}\n"
            "Run this on the machine that holds the Task 10 private artifacts."
        )
    frozen = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    # Verify prediction arrays against the frozen hashes before trusting them.
    expected = frozen["prediction_hashes"]
    verified, mismatched, missing = {}, {}, []
    for name in ("M1_evaluation.npy", "M2_evaluation.npy"):
        path = PRIVATE / name
        if not path.is_file():
            missing.append(name)
            continue
        actual = digest(path)
        if expected.get(name) == actual:
            verified[name] = actual
        else:
            mismatched[name] = {"expected": expected.get(name), "actual": actual}
    if missing:
        sys.exit(f"Missing frozen prediction arrays: {missing}")
    if mismatched:
        sys.exit(
            "Frozen prediction array hash mismatch; refusing to proceed.\n"
            + json.dumps(mismatched, indent=2)
        )

    evaluation = np.load(PRIVATE / "evaluation.npy")
    m1 = np.load(PRIVATE / "M1_evaluation.npy")
    m2 = np.load(PRIVATE / "M2_evaluation.npy")
    if not (len(evaluation) == len(m1) == len(m2)):
        sys.exit("Row count mismatch between risk array and prediction arrays")

    seen = np.isin(evaluation["vintage"], spec["splits"]["primary_vintages"])
    return frozen, spec, evaluation, m1, m2, seen, verified


def sa01_month(evaluation, m1, m2, seen, frozen):
    """Calendar-month stratified payoff AUC, frozen eligibility rule, no fitting."""
    data = evaluation[seen]
    p1, p2 = m1[seen], m2[seen]
    y = np.asarray(data["event"], int)
    months = np.asarray(data["month"], int)

    rows, excluded = [], []
    for mo in np.unique(months):
        mask = months == mo
        yy = y[mask]
        cases = int(np.count_nonzero(yy == PAYOFF))
        controls = int(np.count_nonzero(yy != PAYOFF))
        if cases < MIN_CAUSE_EVENTS or controls < MIN_CAUSE_EVENTS:
            excluded.append(
                {"month": label(int(mo)), "cases": cases, "controls": controls,
                 "reason": "frozen auc support rule: fewer than 20 in a class"}
            )
            continue
        rows.append(
            {
                "month": label(int(mo)),
                "intervals": int(mask.sum()),
                "cases": cases,
                "controls": controls,
                "within_pairs": cases * controls,
                "M1_payoff_auc": float(roc_auc_score(yy == PAYOFF, p1[mask][:, PAYOFF])),
                "M2_payoff_auc": float(roc_auc_score(yy == PAYOFF, p2[mask][:, PAYOFF])),
            }
        )
    for r in rows:
        r["gap"] = r["M2_payoff_auc"] - r["M1_payoff_auc"]

    if not rows:
        return {"status": "INCONCLUSIVE_DUE_TO_EVENT_SUPPORT", "eligible_months": 0,
                "excluded_months": excluded}

    wp = sum(r["within_pairs"] for r in rows)
    pooled1 = float(roc_auc_score(y == PAYOFF, p1[:, PAYOFF]))
    pooled2 = float(roc_auc_score(y == PAYOFF, p2[:, PAYOFF]))
    tot_pairs = int(np.count_nonzero(y == PAYOFF)) * int(np.count_nonzero(y != PAYOFF))
    between_pairs = tot_pairs - wp

    agg = {}
    for key, pooled in (("M1", pooled1), ("M2", pooled2)):
        pair_w = sum(r["within_pairs"] * r[f"{key}_payoff_auc"] for r in rows) / wp
        equal_w = sum(r[f"{key}_payoff_auc"] for r in rows) / len(rows)
        between = (pooled * tot_pairs - wp * pair_w) / between_pairs if between_pairs else None
        agg[key] = {
            "pooled": pooled,
            "within_month_pair_weighted": pair_w,
            "within_month_equal_weighted": equal_w,
            "between_month_solved": between,
        }

    gain_pooled = agg["M2"]["pooled"] - agg["M1"]["pooled"]
    gain_within = agg["M2"]["within_month_pair_weighted"] - agg["M1"]["within_month_pair_weighted"]
    if abs(gain_within) < 0.01:
        interpretation = "WITHIN_PERIOD_GAIN_NEGLIGIBLE"
    elif gain_within < 0:
        interpretation = "WITHIN_PERIOD_GAIN_REVERSED"
    else:
        interpretation = "WITHIN_PERIOD_GAIN_PRESENT"

    return {
        "status": "EXECUTED",
        "pooled_reconciliation": {
            "computed_M1": pooled1,
            "frozen_M1": frozen["primary"]["M1"]["scores"]["payoff_auc"],
            "computed_M2": pooled2,
            "frozen_M2": frozen["primary"]["M2"]["scores"]["payoff_auc"],
            "M1_matches": abs(pooled1 - frozen["primary"]["M1"]["scores"]["payoff_auc"]) < 1e-9,
            "M2_matches": abs(pooled2 - frozen["primary"]["M2"]["scores"]["payoff_auc"]) < 1e-9,
            "note": "If these do not match, the environment differs from the frozen run and the "
                    "month-level numbers must not be used.",
        },
        "eligibility_rule": f"cases >= {MIN_CAUSE_EVENTS} and controls >= {MIN_CAUSE_EVENTS}",
        "eligible_months": len(rows),
        "excluded_month_count": len(excluded),
        "excluded_months": excluded,
        "pair_structure": {
            "total_pairs": tot_pairs,
            "within_month_pairs": wp,
            "between_month_pairs": between_pairs,
            "between_month_pair_share": between_pairs / tot_pairs if tot_pairs else None,
        },
        "per_month": rows,
        "aggregates": agg,
        "gains": {
            "pooled": gain_pooled,
            "within_month_pair_weighted": gain_within,
            "within_month_equal_weighted": agg["M2"]["within_month_equal_weighted"]
            - agg["M1"]["within_month_equal_weighted"],
            "between_month_solved": (agg["M2"]["between_month_solved"] - agg["M1"]["between_month_solved"])
            if agg["M1"]["between_month_solved"] is not None else None,
            "within_share_of_pooled_gain": gain_within / gain_pooled if gain_pooled else None,
        },
        "per_month_gap_signs": {
            "positive": sum(1 for r in rows if r["gap"] > 0),
            "negative": sum(1 for r in rows if r["gap"] < 0),
        },
        "interpretation": interpretation,
    }


def sa06_entry(evaluation, seen, frozen, spec):
    """CIF landmark entry histogram. Replicates cif.landmarks() entry selection only."""
    data = evaluation[seen]
    ids, starts, counts = np.unique(data["facility"], return_index=True, return_counts=True)
    if not np.all(data["facility"][starts + counts - 1] == ids):
        sys.exit("Facility histories are not contiguous; cannot replicate landmark selection")
    entry = np.asarray(data["month"][starts], int)

    cut = ordinal("2026-02")
    by_year, by_month = {}, {}
    for e in entry:
        by_year[str(e // 12)] = by_year.get(str(e // 12), 0) + 1
        by_month[label(int(e))] = by_month.get(label(int(e)), 0) + 1
    modal_year = max(by_year, key=by_year.get)
    modal_month = max(by_month, key=by_month.get)
    n = len(entry)

    return {
        "status": "EXECUTED",
        "landmark_reconciliation": {
            "computed_landmarks": n,
            "frozen_landmarks": frozen["cif"]["landmarks"],
            "matches": n == frozen["cif"]["landmarks"],
        },
        "entry_year_distribution": dict(sorted(by_year.items())),
        "entry_month_distribution": dict(sorted(by_month.items())),
        "earliest_entry": label(int(entry.min())),
        "latest_entry": label(int(entry.max())),
        "median_entry": label(int(np.median(entry))),
        "distinct_entry_months": int(len(by_month)),
        "modal_year": modal_year,
        "share_in_modal_year": by_year[modal_year] / n,
        "modal_month": modal_month,
        "share_in_modal_month": by_month[modal_month] / n,
        "span_months": int(entry.max() - entry.min() + 1),
        "per_horizon_truncation_check": {
            str(h): {
                "latest_permitted_entry": label(cut - h + 1),
                "computed_truncated": int(np.count_nonzero(entry + h - 1 > cut)),
                "frozen_truncated": frozen["cif"]["horizons"][str(h)]["calendar_truncated_landmarks"],
            }
            for h in spec["cif"]["horizons"]
        },
    }


def main() -> None:
    frozen, spec, evaluation, m1, m2, seen, verified = load_frozen()
    out = {
        "version": "task18-local-closure-v1",
        "purpose": "Close the two Task 18 residual gaps using frozen arrays. Aggregates only.",
        "registration_sha256_lf": REGISTRATION_SHA256_LF,
        "no_model_fitted": True,
        "no_prediction_regenerated": True,
        "no_calibration_fitted": True,
        "no_loan_level_values_emitted": True,
        "frozen_arrays_verified_sha256": verified,
        "population": {
            "seen_vintage_intervals": int(seen.sum()),
            "frozen_seen_vintage_intervals": frozen["split_counts"]["evaluation_seen"]["intervals"],
        },
        "SA01_month": sa01_month(evaluation, m1, m2, seen, frozen),
        "SA06_entry": sa06_entry(evaluation, seen, frozen, spec),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    s1, s6 = out["SA01_month"], out["SA06_entry"]
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    if s1.get("status") == "EXECUTED":
        rec = s1["pooled_reconciliation"]
        print(f"SA01-M pooled reconciles: M1 {rec['M1_matches']}  M2 {rec['M2_matches']}")
        print(f"  eligible months {s1['eligible_months']} (excluded {s1['excluded_month_count']})")
        print(f"  between-month pair share {s1['pair_structure']['between_month_pair_share']:.4f}")
        print(f"  pooled gain          {s1['gains']['pooled']:+.4f}")
        print(f"  within-month gain    {s1['gains']['within_month_pair_weighted']:+.4f}")
        print(f"  between-month gain   {s1['gains']['between_month_solved']:+.4f}")
        print(f"  -> {s1['interpretation']}")
    print()
    print(f"SA06-H landmarks reconcile: {s6['landmark_reconciliation']['matches']}")
    print(f"  entry span {s6['earliest_entry']} .. {s6['latest_entry']} ({s6['span_months']} months)")
    print(f"  median entry {s6['median_entry']}; modal year {s6['modal_year']} "
          f"({s6['share_in_modal_year']:.4f}); modal month {s6['modal_month']} "
          f"({s6['share_in_modal_month']:.4f})")
    print("\nReview the JSON, then attach it to the conversation.")


if __name__ == "__main__":
    main()

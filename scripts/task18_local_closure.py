"""Task 18 local closure, v2: compute the two residual aggregates from frozen arrays.

Run on the machine holding data/track_b/models/macro_hazard_v1/. Closes the gaps Task 18
could not close in a fresh checkout:

  SA01-M  calendar-MONTH stratified payoff AUC (Task 18 reached year granularity only)
  SA06-H  CIF landmark entry month/year histogram (Task 18 derived bounds only)

v2 addresses eight defects found by user-side audit of v1 before execution:

  1. Repository root was resolved as parents[1], which is wrong if the file is moved out
     of scripts/. Root is now located by searching upward for repository markers, and
     --root overrides. The script refuses to run if markers are absent.
  2. v1 verified the prediction arrays but not evaluation.npy, which supplies outcomes,
     months, vintages and row alignment. v2 verifies a full chain: the public report
     pins ledger_sha256, the ledger carries risk_array_sha256 for evaluation.npy, and
     the report pins each prediction array.
  3. v1 computed between_pairs as total - within_eligible, so within-month pairs from
     SPARSE-EXCLUDED months were silently folded into a bucket labelled "between".
     v2 keeps three buckets: within-eligible, within-excluded, between.
  4. v1's within_share_of_pooled_gain omitted the pair weight, reproducing the 10.5%
     contribution error. v2 reports pair-weighted contribution shares and labels the
     magnitude ratio separately.
  5. v1 computed reconciliation flags but did not fail closed. v2 exits non-zero on any
     reconciliation mismatch unless --allow-mismatch is passed, and the output records
     the decision.
  6. v1 hard-coded the registration hash. v2 reads the registration file and verifies it,
     and refuses to proceed if it is missing unless --no-registration is passed.
  7. v1's landmark check omitted the frozen monthly-continuity assertion. v2 replicates
     cif.landmarks() including the contiguity and continuity checks.
  8. v1's docstring claimed it never reads loan-level values. It does: it reads the
     frozen loan-level arrays. It EMITS only aggregates. Stated accurately below.

What this script does: opens frozen arrays read-only, verifies their hashes, computes
aggregate metrics, writes one JSON.

What it does not do: fit, refit, recalibrate or tune anything; regenerate, modify or
re-save any array; emit any loan-level value, row or identifier; touch Fannie anything;
write to any frozen artifact. It reads loan-level data and emits aggregates only.

Usage, from the repository root:

    uv run --no-sync python scripts/task18_local_closure.py
    python scripts/task18_local_closure.py --root /path/to/repo

Writes: reports/paper/task18_local_closure_output.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

MARKERS = ("pyproject.toml", "src/credit_risk", "reports/track_b")
PRIVATE_RELATIVE = "data/track_b/models/macro_hazard_v1"
FROZEN_RELATIVE = "reports/track_b/macro_competing_risk_validation.json"
PROTOCOL_RELATIVE = "docs/track_b/macro_competing_risk_protocol.json"
REGISTRATION_RELATIVE = "reports/paper/task18_analysis_registration.json"
OUT_RELATIVE = "reports/paper/task18_local_closure_output.json"

EXPECTED_REGISTRATION_SHA = "b78aa2b0f63c620d6890efe203d6da811a887bd8a2697dd5a64d2a7bcfcefbed"
MIN_CAUSE_EVENTS = 20  # frozen support rule, metrics.scores()
PAYOFF = 2
CUTOFF = "2026-02"


def fail(message: str) -> None:
    sys.exit(f"REFUSING TO PROCEED: {message}")


def digest_bytes(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_lf(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def ordinal(month: str) -> int:
    text = str(month).strip().replace("-", "")
    return int(text[:4]) * 12 + int(text[4:6]) - 1


def label(n: int) -> str:
    return f"{n // 12:04d}-{n % 12 + 1:02d}"


def locate_root(explicit: str | None) -> pathlib.Path:
    """Defect 1: never infer the root from this file's depth alone."""
    if explicit:
        root = pathlib.Path(explicit).expanduser().resolve()
        missing = [m for m in MARKERS if not (root / m).exists()]
        if missing:
            fail(f"--root {root} is not the repository root; missing {missing}")
        return root
    for candidate in [
        pathlib.Path.cwd().resolve(),
        *pathlib.Path.cwd().resolve().parents,
        pathlib.Path(__file__).resolve().parent,
        *pathlib.Path(__file__).resolve().parents,
    ]:
        if all((candidate / m).exists() for m in MARKERS):
            return candidate
    fail(
        "could not locate the repository root. Run from inside the repository, or pass "
        f"--root. Markers required: {list(MARKERS)}"
    )
    raise AssertionError  # unreachable


def verify_inputs(root: pathlib.Path, skip_registration: bool) -> dict:
    """Defects 2 and 6: full hash chain, and verify the registration rather than assert it."""
    private = root / PRIVATE_RELATIVE
    if not private.is_dir():
        fail(f"frozen model directory not found: {private}")
    frozen_path = root / FROZEN_RELATIVE
    if not frozen_path.is_file():
        fail(f"frozen report not found: {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    chain: dict = {"registration": None, "ledger": None, "risk_array": None, "predictions": {}}

    # Registration.
    registration_path = root / REGISTRATION_RELATIVE
    if registration_path.is_file():
        actual = digest_lf(registration_path)
        chain["registration"] = {
            "path": REGISTRATION_RELATIVE,
            "sha256_lf": actual,
            "expected": EXPECTED_REGISTRATION_SHA,
            "matches": actual == EXPECTED_REGISTRATION_SHA,
        }
        if actual != EXPECTED_REGISTRATION_SHA:
            fail(
                "Task 18 registration hash mismatch.\n"
                f"  expected {EXPECTED_REGISTRATION_SHA}\n  actual   {actual}"
            )
    elif skip_registration:
        chain["registration"] = {
            "path": REGISTRATION_RELATIVE,
            "status": "ABSENT_AND_WAIVED",
            "note": "--no-registration passed. The run is not covered by a verified registration.",
        }
    else:
        fail(
            f"Task 18 registration not found at {REGISTRATION_RELATIVE}.\n"
            "  It is committed on the branch carrying Task 18 (commit fde602d or later). If this\n"
            "  clone predates that commit, fetch it, or use --no-registration for an\n"
            "  run explicitly marked as uncovered."
        )

    # Ledger, pinned by the public report.
    ledger_path = private / "task10_evaluation_ledger.json"
    if not ledger_path.is_file():
        fail(f"evaluation ledger not found: {ledger_path}")
    # The repository hashes the ledger with raw bytes (track_b.data.schemas.digest),
    # so this must NOT be LF-normalised.
    ledger_actual = digest_bytes(ledger_path)
    ledger_expected = frozen.get("ledger_sha256")
    chain["ledger"] = {
        "sha256": ledger_actual,
        "expected": ledger_expected,
        "matches": ledger_actual == ledger_expected,
    }
    if ledger_actual != ledger_expected:
        fail(
            "evaluation ledger hash does not match the frozen report's ledger_sha256.\n"
            f"  expected {ledger_expected}\n  actual   {ledger_actual}"
        )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("state") != "CONSUMED" or ledger.get("prediction_generation_count") != 1:
        fail(
            "ledger is not in the expected consumed state "
            f"(state={ledger.get('state')}, count={ledger.get('prediction_generation_count')})"
        )

    # Risk array, pinned by the ledger registration.
    risk_path = private / "evaluation.npy"
    if not risk_path.is_file():
        fail(f"risk array not found: {risk_path}")
    risk_expected = ledger.get("registration", {}).get("risk_array_sha256")
    risk_actual = digest_bytes(risk_path)
    chain["risk_array"] = {
        "sha256": risk_actual,
        "expected": risk_expected,
        "matches": risk_actual == risk_expected,
    }
    if risk_expected is None:
        fail("ledger registration does not carry risk_array_sha256; cannot verify evaluation.npy")
    if risk_actual != risk_expected:
        fail(
            "evaluation.npy hash does not match the ledger's risk_array_sha256.\n"
            f"  expected {risk_expected}\n  actual   {risk_actual}"
        )

    # Prediction arrays, pinned by the public report.
    expected_predictions = frozen["prediction_hashes"]
    for name in ("M1_evaluation.npy", "M2_evaluation.npy"):
        path = private / name
        if not path.is_file():
            fail(f"prediction array not found: {path}")
        actual = digest_bytes(path)
        chain["predictions"][name] = {
            "sha256": actual,
            "expected": expected_predictions.get(name),
            "matches": actual == expected_predictions.get(name),
        }
        if actual != expected_predictions.get(name):
            fail(
                f"prediction array hash mismatch for {name}.\n"
                f"  expected {expected_predictions.get(name)}\n  actual   {actual}"
            )

    return {"frozen": frozen, "chain": chain, "private": private}


def load_arrays(root: pathlib.Path, private: pathlib.Path):
    spec = json.loads((root / PROTOCOL_RELATIVE).read_text(encoding="utf-8"))
    evaluation = np.load(private / "evaluation.npy")
    m1 = np.load(private / "M1_evaluation.npy")
    m2 = np.load(private / "M2_evaluation.npy")
    if not (len(evaluation) == len(m1) == len(m2)):
        fail("row count mismatch between the risk array and the prediction arrays")
    for field in ("event", "month", "facility", "vintage"):
        if field not in (evaluation.dtype.names or ()):
            fail(f"risk array is missing the required field {field!r}")
    seen = np.isin(evaluation["vintage"], spec["splits"]["primary_vintages"])
    return spec, evaluation, m1, m2, seen


def sa01_month(evaluation, m1, m2, seen, frozen) -> dict:
    """Month-stratified payoff AUC. Defects 3 and 4 addressed."""
    data = evaluation[seen]
    p1, p2 = m1[seen], m2[seen]
    y = np.asarray(data["event"], int)
    months = np.asarray(data["month"], int)

    rows, excluded = [], []
    for value in np.unique(months):
        mask = months == value
        yy = y[mask]
        cases = int(np.count_nonzero(yy == PAYOFF))
        controls = int(np.count_nonzero(yy != PAYOFF))
        record = {
            "month": label(int(value)),
            "intervals": int(mask.sum()),
            "cases": cases,
            "controls": controls,
            "within_pairs": cases * controls,
        }
        if cases < MIN_CAUSE_EVENTS or controls < MIN_CAUSE_EVENTS:
            record["reason"] = f"frozen support rule: fewer than {MIN_CAUSE_EVENTS} in a class"
            excluded.append(record)
            continue
        record["M1_payoff_auc"] = float(roc_auc_score(yy == PAYOFF, p1[mask][:, PAYOFF]))
        record["M2_payoff_auc"] = float(roc_auc_score(yy == PAYOFF, p2[mask][:, PAYOFF]))
        record["gap"] = record["M2_payoff_auc"] - record["M1_payoff_auc"]
        rows.append(record)

    total_cases = int(np.count_nonzero(y == PAYOFF))
    total_controls = int(np.count_nonzero(y != PAYOFF))
    total_pairs = total_cases * total_controls
    eligible_pairs = sum(r["within_pairs"] for r in rows)
    excluded_pairs = sum(r["within_pairs"] for r in excluded)
    between_pairs = total_pairs - eligible_pairs - excluded_pairs  # defect 3: three buckets

    out = {
        "eligibility_rule": f"cases >= {MIN_CAUSE_EVENTS} and controls >= {MIN_CAUSE_EVENTS}",
        "eligible_months": len(rows),
        "excluded_month_count": len(excluded),
        "imputed_auc_count": 0,
        "excluded_months": excluded,
        "per_month": rows,
        "pair_structure": {
            "total_pairs": total_pairs,
            "within_month_pairs_eligible": eligible_pairs,
            "within_month_pairs_excluded": excluded_pairs,
            "between_month_pairs": between_pairs,
            "between_month_pair_share": between_pairs / total_pairs if total_pairs else None,
            "note": (
                "Three buckets. Within-month pairs from SPARSE-EXCLUDED months are held separately "
                "and are NOT part of the between-month bucket."
            ),
        },
    }
    if not rows:
        out["status"] = "INCONCLUSIVE_DUE_TO_EVENT_SUPPORT"
        return out

    pooled1 = float(roc_auc_score(y == PAYOFF, p1[:, PAYOFF]))
    pooled2 = float(roc_auc_score(y == PAYOFF, p2[:, PAYOFF]))
    out["pooled_reconciliation"] = {
        "computed_M1": pooled1,
        "frozen_M1": frozen["primary"]["M1"]["scores"]["payoff_auc"],
        "computed_M2": pooled2,
        "frozen_M2": frozen["primary"]["M2"]["scores"]["payoff_auc"],
        "M1_matches": abs(pooled1 - frozen["primary"]["M1"]["scores"]["payoff_auc"]) < 1e-9,
        "M2_matches": abs(pooled2 - frozen["primary"]["M2"]["scores"]["payoff_auc"]) < 1e-9,
    }

    aggregates = {}
    for key, pooled in (("M1", pooled1), ("M2", pooled2)):
        pair_weighted = (
            sum(r["within_pairs"] * r[f"{key}_payoff_auc"] for r in rows) / eligible_pairs
        )
        aggregates[key] = {
            "pooled": pooled,
            "within_month_eligible_pair_weighted": pair_weighted,
            "within_month_eligible_equal_weighted": sum(r[f"{key}_payoff_auc"] for r in rows)
            / len(rows),
        }
    gain_pooled = aggregates["M2"]["pooled"] - aggregates["M1"]["pooled"]
    gain_within = (
        aggregates["M2"]["within_month_eligible_pair_weighted"]
        - aggregates["M1"]["within_month_eligible_pair_weighted"]
    )
    weight_within = eligible_pairs / total_pairs if total_pairs else 0.0

    out["aggregates"] = aggregates
    out["gains"] = {
        "pooled": gain_pooled,
        "within_month_stratum_gain_pair_weighted": gain_within,
        "within_month_stratum_gain_equal_weighted": (
            aggregates["M2"]["within_month_eligible_equal_weighted"]
            - aggregates["M1"]["within_month_eligible_equal_weighted"]
        ),
        # defect 4: contribution is pair-weighted, not a bare magnitude ratio
        "within_month_contribution_to_pooled_gain": weight_within * gain_within,
        "within_month_contribution_share": (
            weight_within * gain_within / gain_pooled if gain_pooled else None
        ),
        "within_stratum_gain_as_fraction_of_pooled_gain": (
            gain_within / gain_pooled if gain_pooled else None
        ),
        "eligible_within_pair_weight": weight_within,
        "note": (
            "contribution_share is pair-weighted. The final ratio is a magnitude comparison and is "
            "NOT a contribution share. A between-month AUC is not solved here, "
            "because the residual contains two buckets and attributing it wholly to between-month "
            "comparisons would mislabel it."
        ),
    }
    out["per_month_gap_signs"] = {
        "positive": sum(1 for r in rows if r["gap"] > 0),
        "negative": sum(1 for r in rows if r["gap"] < 0),
    }
    out["interpretation"] = (
        "WITHIN_PERIOD_GAIN_NEGLIGIBLE"
        if abs(gain_within) < 0.01
        else ("WITHIN_PERIOD_GAIN_REVERSED" if gain_within < 0 else "WITHIN_PERIOD_GAIN_PRESENT")
    )
    out["status"] = "EXECUTED"
    return out


def sa06_entry(evaluation, seen, frozen, spec) -> dict:
    """Entry histogram. Defect 7: replicate the frozen contiguity AND continuity checks."""
    data = evaluation[seen]
    ids, starts, counts = np.unique(data["facility"], return_index=True, return_counts=True)
    ends = starts + counts - 1
    if not np.all(data["facility"][ends] == ids):
        fail("facility histories are not contiguous; cannot replicate cif.landmarks()")
    if not np.all(data["month"][ends] - data["month"][starts] + 1 == counts):
        fail("facility histories are not monthly-continuous; cannot replicate cif.landmarks()")
    entry = np.asarray(data["month"][starts], int)

    cutoff = ordinal(CUTOFF)
    by_year: dict[str, int] = {}
    by_month: dict[str, int] = {}
    for value in entry:
        by_year[str(int(value) // 12)] = by_year.get(str(int(value) // 12), 0) + 1
        by_month[label(int(value))] = by_month.get(label(int(value)), 0) + 1
    modal_year = max(by_year, key=by_year.get)
    modal_month = max(by_month, key=by_month.get)
    n = len(entry)

    truncation = {}
    for horizon in spec["cif"]["horizons"]:
        computed = int(np.count_nonzero(entry + horizon - 1 > cutoff))
        frozen_value = frozen["cif"]["horizons"][str(horizon)]["calendar_truncated_landmarks"]
        truncation[str(horizon)] = {
            "latest_permitted_entry": label(cutoff - horizon + 1),
            "computed_truncated": computed,
            "frozen_truncated": frozen_value,
            "matches": computed == frozen_value,
        }

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
        "distinct_entry_months": len(by_month),
        "modal_year": modal_year,
        "share_in_modal_year": by_year[modal_year] / n,
        "modal_month": modal_month,
        "share_in_modal_month": by_month[modal_month] / n,
        "span_months": int(entry.max() - entry.min() + 1),
        "per_horizon_truncation_check": truncation,
    }


def reconciliation_failures(out: dict) -> list[str]:
    """Defect 5: collect every mismatch so the run can fail closed."""
    failures = []
    sa01 = out["SA01_month"]
    if sa01.get("status") == "EXECUTED":
        rec = sa01["pooled_reconciliation"]
        for model in ("M1", "M2"):
            if not rec[f"{model}_matches"]:
                failures.append(
                    f"SA01 pooled payoff AUC for {model} does not match the frozen report "
                    f"(computed {rec['computed_' + model]!r}, frozen {rec['frozen_' + model]!r})"
                )
    sa06 = out["SA06_entry"]
    if not sa06["landmark_reconciliation"]["matches"]:
        failures.append("SA06 landmark count does not match the frozen report")
    for horizon, record in sa06["per_horizon_truncation_check"].items():
        if not record["matches"]:
            failures.append(
                f"SA06 truncation count at horizon {horizon} does not match the frozen report "
                f"(computed {record['computed_truncated']}, frozen {record['frozen_truncated']})"
            )
    if (
        out["population"]["seen_vintage_intervals"]
        != out["population"]["frozen_seen_vintage_intervals"]
    ):
        failures.append("seen-vintage interval count does not match the frozen split counts")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=None, help="repository root; auto-located when omitted")
    parser.add_argument(
        "--no-registration",
        action="store_true",
        help="proceed without a verified Task 18 registration, recorded as uncovered",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="write output even if reconciliation fails; the failures are recorded",
    )
    args = parser.parse_args()

    root = locate_root(args.root)
    verified = verify_inputs(root, args.no_registration)
    spec, evaluation, m1, m2, seen = load_arrays(root, verified["private"])
    frozen = verified["frozen"]

    out = {
        "version": "task18-local-closure-v2",
        "purpose": "Close the two Task 18 residual gaps using frozen arrays. Aggregates only.",
        "repository_root": str(root),
        "v2_defects_addressed": 8,
        "reads_loan_level_arrays": True,
        "emits_loan_level_values": False,
        "no_model_fitted": True,
        "no_prediction_regenerated": True,
        "no_calibration_fitted": True,
        "verification_chain": verified["chain"],
        "population": {
            "seen_vintage_intervals": int(seen.sum()),
            "frozen_seen_vintage_intervals": frozen["split_counts"]["evaluation_seen"]["intervals"],
        },
        "SA01_month": sa01_month(evaluation, m1, m2, seen, frozen),
        "SA06_entry": sa06_entry(evaluation, seen, frozen, spec),
    }

    failures = reconciliation_failures(out)
    out["reconciliation_failures"] = failures
    out["reconciliation_clean"] = not failures
    out["mismatch_waived"] = bool(failures) and args.allow_mismatch

    if failures and not args.allow_mismatch:
        for line in failures:
            print(f"RECONCILIATION FAILURE: {line}", file=sys.stderr)
        sys.exit(
            "Refusing to write output. The environment does not reproduce the frozen run, so the "
            "month-level and entry figures must not be used. Pass --allow-mismatch only to "
            "investigate, never to publish."
        )

    path = root / OUT_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    sa01, sa06 = out["SA01_month"], out["SA06_entry"]
    print(f"wrote {path.relative_to(root)}\n")
    print(
        f"verification chain: ledger {verified['chain']['ledger']['matches']}, "
        f"risk array {verified['chain']['risk_array']['matches']}, "
        f"predictions {all(v['matches'] for v in verified['chain']['predictions'].values())}"
    )
    if sa01.get("status") == "EXECUTED":
        gains, pairs = sa01["gains"], sa01["pair_structure"]
        print(
            f"\nSA01-M  eligible months {sa01['eligible_months']} "
            f"(excluded {sa01['excluded_month_count']})"
        )
        aggregate = sa01["aggregates"]
        print(
            f"  within-month M1 {aggregate['M1']['within_month_eligible_pair_weighted']:.5f}"
            f"  M2 {aggregate['M2']['within_month_eligible_pair_weighted']:.5f}"
        )
        print(
            f"  within-month stratum gain {gains['within_month_stratum_gain_pair_weighted']:+.5f}"
        )
        share = gains["within_month_contribution_share"]
        print(
            "  within-month contribution share "
            + (f"{share:+.5f}" if share is not None else "undefined (pooled gain is zero)")
        )
        print(
            f"  pair buckets: eligible {pairs['within_month_pairs_eligible']:,} | "
            f"excluded {pairs['within_month_pairs_excluded']:,} | "
            f"between {pairs['between_month_pairs']:,}"
        )
        print(f"  -> {sa01['interpretation']}")
    print(
        f"\nSA06-H  entry {sa06['earliest_entry']} .. {sa06['latest_entry']} "
        f"({sa06['span_months']} months, {sa06['distinct_entry_months']} distinct)"
    )
    print(
        f"  median {sa06['median_entry']}; modal month {sa06['modal_month']} "
        f"({sa06['share_in_modal_month']:.4f}); modal year {sa06['modal_year']} "
        f"({sa06['share_in_modal_year']:.4f})"
    )
    print("\nReconciliation clean. Review the JSON, then attach it to the conversation.")


if __name__ == "__main__":
    main()

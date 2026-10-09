"""Authorized post-hoc closure: read verified frozen arrays, emit aggregates only."""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PRIVATE = "data/track_b/models/macro_hazard_v1/"
REPORT = "reports/track_b/macro_competing_risk_validation.json"
PROTOCOL = "docs/track_b/macro_competing_risk_protocol.json"
MANIFEST = "docs/paper/literature/task16_preservation_manifest.json"
SUPPORT = 20


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def ordinal(month):
    year, month = map(int, month.split("-"))
    if not 1 <= month <= 12:
        raise ValueError("Invalid calendar month")
    return year * 12 + month - 1


def label(month):
    return f"{int(month) // 12:04d}-{int(month) % 12 + 1:02d}"


def check_equal(actual, expected, description):
    if not np.isclose(actual, expected, rtol=0, atol=1e-9):
        raise ValueError("Reconciliation failed: " + description)


def verify_hashes(root, hashes):
    for name, expected in hashes.items():
        if digest(root / name) != expected:
            raise ValueError("Frozen input hash mismatch: " + name)


def register(root, out):
    report, spec, manifest = read(root / REPORT), read(root / PROTOCOL), read(root / MANIFEST)
    code_hash = digest(Path(__file__))
    public = {name: digest(root / name) for name in (REPORT, PROTOCOL, MANIFEST)}
    private = {}
    for name in ("evaluation.npy", "M1_evaluation.npy", "M2_evaluation.npy"):
        expected = manifest["private_byte_hashes"][PRIVATE + name]
        if name != "evaluation.npy" and report["prediction_hashes"][name] != expected:
            raise ValueError("Public prediction manifests disagree")
        private[PRIVATE + name] = expected
    if spec != report["protocol"]:
        raise ValueError("Protocol differs from frozen report")
    record = dict(
        version="task18-local-closure-corrected-v2",
        registered_at_utc=datetime.now(UTC).isoformat(),
        repository_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        authorization=(
            "User explicitly authorized corrected script preparation "
            "and execution after the external-review audit."
        ),
        classification="AUTHORIZED_POST_HOC_FROZEN_ARRAY_CLOSURE",
        original_task18_registration_available=False,
        original_registration_hash_is_not_claimed_as_verified=True,
        frozen_outputs_will_not_be_overwritten=True,
        code_sha256=code_hash,
        public_input_hashes=public,
        private_input_hashes=private,
        population="Task 10 primary seen-vintage evaluation only",
        methods={
            "SA01_month": (
                "Month AUC: cases and controls >=20. Full-primary AUC reconciled. "
                "Exact eligible-month population decomposition only. Sparse months excluded. "
                "Pair weights applied to contributions; no significance/materiality threshold."
            ),
            "SA06_entry": (
                "First interval per sorted contiguous history; continuity and count gates. "
                "Month/year histogram, central order statistics, modes and 24m endpoints."
            ),
        },
        prohibited=[
            "fitting",
            "calibration",
            "prediction regeneration",
            "new source outcomes",
            "Fannie",
            "model promotion",
            "holdout ledger API",
            "bootstrap",
            "causal mechanism claim",
        ],
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / "local_closure_registration.json"
    write_new(path, record)
    seal = out / "local_closure_registration.sha256"
    with seal.open("x", encoding="utf-8") as handle:
        handle.write(digest(path) + "\n")
    return record


def validate_arrays(data, p1, p2):
    required = {"facility", "vintage", "month", "event", "role"}
    if data.dtype.names is None or not required <= set(data.dtype.names):
        raise ValueError("Risk array schema mismatch")
    if not len(data) or not np.isin(data["event"], [0, 1, 2]).all():
        raise ValueError("Invalid event support")
    if not np.all(data["role"] == 1):
        raise ValueError("Risk array includes non-evaluation roles")
    for p in (p1, p2):
        if (
            p.shape != (len(data), 3)
            or not np.isfinite(p).all()
            or (p < 0).any()
            or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-10)
        ):
            raise ValueError("Invalid aligned probabilities")


def month_analysis(data, p1, p2, frozen):
    target = data["event"] == 2
    full = {m: float(roc_auc_score(target, p[:, 2])) for m, p in (("M1", p1), ("M2", p2))}
    for m in full:
        check_equal(full[m], frozen["primary"][m]["scores"]["payoff_auc"], m + " full AUC")
    rows, excluded, eligible = [], [], np.zeros(len(data), dtype=bool)
    for month in np.unique(data["month"]):
        mask = data["month"] == month
        cases, controls = int(target[mask].sum()), int((~target[mask]).sum())
        if min(cases, controls) < SUPPORT:
            excluded.append(
                dict(
                    month=label(month),
                    cases=cases,
                    controls=controls,
                    reason="Frozen 20-event/class AUC support rule",
                )
            )
            continue
        eligible |= mask
        a1, a2 = (float(roc_auc_score(target[mask], p[mask, 2])) for p in (p1, p2))
        rows.append(
            dict(
                month=label(month),
                cases=cases,
                controls=controls,
                intervals=int(mask.sum()),
                within_pairs=cases * controls,
                M1_auc=a1,
                M2_auc=a2,
                gap=a2 - a1,
            )
        )
    result = dict(
        status="EXECUTED" if rows else "INCONCLUSIVE_DUE_TO_EVENT_SUPPORT",
        full_primary_auc_reconciled=full,
        eligible_months=len(rows),
        excluded_months=excluded,
        per_month=rows,
        eligible_intervals=int(eligible.sum()),
        total_intervals=len(data),
        interpretation="Descriptive post-hoc scores; no inference or causal attribution",
    )
    if not rows:
        return result
    within = sum(r["within_pairs"] for r in rows)
    cases = int(target[eligible].sum())
    total = cases * (int(eligible.sum()) - cases)
    between = total - within
    weight = within / total
    agg = {}
    for m, p in (("M1", p1), ("M2", p2)):
        pooled = float(roc_auc_score(target[eligible], p[eligible, 2]))
        within_auc = sum(r["within_pairs"] * r[m + "_auc"] for r in rows) / within
        between_auc = (pooled * total - within_auc * within) / between if between else None
        if between_auc is not None and not -1e-9 <= between_auc <= 1 + 1e-9:
            raise ValueError("Invalid solved between-month AUC")
        agg[m] = dict(
            eligible_population_pooled=pooled,
            within_month_pair_weighted=within_auc,
            within_month_equal_weighted=float(np.mean([r[m + "_auc"] for r in rows])),
            between_month_solved=between_auc,
        )
    pooled_gain = agg["M2"]["eligible_population_pooled"] - agg["M1"]["eligible_population_pooled"]
    within_gain = agg["M2"]["within_month_pair_weighted"] - agg["M1"]["within_month_pair_weighted"]
    between_gain = (
        agg["M2"]["between_month_solved"] - agg["M1"]["between_month_solved"] if between else None
    )
    within_contribution = weight * within_gain
    between_contribution = (1 - weight) * between_gain if between else 0.0
    check_equal(
        within_contribution + between_contribution, pooled_gain, "weighted AUC gain decomposition"
    )
    result.update(
        decomposition_population=(
            "Eligible calendar months only; not the full-primary pooled AUC decomposition"
        ),
        pair_structure=dict(
            total_pairs=total,
            within_month_pairs=within,
            between_month_pairs=between,
            within_pair_share=weight,
            between_pair_share=1 - weight,
        ),
        aggregates=agg,
        gains=dict(
            eligible_pooled=pooled_gain,
            within_month_pair_weighted=within_gain,
            between_month=between_gain,
            within_weighted_contribution=within_contribution,
            between_weighted_contribution=between_contribution,
            within_contribution_share=within_contribution / pooled_gain
            if abs(pooled_gain) > 1e-12
            else None,
        ),
        per_month_gap_signs=dict(
            positive=sum(r["gap"] > 0 for r in rows),
            negative=sum(r["gap"] < 0 for r in rows),
            zero=sum(r["gap"] == 0 for r in rows),
        ),
    )
    return result


def histogram(values, convert=label):
    keys, counts = np.unique(values, return_counts=True)
    return {convert(k): int(n) for k, n in zip(keys, counts, strict=True)}


def entry_analysis(data, frozen, spec):
    ids, starts, counts = np.unique(data["facility"], return_index=True, return_counts=True)
    ends = starts + counts - 1
    if (
        not np.all(data["facility"][ends] == ids)
        or not np.all(data["month"][ends] - data["month"][starts] + 1 == counts)
        or not np.array_equal(np.lexsort((data["month"], data["facility"])), np.arange(len(data)))
    ):
        raise ValueError("Facility histories must be sorted, contiguous monthly histories")
    entry = data["month"][starts].astype(int)
    if len(entry) != frozen["cif"]["landmarks"] or len(entry) != frozen["cif"]["unique_facilities"]:
        raise ValueError("Landmark reconciliation failed")
    if (
        np.count_nonzero(entry // 12 == 2019)
        != frozen["stability"]["calendar"]["2019"]["M1"]["facilities"]
    ):
        raise ValueError("2019 entry count reconciliation failed")
    truncation = {}
    for h in spec["cif"]["horizons"]:
        count = int(np.count_nonzero(entry + h - 1 > ordinal("2026-02")))
        part = frozen["cif"]["horizons"][str(h)]
        if count != part["calendar_truncated_landmarks"] or len(entry) - count != part["landmarks"]:
            raise ValueError("Horizon truncation reconciliation failed")
        truncation[str(h)] = dict(
            count=count, latest_permitted_entry=label(ordinal("2026-02") - h + 1)
        )
    months = histogram(entry)
    years = histogram(entry // 12, convert=lambda y: str(int(y)))
    ordered = np.sort(entry)
    mode_count = max(months.values())
    return dict(
        status="EXECUTED",
        reconciliations_passed=True,
        landmarks=len(entry),
        entry_month_distribution=months,
        entry_year_distribution=years,
        earliest_entry=label(entry.min()),
        latest_entry=label(entry.max()),
        central_order_statistic_months=[
            label(ordered[(len(entry) - 1) // 2]),
            label(ordered[len(entry) // 2]),
        ],
        modal_months=[m for m, n in months.items() if n == mode_count],
        modal_month_share=mode_count / len(entry),
        distinct_months=len(months),
        per_horizon_truncation=truncation,
        path_24m_endpoint_distribution=histogram(entry + 23),
        retrospective_paths_are_not_prospective_forecasts=True,
    )


def run(root, out):
    output = out / "local_closure_output.json"
    if output.exists():
        raise ValueError("Refusing to overwrite closure output")
    reg_path = out / "local_closure_registration.json"
    if digest(reg_path) != (out / "local_closure_registration.sha256").read_text().strip():
        raise ValueError("Registration hash mismatch")
    reg = read(reg_path)
    if (
        reg["code_sha256"] != digest(Path(__file__))
        or reg["repository_commit"]
        != subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    ):
        raise ValueError("Code or repository differs from registered state")
    verify_hashes(root, reg["public_input_hashes"])
    verify_hashes(root, reg["private_input_hashes"])
    report, spec = read(root / REPORT), read(root / PROTOCOL)
    data, p1, p2 = [
        np.load(root / PRIVATE / n, mmap_mode="r", allow_pickle=False)
        for n in ("evaluation.npy", "M1_evaluation.npy", "M2_evaluation.npy")
    ]
    validate_arrays(data, p1, p2)
    seen = np.isin(data["vintage"], spec["splits"]["primary_vintages"])
    selected, a1, a2 = data[seen], p1[seen], p2[seen]
    expected = report["split_counts"]["evaluation_seen"]
    if (
        len(selected) != expected["intervals"]
        or len(np.unique(selected["facility"])) != expected["facilities"]
        or int((selected["event"] == 1).sum()) != expected["defaults"]
        or int((selected["event"] == 2).sum()) != expected["payoffs"]
    ):
        raise ValueError("Primary population reconciliation failed")
    result = dict(
        version=reg["version"],
        classification=reg["classification"],
        registration_sha256=digest(reg_path),
        inputs_verified=reg["private_input_hashes"],
        population=expected,
        SA01_month=month_analysis(selected, a1, a2, report),
        SA06_entry=entry_analysis(selected, report, spec),
        no_fit_or_prediction_regeneration=True,
        outputs_are_aggregates_only=True,
    )
    verify_hashes(root, reg["private_input_hashes"])
    verify_hashes(root, reg["public_input_hashes"])
    write_new(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "eligible_months": result["SA01_month"]["eligible_months"],
                "gains": result["SA01_month"].get("gains"),
                "entry_months": result["SA06_entry"]["entry_month_distribution"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if args.register:
        record = register(args.root.resolve(), args.output_dir.resolve())
        print(json.dumps({"registered": True, "code_sha256": record["code_sha256"]}))
    else:
        run(args.root.resolve(), args.output_dir.resolve())

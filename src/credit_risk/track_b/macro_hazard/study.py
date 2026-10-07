"""Count gates, registration before fitting, frozen ladder and one temporal session."""

import importlib.metadata
import time

import joblib
import numpy as np

from credit_risk.track_b.data.schemas import digest
from credit_risk.track_b.macro.information import feature_hash
from credit_risk.track_b.macro_support.eligibility import ordinal
from credit_risk.track_b.macro_support.study import immutable_json, read_json
from credit_risk.track_b.multivintage.study import lf_hash

from .cif import evaluate as cif_evaluate
from .data import load
from .ledger import Ledger
from .metrics import SCORES, calibration, paired, scores
from .models import artifact, fit, predict
from .protocol import PRIMARY, RATE, REDUCED, freeze


def counts(data):
    return dict(
        facilities=len(np.unique(data["facility"])),
        intervals=len(data),
        defaults=int(np.count_nonzero(data["event"] == 1)),
        payoffs=int(np.count_nonzero(data["event"] == 2)),
    )


def population(root, spec):
    arrays = load(root)
    development = arrays["development"][arrays["development"]["month"] >= ordinal("2010-09")]
    evaluation = arrays["evaluation"]
    seen = np.isin(evaluation["vintage"], spec["splits"]["primary_vintages"])
    old = read_json(root / "data/track_b/macro_support/counts.json")
    final = read_json(root / "docs/track_b/macro_support_validation_design.json")
    if counts(development) != old["validation_counts"]["PRIMARY"]["development"] or (
        counts(evaluation[seen]) != final["primary_temporal_seen_vintage_counts"]
    ):
        raise ValueError("STOP — Task10 split/event support materially inconsistent with Task9A")
    if set(np.unique(development["facility"])) & set(np.unique(evaluation["facility"])):
        raise ValueError("Facility role leakage")
    return arrays["development"], development, evaluation, seen


def registration(root, spec, evaluation, seen):
    private = root / "data/track_b/models/macro_hazard_v1"
    keys = read_json(private / "facility_keys.json")
    names = ["protocol.py", "data.py", "models.py", "ledger.py", "metrics.py", "cif.py", "study.py"]
    return dict(
        namespace="TASK10_MACRO_HAZARD",
        protocol_sha256=feature_hash(spec),
        eligibility_sha256_lf=spec["eligibility_sha256_lf"],
        risk_array_sha256=digest(private / "evaluation.npy"),
        evaluation_facility_ids=[keys[i] for i in np.unique(evaluation["facility"])],
        primary_facility_ids=[keys[i] for i in np.unique(evaluation["facility"][seen])],
        population_counts=counts(evaluation),
        primary_counts=counts(evaluation[seen]),
        code_sha256_lf={
            n: lf_hash(root / "src/credit_risk/track_b/macro_hazard" / n) for n in names
        },
        prior_support_outcomes_inspected=True,
        virgin_holdout=False,
    )


def train(root):
    spec = freeze(root)
    private = root / "data/track_b/models/macro_hazard_v1"
    reduced, development, evaluation, seen = population(root, spec)
    reg = registration(root, spec, evaluation, seen)
    ledger = Ledger(private / "task10_evaluation_ledger.json", reg)
    fitting = [
        ("M0", development, False, ()),
        ("M1", development, True, ()),
        ("M2", development, True, PRIMARY),
        ("RATE", development, True, RATE),
        ("REDUCED_M1", reduced, True, ()),
        ("REDUCED_M2", reduced, True, REDUCED),
    ]
    results, bindings = {}, {}
    for name, data, mortgage, macro in fitting:
        ledger.check("REGISTERED_BEFORE_FIT")
        print(
            f"Fitting {name}: {len(data):,} development intervals, frozen specification", flush=True
        )
        start = time.monotonic()
        bundle = fit(data, spec, mortgage, macro)
        p = predict(bundle, data)
        results[name] = dict(
            scores=scores(data, p),
            parameters=artifact(bundle),
            training_array_sha256=digest(private / "development.npy"),
            seconds=time.monotonic() - start,
        )
        joblib.dump(bundle, private / (name + ".joblib"))
        bindings[name] = digest(private / (name + ".joblib"))
        print(f"{name} converged: iterations {bundle[1].n_iter_.tolist()}", flush=True)
        immutable_json(private / (name + "_development.json"), results[name])
    lvo = {}
    for vintage in spec["sensitivities"]["leave_vintage_out"]:
        held = development["vintage"] == vintage
        target = development[held]
        support = counts(target)
        if support["defaults"] < 100 or support["payoffs"] < 500:
            lvo[str(vintage)] = dict(status="SPARSE_FOLD_SUPPRESSED", counts=support)
            continue
        reference = 2008 if vintage == 2006 else 2006
        lvo[str(vintage)] = dict(
            status="DEVELOPMENT_ONLY_TRANSPORT_SENSITIVITY",
            counts=support,
            reference=reference,
            heldout_cohort_effect="zero",
            models={},
        )
        for name, macro in [("M1", ()), ("M2", PRIMARY)]:
            ledger.check("REGISTERED_BEFORE_FIT")
            label = f"LVO_{vintage}_{name}"
            print(f"Fitting {label}; no temporal evaluation access", flush=True)
            bundle = fit(development[~held], spec, True, macro, reference=reference)
            p = predict(bundle, target)
            lvo[str(vintage)]["models"][name] = dict(
                scores=scores(target, p), parameters=artifact(bundle)
            )
            joblib.dump(bundle, private / (label + ".joblib"))
            bindings[label] = digest(private / (label + ".joblib"))
        immutable_json(private / (f"lvo_{vintage}.json"), lvo[str(vintage)])
    immutable_json(private / "development_results.json", results)
    immutable_json(private / "leave_vintage_out.json", lvo)
    versions = {
        n: importlib.metadata.version(n)
        for n in ["numpy", "scipy", "scikit-learn", "pandas", "joblib", "threadpoolctl"]
    }
    immutable_json(
        private / "model_manifest.json",
        dict(bindings=bindings, software=versions, protocol_sha256=feature_hash(spec)),
    )
    ledger.freeze_models(bindings)
    print("All development gates passed. Models frozen; temporal ledger unconsumed.", flush=True)


def diagnostics(data, p):
    return dict(
        scores=scores(data, p),
        calibration={
            cause: calibration((data["event"] == code).astype(int), p[:, code])
            for code, cause in [(1, "default"), (2, "payoff")]
        },
    )


def decision(primary, uncertainty, calendar):
    gain = uncertainty["intervals"]["joint_log_loss"]
    calendar_gain = calendar["intervals"]["joint_log_loss"]
    no_degradation = all(
        primary["M2"]["scores"][cause + "_brier"] - primary["M1"]["scores"][cause + "_brier"]
        <= 1e-4
        and primary["M2"]["calibration"][cause]["absolute_mean_rate_error"]
        - primary["M1"]["calibration"][cause]["absolute_mean_rate_error"]
        <= 1e-4
        for cause in ["default", "payoff"]
    )
    if (
        gain["upper"] is not None
        and calendar_gain["upper"] is not None
        and (gain["upper"] < 0 and calendar_gain["upper"] < 0 and no_degradation)
    ):
        return "MACRO INFORMATION IMPROVES TEMPORAL COMPETING-RISK PREDICTION"
    if any(uncertainty["intervals"][n]["delta"] < 0 for n in SCORES):
        return "MACRO INFORMATION PROVIDES LIMITED / MIXED INCREMENTAL VALUE"
    return "NO RELIABLE TEMPORAL MACRO INCREMENT DEMONSTRATED"


def evaluate(root):
    spec = freeze(root)
    private = root / "data/track_b/models/macro_hazard_v1"
    reduced, development, evaluation, seen = population(root, spec)
    ledger_data = read_json(private / "task10_evaluation_ledger.json")
    ledger = object.__new__(Ledger)
    ledger.path, ledger.binding, ledger.data = (
        private / "task10_evaluation_ledger.json",
        ledger_data["registration_sha256"],
        ledger_data,
    )
    reg = registration(root, spec, evaluation, seen)
    manifest = read_json(private / "model_manifest.json")
    for name, expected in manifest["bindings"].items():
        if digest(private / (name + ".joblib")) != expected:
            raise ValueError("Frozen model artifact changed")
    ledger.consume(reg)
    names = ["M0", "M1", "M2", "RATE", "REDUCED_M1", "REDUCED_M2"]
    models = {n: joblib.load(private / (n + ".joblib")) for n in names}
    predictions = {}
    for name in names:
        predictions[name] = predict(models[name], evaluation)
        np.save(private / (name + "_evaluation.npy"), predictions[name])
    primary = {n: diagnostics(evaluation[seen], p[seen]) for n, p in predictions.items()}
    print(
        "Prespecified temporal predictions generated; paired cluster uncertainty next", flush=True
    )
    paired_facility = paired(evaluation[seen], predictions["M1"][seen], predictions["M2"][seen])
    paired_calendar = paired(
        evaluation[seen],
        predictions["M1"][seen],
        predictions["M2"][seen],
        seed=61036,
        unit="calendar_year",
        ranking=False,
    )
    cells = {}
    for label, vector in [
        ("calendar", evaluation["month"] // 12),
        ("vintage", evaluation["vintage"]),
    ]:
        cells[label] = {}
        for value in np.unique(vector[seen]):
            mask = seen & (vector == value)
            cells[label][str(value)] = {
                n: scores(evaluation[mask], predictions[n][mask], suppress=True)
                for n in ["M1", "M2"]
            }
    pandemic = seen & np.isin(evaluation["month"] // 12, [2020, 2021])
    adjacent = seen & np.isin(evaluation["month"] // 12, [2019, 2022])
    period = {
        label: {n: diagnostics(evaluation[mask], predictions[n][mask]) for n in ["M1", "M2"]}
        for label, mask in [("pandemic_2020_2021", pandemic), ("adjacent_2019_2022", adjacent)]
    }
    unseen = {n: diagnostics(evaluation[~seen], predictions[n][~seen]) for n in ["M1", "M2"]}
    table = {
        r["reporting_month"]: r
        for r in read_json(
            root / "data/track_b/macro/processed/task9_api_v5/macro_month_table.json"
        )
    }
    cif = cif_evaluate(
        evaluation[seen], {n: models[n] for n in ["M0", "M1", "M2"]}, table, spec, private
    )
    gfc = {}
    for year in [2007, 2008, 2009]:
        mask = reduced["month"] // 12 == year
        gfc[str(year)] = {
            n: scores(reduced[mask], predict(models[n], reduced[mask]), suppress=True)
            for n in ["REDUCED_M1", "REDUCED_M2"]
        }
    prediction_hashes = {p.name: digest(p) for p in sorted(private.glob("*_evaluation.npy"))}
    prediction_hashes.update({p.name: digest(p) for p in sorted(private.glob("cif_*.npy"))})
    ledger.complete(prediction_hashes)
    result = dict(
        primary=primary,
        paired_facility=paired_facility,
        paired_calendar=paired_calendar,
        stability=cells,
        period_sensitivity=period,
        unseen_vintage=unseen,
        cif=cif,
        gfc_in_sample_descriptive=gfc,
        decision=decision(primary, paired_facility, paired_calendar),
        split_counts=dict(
            development=counts(development),
            reduced_development=counts(reduced),
            evaluation_seen=counts(evaluation[seen]),
            evaluation_unseen=counts(evaluation[~seen]),
        ),
        prediction_hashes=prediction_hashes,
        ledger_sha256=digest(ledger.path),
    )
    immutable_json(private / "temporal_results.json", result)
    print(result["decision"], flush=True)

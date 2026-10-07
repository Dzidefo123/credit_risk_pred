"""One frozen exploratory experiment; no claim of fresh validation or promotion."""

import time

import joblib
import numpy as np

from credit_risk.track_b.data.schemas import digest
from credit_risk.track_b.macro_diagnostics.distributions import shift
from credit_risk.track_b.macro_diagnostics.study import calendar_diagnostics
from credit_risk.track_b.macro_hazard.cif import landmarks
from credit_risk.track_b.macro_hazard.data import macro_join
from credit_risk.track_b.macro_hazard.metrics import calibration, paired, scores
from credit_risk.track_b.macro_hazard.models import BANDS, artifact, predict
from credit_risk.track_b.macro_hazard.protocol import freeze as old_spec
from credit_risk.track_b.macro_hazard.study import counts, population
from credit_risk.track_b.macro_support.eligibility import monthly_text, ordinal
from credit_risk.track_b.macro_support.study import immutable_json, read_json
from credit_risk.track_b.multivintage.study import lf_hash
from credit_risk.track_b.survival.math import aj, curves, horizon_metrics, support

from .audit import PRIVATE, verify
from .features import gap
from .models import CONTEXT, fit
from .protocol import BIN_EDGES, LABEL, MODELS, freeze


def code_hashes(root):
    return {p.name: lf_hash(p) for p in (root / "src/credit_risk/track_b/refinancing").glob("*.py")}


def binding(root, spec):
    return dict(
        prespecification_sha256=digest(
            root / "docs/track_b/refinancing_incentive_prespecification.json"
        ),
        code_sha256_lf=code_hashes(root),
        population_hashes=spec["population_hashes"],
        status="EXPLORATORY_ONLY",
        no_independent_ledger=True,
    )


def train(root):
    verify(root)
    spec = freeze(root)
    private = root / PRIVATE
    if any((private / (n + ".joblib")).exists() for n in MODELS):
        raise ValueError("Task12 fitting already started; no refitting/retuning")
    immutable_json(private / "prefit_registration.json", binding(root, spec))
    _, development, _, _ = population(root, old_spec(root))
    result = {}
    for name in MODELS:
        started = time.perf_counter()
        bundle = fit(development, name)
        path = private / (name + ".joblib")
        joblib.dump(bundle, path)
        evidence = artifact(bundle)
        immutable_json(private / (name + "_coefficients.json"), evidence)
        result[name] = dict(
            sha256=digest(path),
            iterations=bundle[1].n_iter_.tolist(),
            elapsed_seconds=time.perf_counter() - started,
            development=scores(development, predict(bundle, development)),
        )
        print("Task12 frozen development fit", name, result[name]["iterations"], flush=True)
    immutable_json(
        private / "models_manifest.json", dict(binding=binding(root, spec), models=result)
    )


def cells(data, predictions, mask):
    rows = data[mask]
    c = counts(rows)
    supported = c["facilities"] >= 100 and c["payoffs"] >= 20
    return dict(
        counts=c,
        status="SUPPORTED" if supported else "SPARSE_CELL_SUPPRESSED",
        observed_payoff=float((rows["event"] == 2).mean()) if supported else None,
        models={
            n: dict(
                scores=scores(rows, p[mask], suppress=True),
                mean_payoff=float(p[mask, 2].mean()) if supported else None,
            )
            for n, p in predictions.items()
        }
        if supported
        else {},
    )


def binned(data, predictions):
    index = np.searchsorted(BIN_EDGES, gap(data), side="right")
    return {str(i): cells(data, predictions, index == i) for i in range(5) if (index == i).any()}


def grouped(data, predictions):
    band = np.searchsorted([12, 24, 36, 60, 84, 120, 180, 240], data["duration"], side="left")
    year = data["month"] // 12
    masks = {"vintage:" + str(v): data["vintage"] == v for v in np.unique(data["vintage"])}
    masks.update({"duration:" + b: band == i for i, b in enumerate(BANDS)})
    masks.update({"year:" + str(y): year == y for y in np.unique(year)})
    masks.update(
        {"pandemic_2020_2021": (year >= 2020) & (year <= 2021), "post2022_2026": year >= 2022}
    )
    return {n: cells(data, predictions, m) for n, m in masks.items() if m.any()}


def forecast(bundle, entries, table, horizon):
    if np.any(entries["month"] + horizon - 1 > ordinal("2026-02")):
        raise ValueError("Future unobserved macro path")
    rows = entries.copy()
    hazard = []
    for step in range(horizon):
        rows["month"] = entries["month"] + step
        rows["duration"] = entries["duration"] + step
        needed = ["mortgage_30y_level", *CONTEXT] if bundle[0].context else ["mortgage_30y_level"]
        for month in np.unique(rows["month"]):
            rows["macro"][rows["month"] == month] = macro_join(table, monthly_text(month), needed)
        hazard.append(predict(bundle, rows))
    hazards = np.stack(hazard, axis=1)
    return curves(hazards[:, :, 1], hazards[:, :, 2])


def cif(root, data, models):
    entries, subjects = landmarks(data)
    table = {
        r["reporting_month"]: r
        for r in read_json(
            root / "data/track_b/macro/processed/task9_api_v5/macro_month_table.json"
        )
    }
    result = {}
    for h in [12, 24, 36, 60]:
        mask = entries["month"] + h - 1 <= ordinal("2026-02")
        selected = entries[mask]
        observed = subjects.loc[mask].copy().reset_index(drop=True)
        estimate = aj(observed, h)
        status = support(estimate[-1], h, 88, minimum_risk=200)
        part = dict(
            status=status,
            facilities=len(selected),
            calendar_excluded=int((~mask).sum()),
            observed=estimate[-1],
            models={},
        )
        swapped = observed.copy()
        swapped["event_code"] = observed.event_code.map({0: 0, 1: 2, 2: 1})
        payoff_aj = aj(swapped, h)
        if status == "SUPPORTED":
            for n, bundle in models.items():
                prediction = forecast(bundle, selected, table, h)
                np.save(root / PRIVATE / f"cif_{n}_{h}.npy", prediction)
                part["models"][n] = dict(
                    default=horizon_metrics(
                        observed, prediction[:, -1, 1], h, minimum_events=20, estimator=estimate
                    ),
                    payoff=horizon_metrics(
                        swapped, prediction[:, -1, 2], h, minimum_events=20, estimator=payoff_aj
                    ),
                    max_conservation_residual=float(np.max(abs(prediction.sum(axis=2) - 1))),
                    mean_curves=prediction.mean(axis=0).tolist(),
                )
        result[str(h)] = part
        print("Task12 exploratory CIF", h, status, flush=True)
    return result


def decision(results):
    metrics = results["metrics"]
    p1, p2 = metrics["P1"], metrics["P2"]
    comparisons = results["paired"]
    cal = results["calibration"]
    supported = [v for v in results["cif"].values() if v["status"] == "SUPPORTED"]
    gates = dict(
        joint_score=all(
            comparisons[u]["intervals"]["joint_log_loss"]["upper"] is not None
            and comparisons[u]["intervals"]["joint_log_loss"]["upper"] < 0
            for u in ["facility", "calendar_year"]
        ),
        payoff_brier=comparisons["facility"]["intervals"]["payoff_brier"]["upper"] <= 0.0001,
        calibration=cal["P1"]["payoff"]["absolute_mean_rate_error"]
        - cal["P2"]["payoff"]["absolute_mean_rate_error"]
        >= 0.0001,
        payoff_cif=bool(supported)
        and all(
            abs(v["models"]["P2"]["payoff"]["calibration_difference"])
            <= abs(v["models"]["P1"]["payoff"]["calibration_difference"])
            for v in supported
        ),
        default=p2["default_brier"] - p1["default_brier"] <= 0.0001
        and bool(supported)
        and all(
            abs(v["models"]["P2"]["default"]["calibration_difference"])
            - abs(v["models"]["P1"]["default"]["calibration_difference"])
            <= 0.001
            for v in supported
        ),
        stability=all(
            v["models"]["P2"]["scores"]["payoff_brier"]
            - v["models"]["P1"]["scores"]["payoff_brier"]
            <= 0.0001
            for v in results["groups"].values()
            if v["status"] == "SUPPORTED"
        ),
    )
    if all(gates.values()):
        verdict = "REFINANCING INCENTIVE HYPOTHESIS SUPPORTED EXPLORATORILY"
    elif p2["joint_log_loss"] < p1["joint_log_loss"] or p2["payoff_brier"] < p1["payoff_brier"]:
        verdict = "REFINANCING INCENTIVE HYPOTHESIS MIXED EXPLORATORILY"
    else:
        verdict = "REFINANCING INCENTIVE HYPOTHESIS NOT SUPPORTED EXPLORATORILY"
    return dict(verdict=verdict, gates=gates, confirmatory=False)


def explore(root):
    verify(root)
    spec = freeze(root)
    private = root / PRIVATE
    if (private / "results.json").exists() or any(private.glob("prediction_*.npy")):
        raise ValueError("Exploratory scoring already started; no repeats/selection")
    manifest = read_json(private / "models_manifest.json")
    if manifest["binding"] != binding(root, spec):
        raise ValueError("Prespecification/code/data changed after fitting")
    models = {}
    for n in MODELS:
        path = private / (n + ".joblib")
        if digest(path) != manifest["models"][n]["sha256"]:
            raise ValueError("Frozen Task12 model changed")
        models[n] = joblib.load(path)
    _, development, evaluation, seen = population(root, old_spec(root))
    evaluation = evaluation[seen]
    predictions = {n: predict(bundle, evaluation) for n, bundle in models.items()}
    for n, p in predictions.items():
        np.save(private / ("prediction_" + n + ".npy"), p)
    development_predictions = {n: predict(b, development) for n, b in models.items()}
    result = dict(
        label=LABEL,
        validation_status="EXPLORATORY_ONLY",
        virgin_holdout=False,
        counts=dict(development=counts(development), exploratory=counts(evaluation)),
        metrics={n: scores(evaluation, p) for n, p in predictions.items()},
        development=manifest["models"],
        calibration={
            n: {
                c: calibration(evaluation["event"] == code, p[:, code])
                for code, c in [(1, "default"), (2, "payoff")]
            }
            for n, p in predictions.items()
        },
        development_refi_bins=binned(development, development_predictions),
        exploratory_refi_bins=binned(evaluation, predictions),
        gap_shift=shift(gap(development), gap(evaluation)),
        calendar=calendar_diagnostics(evaluation, predictions),
        calendar_refi_bins={
            str(y): binned(
                evaluation[evaluation["month"] // 12 == y],
                {n: p[evaluation["month"] // 12 == y] for n, p in predictions.items()},
            )
            for y in np.unique(evaluation["month"] // 12)
        },
        groups=grouped(evaluation, predictions),
    )
    result["paired"] = {}
    for unit, seed in [("facility", 61201), ("calendar_year", 61202)]:
        interval = paired(
            evaluation,
            predictions["P1"],
            predictions["P2"],
            seed=seed,
            draws=1000,
            unit=unit,
            ranking=False,
        )
        interval["orientation"] = "P2 minus P1; negative proper score is better; EXPLORATORY_ONLY"
        result["paired"][unit] = interval
        print("Task12 paired exploratory resampling", unit, flush=True)
    result["cif"] = cif(root, evaluation, models)
    result["decision"] = decision(result)
    result["prediction_hashes"] = {p.name: digest(p) for p in private.glob("prediction_*.npy")}
    result["next_task"] = (
        "External Fannie Mae replication feasibility and harmonization con"
        "tract; acquisition requires separate authorization"
    )
    immutable_json(private / "results.json", result)
    print(result["decision"], flush=True)

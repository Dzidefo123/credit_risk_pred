"""One-shot Task 5 development, specification freeze, then temporal predictions."""

import hashlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import DMatrix

from credit_risk.track_b.data.annual import peak_memory_bytes
from credit_risk.track_b.data.schemas import digest, load_protocol
from credit_risk.track_b.pd.baseline import cohort, cohort_hash, counts, metrics, split

from .core import (
    CATEGORICAL,
    NUMERIC,
    Ledger,
    apply_calibrator,
    calibrate,
    clustered,
    diagnostic,
    fit_model,
    frame,
    hazard_forecast,
    hazard_periods,
    internal_split,
)


def finite_stat(value):
    return float(value) if np.isfinite(value) else None


def lf_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def score(model, f, registry, static=False):
    return model.predict_proba(frame(f, registry, static))[:, 1]


def psi(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return None
    cuts = np.unique(np.quantile(a, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    edges = np.r_[-np.inf, cuts, np.inf]
    x = np.histogram(a, edges)[0] / len(a)
    y = np.histogram(b, edges)[0] / len(b)
    x = np.maximum(x, 1e-6)
    y = np.maximum(y, 1e-6)
    return float(np.sum((y - x) * np.log(y / x)))


def groups(f, p, mask):
    sub = f.loc[mask]
    if not len(sub):
        return dict(status="empty")
    if counts(sub)["default_loans"] < 20:
        return dict(
            effective_sample=counts(sub),
            status="SUPPRESSED — INSUFFICIENT EVENT LOANS",
            observed_rate=float(sub.binary_default_12m.mean()),
            mean_probability=float(np.asarray(p)[mask].mean()),
        )
    return diagnostic(sub, np.asarray(p)[mask])


def run(root, archive, progress=print):
    root = Path(root).resolve()
    started = time.monotonic()
    private = root / "data/track_b/models/expanded_pd_v1"
    if (private / "evaluation_ledger.json").exists():
        raise ValueError("Evaluation ledger already exists; silent reevaluation prohibited")
    private.mkdir(parents=True, exist_ok=True)
    design = json.loads((root / "docs/track_b/expanded_pd_design.json").read_text(encoding="utf-8"))
    registry = json.loads(
        (root / "docs/track_b/expanded_pd_feature_registry.json").read_text(encoding="utf-8")
    )
    historical = json.loads(
        (root / "reports/track_b/sample_expansion_feasibility.json").read_text(encoding="utf-8")
    )
    source = historical["sample"]["source_sha256"]
    if digest(archive) != source:
        raise ValueError("Source hash mismatch")
    panel_path = root / "data/track_b/processed/expansion_v1/panel.csv"
    if digest(panel_path) != design["panel_sha256"]:
        raise ValueError("Expanded panel mismatch")
    ids = (
        (root / "data/track_b/manifests/expansion_v1/selected_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    if (
        len(ids) != 20000
        or hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest() != design["sample_sha256"]
    ):
        raise ValueError("Frozen expanded sample mismatch")
    old_ids = (
        (root / "data/track_b/manifests/annual_2010_v1/selected_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    if not set(old_ids) <= set(ids):
        raise ValueError("Original nesting changed")
    _, protocol_hash = load_protocol(root)
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], text=True).split(
        "\0"
    )
    preserved = {name: digest(root / name) for name in tracked if name}
    for path in (root / "data/track_b/manifests").glob("**/*.json"):
        preserved[str(path.relative_to(root))] = digest(path)
    preserved[str(panel_path.relative_to(root))] = digest(panel_path)
    cols = list(
        dict.fromkeys(
            [
                *registry["selected"],
                "loan_id",
                "t0",
                "eligible",
                "outcome_status",
                "binary_default_12m",
                "event_offset",
                "observed_followup_months",
            ]
        )
    )
    dtypes = {k: "string" for k in ["loan_id", "t0", "delinquency_state", *CATEGORICAL]}
    panel = pd.read_csv(panel_path, usecols=cols, dtype=dtypes, low_memory=False)
    if set(panel.loan_id) != set(ids):
        raise ValueError("Panel identity mismatch")
    primary = cohort(panel)
    dev, ev = split(primary)
    expected = {
        "primary": dict(landmarks=1241045, loans=19590, positive_landmarks=7153, default_loans=618),
        "development": dict(
            landmarks=478796, loans=13801, positive_landmarks=2560, default_loans=246
        ),
        "evaluation": dict(landmarks=128812, loans=2423, positive_landmarks=1072, default_loans=95),
    }
    reproduced = {
        k: counts(f) for k, f in [("primary", primary), ("development", dev), ("evaluation", ev)]
    }
    if reproduced != expected:
        raise ValueError("STOP — cohort counts differ")
    hashes = {
        k: cohort_hash(f)
        for k, f in [("primary", primary), ("development", dev), ("evaluation", ev)]
    }
    progress("Frozen cohort counts and identities reproduced exactly")
    fit, cal, selection = internal_split(dev)
    parts = {
        k: counts(f) for k, f in [("fit", fit), ("calibration", cal), ("selection", selection)]
    }
    if any(v["default_loans"] < 20 for v in parts.values()):
        raise ValueError("Insufficient fixed internal event support; no reshuffle")
    rawdev, rawev = split(panel)
    riskdev = hazard_periods(rawdev)
    riskev = hazard_periods(rawev)
    hazard_fit, _, hazard_selection = internal_split(riskdev)
    audit = dict(
        cohorts=reproduced,
        hashes=hashes,
        internal_partitions=parts,
        hazard_fit=counts(hazard_fit),
        hazard_selection=counts(hazard_selection),
        count_audit_only=True,
        evaluation_predictions_accessed=False,
    )
    write(private / "prefit_cohort_audit.json", audit)
    models = {}
    calibrators = {}
    selection_evidence = {}
    trials = []
    models["logistic"] = fit_model("logistic", fit, registry, design["logistic"])
    progress("Interpretable logistic benchmark fitted on development fit groups")
    # Hazard benchmark is validated before opening the controlled challenger search.
    models["hazard"] = fit_model("logistic", hazard_fit, registry, design["logistic"])
    selection_evidence["hazard_monthly"] = dict(
        effective_sample=counts(hazard_selection),
        metrics=metrics(
            hazard_selection.binary_default_12m, score(models["hazard"], hazard_selection, registry)
        ),
    )
    progress("One-event-per-facility monthly hazard benchmark fitted")
    candidates = []
    for config in design["xgboost_candidates"]:
        parameters = {**design["xgboost_fixed"], **config}
        model = fit_model("xgboost", fit, registry, parameters)
        m = metrics(selection.binary_default_12m, score(model, selection, registry))
        trials.append(dict(configuration=parameters, effective_sample=counts(selection), metrics=m))
        candidates.append(model)
        progress("Development XGBoost trial complete: " + str(config))
    chosen = (
        1
        if trials[1]["metrics"]["log_loss"] <= 0.99 * trials[0]["metrics"]["log_loss"]
        and trials[1]["metrics"]["brier"] <= trials[0]["metrics"]["brier"]
        else 0
    )
    models["xgboost"] = candidates[chosen]
    selected_xgb = trials[chosen]["configuration"]
    del candidates
    for name in ["logistic", "xgboost"]:
        calibrators[name], selection_evidence[name] = calibrate(
            models[name], cal, selection, registry
        )
        selection_evidence[name]["fit_sample"] = counts(fit)
        selection_evidence[name]["calibration_sample"] = counts(cal)
    models["static_logistic"] = fit_model(
        "logistic", fit, registry, design["logistic"], static=True
    )
    models["static_xgboost"] = fit_model(
        "xgboost",
        fit,
        registry,
        {**design["xgboost_fixed"], **design["xgboost_candidates"][0]},
        static=True,
    )
    for name in ["static_logistic", "static_xgboost"]:
        selection_evidence[name] = dict(
            effective_sample=counts(selection),
            decision="RAW RETAINED",
            metrics=metrics(
                selection.binary_default_12m, score(models[name], selection, registry, static=True)
            ),
        )
    artifacts = {}
    for name, model in models.items():
        path = private / (name + ".joblib")
        joblib.dump(dict(model=model, calibrator=calibrators.get(name)), path)
        artifacts[name] = dict(
            path=str(path.relative_to(root)),
            sha256=digest(path),
            model_type=name,
            seed=design["seed"],
            development_cohort_sha256=hashes["development"],
            fit_cohort_sha256=cohort_hash(hazard_fit if name == "hazard" else fit),
            feature_registry_sha256_lf=lf_hash(
                root / "docs/track_b/expanded_pd_feature_registry.json"
            ),
        )
    specification = dict(
        null_prevalence=float(dev.binary_default_12m.mean()),
        preprocessing=(
            "Development-fit-only median + missing indicators; numeric scaling; "
            "categorical modal imputation and first-category reference OHE"
        ),
        design=design,
        design_sha256_lf=lf_hash(root / "docs/track_b/expanded_pd_design.json"),
        registry_sha256_lf=lf_hash(root / "docs/track_b/expanded_pd_feature_registry.json"),
        artifacts=artifacts,
        xgboost_configuration=selected_xgb,
        calibration_decisions={k: selection_evidence[k]["decision"] for k in calibrators},
        source_code_sha256_lf={
            str(p.relative_to(root)): lf_hash(p) for p in Path(__file__).parent.glob("*.py")
        },
    )
    write(private / "frozen_model_specifications.json", specification)
    ledger = Ledger(private)
    ledger.freeze(specification)
    for artifact in artifacts.values():
        if digest(root / artifact["path"]) != artifact["sha256"]:
            raise ValueError("Frozen model artifact changed")
    progress(
        "All model, calibration and sensitivity specifications frozen; opening "
        "Task5 evaluation once"
    )
    ledger.open()
    predictions = {"null": np.full(len(ev), specification["null_prevalence"])}
    for name in ["logistic", "xgboost"]:
        raw = score(models[name], ev, registry)
        predictions[name + "_raw"] = raw
        predictions[name] = (
            apply_calibrator(calibrators[name], raw)
            if selection_evidence[name]["decision"] == "CALIBRATION RETAINED"
            else raw
        )
    for name in ["static_logistic", "static_xgboost"]:
        predictions[name] = score(models[name], ev, registry, static=True)
    net_pd = hazard_forecast(models["hazard"], ev, registry)
    monthly = score(models["hazard"], riskev, registry)
    np.savez_compressed(
        private / "temporal_predictions.npz",
        **predictions,
        hazard_net_pd=net_pd,
        hazard_monthly=monthly,
    )
    evaluated = {k: diagnostic(ev, p) for k, p in predictions.items()}
    progress("Frozen temporal predictions generated and persisted; computing clustered diagnostics")
    uncertainty = (
        clustered(ev, predictions, **design["bootstrap_as_arguments"])
        if "bootstrap_as_arguments" in design
        else clustered(
            ev, predictions, draws=design["bootstrap"]["draws"], seed=design["bootstrap"]["seed"]
        )
    )
    hazard_uncertainty = clustered(
        riskev,
        {"hazard_monthly": monthly},
        draws=design["bootstrap"]["draws"],
        seed=design["bootstrap"]["seed"],
    )
    calendar = {}
    segments = {}
    for name in ["logistic", "xgboost"]:
        p = predictions[name]
        calendar[name] = {}
        for label, low, high in [
            ("2016-2018", "2016-01", "2018-12"),
            ("2019-2021", "2019-01", "2021-12"),
            ("2022-2026", "2022-01", "2026-12"),
        ]:
            calendar[name][label] = groups(ev, p, ev.t0.between(low, high).to_numpy())
        segments[name] = {}
        for feature, edges in [
            ("delinquency_state", [-1, 0.5, 2]),
            ("orig_ltv", [-np.inf, 80, np.inf]),
            ("orig_credit_score", [-np.inf, 720, 780, np.inf]),
            ("loan_age", [-np.inf, 120, np.inf]),
        ]:
            bins = pd.cut(pd.to_numeric(ev[feature]), edges)
            for level in bins.cat.categories:
                segments[name][feature + ":" + str(level)] = groups(
                    ev, p, bins.eq(level).to_numpy()
                )
    stability = dict(
        feature_psi={
            n: psi(pd.to_numeric(dev[n], errors="coerce"), pd.to_numeric(ev[n], errors="coerce"))
            for n in NUMERIC
        },
        missingness={
            n: dict(development=float(dev[n].isna().mean()), evaluation=float(ev[n].isna().mean()))
            for n in registry["selected"]
        },
        event_rate=dict(
            development=float(dev.binary_default_12m.mean()),
            evaluation=float(ev.binary_default_12m.mean()),
        ),
        prediction_psi={
            n: psi(score(models[n], selection, registry), predictions[n])
            for n in ["logistic", "xgboost"]
        },
        psi_interpretation="Descriptive development-bin statistic; no universal threshold",
    )
    coef = models["logistic"][-1].coef_[0]
    names = models["logistic"][0].get_feature_names_out().tolist()
    coefficients = [
        dict(
            feature=n,
            coefficient=float(c),
            odds_ratio=float(np.exp(c)),
            direction="positive" if c > 0 else "negative",
            scale="one development SD for numeric features; category vs reference for one-hot",
        )
        for n, c in zip(names, coef, strict=True)
    ]
    refs = {
        n: str(categories[0])
        for n, categories in zip(
            CATEGORICAL,
            models["logistic"][0].named_transformers_["categorical"][-1].categories_,
            strict=True,
        )
    }
    # Bounded deterministic facility sample, one earliest evaluation landmark each.
    explanation = ev.sort_values(["loan_id", "t0"]).drop_duplicates("loan_id").copy()
    explanation["_rank"] = explanation.loan_id.map(
        lambda i: hashlib.sha256(("task5-shap:" + i).encode()).hexdigest()
    )
    explanation = explanation.sort_values("_rank").head(1000)
    transformed = models["xgboost"][0].transform(frame(explanation, registry))
    shap = models["xgboost"][-1].get_booster().predict(DMatrix(transformed), pred_contribs=True)
    feature_names = models["xgboost"][0].get_feature_names_out().tolist()
    values = shap[:, :-1]
    explanations = dict(
        effective_sample=counts(explanation),
        unit="one earliest evaluation landmark per deterministic selected facility",
        scale=(
            "raw XGBoost log odds; calibrated probabilities have a further sigmoid "
            "mapping if retained"
        ),
        global_mean_abs=dict(
            zip(feature_names, np.mean(np.abs(values), axis=0).tolist(), strict=True)
        ),
        half_sample_rank_spearman=finite_stat(
            spearmanr(
                np.mean(np.abs(values[::2]), axis=0), np.mean(np.abs(values[1::2]), axis=0)
            ).statistic
        ),
        causal=False,
    )
    top = np.argsort(-np.mean(np.abs(values), axis=0))[:4]
    direction = []
    for idx in top:
        x = transformed[:, idx]
        bins = pd.qcut(x, 4, duplicates="drop")
        for level in bins.categories:
            mask = bins == level
            if mask.any():
                direction.append(
                    dict(
                        feature=feature_names[idx],
                        bin=str(level),
                        rows=int(mask.sum()),
                        mean_standardized_feature=float(x[mask].mean()),
                        mean_shap_logodds=float(values[mask, idx].mean()),
                    )
                )
    explanations["direction_bins"] = direction
    logp = predictions["logistic"]
    xgbp = predictions["xgboost"]
    topn = max(1, len(ev) // 20)
    a = set(np.argsort(-logp, kind="stable")[:topn])
    b = set(np.argsort(-xgbp, kind="stable")[:topn])
    disagreement = np.argsort(-np.abs(logp - xgbp), kind="stable")[: max(1, len(ev) // 100)]
    agreement = dict(
        rank_spearman=finite_stat(spearmanr(logp, xgbp).statistic),
        top_5_percent_overlap=len(a & b) / topn,
        disagreement_top_1_percent_effective_sample=counts(ev.iloc[disagreement]),
        disagreement_observed_rate=float(ev.iloc[disagreement].binary_default_12m.mean()),
        mean_logistic=float(logp[disagreement].mean()),
        mean_xgboost=float(xgbp[disagreement].mean()),
        ids_published=False,
    )
    paired = uncertainty["paired"]["xgboost-minus-logistic"]
    lm, xm = evaluated["logistic"]["metrics"], evaluated["xgboost"]["metrics"]
    stable = all(
        not ("metrics" in calendar["logistic"][k] and "metrics" in calendar["xgboost"][k])
        or calendar["xgboost"][k]["metrics"]["log_loss"]
        <= 1.1 * calendar["logistic"][k]["metrics"]["log_loss"]
        for k in calendar["logistic"]
    )
    promote = (
        paired["brier"]["upper"] < 0
        and paired["log_loss"]["upper"] < 0
        and xm["brier"] <= 0.98 * lm["brier"]
        and xm["log_loss"] <= 0.98 * lm["log_loss"]
        and xm["roc_auc"] >= lm["roc_auc"] - 0.01
        and xm["average_precision"] >= lm["average_precision"] - 0.01
        and stable
    )
    champion = "xgboost" if promote else "logistic"
    cm = evaluated[champion]["metrics"]
    null = evaluated["null"]["metrics"]
    decision = (
        "TEMPORAL VALIDATION SUPPORTS RESEARCH PD WITH MATERIAL LIMITATIONS"
        if cm["roc_auc"] > 0.6
        and cm["average_precision"] > null["average_precision"]
        and cm["brier"] < null["brier"]
        and cm["log_loss"] < null["log_loss"]
        else "NO MATERIAL OUT-OF-TIME PREDICTIVE VALUE"
    )
    ledger.complete(
        [
            "ROC-AUC",
            "Gini",
            "Average Precision",
            "secondary trapezoidal PR-AUC",
            "Brier",
            "log loss",
            "calibration diagnostics",
            "reliability",
            "paired cluster intervals",
            "hazard monthly diagnostics",
            "static sensitivity",
            "calendar/segment diagnostics",
            "TreeSHAP",
        ]
    )
    if any(digest(root / name) != expected for name, expected in preserved.items()):
        raise ValueError("Historical evidence changed")
    if digest(archive) != source:
        raise ValueError("Source changed during study")
    result = dict(
        schema_version="1.0",
        decision=decision,
        champion=champion,
        source_sha256=source,
        sample_sha256=design["sample_sha256"],
        panel_sha256=design["panel_sha256"],
        protocol_sha256_lf=protocol_hash,
        cohort_reproduction=audit,
        model_specifications=specification,
        development_selection=selection_evidence,
        xgboost_trials=trials,
        evaluation_ledger=ledger.data,
        temporal_results=evaluated,
        clustered_uncertainty=uncertainty,
        calendar=calendar,
        segments=segments,
        stability=stability,
        logistic_coefficients=coefficients,
        categorical_references=refs,
        xgboost_explanations=explanations,
        model_agreement=agreement,
        hazard=dict(
            estimand=(
                "monthly default hazard with payoff censored; twelve-month net-default "
                "projection not primary CIF"
            ),
            development_selection=selection_evidence["hazard_monthly"],
            temporal_monthly=diagnostic(riskev, monthly),
            clustered_uncertainty=hazard_uncertainty,
            net_12m=dict(
                effective_sample=counts(ev),
                mean=float(net_pd.mean()),
                minimum=float(net_pd.min()),
                maximum=float(net_pd.max()),
            ),
            events_once_per_facility=True,
        ),
        preservation=dict(
            previous_file_sha256=preserved,
            unchanged=True,
            source_unchanged=True,
            sample_unchanged=True,
        ),
        runtime_seconds=time.monotonic() - started,
        resources={
            "peak_working_set_bytes": peak_memory_bytes(),
            "source_performance_rescanned": False,
        },
        package_versions={
            n: importlib.metadata.version(n)
            for n in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "credit-risk-lab"]
        },
        limitations=[
            (
                "Historical operational knowledge time UNVERIFIED; nominal-time "
                "retrospective research"
            ),
            (
                "2010-vintage facilities only; ageing/survival selection and calendar "
                "drift confounded"
            ),
            (
                "Unknown borrower identity; loan clusters are not proven independent "
                "borrower clusters"
            ),
            "Fixed-fit bootstrap excludes development model uncertainty",
            (
                "Tasks3/4 outcomes and nested Task3 scores previously seen; Task5 "
                "specifications frozen without new predictive evaluation"
            ),
            (
                "Internal development partitions grouped but not time-held-out; "
                "calibration from development cannot guarantee temporal calibration"
            ),
            (
                "Hazard net-risk projection with frozen covariates is not payoff-adjusted "
                "primary cumulative incidence"
            ),
            "No regulatory/IFRS9/IRB/production/fairness/causal/external-validity claim",
        ],
        next_task=(
            "Formal survival and competing-risk modeling of default versus payoff with "
            "the frozen cohort"
        ),
    )
    write(root / "reports/track_b/expanded_pd_validation.json", result)
    from .reporting import render

    render(root, result)
    progress(decision + "; champion=" + champion)
    return result

"""Protocol-first Task6 risk audit, fixed development, one temporal consumption."""

import csv
import hashlib
import importlib.metadata
import json
import sqlite3
import subprocess
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from credit_risk.track_b.data.annual import peak_memory_bytes
from credit_risk.track_b.data.schemas import digest, load_protocol
from credit_risk.track_b.pd.baseline import cohort, cohort_hash, split

from .ledger import Ledger
from .math import aj, horizon_metrics, support
from .models import DYNAMIC, STATIC, coefficients, fit, forecast, likelihood, probabilities
from .risk import (
    effective,
    fingerprint,
    first_endpoint,
    monthly_text,
    raw_history,
    trajectory,
    transition_table,
)


def lf_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepared(root, protocol, registry, private, policy, progress):
    feature_names = list(dict.fromkeys([*STATIC, *DYNAMIC]))
    cols = feature_names + ["loan_id", "t0", "eligible", "outcome_status", "binary_default_12m"]
    dtype = {
        k: "string"
        for k in ["loan_id", "t0", "delinquency_state", "loan_purpose", "occupancy_status"]
    }
    panel = pd.read_csv(
        root / "data/track_b/processed/expansion_v1/panel.csv",
        usecols=cols,
        dtype=dtype,
        low_memory=False,
    )
    old = cohort(panel)
    if cohort_hash(old) != protocol["task5_primary_cohort_sha256"]:
        raise ValueError("Task5 governing cohort changed")
    del old
    features = feature_names
    fields = features + ["loan_id", "t0", "target_month", "duration", "event_code", "next_state"]
    streams = {
        k: (private / (k + "_risk.csv")).open("w", encoding="utf-8", newline="")
        for k in ["development", "evaluation"]
    }
    writers = {
        k: csv.DictWriter(s, fieldnames=fields, lineterminator="\n") for k, s in streams.items()
    }
    for writer in writers.values():
        writer.writeheader()
    subjects = {k: [] for k in ["global", "development", "evaluation"]}
    flow = {k: Counter() for k in subjects}
    raw_events = Counter()
    calendar = {}
    global_interval_count = 0
    cache = root / "data/track_b/processed/expansion_v1/retained_performance.sqlite"
    with sqlite3.connect(cache.resolve().as_uri() + "?mode=ro", uri=True) as db:
        for number, (loan, g) in enumerate(panel.groupby("loan_id", sort=True), 1):
            history = raw_history(
                (
                    raw
                    for (raw,) in db.execute(
                        "SELECT raw FROM performance WHERE loan=? ORDER BY rowid", (str(loan),)
                    )
                ),
                policy,
            )
            endpoint = first_endpoint(history)
            raw_events[endpoint] += 1
            month = next(
                (
                    t
                    for t in sorted(history)
                    if history[t]["category"]
                    in ["default", "payoff", "administrative", "ambiguous"]
                ),
                max(history),
            )
            year = monthly_text(month)[:4]
            calendar.setdefault(year, Counter())[endpoint] += 1
            subject, rows, status = trajectory(g, history, features)
            flow["global"][status] += 1
            if status == "included":
                subjects["global"].append(subject)
                global_interval_count += len(rows)
            assigned = (
                int(hashlib.sha256(("track-b-pd-v1:" + str(loan)).encode()).hexdigest(), 16) % 10
                < 7
            )
            group = "development" if assigned else "evaluation"
            start = None if assigned else "2016-01"
            end = "2014-12" if assigned else None
            subject, rows, status = trajectory(g, history, features, start, end)
            flow[group][status] += 1
            if status == "included":
                subjects[group].append(subject)
                writers[group].writerows(rows)
            if number % 1000 == 0:
                progress(f"Task6 risk construction: {number:,}/20,000 facilities")
    for stream in streams.values():
        stream.close()
    expected = {
        "default": 623,
        "payoff": 18147,
        "administrative": 29,
        "ambiguous": 33,
        "active_or_unknown": 1168,
    }
    if dict(raw_events) != expected:
        raise ValueError("STOP — raw first-event audit differs from Task4")
    frames = {k: pd.DataFrame(v) for k, v in subjects.items()}
    if set(frames["development"].loan_id) & set(frames["evaluation"].loan_id):
        raise ValueError("Facility leakage")
    risk = {
        k: pd.read_csv(
            private / (k + "_risk.csv"),
            dtype={
                n: "string"
                for n in [
                    "loan_id",
                    "t0",
                    "target_month",
                    "delinquency_state",
                    "loan_purpose",
                    "occupancy_status",
                    "next_state",
                ]
            },
            low_memory=False,
        )
        for k in ["development", "evaluation"]
    }
    for _k, f in risk.items():
        if (
            f.duplicated(["loan_id", "target_month"]).any()
            or f.groupby("loan_id").event_code.apply(lambda v: (v > 0).sum()).gt(1).any()
        ):
            raise ValueError("Invalid first-event risk intervals")
    if (
        risk["development"].target_month.gt("2014-12").any()
        or risk["evaluation"].t0.lt("2016-01").any()
    ):
        raise ValueError("Calendar/purge violation")
    audit = dict(
        total_selected_facilities=20000,
        raw_first_endpoints=expected,
        raw_calendar={k: dict(v) for k, v in calendar.items()},
        flow={k: dict(v) for k, v in flow.items()},
        global_cohort=effective(frames["global"]),
        global_risk_intervals=global_interval_count,
        partitions={k: effective(frames[k], risk[k]) for k in risk},
        censor_reasons={
            k: dict(Counter(frames[k].loc[frames[k].event_code.eq(0), "exit_reason"]))
            for k in frames
        },
        delayed_entry_age={
            k: dict(
                minimum=float(f.entry_mortgage_age.min()),
                median=float(f.entry_mortgage_age.median()),
                maximum=float(f.entry_mortgage_age.max()),
            )
            for k, f in frames.items()
        },
        survival_cohort_sha256={
            k: fingerprint(
                f,
                ["loan_id", "entry_month", "entry_time", "exit_time", "event_code", "exit_reason"],
            )
            for k, f in frames.items()
        },
        risk_set_sha256={k: digest(private / (k + "_risk.csv")) for k in risk},
        task5_primary_cohort_sha256=protocol["task5_primary_cohort_sha256"],
    )
    write(private / "risk_audit.json", audit)
    for k, f in frames.items():
        f.to_csv(private / (k + "_subjects.csv"), index=False, lineterminator="\n")
    return frames, risk, audit


def interval_bootstrap(risk, predictions, draws, seed):
    ids, index = np.unique(risk.loan_id.astype(str), return_inverse=True)
    rng = np.random.default_rng(seed)
    values = {name: {k: [] for k in likelihood(risk, p)} for name, p in predictions.items()}
    for _ in range(draws):
        w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))[index]
        for name, p in predictions.items():
            for key, v in likelihood(risk, p, w).items():
                values[name][key].append(v)
    return dict(
        unit="facility, all monthly intervals retained",
        draws=draws,
        seed=seed,
        valid_draws=draws,
        failed_draws=0,
        intervals={
            name: {
                k: dict(lower=float(np.quantile(v, 0.025)), upper=float(np.quantile(v, 0.975)))
                for k, v in vs.items()
            }
            for name, vs in values.items()
        },
    )


def run(root, source, progress=print):
    root = Path(root).resolve()
    private = root / "data/track_b/models/survival_v1"
    started = time.monotonic()
    if (private / "evaluation_ledger.json").exists():
        raise ValueError("Task6 already started; silent repeat prohibited")
    protocol_path = root / "docs/track_b/survival_competing_risk_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    registry = json.loads(
        (root / "docs/track_b/survival_feature_registry.json").read_text(encoding="utf-8")
    )
    if (
        digest(source) != protocol["source_sha256"]
        or digest(root / "data/track_b/processed/expansion_v1/panel.csv")
        != protocol["panel_sha256"]
    ):
        raise ValueError("Frozen source/panel mismatch")
    ids = (
        (root / "data/track_b/manifests/expansion_v1/selected_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    if (
        len(ids) != 20000
        or hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest() != protocol["sample_sha256"]
    ):
        raise ValueError("Sample mismatch")
    base, base_hash = load_protocol(root)
    authorized_tests = [
        "tests/test_track_b_documentation.py",
        "tests/test_track_b_research_design.py",
    ]
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], text=True).split(
        "\0"
    )
    preserved = {
        name: digest(root / name) for name in tracked if name and name not in authorized_tests
    }
    for folder in [
        root / "data/track_b/manifests",
        root / "data/track_b/models/expanded_pd_v1",
        root / "data/track_b/processed/annual_2010_v1",
        root / "data/track_b/processed/expansion_v1",
    ]:
        for path in folder.glob("**/*"):
            if path.is_file():
                preserved[str(path.relative_to(root))] = digest(path)
    ledger = Ledger(private / "evaluation_ledger.json", protocol)
    frames, risk, audit = prepared(root, protocol, registry, private, base["event"], progress)
    dev = risk["development"]
    evalrisk = risk["evaluation"]
    deventry = frames["development"]
    evalentry = frames["evaluation"]
    key = dev.loan_id.map(
        lambda i: (
            int(hashlib.sha256(("task6-development-v1:" + str(i)).encode()).hexdigest(), 16) % 5
        )
    )
    training = dev.loc[key.ne(4)]
    validation = dev.loc[key.eq(4)]
    development = {}
    models = {}
    artifacts = {}
    for view in ["structural", "dynamic"]:
        candidate = fit(training, registry, view)
        development[view] = dict(
            validation_facilities=int(validation.loan_id.nunique()),
            validation_intervals=len(validation),
            metrics=likelihood(validation, probabilities(candidate, validation, registry, view)),
            fixed_specification=True,
        )
        models[view] = fit(dev, registry, view)
        path = private / (view + ".joblib")
        joblib.dump(models[view], path)
        artifacts[view] = dict(
            path=str(path.relative_to(root)),
            sha256=digest(path),
            view=view,
            preprocessing=(
                "Development median/missing indicator/scaling; modal + reference OHE; "
                "current balance log1p in dynamic view"
            ),
            C=1.0,
            seed=61006,
        )
        progress("Task6 jointly fitted default/payoff cause logits: " + view)
    max_training = int(dev.duration.max())
    horizons = protocol["horizons"]
    maximum_model = min(max_training, max(horizons))
    specification = dict(
        protocol_sha256_lf=lf_hash(protocol_path),
        registry_sha256_lf=lf_hash(root / "docs/track_b/survival_feature_registry.json"),
        artifacts=artifacts,
        max_training_duration=max_training,
        model_forecast_duration=maximum_model,
        risk_audit=audit,
        source_code_sha256_lf={
            str(p.relative_to(root)): lf_hash(p) for p in Path(__file__).parent.glob("*.py")
        },
    )
    ledger.freeze(specification)
    write(private / "frozen_specifications.json", specification)
    ledger.open()
    progress("Task6 model freeze complete; own temporal evaluation opened once")
    development_curve = aj(deventry, 60)
    observed = aj(evalentry, 60)
    structural = forecast(models["structural"], evalentry, registry, maximum_model)
    # Dynamic forecasts use only the currently available row: no invented future paths.
    updated = {
        view: probabilities(models[view], evalrisk, registry, view)
        for view in ["structural", "dynamic"]
    }
    np.savez_compressed(
        private / "temporal_predictions.npz", structural_curves=structural, **updated
    )
    point = {}
    nullpred = {
        t: np.full(len(evalentry), development_curve[t - 1]["default_cif"])
        for t in range(1, maximum_model + 1)
    }
    for h in horizons:
        status = support(observed[h - 1], h, max_training)
        entry = dict(
            month=h, status=status, observed=observed[h - 1], effective_sample=effective(evalentry)
        )
        if status == "SUPPORTED":
            pred = structural[:, h - 1, 1]
            entry.update(
                metrics=horizon_metrics(evalentry, pred, h),
                transported_development_AJ_reference=horizon_metrics(evalentry, nullpred[h], h),
                mean_survival=float(structural[:, h - 1, 0].mean()),
                mean_payoff_cif=float(structural[:, h - 1, 2].mean()),
                payoff_calibration_difference=float(
                    structural[:, h - 1, 2].mean() - observed[h - 1]["payoff_cif"]
                ),
            )
        point[str(h)] = entry
    ibs = None
    if maximum_model >= 36 and all(
        support(observed[t - 1], t, max_training) == "SUPPORTED" for t in range(1, 37)
    ):
        scores = [
            horizon_metrics(
                evalentry, structural[:, t - 1, 1], t, minimum_events=10**9, estimator=observed
            )["ipcw_brier"]
            for t in range(1, 37)
        ]
        ibs = float(np.trapezoid(scores, np.arange(1, 37)) / 35)
    # Full facility-bootstrap re-estimates AJ and censoring; models stay fixed.
    rng = np.random.default_rng(protocol["uncertainty"]["seed"])
    draws = protocol["uncertainty"]["replicates"]
    boot = {
        str(h): dict(
            default_cif=[],
            payoff_cif=[],
            survival=[],
            brier=[],
            auc=[],
            calibration_difference=[],
            payoff_calibration_difference=[],
        )
        for h in horizons
    }
    failed = 0
    integrated_boot = []
    for draw in range(draws):
        w = rng.multinomial(len(evalentry), np.full(len(evalentry), 1 / len(evalentry)))
        curve = aj(evalentry, 60, w)
        for h in horizons:
            bucket = boot[str(h)]
            for key in ["default_cif", "payoff_cif", "survival"]:
                bucket[key].append(curve[h - 1][key])
            if point[str(h)]["status"] == "SUPPORTED":
                m = horizon_metrics(
                    evalentry, structural[:, h - 1, 1], h, w, minimum_events=1, estimator=curve
                )
                bucket["brier"].append(m["ipcw_brier"])
                bucket["calibration_difference"].append(m["calibration_difference"])
                bucket["payoff_calibration_difference"].append(
                    float(w @ structural[:, h - 1, 2] / w.sum() - curve[h - 1]["payoff_cif"])
                )
                if (
                    point[str(h)]["metrics"]["cumulative_dynamic_auc"] is not None
                    and m["cumulative_dynamic_auc"] is not None
                ):
                    bucket["auc"].append(m["cumulative_dynamic_auc"])
        if ibs is not None:
            bs = [
                horizon_metrics(
                    evalentry, structural[:, t - 1, 1], t, w, minimum_events=10**9, estimator=curve
                )["ipcw_brier"]
                for t in range(1, 37)
            ]
            integrated_boot.append(float(np.trapezoid(bs, np.arange(1, 37)) / 35))
        if (draw + 1) % 100 == 0:
            progress(f"Task6 facility-bootstrap: {draw + 1}/{draws}")
    intervals = {
        h: {
            k: dict(
                lower=float(np.quantile(v, 0.025)),
                upper=float(np.quantile(v, 0.975)),
                valid_draws=len(v),
            )
            if v
            else None
            for k, v in vals.items()
        }
        for h, vals in boot.items()
    }
    monthly = {
        view: dict(effective_sample=effective(evalentry, evalrisk), metrics=likelihood(evalrisk, p))
        for view, p in updated.items()
    }
    monthly_uncertainty = interval_bootstrap(evalrisk, updated, draws, 61006)
    # Reliability: quantiles of development entry forecasts, never evaluation-selected bins.
    reliability = {}
    for h in horizons:
        if point[str(h)]["status"] != "SUPPORTED":
            continue
        dp = forecast(models["structural"], deventry, registry, h)[:, h - 1, 1]
        edges = np.unique(np.quantile(dp, [1 / 3, 2 / 3]))
        groups = np.digitize(structural[:, h - 1, 1], edges)
        cells = []
        for group in range(len(edges) + 1):
            mask = groups == group
            sub = evalentry.loc[mask]
            if sub.empty:
                continue
            obs = aj(sub, h)[-1]
            cases = int(((sub.event_code == 1) & (sub.exit_time <= h)).sum())
            cells.append(
                dict(
                    group=group,
                    facilities=len(sub),
                    default_facilities_by_horizon=cases,
                    at_risk=obs["at_risk"],
                    mean_predicted=float(structural[mask, h - 1, 1].mean()),
                    observed_default_cif=obs["default_cif"],
                    support="sparse" if cases < 10 else "descriptive",
                )
            )
        reliability[str(h)] = cells
    ambiguous_free = evalentry.loc[evalentry.exit_reason.ne("ambiguous")]
    sensitivity = dict(
        treatment=(
            "Exclude ambiguous endpoint facilities; primary censored before ambiguous "
            "month. No relabeling or refitting"
        ),
        facilities=len(ambiguous_free),
        curves_at_horizons={str(h): aj(ambiguous_free, h)[-1] for h in horizons},
    )
    # Task5 bridge reads cached scores; never instantiates or reevaluates its models.
    bridge = {}
    if point["12"]["status"] == "SUPPORTED":
        cols = [
            "loan_id",
            "t0",
            "eligible",
            "outcome_status",
            "binary_default_12m",
            "delinquency_state",
        ]
        panel = pd.read_csv(
            root / "data/track_b/processed/expansion_v1/panel.csv",
            usecols=cols,
            dtype={"loan_id": "string", "t0": "string", "delinquency_state": "string"},
        )
        _, old_eval = split(cohort(panel))
        old_score = np.load(root / "data/track_b/models/expanded_pd_v1/temporal_predictions.npz")[
            "logistic"
        ]
        lookup = {
            (str(loan), str(month)): float(p)
            for loan, month, p in zip(old_eval.loan_id, old_eval.t0, old_score, strict=True)
        }
        matched = [
            (i, lookup[(str(row.loan_id), str(row.entry_month))])
            for i, row in evalentry.reset_index(drop=True).iterrows()
            if (str(row.loan_id), str(row.entry_month)) in lookup
        ]
        index = np.array([i for i, p in matched])
        oldp = np.array([p for i, p in matched])
        newp = structural[index, 11, 1]
        correlation = float(spearmanr(oldp, newp).statistic)
        bridge = dict(
            matched_facilities=len(index),
            task5_mean_pd=float(oldp.mean()),
            task6_mean_default_cif=float(newp.mean()),
            mean_difference=float((newp - oldp).mean()),
            rank_spearman=correlation if np.isfinite(correlation) else None,
            cached_Task5_scores_only=True,
            Task5_ledger_unchanged=True,
            interpretation=(
                "One conditional window-entry per facility, explicit payoff, risk-set "
                "selection and censoring differ from repeated Task5 landmarks; no forced "
                "equality"
            ),
        )
    ledger.complete()
    if (
        any(digest(root / name) != expected for name, expected in preserved.items())
        or digest(source) != protocol["source_sha256"]
    ):
        raise ValueError("Previous evidence/source changed")
    result = dict(
        schema_version="1.0",
        decision="COMPETING-RISK SURVIVAL FOUNDATION ESTABLISHED WITH MATERIAL LIMITATIONS",
        protocol=protocol,
        protocol_sha256_lf=lf_hash(protocol_path),
        base_protocol_sha256_lf=base_hash,
        source_sha256=protocol["source_sha256"],
        sample_sha256=protocol["sample_sha256"],
        survival_audit=audit,
        model_specifications=specification,
        development_validation=development,
        evaluation_ledger=ledger.data,
        nonparametric_evaluation=observed,
        nonparametric_development=development_curve,
        horizon_results=point,
        uncertainty=dict(
            requested_draws=draws,
            valid_draws=draws,
            failed_draws=failed,
            seed=61006,
            unit="facility",
            models_fixed=True,
            intervals=intervals,
        ),
        integrated_brier_1_36=ibs,
        integrated_brier_interval={
            "lower": float(np.quantile(integrated_boot, 0.025)),
            "upper": float(np.quantile(integrated_boot, 0.975)),
            "valid_draws": len(integrated_boot),
        }
        if integrated_boot
        else None,
        reliability=reliability,
        monthly_current_state=monthly,
        monthly_uncertainty=monthly_uncertainty,
        transition_diagnostics=dict(
            development=transition_table(dev), evaluation=transition_table(evalrisk)
        ),
        cause_logits={view: coefficients(model) for view, model in models.items()},
        ambiguity_sensitivity=sensitivity,
        task5_bridge=bridge,
        preservation=dict(
            previous_file_sha256=preserved,
            unchanged=True,
            authorized_historical_test_updates=authorized_tests,
        ),
        resources=dict(
            runtime_seconds=time.monotonic() - started,
            peak_working_set_bytes=peak_memory_bytes(),
            source_performance_rescanned=False,
        ),
        package_versions={
            n: importlib.metadata.version(n)
            for n in ["numpy", "pandas", "scipy", "scikit-learn", "credit-risk-lab"]
        },
        limitations=[
            (
                "Conditional calendar-window entry differs in mortgage seasoning/survival "
                "selection between development and evaluation"
            ),
            (
                "No pre-entry exposure fabricated; this is not origination-lifetime "
                "default incidence"
            ),
            "Historical operational knowledge time UNVERIFIED",
            (
                "Dynamic current-state predictions are next-month only; multi-month "
                "dynamic CIF not identified without future-state assumptions"
            ),
            (
                "Training follow-up does not support unrestricted long-horizon "
                "extrapolation; tail rules applied"
            ),
            (
                "Marginal IPCW requires noninformative censoring; informative censoring "
                "not ruled out"
            ),
            (
                "Facility bootstrap conditions on fitted models and does not identify "
                "borrower dependence or future macro uncertainty"
            ),
            (
                "Task4/5 outcomes already inspected; Task6 has its own versioned sealed "
                "predictive protocol, not virgin data"
            ),
            (
                "No regulatory lifetime "
                "PD/IFRS9/IRB/production/causal/fairness/external-vintage validity"
            ),
        ],
        next_task=(
            "Vintage-aware macroeconomic data provenance and identification design, "
            "before any stress-model fitting"
        ),
    )
    write(root / "reports/track_b/survival_competing_risk_validation.json", result)
    from .reporting import render

    render(root, result)
    progress(result["decision"])
    return result

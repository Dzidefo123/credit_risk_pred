"""Logged post-evaluation descriptive audit; never fits or predicts."""

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_risk.track_b.data.schemas import digest
from credit_risk.track_b.pd.baseline import cohort, counts, split

from .core import hazard_periods
from .study import lf_hash


def categorical_psi(a, b, levels=("00", "01", "02")):
    x = np.array([a.eq(k).mean() for k in levels])
    y = np.array([b.eq(k).mean() for k in levels])
    x = np.maximum(x, 1e-6)
    y = np.maximum(y, 1e-6)
    return float(np.sum((y - x) * np.log(y / x)))


def supplement(root):
    root = Path(root)
    path = root / "reports/track_b/expanded_pd_validation.json"
    r = json.loads(path.read_text(encoding="utf-8"))
    if "post_evaluation_diagnostic_supplement" in r:
        raise ValueError("Diagnostic supplement already recorded")
    cols = [
        "loan_id",
        "t0",
        "eligible",
        "outcome_status",
        "binary_default_12m",
        "delinquency_state",
        "loan_age",
        "event_offset",
        "observed_followup_months",
    ]
    panel = pd.read_csv(
        root / "data/track_b/processed/expansion_v1/panel.csv",
        usecols=cols,
        dtype={"loan_id": "string", "t0": "string", "delinquency_state": "string"},
    )
    dev, ev = split(cohort(panel))
    _, raw_eval = split(panel)
    risk = hazard_periods(raw_eval)
    states = {
        name: {
            str(k): dict(**counts(g), fraction=len(g) / len(f))
            for k, g in f.groupby("delinquency_state")
        }
        for name, f in [
            ("development", dev),
            ("evaluation", ev),
            ("hazard_monthly_evaluation", risk),
        ]
    }
    artifact = r["model_specifications"]["artifacts"]["xgboost"]
    model_path = root / artifact["path"]
    if digest(model_path) != artifact["sha256"]:
        raise ValueError("Frozen model hash mismatch")
    model = joblib.load(model_path)["model"]
    names = model[0].get_feature_names_out().tolist()
    scores = model[-1].get_booster().get_score(importance_type="weight")
    splits = {name: int(scores.get("f" + str(i), 0)) for i, name in enumerate(names)}
    data = dict(
        recorded_at=datetime.now(UTC).isoformat(),
        reason=(
            "Interpretation audit after initial frozen evaluation; no model or prediction changes"
        ),
        models_modified=False,
        predictions_regenerated=False,
        primary_metrics_recomputed=False,
        metrics_added=[
            "categorical delinquency PSI",
            "current-state exposure/event counts",
            "frozen booster split usage",
        ],
        source_sha256_lf=lf_hash(Path(__file__)),
        delinquency_state_counts=states,
        categorical_delinquency_psi=categorical_psi(dev.delinquency_state, ev.delinquency_state),
        numeric_quantile_psi_warning=(
            "Mostly-zero delinquency collapses development deciles at zero; reported "
            "numeric PSI=0 is uninformative, not proof of stability"
        ),
        frozen_xgboost_split_counts=splits,
        loan_age_ranges={
            k: [float(f.loan_age.min()), float(f.loan_age.max())]
            for k, f in [("development", dev), ("evaluation", ev)]
        },
        interpretation=[
            (
                "This bounded, strongly regularized challenger did not split current "
                "delinquency; failure is specific to its prespecified architecture, not a "
                "universal failure of boosting"
            ),
            (
                "Monthly hazard discrimination is largely recognition of imminent movement "
                "across the delinquency default proxy boundary, not evidence of reliable "
                "twelve-month or lifetime forecasting"
            ),
            (
                "Hazard net-risk projection freezes delinquency; future distress evolution "
                "and payoff are not modeled. Its low projected mean must not be called "
                "calibrated primary CIF"
            ),
            (
                "Static sensitivity removes both age and delinquency, so the difference "
                "does not isolate the causal effect of delinquency"
            ),
            (
                "Raw probabilities are materially too low out of time; development sigmoid "
                "hurt Brier despite small log-loss improvements, so raw remained frozen. "
                "No post-evaluation recalibration permitted here"
            ),
        ],
    )
    ledger_path = root / "data/track_b/models/expanded_pd_v1/evaluation_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger["state"] != "CONSUMED" or ledger["post_evaluation_model_change"]:
        raise ValueError("Invalid completed ledger")
    ledger["post_evaluation_diagnostic_supplement"] = data
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    r["evaluation_ledger"] = ledger
    r["post_evaluation_diagnostic_supplement"] = data
    r["limitations"].extend(data["interpretation"])
    path.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = root / "reports/track_b/EXPANDED_PD_VALIDATION.md"
    text = report.read_text(encoding="utf-8")
    summary = (
        "\n\nLogistic improves temporal probability quality over the null, but "
        "substantially underpredicts risk: mean 0.396% versus observed 0.832%; "
        "CITL +0.987, slope 0.768. No claim of satisfactory temporal calibration. "
        "This specific XGBoost challenger is inferior on paired "
        "discrimination/probability-quality intervals and used no delinquency "
        "splits. Origination-only AP is 0.0195 versus full-model 0.176; removing "
        "both ageing and delinquency materially changes the question.\n\n"
    )
    text = text.replace("## Research Question", summary + "## Research Question", 1)
    text += (
        "\n## Logged descriptive interpretation audit\n\n"
        + "\n".join("- " + s for s in data["interpretation"])
        + "\n\n"
        + data["numeric_quantile_psi_warning"]
        + ". Categorical PSI: "
        + str(data["categorical_delinquency_psi"])
        + (
            ".\n\nNo model change, no new predictions and no recomputation of primary "
            "metrics. This supplement was logged after evaluation and did not "
            "influence frozen model selection.\n"
        )
    )
    report.write_text(text, encoding="utf-8")
    return data


if __name__ == "__main__":
    supplement(Path(__file__).resolve().parents[4])

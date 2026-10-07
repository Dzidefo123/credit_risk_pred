"""Post-evaluation time-support audit using cached probabilities only."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .models import likelihood
from .study import interval_bootstrap, lf_hash


def supplement(root):
    root = Path(root)
    path = root / "reports/track_b/survival_competing_risk_validation.json"
    r = json.loads(path.read_text(encoding="utf-8"))
    if "time_support_audit" in r:
        raise ValueError("Supplement already recorded")
    private = root / "data/track_b/models/survival_v1"
    risk = pd.read_csv(private / "evaluation_risk.csv", dtype={"loan_id": "string"})
    cache = np.load(private / "temporal_predictions.npz")
    limit = r["model_specifications"]["max_training_duration"]
    supported = risk.duration.le(limit).to_numpy()
    probabilities = {view: cache[view][supported] for view in ["structural", "dynamic"]}
    sub = risk.loc[supported]
    new = dict(
        recorded_at=datetime.now(UTC).isoformat(),
        models_modified=False,
        predictions_regenerated=False,
        primary_horizon_metrics_recomputed=False,
        source_sha256_lf=lf_hash(Path(__file__)),
        maximum_development_duration=limit,
        all_intervals=len(risk),
        supported_intervals=len(sub),
        outside_training_duration_intervals=int((~supported).sum()),
        unseen_duration_band_intervals=int(risk.duration.gt(60).sum()),
        supported_facilities=int(sub.loan_id.nunique()),
        supported_default_events=int(sub.event_code.eq(1).sum()),
        supported_payoff_events=int(sub.event_code.eq(2).sum()),
        supported_time_monthly_metrics={
            view: likelihood(sub, p) for view, p in probabilities.items()
        },
        supported_time_monthly_uncertainty=interval_bootstrap(sub, probabilities, 400, 61006),
        interpretation=(
            "All-period monthly scores include time extrapolation; bands61+ were "
            "unseen in development and encoded as reference by the frozen pipeline. "
            "Supported-duration subset is a new descriptive diagnostic using cached "
            "scores, not a model revision or primary CIF reevaluation."
        ),
    )
    r["time_support_audit"] = new
    r["limitations"].extend(
        [
            (
                "Later-duration categories were unseen during development; full-period "
                "updated-state scores are exploratory outside the trained time basis"
            ),
            (
                "Unobserved history before first servicing record cannot establish "
                "first-ever default; endpoints are first observed research events"
            ),
            (
                "At36 months structural Brier did not improve on the transported "
                "development AJ reference; payoff CIF is materially miscalibrated at12 "
                "and36 months"
            ),
        ]
    )
    ledger_path = private / "evaluation_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger["state"] != "CONSUMED":
        raise ValueError("Own evaluation must already be consumed")
    ledger["post_evaluation_diagnostic_supplement"] = dict(
        recorded_at=new["recorded_at"],
        purpose="Duration-basis support warning review",
        models_modified=False,
        predictions_regenerated=False,
        primary_horizon_metrics_recomputed=False,
    )
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    r["evaluation_ledger"] = ledger
    path.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md = root / "reports/track_b/SURVIVAL_COMPETING_RISK_VALIDATION.md"
    text = md.read_text(encoding="utf-8")
    intro = (
        "\n\nInterpretation review: raw audit reconciles, risk sets and probability "
        "identities are valid, but this is not a calibrated lifetime-PD model. "
        "At12 months predicted default CIF is0.329% versus observed0.578%, and "
        "payoff CIF9.42% versus18.61%. At36 months default means nearly match, "
        "while payoff is overpredicted; model Brier is slightly worse than the "
        "transported development AJ reference. Only54 development months are "
        "available: modeled60-month forecasts are suppressed despite941 evaluation "
        "facilities still at risk. Dynamic multimonth CIF remains unidentified "
        "without a future-state model.\n\n"
    )
    text = text.replace("## Research Question", intro + "## Research Question", 1)
    text += (
        "\n## Logged time-basis support diagnostic\n\n"
        + new["interpretation"]
        + "\n\n"
        + json.dumps(new, indent=2)
        + (
            "\n\nMonthly dynamic information improves imminent default scoring but can "
            "worsen payoff/joint likelihood. It is not automatically a superior "
            "structural forecast. Unseen-category handling must not be mistaken for "
            "supported long-horizon prediction. No model/primary metric/prediction "
            "regeneration; the supplement is logged in the separate Task6 ledger.\n"
        )
    )
    md.write_text(text, encoding="utf-8")
    return new


if __name__ == "__main__":
    supplement(Path(__file__).resolve().parents[4])

"""One landmark/facility, rolling historical PIT paths, AJ and Task6 pooled IPCW."""

import numpy as np
import pandas as pd

from credit_risk.track_b.macro_support.eligibility import monthly_text, ordinal
from credit_risk.track_b.survival.math import aj, curves, horizon_metrics, support

from .data import macro_join
from .models import predict


def landmarks(data):
    ids, starts, counts = np.unique(data["facility"], return_index=True, return_counts=True)
    ends = starts + counts - 1
    if not np.all(data["facility"][ends] == ids) or not np.all(
        data["month"][ends] - data["month"][starts] + 1 == counts
    ):
        raise ValueError("Landmark risk rows must be contiguous, sorted facility histories")
    subjects = pd.DataFrame(
        dict(
            facility=ids,
            entry_time=0,
            exit_time=counts,
            event_code=data["event"][ends],
            first_month=data["month"][starts],
        )
    )
    return data[starts].copy(), subjects


def path_forecast(bundle, entries, table, horizon, cutoff="2026-02"):
    if np.any(entries["month"] + horizon - 1 > ordinal(cutoff)):
        raise ValueError("Future macro path beyond frozen reporting cutoff")
    rows = entries.copy()
    ds, ps = [], []
    first_month = entries["month"].copy()
    first_duration = entries["duration"].copy()
    for step in range(horizon):
        rows["month"] = first_month + step
        rows["duration"] = first_duration + step
        for month in np.unique(rows["month"]):
            key = monthly_text(month)
            if key not in table:
                raise ValueError("Required historical macro month unavailable")
            macro = table[key]["features"]
            needed = bundle[0].macro
            if any(macro[n]["status"] != "AVAILABLE" for n in needed):
                raise ValueError("Incomplete historical PIT macro path")
            rows["macro"][rows["month"] == month] = macro_join(table, key, needed)
        p = predict(bundle, rows)
        ds.append(p[:, 1])
        ps.append(p[:, 2])
    return curves(np.column_stack(ds), np.column_stack(ps))


def evaluate(data, models, table, spec, private):
    entries, subjects = landmarks(data)
    result = dict(
        landmarks=len(entries),
        unique_facilities=len(entries),
        unique_default_facilities=int(subjects.event_code.eq(1).sum()),
        unique_payoff_facilities=int(subjects.event_code.eq(2).sum()),
        interpretation="Rolling historical PIT-path CIF; not a prospective t0 forecast",
        censoring_assumption="Pooled independent administrative censoring, unverified",
        horizons={},
    )
    for h in spec["cif"]["horizons"]:
        mask = entries["month"] + h - 1 <= ordinal("2026-02")
        chosen = entries[mask]
        observed = subjects.loc[mask].copy().reset_index(drop=True)
        estimate = aj(observed, h)
        status = support(estimate[-1], h, 88, minimum_risk=200)
        part = dict(
            facilities=len(chosen),
            landmarks=len(chosen),
            calendar_truncated_landmarks=int((~mask).sum()),
            status=status,
            observed=estimate,
            models={},
        )
        swapped = observed.copy()
        swapped["event_code"] = observed.event_code.map({0: 0, 1: 2, 2: 1})
        swapped_estimate = aj(swapped, h)
        if status == "SUPPORTED":
            for name, bundle in models.items():
                forecast = path_forecast(bundle, chosen, table, h)
                np.save(private / f"cif_{name}_{h}.npy", forecast)
                part["models"][name] = dict(
                    default=horizon_metrics(
                        observed, forecast[:, -1, 1], h, minimum_events=20, estimator=estimate
                    ),
                    payoff=horizon_metrics(
                        swapped,
                        forecast[:, -1, 2],
                        h,
                        minimum_events=20,
                        estimator=swapped_estimate,
                    ),
                    mean_curves=forecast.mean(axis=0).tolist(),
                    max_conservation_residual=float(np.max(abs(forecast.sum(axis=2) - 1))),
                )
        result["horizons"][str(h)] = part
    return result

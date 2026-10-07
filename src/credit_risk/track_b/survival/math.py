"""Monthly probability-conserving competing risks and censor-adjusted metrics."""

import numpy as np
from sklearn.metrics import roc_auc_score


def curves(default, payoff):
    d, p = np.asarray(default, float), np.asarray(payoff, float)
    if (
        d.shape != p.shape
        or d.ndim != 2
        or not np.isfinite(d).all()
        or not np.isfinite(p).all()
        or (d < 0).any()
        or (p < 0).any()
        or (d + p > 1 + 1e-12).any()
    ):
        raise ValueError("Invalid mutually exclusive monthly hazards")
    s = np.ones(len(d))
    fd = np.zeros(len(d))
    fp = np.zeros(len(d))
    out = []
    for j in range(d.shape[1]):
        before = s.copy()
        fd += before * d[:, j]
        fp += before * p[:, j]
        s = before * np.maximum(0, 1 - d[:, j] - p[:, j])
        out.append(np.column_stack([s, fd, fp]))
    result = np.stack(out, axis=1)
    if not np.allclose(result.sum(axis=2), 1, atol=1e-10):
        raise ValueError("Probability conservation failed")
    return result


def aj(subjects, horizon, weights=None):
    entry = subjects.entry_time.to_numpy(int)
    exit = subjects.exit_time.to_numpy(int)
    event = subjects.event_code.to_numpy(int)
    if (entry >= exit).any() or (entry < 0).any() or not np.isin(event, [0, 1, 2]).all():
        raise ValueError("Invalid entry/exit/event")
    w = np.ones(len(subjects)) if weights is None else np.asarray(weights, float)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid facility weights")
    s = 1.0
    fd = 0.0
    fp = 0.0
    net = 1.0
    g = 1.0
    out = []
    for t in range(1, horizon + 1):
        risk = float(w[(entry < t) & (exit >= t)].sum())
        dd = float(w[(exit == t) & (event == 1)].sum())
        dp = float(w[(exit == t) & (event == 2)].sum())
        cc = float(w[(exit == t) & (event == 0)].sum())
        before = s
        gbefore = g
        if risk:
            fd += before * dd / risk
            fp += before * dp / risk
            s *= 1 - (dd + dp) / risk
            net *= 1 - dd / risk
            # Censors occur after last confirmed status; event removals precede tied censoring.
            censor_risk = risk - dd - dp
            if censor_risk > 0:
                g *= 1 - cc / censor_risk
        out.append(
            dict(
                month=t,
                at_risk=risk,
                default_events=dd,
                payoff_events=dp,
                censored=cc,
                survival=s,
                default_cif=fd,
                payoff_cif=fp,
                naive_net_default=1 - net,
                censor_survival_before=gbefore,
                censor_survival_after=g,
            )
        )
    if any(abs(r["survival"] + r["default_cif"] + r["payoff_cif"] - 1) > 1e-10 for r in out):
        raise ValueError("AJ conservation failed")
    return out


def horizon_metrics(subjects, predicted, horizon, weights=None, minimum_events=10, estimator=None):
    if subjects.entry_time.ne(0).any():
        raise ValueError("IPCW currently requires window-relative entry zero")
    w = np.ones(len(subjects)) if weights is None else np.asarray(weights, float)
    estimator = aj(subjects, horizon, w) if estimator is None else estimator[:horizon]
    exit = subjects.exit_time.to_numpy(int)
    event = subjects.event_code.to_numpy(int)
    before = np.r_[1, [r["censor_survival_before"] for r in estimator]]
    if before[horizon] <= 0:
        raise ValueError("Insufficient censoring support")
    observed_event = (event > 0) & (exit <= horizon)
    survived = exit >= horizon
    known = observed_event | survived
    y = ((event == 1) & (exit <= horizon)).astype(int)
    time = np.where(observed_event, exit, horizon)
    ipcw = np.zeros(len(subjects))
    if (before[time[known]] <= 0).any():
        raise ValueError("Zero event-time censoring support")
    ipcw[known] = 1 / before[time[known]]
    effective = w * ipcw
    p = np.asarray(predicted, float)
    score = float(np.sum(effective * (y - p) ** 2) / w.sum())
    cases = int(np.count_nonzero((w > 0) & (y == 1)))
    auc = None
    if cases >= minimum_events and np.count_nonzero((effective > 0) & (y == 0)) >= minimum_events:
        auc = float(roc_auc_score(y, p, sample_weight=effective))
    observed = estimator[-1]["default_cif"]
    return dict(
        ipcw_brier=score,
        cumulative_dynamic_auc=auc,
        auc_status="estimated" if auc is not None else "INSUFFICIENT DEFAULT EVENT SUPPORT",
        default_facilities_by_horizon=cases,
        observed_default_cif=observed,
        mean_predicted_cif=float(w @ p / w.sum()),
        calibration_difference=float(w @ p / w.sum() - observed),
        at_risk=estimator[-1]["at_risk"],
        censor_survival=before[horizon],
        known_status_facilities=int(np.count_nonzero(known & (w > 0))),
        ipcw_weight_sum=float(effective.sum()),
        cohort_size=float(w.sum()),
        controls="All non-defaults including prior competing payoff; not survivor-only controls",
    )


def support(row, horizon, max_training, minimum_risk=200):
    if row["at_risk"] < minimum_risk or row["censor_survival_before"] < 0.1:
        return "INSUFFICIENT TAIL SUPPORT"
    if horizon > max_training:
        return "INSUFFICIENT TRAINING-TIME SUPPORT"
    return "SUPPORTED"

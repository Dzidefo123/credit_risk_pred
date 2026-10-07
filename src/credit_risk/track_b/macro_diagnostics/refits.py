"""Isolated post-validation coefficient diagnostics; no replacement-model scoring."""

import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from credit_risk.track_b.data.schemas import digest
from credit_risk.track_b.macro_hazard.models import artifact, fit
from credit_risk.track_b.macro_hazard.protocol import PRIMARY
from credit_risk.track_b.macro_hazard.study import counts, population
from credit_risk.track_b.macro_support.study import immutable_json, read_json

from .protocol import PRIVATE, WINDOWS, freeze
from .verification import verify


def diagnostic_view(data, start, end):
    year = data["month"] // 12
    selected = data[(year >= start) & (year <= end)].copy()
    selected["role"] = 0
    return selected


def run(root):
    verify(root)
    freeze(root)
    private = root / PRIVATE
    private.mkdir(parents=True, exist_ok=True)
    if (private / "diagnostic_refits.json").exists():
        raise ValueError("Diagnostic refits already sealed; no adaptive rerun")
    spec = read_json(root / "docs/track_b/macro_competing_risk_protocol.json")
    _, development, evaluation, seen = population(root, spec)
    common = read_json(root / "data/track_b/models/macro_hazard_v1/M2_development.json")[
        "parameters"
    ]
    results = {}
    for start, end in WINDOWS:
        data = diagnostic_view(development if end <= 2017 else evaluation[seen], start, end)
        key = f"{start}_{end}"
        result = dict(
            label="DIAGNOSTIC_REFIT_ONLY",
            counts=counts(data),
            months=len(np.unique(data["month"])),
            original_role="development" if end <= 2017 else "consumed_temporal_evaluation",
            original_role_arrays_mutated=False,
            replacement_scored=False,
        )
        print("Diagnostic coefficient fit " + key + "; not a replacement model", flush=True)
        if (
            result["counts"]["defaults"] < 20
            or result["counts"]["payoffs"] < 100
            or result["months"] < 24
        ):
            result["status"] = "INSUFFICIENT_DIAGNOSTIC_SUPPORT"
            results[key] = result
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                bundle = fit(data, spec, True, PRIMARY)
            result.update(
                status="CONVERGED",
                parameters=artifact(bundle),
                warnings=[str(w.message) for w in caught],
            )
            encoder, model = bundle
            coefficients = []
            for code, cause in [(1, "default"), (2, "payoff")]:
                for name in PRIMARY:
                    i = encoder.names.index(name)
                    ci = common["feature_order"].index(name)
                    beta = float(model.coef_[code, i] - model.coef_[0, i])
                    native = beta / encoder.parameters["scales"][i]
                    common_beta = native * common["preprocessing"]["scales"][ci]
                    coefficients.append(
                        dict(
                            cause=cause,
                            feature=name,
                            window_sd_log_odds=beta,
                            native_unit_log_odds=native,
                            common_task10_sd_log_odds=common_beta,
                            common_task10_sd_odds_ratio=float(np.exp(common_beta))
                            if abs(common_beta) < 700
                            else None,
                        )
                    )
            result["macro_coefficients"] = coefficients
            artifact_path = private / (key + ".joblib")
            joblib.dump(bundle, artifact_path)
            result["artifact_sha256"] = digest(artifact_path)
            print(key + " diagnostic fit converged " + str(model.n_iter_.tolist()), flush=True)
        except ValueError as exc:
            result.update(status="DIAGNOSTIC_FIT_INVALID_NOT_REPAIRED", reason=str(exc))
            print(key + " diagnostic fit unsupported: " + str(exc), flush=True)
        results[key] = result
        immutable_json(private / (key + "_coefficients.json"), result)
    immutable_json(private / "diagnostic_refits.json", results)
    verify(root)

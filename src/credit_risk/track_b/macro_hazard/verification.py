"""Task10 preservation and artifact-only deterministic verification replay."""

import numpy as np

from credit_risk.track_b.data.schemas import digest
from credit_risk.track_b.macro_support.study import immutable_json, read_json
from credit_risk.track_b.macro_support.verification import verify as prior_verify
from credit_risk.track_b.multivintage.study import lf_hash


def preservation(root, source_dir):
    frozen = read_json(root / "docs/track_b/macro_competing_risk_preservation_manifest.json")
    for n, h in frozen["public_lf_hashes"].items():
        if lf_hash(root / n) != h:
            raise ValueError("Task2–9A public evidence changed: " + n)
    for n, h in frozen["private_byte_hashes"].items():
        if digest(root / n) != h:
            raise ValueError("Prior private/frozen evidence changed: " + n)
    return dict(
        status="PASSED",
        prior_public_lf_hashes=len(frozen["public_lf_hashes"]),
        prior_private_byte_hashes=len(frozen["private_byte_hashes"]),
        task9a_and_earlier=prior_verify(root, source_dir),
        prior_ledgers_reused=False,
        prior_models_regenerated=False,
        retained_metrics_only=dict(auc=0.868152, brier=0.048545, log_loss=0.176030),
    )


def replay(root):
    import joblib

    from .cif import landmarks, path_forecast
    from .metrics import scores
    from .models import predict
    from .protocol import freeze
    from .study import population, registration

    private = root / "data/track_b/models/macro_hazard_v1"
    spec = freeze(root)
    _, _, evaluation, seen = population(root, spec)
    ledger = read_json(private / "task10_evaluation_ledger.json")
    from credit_risk.track_b.macro.information import feature_hash

    if (
        ledger["state"] != "CONSUMED"
        or feature_hash(registration(root, spec, evaluation, seen))
        != (ledger["registration_sha256"])
    ):
        raise ValueError("Consumed ledger/input registration changed")
    manifest = read_json(private / "model_manifest.json")
    if manifest["bindings"] != ledger["model_hashes"]:
        raise ValueError("Model manifest differs from pre-evaluation ledger binding")
    for name, expected in manifest["bindings"].items():
        if digest(private / (name + ".joblib")) != expected:
            raise ValueError("Frozen model artifact changed before replay")
    for name, expected in ledger["prediction_hashes"].items():
        if digest(private / name) != expected:
            raise ValueError("Consumed prediction artifact changed before replay")
    result = read_json(private / "temporal_results.json")
    max_error = 0.0
    table = {
        r["reporting_month"]: r
        for r in read_json(
            root / "data/track_b/macro/processed/task9_api_v5/macro_month_table.json"
        )
    }
    entries, _ = landmarks(evaluation[seen])
    for name in ["M0", "M1", "M2", "RATE", "REDUCED_M1", "REDUCED_M2"]:
        model = joblib.load(private / (name + ".joblib"))
        p = predict(model, evaluation)
        original = np.load(private / (name + "_evaluation.npy"))
        error = float(np.max(abs(p - original)))
        max_error = max(max_error, error)
        if not np.allclose(p, original, atol=1e-12, rtol=1e-10):
            raise ValueError("Deterministic prediction replay failed")
        actual = scores(evaluation[seen], p[seen])
        expected = result["primary"][name]["scores"]
        for k, v in expected.items():
            if isinstance(v, (int, float)) and not np.isclose(actual[k], v, atol=1e-12, rtol=1e-10):
                raise ValueError("Aggregate metric replay failed")
        if name in ["M0", "M1", "M2"]:
            from credit_risk.track_b.macro_support.eligibility import ordinal

            for h in spec["cif"]["horizons"]:
                q = private / f"cif_{name}_{h}.npy"
                if q.exists():
                    mask = entries["month"] + h - 1 <= ordinal("2026-02")
                    regenerated = path_forecast(model, entries[mask], table, h)
                    if not np.allclose(regenerated, np.load(q), atol=1e-12, rtol=1e-10):
                        raise ValueError("CIF replay failed")
    immutable_json(
        private / "reproducibility.json",
        dict(
            status="PASSED",
            max_prediction_absolute_error=max_error,
            atol=1e-12,
            rtol=1e-10,
            same_frozen_models=True,
            verification_only_not_new_scientific_evaluation=True,
            primary_ledger_consumption_count=ledger["prediction_generation_count"],
            ledger_unchanged=True,
            new_fitting_or_model_selection=False,
        ),
    )

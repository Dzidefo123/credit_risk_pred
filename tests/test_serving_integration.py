"""Train only fresh synthetic fixtures, then verify the full frozen serving lifecycle."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from credit_risk.api.main import create_app
from credit_risk.data.validation import ORIGINATION_FEATURES, ORIGINATION_TARGET
from credit_risk.models.pd import run_origination_experiment, split_origination
from credit_risk.utils.config import ModelConfig, XGBoostConfig
from credit_risk.validation.holdout_registry import HoldoutRegistry
from credit_risk.validation.runner import digest, run_validation
from credit_risk.validation.settings import ValidationConfig


@pytest.fixture(scope="module")
def serving_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-serving")
    registry_path = root / "holdout_registry.json"
    HoldoutRegistry.initialize(registry_path)
    rng = np.random.default_rng(112)
    n = 400
    frame = pd.DataFrame({name: rng.integers(0, 6, n) for name in ORIGINATION_FEATURES})
    frame["age"] = rng.integers(21, 80, n)
    frame["MonthlyIncome"] = rng.lognormal(8, 0.5, n)
    frame["DebtRatio"] = rng.uniform(0, 1, n)
    frame["RevolvingUtilizationOfUnsecuredLines"] = rng.uniform(0, 1, n)
    frame[ORIGINATION_TARGET] = (
        (frame.NumberOfTimes90DaysLate >= 4) & (rng.random(n) > 0.3)
    ).astype(int)
    source, run, validation = root / "synthetic.csv", root / "run", root / "validation"
    frame.to_csv(source, index=False)
    config = ModelConfig(xgboost=XGBoostConfig(n_estimators=8, n_jobs=1))
    run_origination_experiment(source, run, config)
    run_validation(
        source,
        run,
        validation,
        ValidationConfig(bootstrap_samples=20, minimum_calibration_events=4),
        registry_path=registry_path,
    )
    (root / "policy.yaml").write_text("policies: [{}]\n", encoding="utf-8")
    settings = root / "serving.yaml"
    settings.write_text(
        "source_csv: synthetic.csv\nrun_dir: run\n"
        "validation_dir: validation\npolicy_config: policy.yaml\n",
        encoding="utf-8",
    )
    loaded = pd.read_csv(source)
    splits = split_origination(loaded, config)
    return settings, loaded, splits


def applicant(frame, position):
    values = frame.iloc[position].loc[list(ORIGINATION_FEATURES)].to_dict()
    for name in values:
        values[name] = int(values[name]) if name.startswith("Number") else float(values[name])
    return {"application_id": "synthetic-integration", "features": values}


def test_verified_bundle_serves_without_fitting_or_changing_evidence(serving_bundle, monkeypatch):
    settings, frame, splits = serving_bundle
    root = settings.parent
    protected = [
        p for folder in ("run", "validation") for p in (root / folder).iterdir() if p.is_file()
    ]
    before = {p: digest(p) for p in protected}

    def forbidden_fit(*args, **kwargs):
        pytest.fail("Serving must never fit the base model")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    request = applicant(frame, splits["development"][0])
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        score = client.post("/score", json=request)
        decision = client.post("/decision", json=request)
        assert score.status_code == decision.status_code == 200
        assert score.json()["pd"] == decision.json()["pd"]
        assert score.json()["model_sha256"] == digest(
            root / "validation" / f"{client.app.state.service.candidate}_selected.joblib"
        )
        assert decision.json()["research_only"]
    assert {p: digest(p) for p in protected} == before
    assert json.loads((root / "run/test_consumption.json").read_text(encoding="utf-8"))


def test_new_fixture_holdout_is_rejected_before_prediction(serving_bundle, monkeypatch):
    settings, frame, splits = serving_bundle
    with TestClient(create_app(settings)) as client:

        def forbidden_predict(*args, **kwargs):
            pytest.fail("Reserved profiles must be rejected before scoring")

        monkeypatch.setattr(client.app.state.service.model, "predict_proba", forbidden_predict)
        response = client.post("/score", json=applicant(frame, splits["test"][0]))
        assert response.status_code == 422


@pytest.mark.parametrize("artifact", ["synthetic.csv", "selected_model"])
def test_tampered_bundle_is_unavailable_before_deserialization(
    serving_bundle, monkeypatch, artifact
):
    settings, frame, splits = serving_bundle
    root = settings.parent
    selection = json.loads((root / "validation/selection.json").read_text(encoding="utf-8"))
    path = (
        root / artifact
        if artifact == "synthetic.csv"
        else root / "validation" / f"{selection['preferred_candidate']}_selected.joblib"
    )
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"tampered")
        monkeypatch.setattr(
            joblib,
            "load",
            lambda *args, **kwargs: pytest.fail("Integrity must precede deserialization"),
        )
        with TestClient(create_app(settings)) as client:
            assert client.get("/health").status_code == 503
            assert (
                client.post(
                    "/decision", json=applicant(frame, splits["development"][0])
                ).status_code
                == 503
            )
    finally:
        path.write_bytes(original)

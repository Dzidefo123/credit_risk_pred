"""API contracts, startup isolation, shared policy and failure behavior."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_risk.api.main import create_app
from credit_risk.api.schemas import ApplicantRequest
from credit_risk.api.service import ScoringService
from credit_risk.data.validation import ORIGINATION_FEATURES
from credit_risk.decisioning.policy import decide
from credit_risk.decisioning.settings import CreditStrategy


def payload():
    return {
        "application_id": "demo-001",
        "features": {
            "RevolvingUtilizationOfUnsecuredLines": 0.2,
            "age": 43.0,
            "NumberOfTime30_59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.2,
            "MonthlyIncome": 5000.0,
            "NumberOfOpenCreditLinesAndLoans": 3,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60_89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 1,
        },
    }


def service(probability=0.01):
    class FrozenModel:
        def predict_proba(self, x):
            assert list(x.columns) == list(ORIGINATION_FEATURES)
            return np.array([[1 - probability, probability]])

    return ScoringService(
        FrozenModel(),
        CreditStrategy(),
        "xgboost",
        "sigmoid",
        "a" * 64,
        "Inherited SeriousDlqin2yrs; source two-year delinquency label",
    )


def test_lifespan_loads_once_and_routes_share_policy():
    loads = []
    scorer = service()

    def load(path):
        loads.append(path)
        return scorer

    app = create_app("unused.yaml", load)
    assert loads == []
    with TestClient(app) as client:
        assert len(loads) == 1
        assert client.get("/health").json()["status"] == "ready"
        p = client.post("/score", json=payload()).json()
        d = client.post("/decision", json=payload()).json()
        assert p["pd"] == d["pd"] == 0.01 and p["risk_grade"] == "G1"
        assert p["research_only"] and p["model_sha256"] == "a" * 64
        assert d["decision"] == "APPROVE" and d["recommended_limit"] > 0
        frame = pd.DataFrame([payload()["features"]], columns=ORIGINATION_FEATURES)
        expected = decide(frame, [0.01], scorer.policy).iloc[0]
        assert d["recommended_limit"] == expected.recommended_limit
        assert d["expected_loss_proxy"] == expected.expected_loss_proxy
        assert d["policy_sha256"] == scorer.policy_sha256
        assert len(loads) == 1
    assert app.state.service is None


@pytest.mark.parametrize(
    "probability,expected",
    [(0.03, "MANUAL_REVIEW"), (0.1, "DECLINE"), (0.0, "APPROVE"), (1.0, "DECLINE")],
)
def test_decision_thresholds_and_probability_endpoints(probability, expected):
    with TestClient(create_app(service_loader=lambda path: service(probability))) as client:
        response = client.post("/decision", json=payload())
        assert response.status_code == 200 and response.json()["decision"] == expected
        if expected != "APPROVE":
            assert response.json()["recommended_limit"] == 0.0


def test_missing_income_never_uses_model_imputation_for_policy():
    scorer = service()
    original = scorer.model.predict_proba

    def check_raw(x):
        assert pd.isna(x.MonthlyIncome.iloc[0])
        return original(x)

    scorer.model.predict_proba = check_raw
    request = payload()
    request["features"].pop("MonthlyIncome")
    with TestClient(create_app(service_loader=lambda path: scorer)) as client:
        response = client.post("/decision", json=request)
        assert response.status_code == 200
        d = response.json()
        assert "MonthlyIncome" in d["missing_features"]
        assert d["decision"] == "MANUAL_REVIEW" and d["recommended_limit"] == 0
        assert "INCOME_UNAVAILABLE_OR_ZERO" in d["reason_codes"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", -1),
        ("age", 43.5),
        ("NumberOfDependents", 2**53),
        ("MonthlyIncome", "secret-income-value"),
        ("NumberOfDependents", True),
        ("NumberOfTimes90DaysLate", 1.5),
        ("SeriousDlqin2yrs", 1),
        ("source_row_id", 15),
    ],
)
def test_invalid_and_leaking_fields_rejected_without_echo(field, value):
    request = payload()
    request["features"][field] = value
    with TestClient(create_app(service_loader=lambda path: service())) as client:
        response = client.post("/score", json=request)
        assert response.status_code == 422
        assert "input" not in response.json()["detail"][0]
        assert "secret-income-value" not in response.text


def test_invalid_application_id_and_nonfinite_json_rejected():
    request = payload()
    request["application_id"] = "../escape"
    with TestClient(create_app(service_loader=lambda path: service())) as client:
        assert client.post("/decision", json=request).status_code == 422
        body = json.dumps(payload()).replace('"age": 43.0', '"age": NaN')
        assert (
            client.post(
                "/score", content=body, headers={"content-type": "application/json"}
            ).status_code
            == 422
        )


def test_unavailable_artifact_never_falls_back_to_inherited_model():
    def broken(path):
        raise ValueError("C:/private/secret-model.joblib")

    with TestClient(create_app(service_loader=broken)) as client:
        assert client.get("/health").status_code == 503
        for endpoint in ("/score", "/decision"):
            response = client.post(endpoint, json=payload())
            assert response.status_code == 503 and "secret-model" not in response.text


def test_bad_model_output_fails_closed_and_reserved_group_cannot_score():
    scorer = service(np.nan)
    with TestClient(create_app(service_loader=lambda path: scorer)) as client:
        assert client.post("/decision", json=payload()).status_code == 503
    scorer = service()
    frame = pd.DataFrame([payload()["features"]], columns=ORIGINATION_FEATURES).astype(float)
    scorer.reserved_groups = frozenset(
        [int(pd.util.hash_pandas_object(frame, index=False).iloc[0])]
    )

    def forbidden(x):
        raise AssertionError("Reserved rows must not be scored")

    scorer.model.predict_proba = forbidden
    with TestClient(create_app(service_loader=lambda path: scorer)) as client:
        assert client.post("/score", json=payload()).status_code == 422
    with pytest.raises(ValueError):
        ApplicantRequest.model_validate({**payload(), "label": 1})

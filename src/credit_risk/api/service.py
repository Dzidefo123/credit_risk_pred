"""Verified one-time model loading, read-only prediction and shared policy logic."""

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from threading import RLock

import numpy as np
import pandas as pd

from credit_risk import __version__
from credit_risk.api.schemas import ApplicantRequest, DecisionResponse, ScoreResponse
from credit_risk.api.settings import ServingConfig
from credit_risk.data.validation import ORIGINATION_FEATURES, validate_origination
from credit_risk.decisioning.policy import decide
from credit_risk.decisioning.runner import load_selected_model
from credit_risk.decisioning.settings import CreditStrategy, PolicyComparisonConfig
from credit_risk.utils.config import load_config
from credit_risk.validation.runner import verify_experiment


class ReservedHoldoutRequest(ValueError):
    """The lab cannot reuse original final-test profiles for API demonstrations."""


@dataclass
class ScoringService:
    model: object
    policy: CreditStrategy
    candidate: str
    calibration_method: str
    model_sha256: str
    target_semantics: str
    reserved_groups: frozenset[int] = field(default_factory=frozenset)
    lock: RLock = field(default_factory=RLock, repr=False)

    @property
    def model_version(self):
        return f"{self.candidate}:{self.calibration_method}:{self.model_sha256[:12]}"

    @property
    def policy_sha256(self):
        return sha256(json.dumps(self.policy.model_dump(), sort_keys=True).encode()).hexdigest()

    def evaluate(self, request: ApplicantRequest):
        raw = request.features.model_dump()
        frame = pd.DataFrame([raw], columns=ORIGINATION_FEATURES).astype(float)
        quality = validate_origination(frame, require_target=False)
        fingerprint = int(pd.util.hash_pandas_object(frame, index=False).iloc[0])
        if fingerprint in self.reserved_groups:
            raise ReservedHoldoutRequest("Profile belongs to the reserved research holdout")
        with self.lock:
            probabilities = np.asarray(self.model.predict_proba(frame), dtype=float)
        if (
            probabilities.shape != (1, 2)
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
            or not np.allclose(probabilities.sum(axis=1), 1)
        ):
            raise ValueError("Invalid model probability output")
        pd_value = float(probabilities[0, 1])
        decision = decide(frame, [pd_value], self.policy).iloc[0]
        score = ScoreResponse(
            application_id=request.application_id,
            pd=pd_value,
            risk_grade=decision.risk_grade,
            model_version=self.model_version,
            model_sha256=self.model_sha256,
            package_version=__version__,
            calibration_method=self.calibration_method,
            target_semantics=self.target_semantics,
            missing_features=[name for name in ORIGINATION_FEATURES if raw[name] is None],
            suspicious_inputs=quality.suspicious,
        )
        return score, decision

    def score(self, request):
        return self.evaluate(request)[0]

    def decision(self, request):
        score, result = self.evaluate(request)
        return DecisionResponse(
            **score.model_dump(),
            decision=result.decision,
            recommended_limit=float(result.recommended_limit),
            reason_codes=result.reason_codes.split("|"),
            policy_name=self.policy.name,
            policy_sha256=self.policy_sha256,
            assumed_ead=float(result.assumed_ead),
            expected_loss_proxy=float(result.expected_loss_proxy),
        )


def load_scoring_service(config_path):
    path = Path(config_path).resolve()
    settings = load_config(path, ServingConfig)

    def resolve(value):
        return (path.parent / value).resolve()

    experiment, _, frame, splits, _ = verify_experiment(
        resolve(settings.source_csv), resolve(settings.run_dir)
    )
    model, selection, validation = load_selected_model(
        resolve(settings.validation_dir), resolve(settings.run_dir), experiment
    )
    if validation["source_sha256"] != experiment["source_sha256"]:
        raise ValueError("Selected model source differs")
    policies = load_config(resolve(settings.policy_config), PolicyComparisonConfig)
    policy = next((p for p in policies.policies if p.name == settings.policy_name), None)
    if policy is None:
        raise ValueError("Configured policy name does not exist")
    candidate = selection["preferred_candidate"]
    reserved = pd.util.hash_pandas_object(
        frame.iloc[splits["test"]].loc[:, ORIGINATION_FEATURES].astype(float), index=False
    )
    return ScoringService(
        model,
        policy,
        candidate,
        selection["selected_methods"][candidate],
        validation["artifacts_sha256"][f"{candidate}_selected.joblib"],
        validation["target_semantics"],
        frozenset(int(x) for x in reserved),
    )

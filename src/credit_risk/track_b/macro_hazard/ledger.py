"""Task10 namespace; population/spec/code binding before fit and single evaluation."""

from credit_risk.track_b.macro.information import feature_hash
from credit_risk.track_b.macro_support.study import immutable_json, read_json
from credit_risk.track_b.multivintage.core import write_json


class Ledger:
    def __init__(self, path, registration):
        if path.name != "task10_evaluation_ledger.json" or path.exists():
            raise ValueError("Task10 requires a new independent evaluation ledger")
        if registration.get("namespace") != "TASK10_MACRO_HAZARD":
            raise ValueError("Wrong evaluation namespace")
        self.path = path
        self.binding = feature_hash(registration)
        self.data = dict(
            state="REGISTERED_BEFORE_FIT",
            registration=registration,
            registration_sha256=self.binding,
            prediction_generation_count=0,
            virgin_holdout=False,
        )
        immutable_json(path, self.data)

    def freeze_models(self, models):
        self.check("REGISTERED_BEFORE_FIT")
        self.data.update(state="MODELS_FROZEN", model_hashes=models)
        write_json(self.path, self.data)

    def check(self, state):
        actual = read_json(self.path)
        if actual["state"] != state or feature_hash(actual["registration"]) != self.binding:
            raise ValueError("Ledger state/specification/population integrity gate failed")
        self.data = actual

    def consume(self, registration):
        self.check("MODELS_FROZEN")
        if feature_hash(registration) != self.binding:
            raise ValueError("Specification or population changed")
        self.data.update(state="CONSUMING", prediction_generation_count=1)
        write_json(self.path, self.data)

    def complete(self, prediction_hashes):
        self.check("CONSUMING")
        self.data.update(state="CONSUMED", prediction_hashes=prediction_hashes)
        write_json(self.path, self.data)

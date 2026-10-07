"""Task6 owns its protocol and predictive-evaluation consumption record."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Ledger:
    def __init__(self, path, protocol):
        self.path = Path(path)
        if self.path.exists():
            raise ValueError("Task6 already started; silent repeat prohibited")
        self.data = dict(
            state="PROTOCOL_FROZEN",
            protocol=protocol,
            protocol_sha256=canonical(protocol),
            created_at=datetime.now(UTC).isoformat(),
            prior_access=(
                "Task4/5 endpoints and evaluation outcomes previously inspected; Task6 "
                "protocol frozen before construction/count audit, metrics sealed"
            ),
            post_evaluation_model_change=False,
        )
        self.protocol_hash = self.data["protocol_sha256"]
        self.save()

    def freeze(self, spec):
        if self.data["state"] != "PROTOCOL_FROZEN":
            raise ValueError("Invalid freeze phase")
        self.data.update(
            state="MODEL_FROZEN",
            specification=spec,
            specification_sha256=canonical(spec),
            count_audit_access=True,
        )
        self.spec_hash = self.data["specification_sha256"]
        self.save()

    def open(self):
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            self.data["state"] != "MODEL_FROZEN"
            or canonical(self.data["specification"]) != self.spec_hash
            or canonical(self.data["protocol"]) != self.protocol_hash
        ):
            raise ValueError("Evaluation freeze/integrity gate failed")
        self.data.update(state="ACCESS_STARTED", accessed_at=datetime.now(UTC).isoformat())
        self.save()

    def complete(self):
        if self.data["state"] != "ACCESS_STARTED":
            raise ValueError("Invalid completion phase")
        self.data.update(
            state="CONSUMED",
            prediction_generation_count=1,
            completed_at=datetime.now(UTC).isoformat(),
        )
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")

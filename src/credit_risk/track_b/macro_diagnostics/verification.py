"""Read-only preservation; no Task10 ledger state transitions or old artifact writes."""

from credit_risk.track_b.data.schemas import digest
from credit_risk.track_b.macro_hazard.verification import preservation as earlier
from credit_risk.track_b.macro_support.study import read_json
from credit_risk.track_b.multivintage.study import lf_hash


def verify(root, source_dir=None):
    manifest = read_json(root / "docs/track_b/macro_signal_preservation_manifest.json")
    for name, expected in manifest["public_lf_hashes"].items():
        if lf_hash(root / name) != expected:
            raise ValueError("Frozen Task2–10 public evidence changed: " + name)
    for name, expected in manifest["private_byte_hashes"].items():
        if digest(root / name) != expected:
            raise ValueError("Frozen private evidence/artifact/prediction changed: " + name)
    ledger = read_json(root / "data/track_b/models/macro_hazard_v1/task10_evaluation_ledger.json")
    if ledger["state"] != "CONSUMED" or ledger["prediction_generation_count"] != 1:
        raise ValueError("Task10 consumed ledger boundary changed")
    return dict(
        status="PASSED",
        public_lf_hashes=len(manifest["public_lf_hashes"]),
        private_byte_hashes=len(manifest["private_byte_hashes"]),
        task10_ledger_unchanged=True,
        task10_ledger_api_called=False,
        old_models_mutated=False,
        earlier=earlier(root, source_dir)
        if source_dir is not None
        else "Hash checks only; source archives checked in final verification",
    )

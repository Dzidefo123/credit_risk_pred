"""Verify Task9 frozen evidence using bytes only; requires retained private files."""

import hashlib
import json
import subprocess
from pathlib import Path

from credit_risk.track_b.data.schemas import digest
from credit_risk.track_b.multivintage.core import write_json
from credit_risk.track_b.multivintage.reporting import verify_preservation
from credit_risk.track_b.multivintage.study import lf_hash
from credit_risk.track_b.pit_macro.acquisition import load_versions


def verify(root, source_dir):
    root = Path(root)
    manifest = json.loads(
        (root / "docs/track_b/pit_macro_preservation_manifest.json").read_text(encoding="utf-8")
    )
    baseline = json.loads(
        (root / "data/track_b/macro/manifests/task9_v1/preservation_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    for name, expected in baseline.items():
        if digest(root / name) != expected:
            raise ValueError("Prior tracked bytes changed: " + name)
    for name, spec in manifest["evidence"].items():
        if lf_hash(root / name) != spec["sha256"]:
            raise ValueError("Prior public LF hash changed: " + name)
    for name, expected in manifest["private_byte_hashes"].items():
        if digest(root / name) != expected:
            raise ValueError("Prior private bytes changed: " + name)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != manifest["base_commit"]:
        raise ValueError("Task9 base commit changed")
    freeze = json.loads(
        (root / "data/track_b/macro/manifests/task9_v1/registry_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    for name, key in [
        ("pit_macro_series_registry.json", "registry_sha256_lf"),
        ("pit_macro_research_protocol.json", "protocol_sha256_lf"),
    ]:
        if lf_hash(root / "docs/track_b" / name) != freeze[key]:
            raise ValueError("Pre-acquisition freeze changed: " + name)
    acquisition, rows = load_versions(root)
    prior = verify_preservation(root, Path(source_dir))
    return dict(
        status="PASSED",
        base_commit=head,
        prior_tracked_byte_hashes=len(baseline),
        prior_public_lf_hashes=len(manifest["evidence"]),
        prior_private_byte_hashes=len(manifest["private_byte_hashes"]),
        prior_preservation=prior,
        acquisition_manifest_sha256=hashlib.sha256(
            (root / "data/track_b/macro/manifests/task9_v1/acquisition.json").read_bytes()
        ).hexdigest(),
        normalized_versions=len(rows),
        frozen_series_registry_sha256=acquisition["registry_sha256_lf"],
        model_or_holdout_loaded=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    arguments = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    result = verify(project, arguments.source_dir)
    write_json(project / "reports/track_b/pit_macro_preservation_check.json", result)
    print(json.dumps(result))

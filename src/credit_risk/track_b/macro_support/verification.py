"""Byte/LF preservation verification; does not load models or consume holdouts."""

import subprocess
from pathlib import Path

from credit_risk.track_b.data.schemas import digest
from credit_risk.track_b.multivintage.reporting import verify_preservation
from credit_risk.track_b.multivintage.study import lf_hash

from .study import read_json


def verify(root, source_dir):
    root = Path(root)
    frozen = read_json(root / "docs/track_b/macro_support_preservation_manifest.json")
    for name, expected in frozen["prior_public_lf_hashes"].items():
        if lf_hash(root / name) != expected:
            raise ValueError("Prior public evidence changed: " + name)
    for name, expected in frozen["task9_private_byte_hashes"].items():
        if digest(root / name) != expected:
            raise ValueError("Task9 private evidence changed: " + name)
    old = read_json(root / "docs/track_b/pit_macro_preservation_manifest.json")
    for name, expected in old["private_byte_hashes"].items():
        if digest(root / name) != expected:
            raise ValueError("Frozen prior private evidence changed: " + name)
    baseline = read_json(root / "data/track_b/macro/manifests/task9_v1/preservation_baseline.json")
    for name, expected in baseline.items():
        if digest(root / name) != expected:
            raise ValueError("Prior tracked bytes changed: " + name)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", frozen["base_commit"], "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    historical = read_json(root / "docs/track_b/macro_historical_coverage_resolution.json")
    if (
        digest(root / historical["fhfa"]["raw_document_path"])
        != historical["fhfa"]["raw_document_sha256"]
    ):
        raise ValueError("Historical publication bytes changed")
    return dict(
        status="PASSED",
        prior_public_lf_hashes=len(frozen["prior_public_lf_hashes"]),
        task9_private_byte_hashes=len(frozen["task9_private_byte_hashes"]),
        older_private_byte_hashes=len(old["private_byte_hashes"]),
        prior_tracked_byte_hashes=len(baseline),
        source_and_frozen_artifact_checks=verify_preservation(root, Path(source_dir)),
        locked_holdout_consumed=False,
        models_loaded_or_regenerated=False,
        task9_original_decision_preserved=True,
        replacement_credentials_accessed=False,
    )

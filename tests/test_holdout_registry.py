"""Cross-run holdout isolation, identity limitations and atomic ledger regressions."""

import json
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from credit_risk.data.validation import ORIGINATION_FEATURES
from credit_risk.validation import holdout_registry as module
from credit_risk.validation.holdout_registry import (
    HoldoutRegistry,
    sample_fingerprints,
    source_fingerprint,
)

SOURCE = "a" * 64
OTHER = "b" * 64


@pytest.fixture
def frame():
    frame = pd.DataFrame({name: [1.0, 1.0, 1.0] for name in ORIGINATION_FEATURES})
    frame["age"] = [30, 40, 50]
    frame["MonthlyIncome"] = [1000.0, 2000.0, 3000.0]
    frame["SeriousDlqin2yrs"] = [0, 1, 0]
    frame["source_row_id"] = [1, 2, 3]
    return frame


@pytest.fixture
def registry(tmp_path):
    return HoldoutRegistry.initialize(tmp_path / "registry.json")


def test_source_identity_is_deterministic_and_independent_of_filename(tmp_path):
    first, copied = tmp_path / "first.csv", tmp_path / "copied.csv"
    first.write_bytes(b"fixed source content")
    copied.write_bytes(first.read_bytes())
    assert source_fingerprint(sha256(first.read_bytes()).hexdigest()) == source_fingerprint(
        sha256(copied.read_bytes()).hexdigest()
    )
    assert source_fingerprint(SOURCE) == source_fingerprint(SOURCE)
    assert source_fingerprint(SOURCE) != source_fingerprint(OTHER)


def test_sample_identity_non_mutation_row_order_target_and_index_independence(frame):
    original = frame.copy(deep=True)
    keys = sample_fingerprints(frame)
    assert keys == sample_fingerprints(frame.iloc[::-1])
    changed = frame.assign(SeriousDlqin2yrs=1, source_row_id=[9, 8, 7])
    changed.index = [99, 99, 101]
    assert keys == sample_fingerprints(changed)
    assert keys == sample_fingerprints(frame[list(reversed(frame.columns))])
    assert_frame_equal(frame, original)
    assert len([k for k in keys if k.startswith("raw-v1:")]) == 3


def test_equivalent_numeric_dtypes_and_missing_representations_match(frame):
    assert sample_fingerprints(frame) == sample_fingerprints(frame.astype(float))
    frame.loc[0, "MonthlyIncome"] = float("nan")
    other = frame.astype("Float64")
    assert sample_fingerprints(frame) == sample_fingerprints(other)


def test_new_holdout_reserved_then_consumed_and_persistent(registry, frame):
    keys = sample_fingerprints(frame)
    assert registry.status(SOURCE) == "available"
    token = registry.reserve(SOURCE, keys, "first-run")
    assert registry.status(SOURCE, keys) == "reserved"
    assert registry.reserve(SOURCE, keys, "first-run") == token
    registry.consume(token)
    loaded = HoldoutRegistry(registry.path)
    assert loaded.status(SOURCE) == loaded.status(OTHER, keys) == "consumed"
    with pytest.raises(ValueError, match="consumed.*first-run"):
        loaded.reserve(OTHER, keys, "second-run")
    with pytest.raises(ValueError, match="one-way"):
        loaded.consume(token)


def test_partial_overlap_reordered_and_relabelled_holdout_rejected(registry, frame):
    registry.consume(registry.reserve(SOURCE, sample_fingerprints(frame.iloc[:2]), "original-run"))
    with pytest.raises(ValueError, match="Fresh final evaluation blocked.*original-run"):
        registry.reserve(
            OTHER,
            sample_fingerprints(frame.iloc[1:].iloc[::-1].assign(SeriousDlqin2yrs=0)),
            "copy-run",
        )


def test_same_source_non_overlapping_profiles_allowed(registry, frame):
    registry.consume(registry.reserve(SOURCE, sample_fingerprints(frame.iloc[:1]), "old"))
    new = sample_fingerprints(frame.iloc[1:])
    assert registry.status(SOURCE) == "consumed"
    assert registry.status(SOURCE, new) == "available"
    assert registry.reserve(SOURCE, new, "new")


def test_new_source_new_samples_allowed(registry, frame):
    registry.consume(registry.reserve(SOURCE, sample_fingerprints(frame.iloc[:1]), "old"))
    assert registry.reserve(OTHER, sample_fingerprints(frame.iloc[1:]), "new")


def test_reserved_holdout_blocks_another_owner(registry, frame):
    keys = sample_fingerprints(frame)
    registry.reserve(SOURCE, keys, "owner")
    with pytest.raises(ValueError, match="reserved.*owner"):
        registry.reserve(SOURCE, keys, "different-owner")


def test_historical_legacy_profile_bridge_blocks_new_raw_profile(registry, frame):
    legacy = [k for k in sample_fingerprints(frame.iloc[:1]) if k.startswith("legacy-")]
    registry.consume(registry.reserve(SOURCE, legacy, "historical-metadata"))
    with pytest.raises(ValueError, match="historical-metadata"):
        registry.reserve(OTHER, sample_fingerprints(frame.iloc[:1].astype(float)), "later-copy")


def test_historical_bridge_version_mismatch_fails_closed(registry, frame, monkeypatch):
    registry.reserve(SOURCE, sample_fingerprints(frame), "old")
    monkeypatch.setattr(pd, "__version__", "unsupported-version")
    with pytest.raises(ValueError, match="version differs"):
        registry.check(OTHER, sample_fingerprints(frame.iloc[:1]))


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "{}",
        '{"schema_version":2,"entries":[]}',
        '{"schema_version":true,"entries":[]}',
        '{"schema_version":1}',
        '{"schema_version":1,"entries":"bad"}',
        '{"schema_version":1,"schema_version":1,"entries":[]}',
    ],
)
def test_corrupt_or_unsupported_registry_fails_closed(registry, frame, payload):
    registry.path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt/unsupported"):
        registry.reserve(SOURCE, sample_fingerprints(frame), "never-evaluate")
    assert registry.path.read_text(encoding="utf-8") == payload


def test_missing_registry_is_not_silently_recreated(tmp_path, frame):
    path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        HoldoutRegistry(path).reserve(SOURCE, sample_fingerprints(frame), "run")
    assert not path.exists()


def test_corrupt_entry_status_and_sample_key_are_rejected(registry, frame):
    registry.reserve(SOURCE, sample_fingerprints(frame), "old")
    payload = json.loads(registry.path.read_text())
    for field, value in [
        ("status", "available"),
        ("samples", ["fake-key"]),
        ("source_fingerprint", OTHER),
    ]:
        bad = json.loads(json.dumps(payload))
        bad["entries"][0][field] = value
        registry.path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupt/unsupported"):
            registry.read()


def test_atomic_replace_failure_preserves_registry_and_cleans_temporary_files(
    registry, frame, monkeypatch
):
    before = registry.path.read_bytes()

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        registry.reserve(SOURCE, sample_fingerprints(frame), "run")
    assert registry.path.read_bytes() == before
    assert list(registry.path.parent.iterdir()) == [registry.path]


def test_concurrent_reservations_have_one_winner(registry, frame):
    def reserve(run):
        try:
            return registry.reserve(SOURCE, sample_fingerprints(frame), run)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reserve, ["one", "two"]))
    assert sum(r is not None for r in results) == 1
    assert len(registry.read().entries) == 1


def test_stale_lock_fails_closed(registry, frame):
    registry.path.with_suffix(".json.lock").write_text("stale", encoding="utf-8")
    busy = HoldoutRegistry(registry.path, lock_timeout=0)
    with pytest.raises(TimeoutError, match="busy/stale"):
        busy.reserve(SOURCE, sample_fingerprints(frame), "never-evaluate")
    assert busy.read().entries == []


def test_consumption_requires_a_real_reservation(registry):
    with pytest.raises(ValueError, match="existing reserved"):
        registry.consume("a" * 64)


@pytest.mark.parametrize("change", ["missing", "empty", "nonfinite", "string", "duplicate"])
def test_invalid_raw_identity_inputs_rejected(frame, change):
    if change == "missing":
        frame = frame.drop(columns="age")
    elif change == "empty":
        frame = frame.iloc[:0]
    elif change == "nonfinite":
        frame.loc[0, "MonthlyIncome"] = float("inf")
    elif change == "string":
        frame["age"] = frame.age.astype(str)
    else:
        frame.columns = ["age"] * len(frame.columns)
    with pytest.raises(ValueError):
        sample_fingerprints(frame)


def test_transformation_is_not_claimed_to_be_detected(registry, frame):
    # Altering every raw value is beyond content-fingerprint matching, not a borrower-ID recovery.
    registry.consume(registry.reserve(SOURCE, sample_fingerprints(frame), "original"))
    altered = frame.copy()
    altered.loc[:, ORIGINATION_FEATURES] = altered.loc[:, ORIGINATION_FEATURES] + 10
    assert registry.status(OTHER, sample_fingerprints(altered)) == "available"


def test_real_historical_seed_is_consumed_without_loading_raw_rows_or_predictions():
    path = Path(__file__).resolve().parents[1] / "reports/holdout_registry.json"
    registry = HoldoutRegistry(path)
    entry = registry.read().entries[0]
    assert entry.status == "consumed" and entry.consumed_at is None
    assert entry.evidence["test_rows"] == "29991"
    assert len(entry.samples) == 29871
    with pytest.raises(ValueError, match="phase4-origination-001"):
        registry.check(entry.source_sha256, [entry.samples[0]])


def test_new_experiment_overlap_blocked_before_loading_or_calibration(
    registry, frame, tmp_path, monkeypatch
):
    from credit_risk.validation import runner

    registry.consume(registry.reserve(SOURCE, sample_fingerprints(frame), "previous-experiment"))
    monkeypatch.setattr(
        runner,
        "verify_experiment",
        lambda *args: ({"source_sha256": SOURCE}, None, frame, {"test": [0, 1, 2]}, None),
    )
    monkeypatch.setattr(
        runner.joblib, "load", lambda *args: pytest.fail("No model loading permitted")
    )
    with pytest.raises(ValueError, match="previous-experiment"):
        runner.run_validation(
            "unread.csv", tmp_path / "new-run", tmp_path / "output", registry_path=registry.path
        )
    assert not (tmp_path / "output").exists()


def test_cli_training_preflight_blocks_consumed_split_before_fit(
    registry, frame, tmp_path, monkeypatch
):
    from credit_risk.cli import main
    from credit_risk.models import pd as modeling

    source = tmp_path / "fixture.csv"
    frame.to_csv(source, index=False)
    digest = sha256(source.read_bytes()).hexdigest()
    registry.consume(registry.reserve(digest, sample_fingerprints(frame), "old"))
    monkeypatch.setattr(module, "default_registry_path", lambda: registry.path)
    monkeypatch.setattr(modeling, "split_origination", lambda *args: {"test": [0, 1, 2]})
    monkeypatch.setattr(
        modeling, "run_origination_experiment", lambda *args: pytest.fail("No fitting allowed")
    )
    assert main(["train", "--csv", str(source), "--output-dir", str(tmp_path / "new-run")]) == 2
    assert not (tmp_path / "new-run").exists()


def test_empty_or_modified_repository_baseline_cannot_become_fresh(registry):
    with pytest.raises(ValueError, match="anchor missing/changed"):
        module.verify_repository_registry(registry.path)
    actual = Path(__file__).resolve().parents[1] / "reports/holdout_registry.json"
    payload = json.loads(actual.read_text(encoding="utf-8"))
    payload["entries"][0]["samples"] = payload["entries"][0]["samples"][1:]
    registry.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="anchor missing/changed"):
        module.verify_repository_registry(registry.path)

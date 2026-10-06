"""ZIP metadata-only inspection; no manual extraction or loan payload reads."""

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from credit_risk.track_b.data.preflight import inspect_zip, record_preflight

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def nested_source(tmp_path):
    source = tmp_path / "historical_data_2010.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_STORED) as outer:
        for q in range(1, 5):
            content = io.BytesIO()
            with zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED) as inner:
                inner.writestr(f"orig_2010Q{q}.txt", "synthetic payload never parsed")
                inner.writestr(f"perf_2010Q{q}.txt", "synthetic payload never parsed")
            outer.writestr(f"historical_data_2010Q{q}.zip", content.getvalue())
    return source


def test_nested_directories_read_without_opening_or_extracting_loan_members(
    nested_source, monkeypatch
):
    before = nested_source.read_bytes()
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda *a, **k: pytest.fail("Loan payload opened"))
    monkeypatch.setattr(
        zipfile.ZipFile, "extractall", lambda *a, **k: pytest.fail("Extraction attempted")
    )
    result = inspect_zip(nested_source)
    assert result["sha256"] == hashlib.sha256(before).hexdigest()
    assert result["member_count"] == 4 and result["records_parsed"] == 0
    assert len(result["archive_inventory"][0]["members"]) == 2
    assert nested_source.read_bytes() == before and not result["copied"] and not result["extracted"]


def test_annual_bundle_stops_without_amending_or_claiming_empirical_results(
    tmp_path, nested_source
):
    for name in [
        "mortgage_research_protocol.json",
        "LONGITUDINAL_DATA_CONTRACT.md",
        "field_dictionary_comparison.json",
    ]:
        p = tmp_path / "docs/track_b" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "docs/track_b" / name, p)
    result = record_preflight(tmp_path, nested_source, official_attestation=True)
    assert result["status"] == "ACQUIRED_PREFLIGHT_STOPPED"
    assert result["feasibility"] == "STOP — DATA/PROTOCOL INCOMPATIBLE"
    assert result["cohort"] is None and result["selected_loans"] is None
    assert result["source_release"] is None and result["actual_column_counts"] is None
    assert not result["loan_payloads_read"] and result["performance_rows_scanned"] == 0
    manifest = json.loads(
        (tmp_path / "data/track_b/manifests/freddie_2010_manifest.json").read_text()
    )
    assert manifest["acquisition_date"] is None and manifest["source_unchanged"]
    assert not (tmp_path / "data/track_b/processed").exists()


def test_flat_metadata_is_not_a_record_schema_verification(tmp_path):
    p = tmp_path / "sample_2010.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("sample_orig_2010.txt", "not loan data")
        z.writestr("sample_perf_2010.txt", "not loan data")
    result = inspect_zip(p)
    assert result["member_count"] == 2 and result["records_parsed"] == 0
    assert result["loan_payloads_read"] is False

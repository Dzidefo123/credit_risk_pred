"""Miniature invented archives: security, resources, ordering and annual sampling."""

import io
import json
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from credit_risk.track_b.data.annual import AnnualReader, GateStop, safe_members
from credit_risk.track_b.data.sampling import select_loans
from credit_risk.track_b.data.schemas import digest

ROOT = Path(__file__).resolve().parents[1]
LIMITS = json.loads((ROOT / "docs/track_b/annual_bundle_acquisition_amendment.json").read_text())[
    "resources"
]


def bundle(tmp_path, compression=zipfile.ZIP_STORED, quarters=(1, 2, 3, 4), inner_extra=None):
    p = tmp_path / "annual.zip"
    with zipfile.ZipFile(p, "w", compression=compression) as outer:
        for q in quarters:
            b = io.BytesIO()
            with zipfile.ZipFile(b, "w", compression=zipfile.ZIP_STORED) as inner:
                inner.writestr(f"orig_2010Q{q}.txt", "synthetic origination; not real records")
                inner.writestr(f"perf_2010Q{q}.txt", "synthetic performance; not real records")
                if inner_extra:
                    inner.writestr(inner_extra, "synthetic")
            outer.writestr(f"historical_data_2010Q{q}.zip", b.getvalue())
    return p


def test_stored_nested_view_requires_no_temporary_quarter(tmp_path):
    source = bundle(tmp_path)
    before = source.read_bytes()
    with AnnualReader(source, tmp_path, LIMITS, digest(source)) as r:
        assert len(r.inventory) == 4
        assert list(r.lines(1, "origination")) == ["synthetic origination; not real records"]
        assert r.temp_bytes_peak == 0 and r.temp_files_created == 0
    assert source.read_bytes() == before


def test_performance_inaccessible_before_frozen_sample_and_no_redraw(tmp_path):
    source = bundle(tmp_path)
    with AnnualReader(source, tmp_path, LIMITS, digest(source)) as r:
        with pytest.raises(GateStop, match="before sample freeze"):
            list(r.lines(1, "performance"))
        r.freeze(["invented-id"])
        assert list(r.lines(1, "performance"))
        with pytest.raises(GateStop, match="replacement"):
            r.freeze(["another-id"])


@pytest.mark.parametrize("quarters", [(1, 2, 3), (1, 2, 3, 3)])
def test_missing_or_duplicate_quarter_rejected(tmp_path, quarters):
    source = bundle(tmp_path, quarters=quarters)
    with pytest.raises(GateStop):
        AnnualReader(source, tmp_path, LIMITS, digest(source))


@pytest.mark.parametrize(
    "extra", ["unexpected.exe", "../escape.txt", "/absolute.txt", "directory/file.txt"]
)
def test_unexpected_executable_or_unsafe_member_rejected(tmp_path, extra):
    source = bundle(tmp_path, inner_extra=extra)
    with pytest.raises(GateStop):
        AnnualReader(source, tmp_path, LIMITS, digest(source))
    assert not (tmp_path / "escape.txt").exists()


def test_compressed_wrapper_cleanup_after_success_and_failure(tmp_path):
    source = bundle(tmp_path, compression=zipfile.ZIP_DEFLATED)
    with AnnualReader(source, tmp_path, LIMITS, digest(source)) as r:
        assert r.temp_files_created == 4 and not list(r.temp_root.glob("*.zip"))
        with pytest.raises(RuntimeError), r.quarter(1):
            raise RuntimeError("synthetic failure")
        assert not list(r.temp_root.glob("*.zip"))
    assert not list((tmp_path / "data/track_b/interim/annual-temporary").glob("*.zip"))


def test_free_disk_guard_before_materialization(tmp_path, monkeypatch):
    source = bundle(tmp_path, compression=zipfile.ZIP_DEFLATED)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=1))
    with pytest.raises(GateStop, match="disk"):
        AnnualReader(source, tmp_path, LIMITS, digest(source))
    assert not list((tmp_path / "data/track_b/interim/annual-temporary").glob("*.zip"))


def test_ratio_size_and_source_hash_guards(tmp_path):
    source = bundle(tmp_path)
    with pytest.raises(GateStop, match="hash"):
        AnnualReader(source, tmp_path, LIMITS, "0" * 64)
    limited = {**LIMITS, "max_quarter_declared_bytes": 1}
    with pytest.raises(GateStop, match="size"):
        AnnualReader(source, tmp_path, limited, digest(source))
    p = tmp_path / "bomb.zip"
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("data.txt", "0" * 100000)
    with zipfile.ZipFile(p) as z, pytest.raises(GateStop, match="ratio"):
        safe_members(z, ["data.txt"], {**LIMITS, "max_compression_ratio": 2})


def test_annual_ranking_has_no_quotas_and_is_invariant_to_rows_quarters_chunks():
    quarters = [[f"F10Q{q}{i:07d}" for i in range(400)] for q in range(1, 5)]
    annual = [i for q in quarters for i in q]
    expected = select_loans(annual)
    assert len(expected) == 1000
    assert expected == select_loans(i for q in reversed(quarters) for i in reversed(q))
    assert expected == select_loans(
        i for start in range(0, len(annual), 73) for i in annual[start : start + 73]
    )
    from hashlib import sha256

    assert (
        sha256("\n".join(sorted(expected)).encode()).hexdigest()
        == sha256("\n".join(sorted(select_loans(reversed(annual)))).encode()).hexdigest()
    )


def test_full_phase_order_and_nonselected_discard_are_fixture_only(tmp_path, monkeypatch):
    from test_track_b_panel import origin, performance

    from credit_risk.track_b.data.annual_run import run_annual

    for name in [
        "mortgage_research_protocol.json",
        "field_dictionary_comparison.json",
        "LONGITUDINAL_DATA_CONTRACT.md",
        "annual_bundle_acquisition_amendment.json",
    ]:
        p = tmp_path / "docs/track_b" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "docs/track_b" / name, p)
    ids = [f"F10Q{q}{i:07d}" for q in range(1, 5) for i in range(260)]
    selected = set(select_loans(ids))
    p = tmp_path / "synthetic_annual.zip"
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_STORED) as outer:
        for q in range(1, 5):
            buffer = io.BytesIO()
            qids = [i for i in ids if i[4] == str(q)]
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as inner:
                inner.writestr(
                    f"orig_2010Q{q}.txt",
                    "\n".join("|".join(origin(i)) for i in reversed(qids)) + "\n",
                )
                rows = ["|".join(performance(k, loan=i)) for i in qids for k in range(20)]
                rows += [
                    i + "|deliberately unparsed nonselected attributes"
                    for i in qids
                    if i not in selected
                ]
                inner.writestr(f"perf_2010Q{q}.txt", "\n".join(rows) + "\n")
            outer.writestr(f"historical_data_2010Q{q}.zip", buffer.getvalue())
    a = tmp_path / "docs/track_b/annual_bundle_acquisition_amendment.json"
    amendment = json.loads(a.read_text())
    amendment["source_sha256"] = digest(p)
    a.write_text(json.dumps(amendment))
    real = AnnualReader.text_stream

    def guarded(reader, q, kind):
        if kind == "performance":
            assert (
                tmp_path / "data/track_b/manifests/annual_2010_v1/sample_manifest.json"
            ).is_file()
            assert reader.sample == frozenset(selected)
        return real(reader, q, kind)

    monkeypatch.setattr(AnnualReader, "text_stream", guarded)
    result = run_annual(tmp_path, p, progress=lambda message: None, source_kind="synthetic_fixture")
    assert (
        result["status"] == "FIXTURE_ONLY"
    )  # Temporary invented fixture run, never published evidence.
    assert result["sample"]["eligible_annual_universe"] == 1040
    assert result["sample"]["selected_count"] == 1000
    assert result["cohort"]["monthly_performance_records"] == 20000
    assert sum(c["performance_rows_scanned"] for c in result["quarter_counts"].values()) == 20840
    assert result["resources"]["temporary_bytes_peak"] == 0
    assert not list((tmp_path / "data/track_b/interim/annual-temporary").glob("*.zip"))


def test_symlink_like_entry_rejected(tmp_path):
    p = tmp_path / "link.zip"
    i = zipfile.ZipInfo("data.txt")
    i.create_system = 3
    i.external_attr = 0o120777 << 16
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(i, "synthetic target")
    with zipfile.ZipFile(p) as z, pytest.raises(GateStop, match="Unsafe"):
        safe_members(z, ["data.txt"], LIMITS)


def test_empirical_aggregate_counts_and_frozen_sample_are_internally_consistent():
    a = json.loads(
        (ROOT / "reports/track_b/freddie_2010_data_audit.json").read_text(encoding="utf-8")
    )
    assert a["status"] == "ACQUIRED" and a["feasibility"] == "PROCEED WITH CONDITIONS"
    assert sum(q["valid_ids"] for q in a["quarter_counts"].values()) == a["annual_universe"]
    assert sum(a["sample"]["selected_by_quarter"].values()) == a["sample"]["selected_count"] == 1000
    assert (
        sum(q["selected_rows_retained"] for q in a["quarter_counts"].values())
        == a["cohort"]["monthly_performance_records"]
    )
    assert (
        sum(a["cohort"]["followup"]["status_counts"].values()) == a["cohort"]["eligible_landmarks"]
    )
    assert sum(a["loan_first_observed_events"].values()) == 1000
    contrast = a["followup_selection_contrast"]
    assert (
        contrast["physical_full12"]["landmarks"] + contrast["not_physical_full12"]["landmarks"]
        == contrast["all_eligible"]["landmarks"]
    )
    for evidence in a["assumptions"].values():
        assert evidence["classification"] in {
            "CONFIRMED",
            "CONFIRMED WITH QUALIFICATION",
            "CONTRADICTED",
            "NOT TESTABLE",
        }
    assert a["sample"]["frozen_before_performance_access"]
    assert (
        a["sample"]["sample_set_sha256"]
        == "b40de7596b0d4fbb189c1ceec5176f4461f7d290117250c1a7ec15b605ce7b11"
    )
    # Aggregate evidence only: no licensed row/ID in repository fixtures.
    assert "selected_ids" not in a

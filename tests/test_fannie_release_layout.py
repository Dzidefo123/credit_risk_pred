"""Task13R uses documented metadata and synthetic opaque records, never private loans."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fannie_release", ROOT / "scripts/fannie_release_layout.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
EXTRACT_SPEC = importlib.util.spec_from_file_location(
    "fannie_document_extractor", ROOT / "scripts/extract_fannie_layout_docs.py"
)
extractor = importlib.util.module_from_spec(EXTRACT_SPEC)
EXTRACT_SPEC.loader.exec_module(extractor)


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


@pytest.fixture
def registry():
    return read("docs/track_b/fannie_release_layout_registry.json")


def layout(registry, width):
    return next(x for x in registry["layouts"] if x["expected_physical_width"] == width)


def record(width):
    return b"|".join([b"opaque"] * width) + b"\n"


def test_exact_113_reproduction(registry):
    candidate = layout(registry, 113)
    line = record(113)
    for _ in range(512):
        result = release.validate_structure(line, candidate["layout_id"], registry)
        assert result["field_count"] == 113
    receipt = read("reports/track_b/fannie_release_archive_structure.json")
    prior = read("reports/track_b/fannie_schema_validation.json")
    assert receipt == prior
    assert receipt["members"][0]["records_structurally_checked"] == 512
    assert not receipt["outcome_values_interpreted"]


def test_current_114_representation(registry):
    current = layout(registry, 114)
    assert (
        release.validate_structure(record(114), current["layout_id"], registry)["field_count"]
        == 114
    )
    assert current["fields"][-1]["position"] == 114
    assert current["fields"][-1]["name"].startswith("Origination VantageScore")


def test_registry_schema(registry):
    assert release.registry_contract(registry) == {"status": "PASSED", "layouts": 6}
    for item in registry["layouts"]:
        assert {
            "layout_id",
            "effective_publication_period",
            "expected_physical_width",
            "field_ordering_source",
            "official_documentation",
            "status",
        } <= item.keys()
    assert registry["sources"]


def test_duplicate_layout_id_rejected(registry):
    registry["layouts"].append(copy.deepcopy(registry["layouts"][0]))
    with pytest.raises(ValueError, match="Duplicate layout"):
        release.registry_contract(registry)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "reorder"])
def test_position_uniqueness(registry, mutation):
    fields = copy.deepcopy(layout(registry, 114)["fields"])
    if mutation == "duplicate":
        fields[1]["position"] = 1
    elif mutation == "missing":
        fields.pop()
    else:
        fields[0], fields[1] = fields[1], fields[0]
    with pytest.raises(ValueError, match="position"):
        release.validate_fields(fields, 114)


def test_applicability_filter_not_physical_width(registry):
    current = layout(registry, 114)
    before = copy.deepcopy(current["fields"])
    semantic = release.sf_fields(current["fields"])
    assert len(semantic) == 72
    assert current["sf_applicability_count"] == 73
    assert current["fields"] == before and len(before) == 114
    assert {1, 47, 77, 78, 109, 110, 112, 113}.isdisjoint({f["position"] for f in semantic})


def test_unknown_applicability_rejected(registry):
    fields = copy.deepcopy(layout(registry, 114)["fields"])
    fields[0]["sf_applicability"] = "GUESS"
    with pytest.raises(ValueError, match="applicability"):
        release.validate_fields(fields, 114)


def test_historical_current_exact_diff(registry):
    old, new = layout(registry, 113), layout(registry, 114)
    diff = release.compare_layouts(old["fields"], new["fields"])
    assert len(diff) == 1
    assert diff[0]["position"] == 114 and diff[0]["classification"] == "FIELD_ADDED_LATER"
    assert old["layout_id"] != new["layout_id"]
    assert old["fields"][-1]["sf_applicability"] == "NOT_APPLICABLE"


@pytest.mark.parametrize("width", [0, 1, 55, 70, 72, 108, 110, 112, 115, 200])
def test_unexpected_width_fails_closed(registry, width):
    with pytest.raises(ValueError, match="width mismatch|Invalid delimiter"):
        release.validate_structure(record(width), layout(registry, 113)["layout_id"], registry)


def test_unknown_layout_fails_even_at_known_width(registry):
    with pytest.raises(ValueError, match="Unknown"):
        release.validate_structure(record(113), "guessed-2010-parser", registry)


def test_incomplete_2020_adapter_disabled(registry):
    with pytest.raises(ValueError, match="unordered"):
        release.validate_structure(record(108), layout(registry, 108)["layout_id"], registry)


def test_no_blank_padding(registry):
    line = record(113)
    original = bytes(line)
    with pytest.raises(ValueError, match="no padding"):
        release.validate_structure(line, layout(registry, 114)["layout_id"], registry)
    assert line == original
    status = release.canonical_metadata(layout(registry, 113))
    assert (
        status["provider_width"] == 113 and status["vantage_score_status"] == "STRUCTURAL_ABSENCE"
    )
    assert status["no_provider_record_mutation"]


def test_ordering_hash_mutation_rejected(registry):
    layout(registry, 113)["fields"][0]["name"] = "Changed"
    with pytest.raises(ValueError, match="Ordering"):
        release.registry_contract(registry)


def test_no_outcome_reader_called(registry, monkeypatch):
    monkeypatch.setattr(release.task13, "event_state", lambda *args: pytest.fail("Outcome access"))
    result = release.validate_structure(record(113), layout(registry, 113)["layout_id"], registry)
    assert set(result) == {"field_count", "line_ending", "ascii"}


@pytest.mark.parametrize(
    "flag",
    [
        "no_outcome_access",
        "no_new_performance_archive_acquisition",
        "no_sample_freeze",
        "no_model_fitting",
        "no_task13a_authorization",
    ],
)
def test_reconciliation_scope(flag):
    report = read("reports/track_b/fannie_release_layout_reconciliation.json")
    assert report[flag] is True


def test_no_downloader_or_model_dependencies():
    for file in ["scripts/fannie_release_layout.py", "scripts/extract_fannie_layout_docs.py"]:
        source = (ROOT / file).read_text()
        assert "urlopen" not in source and "requests." not in source
        assert "sklearn" not in source and "xgboost" not in source
        assert ".fit(" not in source and "extractall" not in source


def test_no_sample_population_created(registry):
    assert not registry["research_authorized"]
    assert registry["protocol_status"] == "DRAFT_NOT_YET_AUTHORIZED"
    result = read("reports/track_b/fannie_release_layout_reconciliation.json")
    assert not result["sample_comparison"]["loan_records_acquired"]
    assert result["sample_comparison"]["status"] == "UNAVAILABLE_HTTP403"


def test_unverified_archive_binding_stops(registry):
    binding = read("reports/track_b/fannie_release_layout_reconciliation.json")["archive_binding"]
    with pytest.raises(ValueError, match="binding unverified"):
        release.research_gate(binding, registry)


def test_verified_binding_still_cannot_authorize_task13a(registry):
    binding = dict(
        provider_release_verified=True, ordering_verified=True, archive_sha256_verified=True
    )
    with pytest.raises(ValueError, match="cannot authorize"):
        release.research_gate(binding, registry)


def test_forbidden_activation(registry):
    registry["research_authorized"] = True
    with pytest.raises(ValueError, match="activation"):
        release.registry_contract(registry)


def test_date_clocks_separate(registry):
    row = next(f for f in registry["date_bound_fields"] if f["position"] == 114)
    assert row["sf_publication"] == "2026-10"
    assert row["sf_activity"] == "2026-05"
    assert row["crt_activity"] == "2026-08"
    assert release.date_mentions("SF: May 2026; CRT: August 2026") == ["2026-05", "2026-08"]


def test_document_table_metadata_only():
    row = [
        114,
        "Synthetic field",
        "definition",
        None,
        "May 2026",
        None,
        "NA",
        "NA",
        "NA",
        "NUMERIC",
        "3",
    ]
    result = extractor.metadata_row(row, "synthetic-document")
    assert result["position"] == 114 and result["sf_applicability"] == "NA"
    assert extractor.metadata_row(["opaque"] * 11, "synthetic-document") is None


def test_primary_predictors_do_not_use_vantagescore():
    protocol = read("docs/track_b/fannie_external_replication_protocol.json")
    assert len(protocol["predictor_mapping"]) == 19
    assert all(114 not in p.get("positions", []) for p in protocol["predictor_mapping"].values())
    assert "orig_vantagescore" not in protocol["predictor_mapping"]


def test_legacy_movement_map():
    mapping = read("docs/track_b/fannie_2020_legacy_field_mapping.json")
    assert len(mapping["moves"]) == 56
    assert mapping["legacy_unique_mapped_positions"] == 55
    assert len(mapping["new_sf_positions"]) == 15
    assert (
        len({m["enhanced_position"] for m in mapping["moves"]} | set(mapping["new_sf_positions"]))
        == 70
    )


def test_documentary_vintage_support_not_population_claim():
    report = read("reports/track_b/fannie_release_layout_reconciliation.json")
    assert [v["year"] for v in report["vintage_availability"]] == [
        2006,
        2008,
        2010,
        2014,
        2018,
        2020,
        2022,
    ]
    assert all(not v["origination_year_coverage_verified"] for v in report["vintage_availability"])
    assert report["license"]["exact_accepted_version"] == "LICENSE_PROVENANCE_UNRESOLVED"


def test_prior_public_hashes_without_private_data():
    frozen = read("docs/track_b/fannie_release_preservation_manifest.json")
    assert len(frozen["public_lf_hashes"]) == 490
    for name, expected in frozen["public_lf_hashes"].items():
        assert (
            hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            == expected
        )


def test_report_sections():
    text = (ROOT / "reports/track_b/FANNIE_RELEASE_LAYOUT_RECONCILIATION.md").read_text(
        encoding="utf-8"
    )
    assert text.count("\n## ") == 17
    assert "FANNIE RELEASE LAYOUT RECONCILED WITH MATERIAL LIMITATIONS" in text

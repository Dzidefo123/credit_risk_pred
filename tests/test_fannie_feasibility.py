"""Synthetic Task13 contracts only: never require or open licensed mortgage data."""

import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fannie_feasibility", ROOT / "scripts/fannie_feasibility.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def load(name):
    return json.loads((ROOT / "docs/track_b" / name).read_text(encoding="utf-8"))


def fixture_root(tmp_path):
    dest = tmp_path / "docs/track_b"
    dest.mkdir(parents=True)
    for name in [
        "freddie_fannie_field_crosswalk.json",
        "fannie_external_replication_protocol.json",
    ]:
        shutil.copyfile(ROOT / "docs/track_b" / name, dest / name)
    return tmp_path


def amend(root, name, mutate):
    path = root / "docs/track_b" / name
    obj = json.loads(path.read_text(encoding="utf-8"))
    mutate(obj)
    path.write_text(json.dumps(obj))


def test_crosswalk_contracts():
    assert audit.check_contracts(ROOT) == {"status": "PASSED", "crosswalk_fields": 56, "models": 6}


def test_duplicate_canonical_rejected(tmp_path):
    root = fixture_root(tmp_path)
    amend(root, "freddie_fannie_field_crosswalk.json", lambda c: c["fields"].append(c["fields"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        audit.check_contracts(root)


def test_direct_unit_mismatch_rejected(tmp_path):
    root = fixture_root(tmp_path)

    def mutate(c):
        f = next(x for x in c["fields"] if x["canonical"] == "orig_interest_rate")
        f["units"]["fannie"] = "fraction"

    amend(root, "freddie_fannie_field_crosswalk.json", mutate)
    with pytest.raises(ValueError, match="units"):
        audit.check_contracts(root)


def test_missing_crosswalk_metadata_rejected(tmp_path):
    root = fixture_root(tmp_path)
    amend(root, "freddie_fannie_field_crosswalk.json", lambda c: c["fields"][0].pop("availability"))
    with pytest.raises(ValueError, match="crosswalk"):
        audit.check_contracts(root)


@pytest.mark.parametrize(
    "state",
    ["AT_RISK", "DEFAULT_PROXY", "PAYOFF_OR_MATURITY", "ADMINISTRATIVE_EXIT", "AMBIGUOUS_EXIT"],
)
def test_event_state_completeness(tmp_path, state):
    root = fixture_root(tmp_path)
    amend(
        root,
        "fannie_external_replication_protocol.json",
        lambda p: p["events"]["states"].remove(state),
    )
    with pytest.raises(ValueError, match="Incomplete event"):
        audit.check_contracts(root)


@pytest.mark.parametrize("code", ["", "01", "02", "03", "06", "09", "15", "16", "96"])
def test_primary_termination_coverage(tmp_path, code):
    root = fixture_root(tmp_path)
    amend(
        root,
        "fannie_external_replication_protocol.json",
        lambda p: p["events"]["termination_mapping"].pop(code),
    )
    with pytest.raises(ValueError, match="termination"):
        audit.check_contracts(root)


@pytest.mark.parametrize(
    ("dlq", "code", "expected"),
    [
        ("00", "", "AT_RISK"),
        ("03", "", "DEFAULT_PROXY"),
        ("99", "", "DEFAULT_PROXY"),
        ("XX", "", "AMBIGUOUS_EXIT"),
        ("", "", "AMBIGUOUS_EXIT"),
        ("00", "01", "PAYOFF_OR_MATURITY"),
        ("03", "01", "AMBIGUOUS_EXIT"),
        ("XX", "02", "DEFAULT_PROXY"),
        ("00", "06", "ADMINISTRATIVE_EXIT"),
        ("03", "06", "DEFAULT_PROXY"),
        ("00", "15", "AMBIGUOUS_EXIT"),
        ("03", "15", "DEFAULT_PROXY"),
        ("00", "16", "ADMINISTRATIVE_EXIT"),
        ("00", "96", "ADMINISTRATIVE_EXIT"),
    ],
)
def test_synthetic_event_semantics(dlq, code, expected):
    assert audit.event_state(dlq, code, "2010-09", "2010-09" if code else "") == expected


def test_date_conflict_first():
    assert audit.event_state("03", "09", "2010-09", "2010-08") == "AMBIGUOUS_EXIT"


@pytest.mark.parametrize("code", ["97", "98", "unknown"])
def test_crt_or_unknown_termination_rejected(code):
    with pytest.raises(ValueError, match="Unknown"):
        audit.event_state("00", code, "2010-09", "2010-09")


@pytest.mark.parametrize(
    ("release", "count"),
    [
        ("glossary-2026-09-10", 113),
        ("guessed-legacy", 113),
        ("guessed-legacy", 114),
    ],
)
def test_unmatched_release_rejected(release, count):
    with pytest.raises(ValueError, match="documentation"):
        audit.release_adapter(release, count)


def test_current_documented_release():
    assert audit.release_adapter("glossary-2026-09-10", 114) == "merged-114"


def test_no_predictor_search_or_missing_coverage():
    p = load("fannie_external_replication_protocol.json")
    old = load("macro_competing_risk_protocol.json")
    assert {m: p["models"][m] for m in ["M0", "M1", "M2"]} == old["ladder"]
    assert p["models"]["P2"] == p["models"]["P1"] + ["REFI_POS", "REFI_NEG"]
    assert p["macro"]["no_reacquisition"]
    assert p["sensitivities"]["no_primary_promotion"]
    assert all(x["classification"] != "UNAVAILABLE" for x in p["predictor_mapping"].values())


def test_unmapped_predictor_rejected(tmp_path):
    root = fixture_root(tmp_path)
    amend(
        root,
        "fannie_external_replication_protocol.json",
        lambda p: p["predictor_mapping"].pop("orig_ltv"),
    )
    with pytest.raises(ValueError, match="predictor"):
        audit.check_contracts(root)


@pytest.mark.parametrize(
    "name",
    [
        "data/track_b/fannie/raw/a.csv",
        "reports/fannie.zip",
        "docs/fannie.parquet",
        ".env",
        "../records.json",
    ],
)
def test_licensed_path_exclusion(name):
    assert not audit.public_path(name)


def test_public_metadata_path():
    assert audit.public_path("reports/track_b/fannie_schema_validation.json")


@pytest.mark.parametrize("name", ["reports/fannie.zip", "data/fannie/raw.csv"])
def test_governance_rejects_tracked_archive_or_data(tmp_path, monkeypatch, name):
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *a, **k: (name + "\0").encode())
    with pytest.raises(ValueError, match="Licensed-data path"):
        audit.governance(tmp_path)


def test_governance_rejects_identifier_in_public_evidence(tmp_path, monkeypatch):
    name = "fannie_evidence.json"
    (tmp_path / name).write_text(json.dumps({"identifier": "1" * 12}))
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *a, **k: (name + "\0").encode())
    with pytest.raises(ValueError, match="exposure"):
        audit.governance(tmp_path)


def test_inspection_limit_cannot_expand(tmp_path):
    p = load("fannie_schema_inspection_policy.json")
    p["maximum_records_per_member"] = 513
    with pytest.raises(ValueError, match="Unbounded"):
        audit.inspect_archive(tmp_path / "2010Q1.zip", p)


def test_identifier_and_row_redaction():
    assert "[REDACTED_ID]" in audit.redact("Loan " + "1" * 12)
    assert "[REDACTED_ID]" in audit.redact("Loan " + "A1" * 6)
    assert "[REDACTED_ID]" in audit.redact(json.dumps({"loan_id": "a" * 12}))
    assert "[REDACTED_ID]" in audit.redact("F10Q1" + "x" * 7)
    assert audit.redact("|".join(["synthetic"] * 113)) == "[REDACTED_ROW]"
    assert audit.redact("a" * 64) == "a" * 64


def test_draft_cannot_activate(tmp_path):
    root = fixture_root(tmp_path)
    amend(
        root,
        "fannie_external_replication_protocol.json",
        lambda p: p.update(outcome_access_authorized=True),
    )
    with pytest.raises(ValueError, match="not authorized"):
        audit.check_contracts(root)


@pytest.mark.parametrize("line", [b"a,b\n", b"a|\xff\n", b"a|b", b"a|\x00\n"])
def test_malformed_structure_fails_without_values(line):
    with pytest.raises(ValueError, match="withheld"):
        audit.structural_line(line)


def test_structural_line_does_not_return_fields():
    assert audit.structural_line(b"opaque|opaque|\r\n") == {
        "field_count": 3,
        "line_ending": "CRLF",
        "ascii": True,
    }


def test_bounded_prefix_never_interprets_events(tmp_path, monkeypatch):
    archive = tmp_path / "2010Q1.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("2010Q1.csv", b"|".join([b"opaque"] * 113) + b"\nmalformed later")
    policy = load("fannie_schema_inspection_policy.json")
    policy["maximum_records_per_member"] = 1
    monkeypatch.setattr(audit, "event_state", lambda *a: pytest.fail("Outcome interpretation"))
    result = audit.inspect_archive(archive, policy)
    assert not result["outcome_values_interpreted"]
    assert not result["full_population_schema_validated"]
    assert result["members"][0]["field_counts"] == [113]
    assert result["members"][0]["records_structurally_checked"] == 1


def test_archive_traversal_no_payload_access(tmp_path):
    archive = tmp_path / "2010Q1.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../2010Q1.csv", "must not open")
    with pytest.raises(ValueError, match="no payload"):
        audit.inspect_archive(archive, load("fannie_schema_inspection_policy.json"))


def test_no_network_or_performance_acquisition():
    source = (ROOT / "scripts/fannie_feasibility.py").read_text(encoding="utf-8")
    assert "urlopen" not in source and "requests." not in source
    assert "extractall" not in source and ".extract(" not in source
    registry = load("fannie_source_registry.json")
    assert registry["no_loan_data_downloaded_by_task13"]
    assert registry["private_archive"]["authorization"].endswith("ONLY")


def test_missingness_and_crt_boundaries():
    fields = {f["canonical"]: f for f in load("freddie_fannie_field_crosswalk.json")["fields"]}
    for n in ["repurchase_date", "interest_bearing_upb", "net_credit_event_loss"]:
        assert fields[n]["classification"] == "UNAVAILABLE"
    assert len(fields["current_upb"]["missing"]["states"]) == 6
    assert (
        fields["orig_vantagescore"]["units"]["canonical"]
        != fields["orig_credit_score"]["units"]["canonical"]
    )


def test_every_declared_freddie_field_has_pinned_position():
    fields = load("freddie_fannie_field_crosswalk.json")["fields"]
    for field in fields:
        if field["freddie"]["field"]:
            assert field["freddie"]["positions"], field["canonical"]


def test_prior_public_preservation_without_private_data():
    import hashlib

    frozen = load("fannie_preservation_manifest.json")
    for name, expected in frozen["public_lf_hashes"].items():
        assert (
            hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            == expected
        )


def test_report_required_sections():
    report = (ROOT / "reports/track_b/FANNIE_EXTERNAL_REPLICATION_FEASIBILITY.md").read_text(
        encoding="utf-8"
    )
    assert report.count("\n## ") == 25
    assert "FANNIE EXTERNAL REPLICATION REQUIRES SOURCE CLARIFICATION" in report

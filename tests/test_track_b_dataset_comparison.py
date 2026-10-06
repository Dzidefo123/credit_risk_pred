"""Documentation-only mortgage field crosswalk completeness and scope checks."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/track_b"


def review():
    return json.loads((DOCS / "field_dictionary_comparison.json").read_text(encoding="utf-8"))


def contract_keys():
    body = (DOCS / "LONGITUDINAL_DATA_CONTRACT.md").read_text(encoding="utf-8")
    result = []
    for part in re.split(r"^### Entity: ", body, flags=re.MULTILINE)[1:]:
        entity = part.splitlines()[0]
        lines = part.splitlines()
        first = next(i for i, line in enumerate(lines) if line.startswith("|"))
        for line in lines[first + 2 :]:
            if not line.startswith("|"):
                break
            result.append((entity, line.split("|")[1].strip()))
    return result


def test_all_77_contract_fields_mapped_once_with_explicit_evidence():
    data = review()
    keys = [(r["entity"], r["field"]) for r in data["fields"]]
    assert len(keys) == 77 and len(set(keys)) == 77 and keys == contract_keys()
    for row in data["fields"]:
        assert row["qualification"] and "pending" not in row
        for provider in ("freddie", "fannie"):
            assert row[provider]["status"] in data["status_definitions"]
            assert row[provider]["locator"]
    payload = (DOCS / "LONGITUDINAL_DATA_CONTRACT.md").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(payload).hexdigest() == data["contract_sha256_lf"]


def test_supported_and_derivable_do_not_assert_empirical_acceptance():
    data = review()
    assert "not verified record quality" in data["status_definitions"]["SUPPORTED"]
    assert "No values derived" in data["status_definitions"]["DERIVABLE"]
    assert data["scope"]["record_level_data_accessed"] is False
    assert data["scope"]["dataset_selected"] is False
    assert data["scope"]["models_implemented"] is False
    assert "Release 47" in data["scope"]["freddie"]
    assert "not CAS/CIRT" in data["scope"]["fannie"]


def test_material_identity_accounting_cashflow_and_revolving_gaps():
    rows = {(r["entity"], r["field"]): r for r in review()["fields"]}
    for key in [
        ("Borrower", "borrower_id"),
        ("Facility / contractual version", "effective_interest_rate"),
        ("Facility / contractual version", "credit_limit"),
        ("Observation / snapshot", "available_limit"),
        ("Recovery / workout cash flow", "cash_flow_id"),
        ("Origination risk / policy reference", "sicr_policy_version"),
    ]:
        assert all(rows[key][p]["status"] == "ABSENT" for p in ("freddie", "fannie"))
    assert (
        rows[("Facility / contractual version", "origination_date")]["freddie"]["status"]
        == "PARTIAL"
    )
    assert (
        rows[("Facility / contractual version", "origination_date")]["fannie"]["status"]
        == "SUPPORTED"
    )
    assert (
        rows[("Default episode / impairment event", "default_date")]["freddie"]["status"]
        == "DERIVABLE"
    )
    assert (
        "not a supplied universal default"
        in rows[("Default episode / impairment event", "default_date")]["qualification"]
    )


def test_fannie_crt_only_fields_are_not_treated_as_public_sf_support():
    rows = {r["field"]: r for r in review()["fields"] if r["entity"] == "Observation / snapshot"}
    assert "71/113 SF NA" in rows["score_or_rating"]["fannie"]["locator"]
    assert rows["score_or_rating"]["fannie"]["status"] == "PARTIAL"
    assert "48/50 SF NA" in rows["payment_history"]["fannie"]["locator"]
    assert rows["arrears_amount"]["fannie"]["status"] == "ABSENT"
    assert "85 SF NA" in rows["arrears_amount"]["fannie"]["locator"]


def test_all_component_decisions_separate_and_qualified():
    data = review()
    rows = {r["component"]: r for r in data["component_assessments"]}
    assert set(rows) == {
        "12-month PD",
        "Lifetime PD",
        "LGD",
        "EAD",
        "SICR",
        "IFRS 9 staging",
        "ECL",
        "Stress testing",
    }
    for row in rows.values():
        assert row["scope"]
        for p in ("freddie", "fannie"):
            assert row[p] in {"SUPPORTED", "PARTIALLY SUPPORTED", "UNSUPPORTED"}
    assert rows["ECL"]["freddie"] == rows["IFRS 9 staging"]["fannie"] == "UNSUPPORTED"
    assert "revolving CCF" in rows["EAD"]["scope"]


@pytest.mark.parametrize(
    "name", ["FIELD_DICTIONARY_COMPARISON.md", "DATASET_COMPARATIVE_ASSESSMENT.md"]
)
def test_comparison_docs_relative_links_and_no_redistributed_data(name):
    body = (DOCS / name).read_text(encoding="utf-8")
    assert "No" in body and "Track" in body
    for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", body):
        url = urlsplit(link)
        if url.scheme:
            assert url.scheme == "https"
            assert url.netloc in {
                "www.freddiemac.com",
                "capitalmarkets.fanniemae.com",
                "github.com",
            }
        else:
            assert (DOCS / url.path).resolve().is_file(), link
    assert "C:\\" not in body


def test_rendered_crosswalk_matches_all_reviewed_rows_and_is_one_table():
    body = (DOCS / "FIELD_DICTIONARY_COMPARISON.md").read_text(encoding="utf-8")
    for row in review()["fields"]:
        expected = f"| {row['entity']} / `{row['field']}` | {row['freddie']['status']} |"
        assert body.count(expected) == 1
    assert not re.search(r"\|[^\n]*\|\n\n\|", body)


def test_source_versions_and_unavailable_hash_are_honest():
    data = review()
    sources = data["sources"]
    assert "September" not in sources["FM_GUIDE"]["version"]
    assert sources["FN_DICT"]["sha256"] is None
    assert "403" in sources["FN_DICT"]["hash_note"]
    for s in sources.values():
        assert s["url"].startswith("https://") and s["reviewed_on"] == data["review_date"]

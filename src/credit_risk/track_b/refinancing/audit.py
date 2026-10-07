"""Evidence availability and t0 coupon audit before any Task12 fit."""

import csv
from collections import Counter

import numpy as np

from credit_risk.track_b.data.schemas import digest
from credit_risk.track_b.macro_diagnostics.verification import verify as earlier
from credit_risk.track_b.macro_hazard.protocol import freeze as old_spec
from credit_risk.track_b.macro_hazard.study import population
from credit_risk.track_b.macro_support.eligibility import ordinal
from credit_risk.track_b.macro_support.study import immutable_json, read_json
from credit_risk.track_b.multivintage.study import lf_hash

from .features import checked_market, gap, proxy_status

PRIVATE = "data/track_b/models/refinancing_v1"


def evidence_status(candidates):
    """Inspected outcomes never qualify, irrespective of vintage or a renamed split."""
    independent = [
        c
        for c in candidates
        if c.get("sealed_before_fit") and c["accessible"] and not c["inspected"]
    ]
    if independent:
        return (
            "INDEPENDENT_EXTERNAL_AVAILABLE"
            if all(c["external"] for c in independent)
            else ("INDEPENDENT_CONFIRMATORY_AVAILABLE")
        )
    return (
        "EXPLORATORY_ONLY"
        if any(c["accessible"] for c in candidates)
        else ("INSUFFICIENT_FOR_NEW_VALIDATION")
    )


def verify(root, source_dir=None):
    manifest = read_json(root / "docs/track_b/refinancing_preservation_manifest.json")
    for name, sha in manifest["public_lf_hashes"].items():
        if lf_hash(root / name) != sha:
            raise ValueError("Frozen public evidence changed: " + name)
    for name, sha in manifest["private_byte_hashes"].items():
        if digest(root / name) != sha:
            raise ValueError("Frozen private evidence changed: " + name)
    return dict(
        status="PASSED",
        public=len(manifest["public_lf_hashes"]),
        private=len(manifest["private_byte_hashes"]),
        earlier=earlier(root, source_dir),
    )


def audit(root):
    verify(root)
    _, development, evaluation, seen = population(root, old_spec(root))
    groups = {
        "development": development,
        "exploratory_seen": evaluation[seen],
        "previously_inspected_unseen": evaluation[~seen],
    }
    table = {
        r["reporting_month"]: r
        for r in read_json(
            root / "data/track_b/macro/processed/task9_api_v5/macro_month_table.json"
        )
    }
    for data in groups.values():
        gap(data)
        for month in np.unique(data["month"]):
            from credit_risk.track_b.macro_support.eligibility import monthly_text

            market = checked_market(table, monthly_text(month))
            if not np.all(data["macro"][data["month"] == month, 3] == market):
                raise ValueError("Frozen row rate differs from checked Task9 PIT market rate")
    candidates = [
        dict(name=k, inspected=True, accessible=True, sealed_before_fit=False, external=False)
        for k in groups
    ]
    candidates += [
        dict(
            name="later_than_2026_02",
            inspected=False,
            accessible=False,
            sealed_before_fit=False,
            external=False,
        ),
        dict(
            name="Fannie_Mae_external",
            inspected=False,
            accessible=False,
            sealed_before_fit=False,
            external=True,
        ),
    ]
    keys = read_json(root / "data/track_b/models/macro_hazard_v1/facility_keys.json")
    lookup = {}
    for group, data in groups.items():
        facilities, first, lengths = np.unique(
            data["facility"], return_index=True, return_counts=True
        )
        for facility, start, length in zip(facilities, first, lengths, strict=True):
            # All source joins refer to previous-month t0, never the outcome month.
            rows = data[start : start + length]
            lookup[keys[facility]] = (
                group,
                set((rows["month"] - 1).tolist()),
                float(rows["numeric"][0, 4]),
            )
    counters = {g: Counter() for g in groups}
    vintage_counters = {}
    inputs = {}
    for vintage in [2006, 2008, 2010, 2014, 2018, 2020, 2022]:
        zone = "processed/v1" if vintage in [2010, 2014, 2018] else "recovery_v1"
        path = root / f"data/track_b/multivintage/{zone}/{vintage}/monthly.csv"
        inputs[path.relative_to(root).as_posix()] = digest(path)
        last, modified = None, False
        local = {g: Counter() for g in groups}
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                loan = row["loan_id"]
                if loan != last:
                    last, modified = loan, False
                flag = row["modification_flag"]
                modified |= flag == "Y"
                key = f"{vintage}:{loan}"
                if key not in lookup:
                    continue
                group, months, rate = lookup[key]
                if ordinal(row["reporting_month"]) not in months:
                    continue
                c = local[group]
                c["matched_t0_intervals"] += 1
                c["modified_by_t0"] += int(modified)
                c["modification_flag_blank"] += int(not flag)
                c["modification_flag_unknown"] += int(flag not in {"", "Y", "N", "P"})
                status = proxy_status(row["current_interest_rate"], rate, modified)
                c[status] += 1
                if row["current_interest_rate"]:
                    try:
                        current = float(row["current_interest_rate"])
                        if np.isfinite(current) and 0 < current < 100:
                            c["current_original_differ"] += int(abs(current - rate) > 1e-6)
                    except ValueError:
                        pass
        for g in groups:
            counters[g].update(local[g])
        vintage_counters[str(vintage)] = {g: dict(v) for g, v in local.items()}
        print("Task12 predictor-only coupon audit", vintage, flush=True)
    for g, data in groups.items():
        if counters[g]["matched_t0_intervals"] != len(data):
            raise ValueError("Incomplete t0 modification/current-rate audit")
    output = dict(
        validation_status=evidence_status(candidates),
        candidates=candidates,
        no_new_ledger=True,
        virgin_holdout=False,
        old_ledger_api_called=False,
        unselected_source_records=(
            "Potential future population, not audited/sealed "
            "independent validation in this experiment"
        ),
        contract_field="orig_interest_rate",
        units="percent; difference percentage points",
        rate_semantics="ORIGINAL_CONTRACT_RATE_PROXY_GAP",
        current_rate_field="current_interest_rate",
        modification_field="modification_flag",
        knowledge_time="Freddie historical knowledge time UNVERIFIED; nominal t0 only",
        current_rate_primary=False,
        current_rate_reason=(
            "Harmonized field exists but historical release/modification timing unverified"
        ),
        modification_audit={g: dict(v) for g, v in counters.items()},
        modification_by_vintage=vintage_counters,
        input_hashes=inputs,
        macro_table_sha256=digest(
            root / "data/track_b/macro/processed/task9_api_v5/macro_month_table.json"
        ),
    )
    immutable_json(root / PRIVATE / "audit.json", output)
    return output

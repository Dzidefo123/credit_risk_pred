"""Replay Task9A interval membership, project static predictors, seal risk arrays."""

import csv
import hashlib
from collections import Counter

import numpy as np

from credit_risk.track_b.data.schemas import digest
from credit_risk.track_b.macro_support.eligibility import (
    eligible_intervals,
    ordinal,
    validation_role,
)
from credit_risk.track_b.macro_support.study import histories, immutable_json, read_json
from credit_risk.track_b.pit_macro.engine import previous_month_end

from .protocol import CATEGORICAL, NUMERIC

MACRO = (
    "unemployment_level",
    "unemployment_change_3m",
    "treasury_10y_level",
    "mortgage_30y_level",
    "mortgage_treasury_spread",
    "hpi_yoy",
    "cpi_yoy",
    "gdp_qoq",
)
DTYPE = np.dtype(
    [
        ("numeric", "f8", (6,)),
        ("categories", "U16", (2,)),
        ("macro", "f8", (8,)),
        ("facility", "i4"),
        ("vintage", "i2"),
        ("duration", "i2"),
        ("month", "i4"),
        ("event", "u1"),
        ("role", "u1"),
    ]
)
EVENT = {"none": 0, "default": 1, "payoff": 2}


def macro_join(table, month, needed):
    if month > "2026-02" or month not in table:
        raise ValueError("Macro join beyond frozen eligible reporting boundary")
    t0 = str(previous_month_end(month))
    features = table[month]["features"]
    for name in needed:
        feature = features[name]
        if feature["status"] != "AVAILABLE" or feature.get("t0") != t0 or not feature.get("inputs"):
            raise ValueError("Unavailable/unverified PIT feature")
        for source in feature["inputs"]:
            if source.get("representation") != "vintage" or source.get("current_revised", False):
                raise ValueError("Current-revised macro prohibited")
            if not source["archive_start"] <= t0 <= source["archive_end"] or any(
                source[n] is None or source[n] > t0
                for n in ["reference_period", "publication_upper_bound", "revision_upper_bound"]
            ):
                raise ValueError("Future macro information at interval assessment")
    return [
        features[n].get("value") if features[n].get("value") is not None else np.nan for n in MACRO
    ]


def block(month, role, design, spec):
    key = "development" if role == "development" else "evaluation"
    lower, upper = spec["splits"][
        "reduced_development" if key == "development" and design == "REDUCED" else key
    ]
    return key if lower <= month <= upper else None


def prepare(root, spec):
    private = root / "data/track_b/models/macro_hazard_v1"
    private.mkdir(parents=True, exist_ok=True)
    if (private / "data_audit.json").exists():
        raise ValueError("Risk construction already sealed; use existing audited arrays")
    support = read_json(root / "data/track_b/macro_support/support.json")
    expected = read_json(root / "data/track_b/macro_support/counts.json")
    retained = read_json(root / "data/track_b/macro/processed/task9_api_v5/macro_month_table.json")
    mapping = {r["reporting_month"]: r for r in retained}
    vectors = {
        month: macro_join(
            mapping,
            month,
            support["designs"][
                "PRIMARY" if month in support["designs"]["PRIMARY"]["support_months"] else "REDUCED"
            ]["features"],
        )
        for month in support["designs"]["REDUCED"]["support_months"]
    }
    sizes = {
        "development": expected["validation_counts"]["REDUCED"]["development"]["intervals"],
        "evaluation": expected["validation_counts"]["PRIMARY"]["temporal_evaluation"]["intervals"],
    }
    arrays = {
        k: np.lib.format.open_memmap(private / (k + ".npy"), mode="w+", dtype=DTYPE, shape=(n,))
        for k, n in sizes.items()
    }
    positions = Counter()
    hashes = {n: hashlib.sha256() for n in support["designs"]}
    global_counts = {n: Counter() for n in support["designs"]}
    facility_keys = []
    missing = {k: Counter() for k in arrays}
    for vintage in [2006, 2008, 2010, 2014, 2018, 2020, 2022]:
        zone = "processed/v1" if vintage in [2010, 2014, 2018] else "recovery_v1"
        folder = root / f"data/track_b/multivintage/{zone}/{vintage}"
        with (folder / "origination.csv").open(encoding="utf-8", newline="") as stream:
            original = {r["loan_id"]: r for r in csv.DictReader(stream)}
        for loan, history in histories(folder / "monthly.csv"):
            key = f"{vintage}:{loan}"
            facility = len(facility_keys)
            facility_keys.append(key)
            origin = original[loan]
            numeric = np.array([float(origin[n]) if origin[n] else np.nan for n in NUMERIC])
            if numeric[3] < 0:
                raise ValueError("Negative origination UPB")
            numeric[3] = np.log1p(numeric[3])
            categories = [origin[n] or "__MISSING__" for n in CATEGORICAL]
            if any(len(c) > 16 for c in categories):
                raise ValueError("Category storage truncation")
            role = validation_role(vintage, loan, spec["splits"]["role_salt"])
            for name, design in support["designs"].items():
                risk = eligible_intervals(
                    history, mapping, design["features"], support["reporting_cutoff"]
                )
                global_counts[name]["facilities"] += bool(risk)
                global_counts[name]["intervals"] += len(risk)
                global_counts[name]["defaults"] += bool(risk and risk[-1]["event"] == "default")
                global_counts[name]["payoffs"] += bool(risk and risk[-1]["event"] == "payoff")
                for row in risk:
                    month = row["target_month"]
                    hashes[name].update(f"{vintage}:{loan}:{row['t0']}:{month}\n".encode())
                    split = block(month, role, name, spec)
                    # Store reduced development once; primary is its frozen calendar subset.
                    if (
                        split is None
                        or (split == "development" and name != "REDUCED")
                        or (split == "evaluation" and name != "PRIMARY")
                    ):
                        continue
                    values = vectors[month]
                    array = arrays[split]
                    i = positions[split]
                    if i >= len(array):
                        raise ValueError("Task9A count exceeded")
                    array[i] = (
                        numeric,
                        categories,
                        values,
                        facility,
                        vintage,
                        row["duration"],
                        ordinal(month),
                        EVENT[row["event"]],
                        split == "evaluation",
                    )
                    positions[split] += 1
                    for n, value in zip(NUMERIC, numeric, strict=True):
                        missing[split][f"{vintage}:{n}"] += int(np.isnan(value))
        print(f"Task10 risk replay {vintage}: frozen intervals and static predictors", flush=True)
    for k, array in arrays.items():
        if positions[k] != len(array):
            raise ValueError("Task9A split count mismatch")
        array.flush()
    if {n: h.hexdigest() for n, h in hashes.items()} != expected["interval_key_sha256"]:
        raise ValueError("Task9A full risk membership did not reproduce")
    immutable_json(private / "facility_keys.json", facility_keys)
    result = dict(
        full_population_counts={n: dict(c) for n, c in global_counts.items()},
        full_interval_fingerprints={n: h.hexdigest() for n, h in hashes.items()},
        stored_rows=dict(positions),
        missing_numeric_by_split_vintage={k: dict(c) for k, c in missing.items()},
        array_byte_hashes={k: digest(private / (k + ".npy")) for k in arrays},
        facility_keys_sha256=digest(private / "facility_keys.json"),
        static_origination_only=True,
        future_loan_states_used=False,
    )
    immutable_json(private / "data_audit.json", result)
    return result


def load(root):
    private = root / "data/track_b/models/macro_hazard_v1"
    audit = read_json(private / "data_audit.json")
    for k, expected in audit["array_byte_hashes"].items():
        if digest(private / (k + ".npy")) != expected:
            raise ValueError("Frozen Task10 risk data changed")
    if digest(private / "facility_keys.json") != audit["facility_keys_sha256"]:
        raise ValueError("Facility population keys changed")
    return {
        k: np.load(private / (k + ".npy"), mmap_mode="r") for k in ["development", "evaluation"]
    }

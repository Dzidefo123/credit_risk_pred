"""Descriptive selected-history audit; no outcome modeling or prediction access."""

import csv
import sqlite3
from collections import Counter

from credit_risk.track_b.data.schemas import ORIGINATION, PERFORMANCE, load_protocol

from .core import age_band, fresh_counts, horizon_support, missing_tally, quantiles, trajectory
from .schema import parse


def stringify(value):
    return "" if value is None else str(value)


def audit(root, output, cache, ids, vintage, plan):
    protocol, _ = load_protocol(root)
    origdb = sqlite3.connect(
        (output / "origination.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    perfdb = sqlite3.connect(cache.resolve().as_uri() + "?mode=ro", uri=True)
    missing = fresh_counts()
    facility_any = Counter()
    facility_all = Counter()
    endpoints = Counter()
    integrity = Counter()
    spans = []
    prefixes = []
    lengths = []
    ages = []
    support = Counter()
    known = Counter()
    age_calendar = Counter()
    source_calendar = Counter()
    age_residual = Counter()
    distributions = {
        k: []
        for k in [
            "orig_credit_score",
            "orig_ltv",
            "orig_dti",
            "orig_upb",
            "orig_interest_rate",
            "original_loan_term",
        ]
    }
    categories = {k: Counter() for k in ["loan_purpose", "occupancy_status"]}
    first_months = []
    last_months = []
    rows_total = found = 0
    extras = [
        "vintage",
        "event_category",
        "post_research_endpoint",
        "post_source_termination",
        "analytical_prefix",
        "pandemic_regime",
        "months_since_first_payment_proxy",
    ]
    with (
        (output / "origination.csv").open("w", encoding="utf-8", newline="") as os,
        (output / "monthly.csv").open("w", encoding="utf-8", newline="") as ps,
    ):
        ow = csv.DictWriter(os, fieldnames=["vintage", *ORIGINATION], lineterminator="\n")
        pw = csv.DictWriter(ps, fieldnames=[*PERFORMANCE, *extras], lineterminator="\n")
        ow.writeheader()
        pw.writeheader()
        for number, loan in enumerate(sorted(ids), 1):
            raworig = origdb.execute("SELECT raw FROM origination WHERE loan=?", (loan,)).fetchone()
            if raworig is None:
                raise ValueError("Selected origination linkage missing")
            otokens = raworig[0].split("|")
            orig = parse(otokens, "origination", vintage)
            ow.writerow(dict(vintage=vintage, **{k: stringify(orig[k]) for k in ORIGINATION}))
            omissing = set()
            missing_tally(otokens, ORIGINATION, missing, omissing)
            facility_any.update(omissing)
            facility_all.update(omissing)
            for k in distributions:
                if orig[k] is not None:
                    distributions[k].append(orig[k])
            for k in categories:
                categories[k][orig[k]] += 1
            rawrows = [
                r[0]
                for r in perfdb.execute(
                    "SELECT raw FROM performance WHERE loan=? ORDER BY rowid", (loan,)
                )
            ]
            parsed = [parse(line.split("|"), "performance", vintage) for line in rawrows]
            info = trajectory(
                parsed, protocol["event"], plan["horizons"], plan["horizon_support_rule"]
            )
            endpoints[info["endpoint"]] += 1
            integrity.update(info["findings"])
            spans.append(info["observed_span"])
            prefixes.append(info["risk_exit"])
            lengths.append(info.get("recorded_months", 0))
            if not parsed:
                integrity["selected_ids_missing_performance"] += 1
                for k in PERFORMANCE:
                    facility_all[k] += 1
                    facility_any[k] += 1
                continue
            found += 1
            first_months.append(info["first"])
            last_months.append(info["last"])
            support.update({str(h): int(info["at_risk"][h]) for h in plan["horizons"]})
            known.update({str(h): int(info["support"][h]) for h in plan["horizons"]})
            per_facility = fresh_counts()
            anymissing = set()
            for raw in rawrows:
                temp = set()
                missing_tally(raw.split("|"), PERFORMANCE, per_facility, temp)
                anymissing.update(temp)
            for k, c in per_facility.items():
                missing[k].update(c)
                facility_all[k] += c.get("OBSERVED", 0) == 0
            facility_any.update(anymissing)
            for row, ann in zip(
                sorted(parsed, key=lambda r: r["reporting_month"]), info["annotations"], strict=True
            ):
                month = row["reporting_month"]
                age = row["loan_age"]
                source_calendar[str(month.year)] += 1
                if age is not None:
                    ages.append(age)
                    residual = month.ordinal - orig["first_payment_month"].ordinal - age
                    age_residual[str(int(residual))] += 1
                if ann["analytical_prefix"] and age is not None and age >= 0:
                    age_calendar[(str(month.year), age_band(age))] += 1
                pw.writerow(
                    {
                        **{k: stringify(row[k]) for k in PERFORMANCE},
                        "vintage": vintage,
                        **ann,
                        "months_since_first_payment_proxy": month.ordinal
                        - orig["first_payment_month"].ordinal,
                    }
                )
                rows_total += 1
            if number % 5000 == 0:
                print(f"{vintage} audit {number:,}/{len(ids):,}", flush=True)
    origdb.close()
    perfdb.close()
    field_missing = {
        k: dict(
            row_states=dict(c),
            facilities_with_any_nonobserved=int(facility_any[k]),
            facilities_with_all_nonobserved=int(facility_all[k]),
            structurally_unavailable=False,
        )
        for k, c in sorted(missing.items())
    }
    return dict(
        vintage=vintage,
        facilities=len(ids),
        selected_ids_found=found,
        selected_ids_missing=len(ids) - found,
        rows=rows_total,
        integrity=dict(integrity),
        first_endpoints=dict(endpoints),
        field_missingness=field_missing,
        distributions={k: quantiles(v) for k, v in distributions.items()},
        categorical_distributions={k: dict(c) for k, c in categories.items()},
        observed_month_span=quantiles(spans),
        recorded_months=quantiles(lengths),
        contiguous_analytical_followup=quantiles(prefixes),
        provider_age=quantiles(ages),
        calendar=dict(
            first=min(first_months) if first_months else None,
            last=max(last_months) if last_months else None,
            source_rows_by_year=dict(sorted(source_calendar.items())),
        ),
        provider_age_vs_first_payment_proxy_residual=dict(sorted(age_residual.items())),
        age_calendar_support=[
            dict(calendar_year=y, provider_age_band=a, rows=n)
            for (y, a), n in sorted(age_calendar.items())
        ],
        horizons={
            str(h): dict(
                at_risk=int(support[str(h)]),
                known_status=int(known[str(h)]),
                status=horizon_support(
                    support[str(h)], known[str(h)], plan["horizon_support_rule"]
                ),
            )
            for h in plan["horizons"]
        },
        source_operational_knowledge_time="UNVERIFIED; retrospective current-release disclosure",
        borrower_independence="NOT ESTABLISHED; facility key is not borrower key",
    )

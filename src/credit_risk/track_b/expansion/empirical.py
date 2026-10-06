"""One-shot expansion scan; unchanged panel/cohort semantics, bounded disk spool."""

import csv
import hashlib
import heapq
import json
import sqlite3
import subprocess
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from credit_risk.track_b.data.annual import (
    AnnualReader,
    GateStop,
    load_amendment,
    peak_memory_bytes,
)
from credit_risk.track_b.data.freddie import parse_row
from credit_risk.track_b.data.panel import build_panel, event_category
from credit_risk.track_b.data.sampling import SALT
from credit_risk.track_b.data.schemas import FEATURES, digest
from credit_risk.track_b.pd.baseline import cohort, split

from .planning import SAMPLE, SOURCE, lf_hash


def ranking(identifiers, n):
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 20000:
        raise GateStop("Invalid expansion size")
    ids = set(identifiers)
    if any(not isinstance(i, str) or not i for i in ids):
        raise GateStop("Identifier-only selection required")
    if len(ids) < n:
        raise GateStop("Insufficient identifier universe")
    return heapq.nsmallest(
        n, ids, key=lambda i: (hashlib.sha256((SALT + ":" + i).encode()).hexdigest(), i)
    )


def set_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def nested(ids, original):
    if (
        len(original) != 1000
        or set_hash(original) != SAMPLE
        or set(ids[:1000]) != set(original)
        or not set(original) <= set(ids)
    ):
        raise GateStop("Original first-1000 preservation failed")


def load_frozen_plan(root):
    root = Path(root)
    path = root / "docs/track_b/sample_expansion_amendment.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    marker = root / "data/track_b/manifests/expansion_v1/phase_a_test_attestation.json"
    if not marker.exists():
        raise GateStop("Phase A tests not attested; no new outcomes accessible")
    attested = json.loads(marker.read_text(encoding="utf-8"))
    if attested.get("passed") is not True or attested.get("amendment_sha256_lf") != lf_hash(path):
        raise GateStop("Phase A test/hash gate failed")
    for name, expected in attested["code_sha256_lf"].items():
        if lf_hash(root / name) != expected:
            raise GateStop("Tested expansion code changed")
    if (
        plan["chosen_n"] != 20000
        or plan["phase"] != "A_FROZEN"
        or plan["models_permitted"] is not False
    ):
        raise GateStop("Frozen expansion design changed")
    for name, expected in plan["planning_inputs_sha256_lf"].items():
        if lf_hash(root / name) != expected:
            raise GateStop("Prior aggregate evidence changed")
    return plan, lf_hash(path)


class ExpansionReader(AnnualReader):
    def __init__(self, source, root, limits, expected_hash, plan):
        self.chosen_n = plan["chosen_n"]
        self.scanned_performance = set()
        super().__init__(source, root, limits, expected_hash)

    def freeze(self, identifiers):
        ids = frozenset(identifiers)
        if len(ids) != self.chosen_n:
            raise GateStop("Frozen N mismatch")
        if self.sample is not None and self.sample != ids:
            raise GateStop("Adaptive replacement prohibited")
        self.sample = ids

    def lines(self, q, kind):
        if kind == "performance":
            if self.sample is None:
                raise GateStop("Performance inaccessible before ID freeze")
            if q in self.scanned_performance:
                raise GateStop("Second performance scan prohibited")
            self.scanned_performance.add(q)
        yield from super().lines(q, kind)


class Tally:
    def __init__(self):
        self.rows = 0
        self.loans = set()
        self.positives = 0
        self.events = set()

    def add(self, f):
        self.rows += len(f)
        self.loans.update(f.loan_id)
        positive = f.binary_default_12m.eq(1)
        self.positives += int(positive.sum())
        self.events.update(f.loc[positive, "loan_id"])

    def result(self):
        return dict(
            landmarks=self.rows,
            loans=len(self.loans),
            positive_landmarks=self.positives,
            default_loans=len(self.events),
        )


def decision(development, evaluation, blocking=False):
    if blocking:
        return "EXPANSION SUPPORT INADEQUATE"
    if development >= 150 and evaluation >= 50:
        return "EXPANSION SUPPORT ADEQUATE"
    if development >= 100 and evaluation >= 30:
        return "EXPANSION SUPPORT MARGINAL"
    return "EXPANSION SUPPORT INADEQUATE"


def normalized(f):
    f = f.sort_values(["loan_id", "t0"]).reset_index(drop=True).copy()
    for name in f:
        if (
            name in ["binary_default_12m", "event_offset", "observed_followup_months"]
            or name in FEATURES
            and name
            not in [
                "delinquency_state",
                "occupancy_status",
                "property_state",
                "property_type",
                "loan_purpose",
                "amortization_type",
                "modification_flag",
                "assistance_plan",
                "payment_deferral_flag",
            ]
        ):
            f[name] = pd.to_numeric(f[name], errors="raise").astype("Float64")
        else:
            f[name] = f[name].astype("string").fillna("")
    return f


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(root, source, progress=print):
    root, source = Path(root).resolve(), Path(source).resolve()
    plan, plan_hash = load_frozen_plan(root)
    protocol, protocol_hash, old_amendment, _ = load_amendment(root)
    if protocol_hash != plan["base_protocol_sha256_lf"]:
        raise GateStop("Scientific protocol changed")
    private = root / "data/track_b/manifests/expansion_v1"
    output = root / "data/track_b/processed/expansion_v1"
    if (private / "scan_started.json").exists() or output.exists():
        raise GateStop("One-shot scan already started; no adaptive retry or overwrite")
    # Preserve all pre-existing tracked evidence and private Task 2/3 inputs.
    names = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], text=True).split(
        "\0"
    )
    old_paths = [
        root / name
        for name in names
        if name
        and not name.startswith(
            (
                "src/credit_risk/track_b/expansion/",
                "docs/track_b/SAMPLE_EXPANSION",
                "docs/track_b/sample_expansion",
                "reports/track_b/SAMPLE_EXPANSION",
                "reports/track_b/sample_expansion",
                "tests/test_track_b_expansion",
                "scripts/run_track_b_expansion",
            )
        )
    ]
    old_paths.extend((root / "data/track_b/manifests/annual_2010_v1").glob("*"))
    old_panel = root / "data/track_b/processed/annual_2010_v1/panel.csv"
    old_paths.append(old_panel)
    preserved = {str(p.relative_to(root)): digest(p) for p in old_paths if p.is_file()}
    original = (
        (root / "data/track_b/manifests/annual_2010_v1/selected_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    if set_hash(original) != SAMPLE:
        raise GateStop("Original sample changed")
    output.mkdir(parents=True)
    limits = {
        **old_amendment["resources"],
        **{k: v for k, v in plan["limits"].items() if k in old_amendment["resources"]},
    }
    started = time.monotonic()
    universe = {}
    quarters = {}
    selected = {}
    quarter_counts = {}
    db = sqlite3.connect(output / "retained_performance.sqlite")
    db.execute("CREATE TABLE performance (loan TEXT NOT NULL, raw TEXT NOT NULL)")
    raw_rows = 0
    found = set()
    with ExpansionReader(source, root, limits, SOURCE, plan) as reader:
        for q in range(1, 5):
            count = 0
            for line in reader.lines(q, "origination"):
                tokens = next(csv.reader([line], delimiter="|", strict=True))
                row = parse_row(tokens, "origination")
                loan = row["loan_id"]
                if loan in universe or loan[4] != str(q):
                    raise GateStop("Duplicate/quarter mismatch in universe")
                universe[loan] = q
                count += 1
            quarter_counts[str(q)] = dict(origination_rows=count)
            progress(f"Origination Q{q}: {count:,} IDs")
        if len(universe) != 1820190:
            raise GateStop("Annual identifier universe changed")
        ids = ranking(universe, plan["chosen_n"])
        nested(ids, original)
        quarters = dict(Counter(str(universe[i]) for i in ids))
        sample = dict(
            chosen_n=len(ids),
            sample_set_sha256=set_hash(ids),
            original_sample_sha256=SAMPLE,
            original_first_1000_exact=True,
            original_subset=True,
            annual_universe=len(universe),
            selected_by_quarter=quarters,
            algorithm=plan["sampling"],
            amendment_sha256_lf=plan_hash,
            source_sha256=SOURCE,
            frozen_at=datetime.now(UTC).isoformat(),
            frozen_before_performance_access=True,
        )
        write_json(private / "sample_manifest.json", sample)
        (private / "selected_ids.txt").write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
        reader.freeze(ids)
        progress(f"Expanded IDs frozen: {len(ids):,}; SHA256={sample['sample_set_sha256']}")
        for q in range(1, 5):
            for line in reader.lines(q, "origination"):
                tokens = next(csv.reader([line], delimiter="|", strict=True))
                if tokens[19].strip() in reader.sample:
                    row = parse_row(tokens, "origination")
                    selected[row["loan_id"]] = row
        # This durable marker makes a second source-performance scan fail closed.
        write_json(
            private / "scan_started.json",
            dict(
                amendment_sha256_lf=plan_hash,
                sample_manifest_sha256=digest(private / "sample_manifest.json"),
                started_at=datetime.now(UTC).isoformat(),
            ),
        )
        for q in range(1, 5):
            scanned = retained = 0
            batch = []
            for line in reader.lines(q, "performance"):
                scanned += 1
                loan = line.partition("|")[0].strip()
                if universe.get(loan) != q:
                    raise GateStop("Unmatched/quarter-mismatched performance ID")
                if loan in reader.sample:
                    parse_row(next(csv.reader([line], delimiter="|", strict=True)), "performance")
                    batch.append((loan, line))
                    found.add(loan)
                    retained += 1
                    raw_rows += 1
                    if raw_rows > plan["limits"]["max_retained_performance_rows"]:
                        raise GateStop("Retention row cap exceeded; no truncation")
                    if len(batch) >= 5000:
                        db.executemany("INSERT INTO performance VALUES (?,?)", batch)
                        db.commit()
                        batch = []
                if scanned % 1000000 == 0:
                    memory = peak_memory_bytes()
                    if memory and memory > plan["limits"]["max_peak_memory_bytes"]:
                        raise GateStop("Memory cap exceeded")
                    if (output / "retained_performance.sqlite").stat().st_size > plan["limits"][
                        "max_cache_bytes"
                    ]:
                        raise GateStop("Cache disk cap exceeded")
                    progress(f"Performance Q{q}: {scanned:,} scanned / {retained:,} retained")
            if batch:
                db.executemany("INSERT INTO performance VALUES (?,?)", batch)
                db.commit()
            quarter_counts[str(q)].update(
                performance_rows_scanned=scanned, performance_rows_retained=retained
            )
        temporary_peak = reader.temp_bytes_peak
        scan_seconds = time.monotonic() - started
    del universe
    db.execute("CREATE INDEX loan_index ON performance(loan)")
    db.commit()
    old = pd.read_csv(
        old_panel,
        dtype={
            c: "string"
            for c in [
                "loan_id",
                "t0",
                "delinquency_state",
                "modification_flag",
                "assistance_plan",
                "payment_deferral_flag",
                "occupancy_status",
                "property_state",
                "property_type",
                "loan_purpose",
                "amortization_type",
            ]
        },
        low_memory=False,
    )
    old_groups = {loan: g for loan, g in old.groupby("loan_id", sort=False)}
    totals = {
        k: Tally()
        for k in [
            "panel",
            "eligible",
            "primary",
            "development",
            "evaluation",
            "purged_2015",
            "unused_other",
        ]
    }
    status = defaultdict(Tally)
    calendar = defaultdict(Tally)
    findings = Counter()
    first_events = Counter()
    raw_default = set()
    panel_path = output / "panel.csv"
    old_equivalent = 0
    with panel_path.open("w", encoding="utf-8", newline="") as stream:
        for number, loan in enumerate(sorted(ids), 1):
            records = [
                parse_row(next(csv.reader([raw], delimiter="|", strict=True)), "performance")
                for (raw,) in db.execute(
                    "SELECT raw FROM performance WHERE loan=? ORDER BY rowid", (loan,)
                )
            ]
            categories = [
                event_category(r, protocol["event"])
                for r in sorted(records, key=lambda r: r["reporting_month"])
            ]
            first_events[
                next(
                    (
                        c
                        for c in categories
                        if c in {"default", "payoff", "administrative", "ambiguous"}
                    ),
                    "active_or_unknown",
                )
            ] += 1
            if "default" in categories:
                raw_default.add(loan)
            panel, integrity = build_panel({loan: selected[loan]}, records, protocol)
            findings.update(integrity["findings"])
            panel.to_csv(stream, index=False, header=number == 1, lineterminator="\n")
            if loan in old_groups:
                pd.testing.assert_frame_equal(
                    normalized(panel), normalized(old_groups[loan]), check_dtype=True
                )
                old_equivalent += 1
            primary = cohort(panel)
            dev, ev = split(primary)
            eligible = panel.loc[panel.eligible]
            purge = primary.loc[primary.t0.str[:4].eq("2015")]
            used = set(dev.t0) | set(ev.t0) | set(purge.t0)
            unused = primary.loc[~primary.t0.isin(used)]
            for key, f in [
                ("panel", panel),
                ("eligible", eligible),
                ("primary", primary),
                ("development", dev),
                ("evaluation", ev),
                ("purged_2015", purge),
                ("unused_other", unused),
            ]:
                totals[key].add(f)
            for key, g in eligible.groupby("outcome_status"):
                status[str(key)].add(g)
            for key, g in eligible.groupby(eligible.t0.str[:4]):
                calendar[str(key)].add(g)
            if number % 1000 == 0:
                progress(f"Panel: {number:,} / {len(ids):,} facilities")
            memory = peak_memory_bytes()
            if memory and memory > plan["limits"]["max_peak_memory_bytes"]:
                raise GateStop("Panel memory cap exceeded")
            if stream.tell() > plan["limits"]["max_panel_bytes"]:
                raise GateStop("Panel disk cap exceeded")
    db.close()
    if old_equivalent != 1000:
        raise GateStop("Original histories not all preserved")
    if digest(source) != SOURCE or any(
        digest(root / name) != expected for name, expected in preserved.items()
    ):
        raise GateStop("Previous evidence/source changed")
    totals = {k: v.result() for k, v in totals.items()}
    blocking = bool(set(ids) - found) or any(
        findings.get(k, 0)
        for k in ["conflicting_duplicate_months", "unexpected_post_terminal_rows"]
    )
    chosen = next(r for r in plan["candidates"] if r["n"] == plan["chosen_n"])
    observed = {
        "overall": totals["primary"]["default_loans"],
        "development": totals["development"]["default_loans"],
        "evaluation": totals["evaluation"]["default_loans"],
    }
    comparison = {
        k: dict(
            expected=chosen["support"][k]["expected"],
            predictive_95=chosen["support"][k]["predictive_95"],
            observed=v,
            position="below"
            if v < chosen["support"][k]["predictive_95"][0]
            else "above"
            if v > chosen["support"][k]["predictive_95"][1]
            else "within",
        )
        for k, v in observed.items()
    }
    result = dict(
        schema_version="1.0",
        decision=decision(observed["development"], observed["evaluation"], blocking),
        amendment_sha256_lf=plan_hash,
        phase_a=plan,
        sample=sample,
        protocol_sha256_lf=protocol_hash,
        quarter_counts=quarter_counts,
        event_support=totals,
        outcomes={k: v.result() for k, v in status.items()},
        calendar={k: v.result() for k, v in calendar.items()},
        first_observed_endpoints=dict(first_events),
        any_qualifying_record_default_loans=len(raw_default),
        integrity_findings=dict(findings),
        selected_history_found=len(found),
        selected_history_missing=len(set(ids) - found),
        planning_vs_observed=comparison,
        resources=dict(
            source_scan_seconds=scan_seconds,
            total_seconds=time.monotonic() - started,
            peak_working_set_bytes=peak_memory_bytes(),
            temporary_source_bytes_peak=temporary_peak,
            rows_scanned=sum(q["performance_rows_scanned"] for q in quarter_counts.values()),
            rows_retained=raw_rows,
            panel_bytes=panel_path.stat().st_size,
            selected_cache_bytes=(output / "retained_performance.sqlite").stat().st_size,
            no_full_text_extraction=True,
            cache_policy=(
                "Private Git-ignored immutable panel/cache; no new source scan needed for "
                "future research"
            ),
        ),
        preservation=dict(
            source_unchanged=True,
            original_sample_unchanged=True,
            original_first_1000_exact=True,
            original_panel_equivalent_loans=old_equivalent,
            previous_file_sha256=preserved,
            previous_files_unchanged=True,
        ),
        panel_sha256=digest(panel_path),
        models_fitted=False,
        expanded_predictive_metrics_computed=False,
        limitations=[
            (
                "Feasibility is distinct-facility event support, not proof of independence "
                "or predictive validity"
            ),
            "Historical operational knowledge time remains UNVERIFIED",
            "Same 2010 vintage and fixed calendar split; survival selection and drift persist",
            (
                "Planning exchangeability/posterior assumptions and unknown borrower "
                "clustering limit probability interpretation"
            ),
            (
                "Calibration stability and actual metric uncertainty require later "
                "validation; no metrics fitted here"
            ),
        ],
    )
    result["next_task"] = (
        "Track B Task 5 — Expanded-Cohort 12-Month PD Development and Temporal Validation"
        if result["decision"] == "EXPANSION SUPPORT ADEQUATE"
        else (
            "Separate protocol decision on event support and research scope; no "
            "further sampling in Task 4"
        )
    )
    write_json(
        private / "panel_manifest.json",
        dict(
            panel_sha256=result["panel_sha256"],
            sample_manifest_sha256=digest(private / "sample_manifest.json"),
            source_sha256=SOURCE,
            protocol_sha256_lf=protocol_hash,
            amendment_sha256_lf=plan_hash,
            rows=totals["panel"]["landmarks"],
            created_at=datetime.now(UTC).isoformat(),
        ),
    )
    write_json(root / "reports/track_b/sample_expansion_feasibility.json", result)
    render(root, result)
    progress(result["decision"])
    return result


def render(root, r):
    p = r["phase_a"]
    parts = [
        "# Sample-expansion feasibility",
        "## Executive Summary",
        r["decision"],
        (
            "No model fitted, expanded AUC or calibration metric computed. This is "
            "event-support feasibility only."
        ),
        "## Motivation",
        (
            "Task 3 had 13 development and 5 temporal evaluation event loans. Repeated "
            "landmarks did not increase independent event support."
        ),
        "## Phase A Planning",
        p["probability_model"],
        "## Event-Support Objective",
        p["objective_rationale"],
        "## Selected Expansion Size",
        str(p["chosen_n"])
        + (
            " loans; fixed before new outcomes, smallest candidate passing the 0.90 "
            "joint lower-bound objective."
        ),
        "## Nested Sampling",
        json.dumps(r["sample"], indent=2),
        "## Phase B Empirical Results",
        "| Cohort | Landmarks | Loans | Positive landmarks | Default loans |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, v in r["event_support"].items():
        parts.append(
            "| "
            + " | ".join(
                [
                    k,
                    *[
                        str(v[x])
                        for x in ["landmarks", "loans", "positive_landmarks", "default_loans"]
                    ],
                ]
            )
            + " |"
        )
    parts.extend(
        [
            "## Temporal Event Support",
            (
                "The original Task 3 hash assignments, development <=2014-12, evaluation "
                ">=2016-01 and 2015 purge are unchanged. Purged/unused rows above are "
                "mutually exclusive; loans/event loans can appear in multiple "
                "unused-period summaries and are not additive."
            ),
            "Outcome status counts (eligible landmarks):",
            json.dumps(r["outcomes"], indent=2),
            "## Planning vs Observed",
            json.dumps(r["planning_vs_observed"], indent=2),
            "## Resource Impact",
            json.dumps(r["resources"], indent=2),
            "## Preservation",
            (
                "Original first-1000 ranking and full original-panel equivalence verified "
                "for all 1,000 loans. Previous tracked/private evidence byte hashes and "
                "source hash unchanged; hashes recorded in JSON. No raw IDs published."
            ),
            "## Limitations",
            "\n".join("- " + s for s in r["limitations"]),
            "## Decision",
            r["decision"],
            "Next task: " + r["next_task"],
            (
                "Planning references: [SciPy "
                "beta-binomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.s"
                "tats.betabinom.html), [Hanley and McNeil "
                "(1982)](https://doi.org/10.1148/radiology.143.1.7063747)."
            ),
        ]
    )
    # Consecutive Markdown table rows must remain consecutive.
    text = "\n\n".join(parts)
    text = text.replace(" |\n\n|", " |\n|").replace("|\n\n|", "|\n|")
    (Path(root) / "reports/track_b/SAMPLE_EXPANSION_FEASIBILITY.md").write_text(
        text + "\n", encoding="utf-8"
    )

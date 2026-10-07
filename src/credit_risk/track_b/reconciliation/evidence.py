"""Survey source origination tokens before deciding any parser compatibility rule."""

import json
import math
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

from credit_risk.track_b.data.freddie import NUMERIC
from credit_risk.track_b.data.schemas import ORIGINATION, digest
from credit_risk.track_b.multivintage.core import write_json
from credit_risk.track_b.multivintage.reader import VintageReader

VERSION = "origination-convention-census-v1"
ID = re.compile(r"([FA])([0-9]{2})Q([1-4])([0-9]{7})")


def embedded(token):
    match = ID.fullmatch(token)
    if not match:
        return None
    product, yy, quarter, _ = match.groups()
    year = 1999 if yy == "99" else 2000 + int(yy)
    return dict(product=product, year=year, quarter=int(quarter), label=f"{year}Q{quarter}")


def rate_class(token):
    if not token:
        return "blank", None
    if token == ".":
        return "dot", None
    try:
        value = float(token)
    except ValueError:
        return "other_nonnumeric", None
    if not math.isfinite(value):
        return "other_nonnumeric", None
    return "numeric", value


def ordinal(token):
    if not re.fullmatch(r"[0-9]{6}", token):
        return None
    year, month = int(token[:4]), int(token[4:])
    return year * 12 + month - 1 if 1900 <= year <= 2200 and 1 <= month <= 12 else None


def context(tokens, info, vintage, quarter):
    fp, maturity = ordinal(tokens[1]), ordinal(tokens[3])
    try:
        term = int(tokens[21])
    except ValueError:
        term = None
    delta = ((info["year"] - vintage) * 4 + info["quarter"] - quarter) if info else None
    return dict(
        embedded_quarter=info["label"] if info else None,
        quarter_delta=delta,
        adjacent_boundary=abs(delta) == 1 if delta is not None else None,
        first_payment_month=tokens[1],
        maturity_month=tokens[3],
        term=term,
        scheduled_term_coherent=(maturity - fp + 1 == term)
        if fp is not None and maturity is not None and term is not None
        else False,
        first_payment_minus_embedded_quarter_start=fp
        - (info["year"] * 12 + (info["quarter"] - 1) * 3)
        if fp is not None and info
        else None,
    )


def inspect(source, root, vintage, plan, preflight):
    private = root / "data/track_b/multivintage/reconciliation_v1"
    path = private / f"census_{vintage}.json"
    ids_path = private / f"identifiers_{vintage}.sqlite"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["source_sha256"] != digest(source) or old["version"] != VERSION:
            raise ValueError("Census checkpoint identity changed")
        if old["identifier_database_sha256"] != digest(ids_path):
            raise ValueError("Census identifier evidence changed")
        return old
    if ids_path.exists():
        raise ValueError("Partial census requires explicit recovery")
    db = sqlite3.connect(ids_path)
    db.execute("CREATE TABLE identifiers(loan TEXT PRIMARY KEY,member_quarter INTEGER)")
    start = time.monotonic()
    members = []
    contexts = Counter()
    dot_contexts = []
    dots_elsewhere = Counter()
    total_duplicates = 0
    with VintageReader(source, root, plan["limits"], preflight["sha256"], vintage) as reader:
        for q in range(1, 5):
            counts, prefixes, rates, other = Counter(), Counter(), Counter(), Counter()
            low, high = None, None
            batch = []
            for n, line in enumerate(reader.lines(q, "origination"), 1):
                counts["rows"] += 1
                tokens = [v.strip() for v in line.split("|")]
                if len(tokens) != len(ORIGINATION):
                    counts["wrong_field_count"] += 1
                    continue
                info = embedded(tokens[19])
                prefixes[info["label"] if info else "MALFORMED"] += 1
                counts["wrong_product"] += bool(info and info["product"] != "F")
                mismatch = info and (info["year"], info["quarter"]) != (vintage, q)
                counts["prefix_mismatch"] += bool(mismatch)
                classification, number = rate_class(tokens[12])
                rates[classification] += 1
                if number is not None:
                    low = number if low is None else min(low, number)
                    high = number if high is None else max(high, number)
                if classification == "other_nonnumeric":
                    other[tokens[12]] += 1
                if mismatch:
                    detail = context(tokens, info, vintage, q)
                    contexts[
                        json.dumps(dict(member=f"{vintage}Q{q}", **detail), sort_keys=True)
                    ] += 1
                    counts["mismatch_term_incoherent"] += not detail["scheduled_term_coherent"]
                    counts["adjacent_boundary_mismatch"] += detail["adjacent_boundary"]
                if classification == "dot":
                    detail = context(tokens, info, vintage, q)
                    detail.update(
                        member=f"{vintage}Q{q}",
                        line=n,
                        columns=len(tokens),
                        delimiter_count=line.count("|"),
                        rate_position=13,
                        ltv_neighbor_numeric=rate_class(tokens[11])[0] == "numeric",
                        channel_neighbor_recognized=tokens[13] in {"R", "B", "C", "T", "9"},
                        amortization_frm=tokens[15] == "FRM",
                    )
                    dot_contexts.append(detail)
                for k, token in zip(ORIGINATION, tokens, strict=True):
                    if k in NUMERIC and k != "orig_interest_rate" and token == ".":
                        dots_elsewhere[k] += 1
                batch.append((tokens[19], q))
                if len(batch) >= 5000:
                    before = db.total_changes
                    db.executemany("INSERT OR IGNORE INTO identifiers VALUES (?,?)", batch)
                    total_duplicates += len(batch) - (db.total_changes - before)
                    db.commit()
                    batch = []
            if batch:
                before = db.total_changes
                db.executemany("INSERT OR IGNORE INTO identifiers VALUES (?,?)", batch)
                total_duplicates += len(batch) - (db.total_changes - before)
                db.commit()
            members.append(
                dict(
                    member=f"orig_{vintage}Q{q}.txt",
                    quarter=q,
                    counts=dict(counts),
                    embedded_quarters=dict(prefixes),
                    rate_tokens=dict(rates),
                    rate_other_tokens=dict(other),
                    numeric_rate_min=low,
                    numeric_rate_max=high,
                    prefix_percentages={k: 100 * v / counts["rows"] for k, v in prefixes.items()},
                )
            )
            print(
                f"{vintage}Q{q}: {counts['rows']:,} rows; "
                f"{counts['prefix_mismatch']} prefix exceptions; {rates['dot']} dot rates",
                flush=True,
            )
    unique = db.execute("SELECT COUNT(*) FROM identifiers").fetchone()[0]
    db.close()
    result = dict(
        version=VERSION,
        vintage=vintage,
        source_sha256=preflight["sha256"],
        members=members,
        unique_identifiers=unique,
        duplicate_identifiers=total_duplicates,
        mismatch_contexts=[dict(**json.loads(k), count=v) for k, v in sorted(contexts.items())],
        dot_contexts=dot_contexts,
        dot_other_numeric_fields=dict(dots_elsewhere),
        seconds=time.monotonic() - start,
        performance_accessed=False,
        samples_frozen=False,
        identifier_database_sha256=digest(ids_path),
    )
    write_json(path, result, True)
    return result


def run(root, source_dir):
    root, source_dir = Path(root), Path(source_dir)
    plan = json.loads(
        (root / "docs/track_b/multi_vintage_protocol.json").read_text(encoding="utf-8")
    )
    answer = []
    for y in plan["vintages"]:
        pre = json.loads(
            (root / f"data/track_b/multivintage/manifests/preflight_{y}.json").read_text(
                encoding="utf-8"
            )
        )
        answer.append(inspect(source_dir / f"historical_data_{y}.zip", root, y, plan, pre))
    write_json(root / "data/track_b/multivintage/reconciliation_v1/census.json", answer)
    return answer

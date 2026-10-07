"""Acquire public Crossref metadata only; never access empirical research inputs."""

import concurrent.futures
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def acquire(entry):
    key, doi, buckets, source, notes = entry
    url = "https://doi.org/" + urllib.parse.quote(doi, safe="/")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/vnd.citationstyles.csl+json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        return {
            "reference_id": key,
            "crossref_url": url,
            "retrieved_date": "2026-10-07",
            "metadata": None,
            "error": str(error),
            "content_source": source,
            "content_review": notes,
            "buckets": buckets.split(),
        }
    payload = json.loads(raw)
    message = payload.get("message", payload)
    fields = [
        "DOI",
        "title",
        "author",
        "container-title",
        "volume",
        "issue",
        "page",
        "article-number",
        "published",
        "published-print",
        "published-online",
        "type",
        "URL",
    ]
    return {
        "reference_id": key,
        "crossref_url": url,
        "retrieved_date": "2026-10-07",
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "metadata": {field: message[field] for field in fields if field in message},
        "content_source": source,
        "content_review": notes,
        "buckets": buckets.split(),
    }


if __name__ == "__main__":
    path = ROOT / "docs/paper/literature/crossref_metadata.json"
    if path.exists():
        raise SystemExit("Metadata snapshot already exists; do not overwrite")
    specs = json.loads(
        (ROOT / "docs/paper/literature/task16_reference_specs.json").read_text(encoding="utf-8")
    )["entries"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(acquire, specs))
    path.write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "attempted": len(records),
                "verified_crossref_records": sum(r["metadata"] is not None for r in records),
            }
        )
    )

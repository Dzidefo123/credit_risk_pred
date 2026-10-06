"""Identifier-only sampling fixed before performance inspection."""

from hashlib import sha256

SALT = "freddie_mortgage_research_v1"


def select_loans(identifiers, n=1000):
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 1000:
        raise ValueError("Sample count must be 1..1000")
    ids = set(identifiers)
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Nonempty loan identifiers required")
    return sorted(
        ids, key=lambda value: (sha256((SALT + ":" + value).encode()).hexdigest(), value)
    )[:n]

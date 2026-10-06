"""Local, fail-closed final-holdout ledger; profile keys are not borrower IDs."""

import json
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from credit_risk.data.validation import ORIGINATION_FEATURES

HISTORICAL_SOURCE_SHA256 = "9d863f88e2ef2190ed608c15c3dd362d1833d940e35f4f2bf22660e04b5f05cb"
HISTORICAL_SAMPLES_SHA256 = "974de197a36ad2a1d5cb986be30ddb4927403ef12c14d6a5d7ed4b430345eebc"

FIELDS = tuple(sorted(ORIGINATION_FEATURES))
LEGACY_INTEGERS = tuple(
    n
    for n in ORIGINATION_FEATURES
    if n
    not in {
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfDependents",
    }
)


def canonical_hash(value):
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def source_fingerprint(file_sha256):
    if (
        not isinstance(file_sha256, str)
        or len(file_sha256) != 64
        or any(c not in "0123456789abcdef" for c in file_sha256)
    ):
        raise ValueError("Source identity requires a lowercase SHA-256")
    return canonical_hash({"scheme": "source-v1", "file_sha256": file_sha256, "fields": FIELDS})


def sample_fingerprints(frame):
    """Hash raw predictors only; exclude target/index, normalize int/float and missing."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.columns.duplicated().any():
        raise ValueError("Sample identities require nonempty data with unique columns")
    if not set(FIELDS).issubset(frame.columns):
        raise ValueError("Missing raw predictor identity fields")
    raw = frame.loc[:, FIELDS]
    if any(
        not pd.api.types.is_numeric_dtype(raw[n]) or pd.api.types.is_bool_dtype(raw[n])
        for n in FIELDS
    ):
        raise ValueError("Raw identity fields must be numeric, not strings or booleans")
    tokens = set()
    for row in raw.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append(None)
            elif not np.isfinite(value):
                raise ValueError("Nonfinite raw predictor cannot establish sample identity")
            elif int(value) == value:
                values.append(str(int(value)))
            else:
                values.append(float(value).hex())
        tokens.add("raw-v1:" + canonical_hash(["raw-profile-v1", FIELDS, values]))
    # Bridge the historical saved pandas predictor groups, without retrieving old rows.
    compatible = pd.Series(True, index=frame.index)
    for name in LEGACY_INTEGERS:
        s = frame[name]
        compatible &= s.notna() & (s % 1 == 0) & (s >= -(2**63)) & (s < 2**63)
    bridge = frame.loc[compatible, ORIGINATION_FEATURES].copy()
    if not bridge.empty:
        for name in ORIGINATION_FEATURES:
            bridge[name] = bridge[name].astype("int64" if name in LEGACY_INTEGERS else "float64")
        groups = pd.util.hash_pandas_object(bridge, index=False)
        tokens.update(f"legacy-pandas-v1:{int(v):016x}" for v in groups)
    return sorted(tokens)


class Entry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reservation_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    samples: list[str] = Field(min_length=1)
    run: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    split: Literal["test"] = "test"
    status: Literal["reserved", "consumed"]
    registered_at: str
    consumed_at: str | None = None
    evidence: dict[str, str] = Field(default_factory=dict)
    legacy_pandas_version: str | None = None

    @model_validator(mode="after")
    def check_identity(self):
        import re

        if self.source_fingerprint != source_fingerprint(self.source_sha256):
            raise ValueError("Source fingerprint disagrees with source digest")
        if self.samples != sorted(set(self.samples)) or any(
            not re.fullmatch(r"raw-v1:[a-f0-9]{64}|legacy-pandas-v1:[a-f0-9]{16}", s)
            for s in self.samples
        ):
            raise ValueError("Malformed or noncanonical sample identities")
        if any(s.startswith("legacy-") for s in self.samples) and not self.legacy_pandas_version:
            raise ValueError("Legacy identities require their pandas version")
        return self


class Ledger(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    entries: list[Entry]

    @model_validator(mode="after")
    def check_unique(self):
        seen, ids = set(), set()
        for entry in self.entries:
            if entry.reservation_id in ids or seen.intersection(entry.samples):
                raise ValueError("Duplicate reservation or overlapping ledger entries")
            seen.update(entry.samples)
            ids.add(entry.reservation_id)
        return self


def default_registry_path():
    root = Path(__file__).resolve().parents[3]
    path = root / "reports/holdout_registry.json"
    if not path.is_file():
        for parent in (Path.cwd(), *Path.cwd().parents):
            candidate = parent / "reports/holdout_registry.json"
            if (parent / "pyproject.toml").is_file() and candidate.is_file():
                verify_repository_registry(candidate)
                return candidate
        raise FileNotFoundError(
            "Repository holdout registry missing; do not initialize an empty replacement"
        )
    verify_repository_registry(path)
    return path


class HoldoutRegistry:
    def __init__(self, path, lock_timeout=5):
        self.path = Path(path).resolve()
        self.lock_timeout = lock_timeout

    def read(self):
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON registry key")
                result[key] = value
            return result

        try:
            payload = json.loads(
                self.path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys
            )
            if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int:
                raise ValueError("Missing or invalid registry schema version")
            return Ledger.model_validate(payload)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Corrupt/unsupported holdout registry {self.path}: {exc}") from exc

    @contextmanager
    def _lock(self):
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        deadline = time.monotonic() + self.lock_timeout
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Holdout registry busy/stale lock: {lock}; no evaluation allowed"
                    ) from None
                time.sleep(0.05)
        try:
            os.write(fd, str(os.getpid()).encode())
            yield
        finally:
            os.close(fd)
            lock.unlink()

    def _write(self, ledger):
        # Validate before an atomic replacement; failure retains the previous complete file.
        ledger = Ledger.model_validate(ledger.model_dump())
        payload = json.dumps(ledger.model_dump(), sort_keys=True, indent=2, allow_nan=False) + "\n"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as f:
                temporary = Path(f.name)
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @staticmethod
    def initialize(path):
        """Explicit empty ledger creation for new projects/isolated fixtures only."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as f:
            f.write(json.dumps(Ledger(schema_version=1, entries=[]).model_dump(), indent=2) + "\n")
        return HoldoutRegistry(path)

    def check(self, source_sha256, samples, run=None):
        source_fingerprint(source_sha256)
        ledger = self.read()
        candidates = set(samples)
        if not candidates:
            raise ValueError("Empty candidate holdout")
        for entry in ledger.entries:
            if entry.legacy_pandas_version and entry.legacy_pandas_version != pd.__version__:
                raise ValueError(
                    "Historical sample bridge pandas version differs; "
                    "non-overlap cannot be verified"
                )
            overlap = candidates.intersection(entry.samples)
            if overlap:
                if (
                    entry.status == "reserved"
                    and entry.run == run
                    and candidates == set(entry.samples)
                    and entry.source_sha256 == source_sha256
                ):
                    continue
                raise ValueError(
                    f"Fresh final evaluation blocked: {len(overlap)} identity keys "
                    f"overlap {entry.status} holdout from {entry.run}; "
                    f"source {entry.source_fingerprint}"
                )
        return ledger

    def reserve(self, source_sha256, samples, run, purpose="fresh final evaluation"):
        samples = sorted(set(samples))
        with self._lock():
            ledger = self.check(source_sha256, samples, run)
            for entry in ledger.entries:
                if (
                    entry.status == "reserved"
                    and entry.run == run
                    and entry.samples == samples
                    and entry.source_sha256 == source_sha256
                ):
                    return entry.reservation_id
            identity = canonical_hash([source_fingerprint(source_sha256), samples, run])
            entry = Entry(
                reservation_id=identity,
                source_fingerprint=source_fingerprint(source_sha256),
                source_sha256=source_sha256,
                samples=samples,
                run=run,
                purpose=purpose,
                status="reserved",
                registered_at=datetime.now(UTC).isoformat(),
                legacy_pandas_version=pd.__version__
                if any(s.startswith("legacy-") for s in samples)
                else None,
            )
            ledger.entries.append(entry)
            self._write(ledger)
            return identity

    def consume(self, reservation_id):
        with self._lock():
            ledger = self.read()
            entry = next((e for e in ledger.entries if e.reservation_id == reservation_id), None)
            if entry is None or entry.status != "reserved":
                raise ValueError(
                    "Consumption requires an existing reserved holdout; consumed access is one-way"
                )
            entry.status = "consumed"
            entry.consumed_at = datetime.now(UTC).isoformat()
            self._write(ledger)

    def status(self, source_sha256, samples=None):
        ledger = self.read()
        selected = [
            e
            for e in ledger.entries
            if (
                e.source_sha256 == source_sha256
                if samples is None
                else set(samples).intersection(e.samples)
            )
        ]
        return (
            "consumed"
            if any(e.status == "consumed" for e in selected)
            else "reserved"
            if selected
            else "available"
        )


def verify_repository_registry(path):
    """Preserve the known migrated baseline; an empty replacement is never fresh."""
    ledger = HoldoutRegistry(path).read()
    historical = [
        e
        for e in ledger.entries
        if e.source_sha256 == HISTORICAL_SOURCE_SHA256
        and e.run == "artifacts/phase4-origination-001"
    ]
    if (
        len(historical) != 1
        or historical[0].status != "consumed"
        or canonical_hash(historical[0].samples) != HISTORICAL_SAMPLES_SHA256
    ):
        raise ValueError("Historical consumed holdout anchor missing/changed; evaluation blocked")
    return ledger

"""Version-qualified secure annual streaming; directory inspection precedes payload access."""

import shutil
import struct
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

from credit_risk.track_b.data.annual import GateStop, safe_members
from credit_risk.track_b.data.preflight import StoredSlice, member_info
from credit_risk.track_b.data.schemas import digest


class VintageReader:
    def __init__(self, source, root, limits, expected_hash, vintage):
        self.vintage = vintage
        self.source = Path(source).resolve()
        self.root = Path(root).resolve()
        self.limits = limits
        self.expected_hash = expected_hash
        self.sample = None
        self.inventory = []
        self.temp_bytes_peak = 0
        self.temp_files_created = 0
        self.temp_root = self.root / "data/track_b/multivintage/interim/temporary"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        if not self.temp_root.resolve().is_relative_to(self.root / "data/track_b"):
            raise GateStop("Temporary zone escapes private data zone")
        if digest(self.source) != expected_hash:
            raise GateStop("Approved root source hash mismatch")
        self.outer = zipfile.ZipFile(self.source)
        try:
            safe_members(
                self.outer,
                [f"historical_data_{self.vintage}Q{q}.zip" for q in range(1, 5)],
                limits,
                True,
            )
            total = 0
            for q in range(1, 5):
                with self.quarter(q) as inner:
                    members = safe_members(
                        inner,
                        [f"orig_{self.vintage}Q{q}.txt", f"perf_{self.vintage}Q{q}.txt"],
                        limits,
                    )
                    total += sum(i.file_size for i in members)
                    self.inventory.append(
                        {
                            "quarter": q,
                            "outer": member_info(
                                self.outer.getinfo(f"historical_data_{self.vintage}Q{q}.zip")
                            ),
                            "members": [member_info(i) for i in members],
                        }
                    )
            if total > limits["max_total_text_bytes"]:
                raise GateStop("Annual declared payload budget exceeded")
            if shutil.disk_usage(self.temp_root).free < limits["free_disk_reserve_bytes"]:
                raise GateStop("Insufficient free disk for outputs")
        except Exception:
            self.close()
            raise

    @contextmanager
    def quarter(self, q):
        if q not in {1, 2, 3, 4}:
            raise GateStop("Unsupported quarter")
        info = self.outer.getinfo(f"historical_data_{self.vintage}Q{q}.zip")
        temporary = None
        view = None
        inner = None
        try:
            if info.compress_type == zipfile.ZIP_STORED:
                with self.source.open("rb") as handle:
                    handle.seek(info.header_offset)
                    header = handle.read(30)
                if header[:4] != b"PK\x03\x04":
                    raise GateStop("Invalid local ZIP header")
                name_size, extra_size = struct.unpack_from("<HH", header, 26)
                view = StoredSlice(
                    self.source, info.header_offset + 30 + name_size + extra_size, info.file_size
                )
                inner = zipfile.ZipFile(view)
            else:
                if info.file_size > self.limits["max_temporary_bytes"]:
                    raise GateStop("Temporary-quarter budget exceeded")
                if (
                    shutil.disk_usage(self.temp_root).free
                    < info.file_size + self.limits["free_disk_reserve_bytes"]
                ):
                    raise GateStop("Insufficient temporary disk")
                handle = tempfile.NamedTemporaryFile(
                    dir=self.temp_root, prefix="quarter-", suffix=".zip", delete=False
                )
                temporary = Path(handle.name)
                self.temp_files_created += 1
                copied = 0
                with handle, self.outer.open(info) as original:
                    while block := original.read(self.limits["chunk_bytes"]):
                        copied += len(block)
                        if copied > info.file_size or copied > self.limits["max_temporary_bytes"]:
                            raise GateStop("Quarter expanded beyond declared budget")
                        handle.write(block)
                if copied != info.file_size:
                    raise GateStop("Quarter size mismatch")
                self.temp_bytes_peak = max(self.temp_bytes_peak, copied)
                inner = zipfile.ZipFile(temporary)
            safe_members(
                inner, [f"orig_{self.vintage}Q{q}.txt", f"perf_{self.vintage}Q{q}.txt"], self.limits
            )
            yield inner
        finally:
            if inner is not None:
                inner.close()
            if view is not None:
                view.close()
            if temporary is not None:
                if not temporary.resolve().is_relative_to(self.temp_root.resolve()):
                    raise GateStop("Unsafe temporary cleanup target")
                temporary.unlink(missing_ok=True)

    def freeze(self, identifiers):
        ids = frozenset(identifiers)
        if not 1 <= len(ids) <= 20000:
            raise GateStop("Invalid frozen sample size")
        if self.sample is not None and self.sample != ids:
            raise GateStop("Adaptive sample replacement prohibited")
        self.sample = ids

    @contextmanager
    def text_stream(self, q, kind):
        if kind not in {"origination", "performance"}:
            raise GateStop("Invalid source kind")
        if kind == "performance" and self.sample is None:
            raise GateStop("Performance rows inaccessible before sample freeze")
        name = f"{'orig' if kind == 'origination' else 'perf'}_{self.vintage}Q{q}.txt"
        with self.quarter(q) as inner, inner.open(name) as raw:
            yield raw

    def lines(self, q, kind):
        with self.text_stream(q, kind) as raw:
            while line := raw.readline(self.limits["max_line_bytes"] + 1):
                if len(line) > self.limits["max_line_bytes"]:
                    raise GateStop("Record exceeds streaming line cap")
                try:
                    yield line.decode("utf-8-sig").rstrip("\r\n")
                except UnicodeDecodeError as exc:
                    raise GateStop("Source encoding not UTF-8 compatible") from exc

    def close(self):
        if hasattr(self, "outer"):
            self.outer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

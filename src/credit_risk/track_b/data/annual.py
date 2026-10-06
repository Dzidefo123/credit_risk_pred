"""Secure nested annual reader with origination-before-performance access control."""

import json
import os
import shutil
import stat
import struct
import tempfile
import zipfile
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

from .preflight import StoredSlice, member_info
from .schemas import digest, load_protocol


class GateStop(ValueError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


def peak_memory_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                *[
                    (n, ctypes.c_size_t)
                    for n in [
                        "PeakWorkingSetSize",
                        "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage",
                        "PagefileUsage",
                        "PeakPagefileUsage",
                    ]
                ],
            ]

        counts = Counters()
        counts.cb = ctypes.sizeof(counts)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counts), counts.cb):
            return int(counts.PeakWorkingSetSize)
        return None
    import resource
    import sys

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def load_amendment(root):
    root = Path(root)
    protocol, base_hash = load_protocol(root)
    path = root / "docs/track_b/annual_bundle_acquisition_amendment.json"
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if (
        amendment["base_protocol_sha256_lf"] != base_hash
        or amendment["sampling"]["salt"] != protocol["protocol_id"]
        or amendment["sampling"]["maximum_loans"] != 1000
        or amendment["models_permitted"] is not False
        or amendment["resources"]["max_malformed_rows"] != 0
    ):
        raise GateStop("Amendment altered frozen scientific rules")
    if any(not isinstance(v, int) or v < 0 for v in amendment["resources"].values()):
        raise GateStop("Invalid operational limits")
    return (
        protocol,
        base_hash,
        amendment,
        sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    )


def safe_members(archive, expected, limits, quarter=False):
    items = archive.infolist()
    names = [i.filename for i in items]
    if len(names) != len(set(names)):
        raise GateStop("Duplicate member names")
    if set(names) != set(expected) or len(names) != len(expected):
        raise GateStop("Missing/unexpected member or unsupported nested structure")
    for item in items:
        name = item.filename
        mode = item.external_attr >> 16
        if (
            name.startswith(("/", "\\"))
            or "/" in name
            or "\\" in name
            or ":" in name
            or ".." in name
            or item.is_dir()
            or stat.S_ISLNK(mode)
            or item.flag_bits & 1
        ):
            raise GateStop("Unsafe/encrypted member")
        if item.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise GateStop("Unsupported ZIP compression")
        compressed = limits[
            "max_quarter_compressed_bytes" if quarter else "max_text_compressed_bytes"
        ]
        expanded = limits["max_quarter_declared_bytes" if quarter else "max_text_declared_bytes"]
        if item.compress_size > compressed or item.file_size > expanded:
            raise GateStop("Member exceeds declared size limits")
        if item.file_size / max(1, item.compress_size) > limits["max_compression_ratio"]:
            raise GateStop("Excessive ZIP compression ratio")
    return items


class AnnualReader:
    def __init__(self, source, root, limits, expected_hash):
        self.source = Path(source).resolve()
        self.root = Path(root).resolve()
        self.limits = limits
        self.expected_hash = expected_hash
        self.sample = None
        self.inventory = []
        self.temp_bytes_peak = 0
        self.temp_files_created = 0
        self.temp_root = self.root / "data/track_b/interim/annual-temporary"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        if not self.temp_root.resolve().is_relative_to(self.root / "data/track_b"):
            raise GateStop("Temporary zone escapes private data zone")
        if digest(self.source) != expected_hash:
            raise GateStop("Approved root source hash mismatch")
        self.outer = zipfile.ZipFile(self.source)
        try:
            safe_members(
                self.outer, [f"historical_data_2010Q{q}.zip" for q in range(1, 5)], limits, True
            )
            total = 0
            for q in range(1, 5):
                with self.quarter(q) as inner:
                    members = safe_members(
                        inner, [f"orig_2010Q{q}.txt", f"perf_2010Q{q}.txt"], limits
                    )
                    total += sum(i.file_size for i in members)
                    self.inventory.append(
                        {
                            "quarter": q,
                            "outer": member_info(
                                self.outer.getinfo(f"historical_data_2010Q{q}.zip")
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
        info = self.outer.getinfo(f"historical_data_2010Q{q}.zip")
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
            safe_members(inner, [f"orig_2010Q{q}.txt", f"perf_2010Q{q}.txt"], self.limits)
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
        if not 1 <= len(ids) <= 1000:
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
        name = f"{'orig' if kind == 'origination' else 'perf'}_2010Q{q}.txt"
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

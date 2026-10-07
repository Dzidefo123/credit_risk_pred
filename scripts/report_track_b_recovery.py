"""Build separate aggregate Task8C audit without modifying prior evidence."""

from pathlib import Path

from credit_risk.track_b.recovery.reporting import build

if __name__ == "__main__":
    report = build(Path(__file__).resolve().parents[1])
    print(report["decision"])

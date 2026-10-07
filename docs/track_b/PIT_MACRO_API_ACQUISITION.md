# Task 9 authenticated acquisition follow-up

The original unauthenticated Task 9 attempt is retained as `task9_v1`. The
successful authenticated run is **`task9_api_v5`**. Its audit is
[PIT_MACRO_DATA_AUDIT_task9_api_v5.md](../../reports/track_b/PIT_MACRO_DATA_AUDIT_task9_api_v5.md)
and its full machine-readable coverage is
[pit_macro_data_audit_task9_api_v5.json](../../reports/track_b/pit_macro_data_audit_task9_api_v5.json).
The original report's zero-coverage statement describes the earlier attempt;
it does not describe this authenticated run.

The exact six-series registry, transformations, geography, assessment convention,
freshness caps and mortgage samples remain frozen. Twenty successful source
requests acquired metadata, vintage-date indexes and real-time-period
observations, yielding 20,009 numeric version/validity-segment rows.

| Series | Numeric version / segment rows | Vintage-index dates |
| --- | --- | --- |
| UNRATE | 499 | 793 |
| DGS10 | 12,651 | 4,993 |
| MORTGAGE30US | 1,160 | 824 |
| USSTHPI | 3,437 | 63 |
| GDPC1 | 856 | 412 |
| CPIAUCSL | 1,406 | 663 |

FRED's JSON observation endpoint limits the number of vintage dates per request.
DGS10 therefore uses three disjoint, inclusive real-time windows, each with at
most 1,999 indexed vintage dates, reserving one slot for the carried starting
snapshot. All windows stay within the frozen overall bounds. The complete
series is admitted only after all partitions succeed and pass validation.
Transport segments are **not economic revision-event counts**.

Every raw response is written exclusively and hashed. The manifest records
redacted request parameters, status, retrieval time, hashes and coverage.
Vintage-index pagination must finish; observation pagination must be complete;
version dates must appear in the provider index; duplicate or overlapping
validity intervals fail validation. No current-only series is substituted.

Runs `task9_api_v2`–`task9_api_v4` retain the original provider-limit and transient
transport failures. They were not overwritten or silently incorporated as
complete series. The successful run is independently reproducible under its
own manifest. All raw/interim/processed/manifests remain Git-ignored under the
existing `/data/track_b/**` rule. Existing `.env` and `.env.*` rules also apply.

## Coverage decision

**PIT MACRO SUPPORT INSUFFICIENT for the full January 2006–March 2026 reporting
window.** Full support exists September 2010–February 2026 (186 months); the
remaining 57 months have partial support. Mortgage-rate and HPI archive coverage
begins in 2010, and March 2026 lacks the required unemployment-change operand.
No sample restriction, imputation, freshness relaxation or substitute series
was applied. Successful acquisition does not authorize redefining the analysis
population or certify all mortgage prediction horizons.

Exact agency publication dates and certified initial releases remain separate
from ALFRED availability evidence. Descriptive retained-vintage comparisons
are reported within matching units/regimes; they are not initial-versus-current
comparisons. Historical backfills are explicitly included in availability-bound
lag summaries. No model was fitted.

## Running a separately versioned acquisition

Configure `FRED_API_KEY` locally. A variable in another PowerShell window is not
automatically inherited by Codex. If configured in the Windows User environment,
use the following in the process launching the script, without printing the key:

```powershell
$env:FRED_API_KEY = [Environment]::GetEnvironmentVariable('FRED_API_KEY', 'User')
.venv\Scripts\python.exe scripts/acquire_track_b_pit_macro.py --run task9_api_v6
.venv\Scripts\python.exe scripts/report_track_b_pit_macro.py --run task9_api_v6
```

Use a new run name; existing acquisitions cannot be overwritten. The script
uses `os.environ` and does not persist the key. Each run retains the frozen
36-request budget, 15-second request timeout and response-size limits.

Official contracts: [vintage-date index](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html),
[real-time observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html).

Exactly one corrective next task: **Track B Task 9A — Prespecify macro-support
eligibility and resolve remaining historical coverage gaps without changing
mortgage samples.** No completion commit or push was made.

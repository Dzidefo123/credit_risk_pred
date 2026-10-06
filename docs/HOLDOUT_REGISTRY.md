# Repository-level holdout registry

Task 3: experimental governance only. No original model training, recalibration,
final evaluation, prediction inspection or frozen-artifact regeneration occurred.

## Why this exists

The old `test_consumption.json` marker is local to a training run. A copied source
or new run directory could have no marker and represent already viewed profiles
as fresh final evaluation. Matching selections also previously allowed a repeat
scoring pass. The final-validation runner now rejects an existing marker before
loading the source/models and independently checks a shared source/sample ledger.
Retrospective review uses retained evidence, not a rescore through `run_validation`.

The ledger is [reports/holdout_registry.json](../reports/holdout_registry.json),
implemented by `credit_risk.validation.holdout_registry`. This complements the
existing [model governance](model_governance.md), source/artifact hashes and
per-run marker; it does not replace or weaken them. The existing governance
checker now checks the ledger and immutable historical sample-set anchor.
No database, external service, dependency or package restructuring was added.

## Identity construction

**Source-v1:** SHA-256 of canonical JSON containing a domain/version tag, the raw
source file's SHA-256 and sorted canonical predictor names. Paths, filenames,
run IDs and predictions do not participate. Byte-identical copies have identical
source identities. A reordered/reformatted/subset/relabelled CSV can have a new
source hash; shared sample matching remains essential.

**Raw-profile-v1:** SHA-256 of canonical compact JSON containing a version tag,
sorted field names and the original predictor values. Exactly these ten fields
participate: RevolvingUtilizationOfUnsecuredLines, age,
NumberOfTime30_59DaysPastDueNotWorse, DebtRatio, MonthlyIncome,
NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate,
NumberRealEstateLoansOrLines, NumberOfTime60_89DaysPastDueNotWorse and
NumberOfDependents. Use the source adapter's documented name aliases first.
No clipping, log transform, imputation, scaling or prediction enters the key.

Missing values map to JSON null; numerically integral values encode as exact
integer strings, so 1 and 1.0 match; other finite numeric values use float.hex.
Nonfinite/inappropriate nonnumeric identity inputs fail. The target, row index,
source row ID and dataframe order are excluded. A revised/flipped outcome must
not let the same predictor profile appear untouched. This deliberately treats
identical predictor profiles with different labels or different unidentifiable
people as the same protected profile. It is conservative sample/profile reuse
protection, **not borrower identity**. Duplicate profiles collapse to one key.

**Historical bridge:** the original saved split assignments contain pandas
64-bit exact-predictor groups, not raw-profile SHA-256 keys. They are stored as
`legacy-pandas-v1:<16 hex digits>`; they are not mislabeled cryptographic hashes.
Candidate raw numeric profiles also produce these bridge keys with the original
canonical feature order and parser dtypes: age, the three past-due counts, open
lines and real-estate lines as int64; utilization, DebtRatio, MonthlyIncome and
dependents as float64. Integer/float re-encodings normalize to these dtypes.
Rows that cannot match those historical integer dtypes do not produce a bridge
key. New raw SHA-256 keys still protect them once reserved. The ledger records
the pandas version and fails closed on bridge version incompatibility; no
cross-version guarantee is invented. Hash collisions can cause conservative
false overlap. The legacy bridge has only 64-bit collision resistance.

## Historical registration, without reopening evaluation

Imported holdout: **29,991 saved test rows, 29,871 distinct saved predictor groups**
from `artifacts/phase4-origination-001`, registered as consumed. Migration checked:

- Frozen experiment source identity against committed Phase 4 evidence.
- The consumed marker's source/locked-access status and its checksum.
- Saved assignment checksum, complete row-position sequence, test row count and
  little-endian row-position digest against the frozen experiment manifest.
- No test group in the saved assignments crosses into another partition.

Only experiment/lock metadata and saved assignment columns were read. No raw
holdout rows, label values, prediction files or model bundles were read to migrate
identities. Evidence digests and the original run/purpose are recorded in the
entry. `registered_at` is the UTC migration timestamp. Historical `consumed_at`
is null because the original event timestamp was not recorded; none is invented.
The full original consumed marker remains unchanged.

The canonical SHA-256 of the imported **complete sample-key set** is anchored in
trusted registry code. Default workflow discovery and `check_governance.py`
reject a missing, empty, altered or no-longer-consumed historical baseline.
This catches accidental ledger resets; it is not a signature or protection
against someone deliberately changing both code and evidence.

## States and lifecycle

Available means no matching reservation/consumption exists; absence is checked,
not silently manufactured from a missing file. Persisted states are reserved and
consumed. At source level, consumed means the source **contains** consumed
profiles, not that all profiles are forbidden. Same-source non-overlap is allowed
where identities establish it. Status can also be queried across source versions
using sample keys. Available does not establish prior non-use in training,
development or calibration: this ledger tracks final-holdout reservations/access,
not every earlier inspection or research decision.

Dataset → fingerprint source → candidate split → check ledger → reserve holdout
→ train/develop/calibrate → freeze selection → mark consumed immediately before
first final prediction → final evaluation → freeze artifacts/results.

Consumption occurs before access, conservatively: failed/interrupted prediction
still leaves samples consumed. Calibration/training failures retain reservation;
there is no automatic release, expiration or unconsume operation. An identical
reservation may be resumed only by its same run owner and identical source/key
set. Another owner is blocked. A consumed entry blocks every fresh evaluation,
including the original owner. Artifact/output guards still apply.

CLI `train` checks/reserves the proposed final split before calling the unchanged
training implementation. `run_validation` checks the old marker, loads/validates
the ledger, verifies the frozen run and reserves raw candidate test profiles
before loading any model or fitting calibrators. It consumes the reservation
before any final prediction and adds the registry reservation ID to **new**
validation metadata. Existing frozen manifests/results were not regenerated.
Low-level training utilities alone do not reserve, but they do not score final
tests; final validation still enforces the ledger. Direct custom scoring code
outside this workflow is not intercepted.

## Persistence, concurrency and failure behavior

JSON schema version 1 uses strict fields, source consistency and canonical unique
sample sets; duplicate JSON keys, unsupported/missing versions, corrupt entries
and overlapping ledger entries fail closed. A missing ledger is never recreated
by normal validation. Explicit `HoldoutRegistry.initialize` is for isolated
fixtures or genuinely new projects, **not a reset of this project's history**.

An exclusive-create sidecar lock serializes read/check/update operations on this
local filesystem. Lock acquisition times out after five seconds. A stale lock
fails closed; an operator must establish that no writer is active before removing
only that sidecar. Updates write a temporary file in the same directory, flush
and fsync it, then atomically replace the ledger. A failed replacement preserves
the prior file and cleans the temporary file. Sidecars are ignored in Git.
This is a single-machine cooperative protocol, not distributed/network-filesystem
coordination or automatic merging of divergent Git ledgers.

The shared ledger is tracked, so reservations/consumptions are reviewable Git
changes. Merge histories by preserving their union and rejecting conflicts;
never select an empty or earlier ledger to clear prior consumption. Separate
unmerged clones can diverge. Installed-wheel use must locate an authorized
repository ledger; absence fails closed. An explicit `registry_path` is available
for isolated fixtures/approved integrations; a caller deliberately selecting an
empty ledger can bypass history, so it is not a security boundary. Default CLI
execution always uses the anchored repository ledger.

## Cases and limits

| Case | Behavior |
| --- | --- |
| New source, new raw profiles | Allowed if registry valid and no matching keys |
| Same source, proven non-overlap | Allowed; source history does not blanket-ban every row |
| Partial/exact consumed profile overlap | Blocked with prior run, state and source identity |
| Reserved overlap owned by another run | Blocked |
| Reordered rows or copied filename/path | Detected through sample/source identities |
| Changed target or numeric dtype representation | Predictor keys persist; overlap detected |
| Arbitrarily transformed, imputed, rounded or edited profiles | Not reliably detectable; require original raw identity linkage |

The registry does not prove unique borrowers, temporal independence, absence of
same-borrower overlap with changed predictors, feature-window integrity, source
authenticity/rights, censoring/maturity, regulatory validation or unseen-model
selection. Profile collisions may block different borrowers; edited profiles can
escape matching. Historical matching is weaker than new raw SHA-256 identity.
Fingerprints are not a claim of anonymization and can permit dictionary attacks.
Protect the ledger as research evidence, not a public borrower identity register.

## Checks and next task

Tests use isolated temporary ledgers and small synthetic fixtures. A read-only
test verifies the committed historical seed without using original records or
predictions. Run:

```console
uv run --no-sync pytest tests/test_holdout_registry.py tests/test_validation.py tests/test_serving_integration.py -q
uv run --no-sync python scripts/check_governance.py
```

Retained historical XGBoost metrics remain AUC 0.868152, Brier 0.048545 and log
loss 0.176030. They are not newly evaluated results.

Exactly one next implementation task: add explicit threshold diagnostics (F1 and
confusion matrices) and validation calibration intercept/slope in additive
modules, tested on fixtures/training-only data. Do not modify frozen metric or
calibrator code and do not reopen the original final holdout. Not implemented here.

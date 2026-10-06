# Freddie 2010 Track B data audit

## Acquisition verified; processing stopped at preflight

The original authorized ZIP was hashed before inspection and rehashed afterward. Only outer/nested ZIP central-directory metadata was inspected. No loan rows were read, No extraction/copy/rename occurred. No panel or outcome statistics were generated.

Original filename: `historical_data_2010.zip`. Byte size: **1,882,675,775**.

Root SHA-256: `a82bc0f1efffccfb494b2b33f61877428bb4a6443c1a73d44bc7ee24d77aa45d`.

Authenticated Freddie Mac/Clarity acquisition is user-attested. Retrieval time was not supplied; ZIP/filesystem timestamps are not release or acquisition proof.

## Verified member inventory

| Member | Compressed bytes | Uncompressed bytes |
| --- | --- | --- |
| historical_data_2010Q1.zip | 340,461,810 | 340,461,810 |
| historical_data_2010Q1.zip / orig_2010Q1.txt | 8,214,739 | 46,698,781 |
| historical_data_2010Q1.zip / perf_2010Q1.txt | 332,246,733 | 2,641,467,789 |
| historical_data_2010Q2.zip | 339,810,443 | 339,810,443 |
| historical_data_2010Q2.zip / orig_2010Q2.txt | 8,192,850 | 46,785,787 |
| historical_data_2010Q2.zip / perf_2010Q2.txt | 331,617,255 | 2,623,274,434 |
| historical_data_2010Q3.zip | 522,375,743 | 522,375,743 |
| historical_data_2010Q3.zip / orig_2010Q3.txt | 11,514,945 | 64,766,530 |
| historical_data_2010Q3.zip / perf_2010Q3.txt | 510,860,460 | 3,954,924,580 |
| historical_data_2010Q4.zip | 680,027,037 | 680,027,037 |
| historical_data_2010Q4.zip / orig_2010Q4.txt | 13,603,582 | 76,618,439 |
| historical_data_2010Q4.zip / perf_2010Q4.txt | 666,423,009 | 5,149,323,563 |



Underlying text members total **14,603,859,903 uncompressed bytes**. No documentation/readme member was found in this inventory. CRC values in JSON are ZIP integrity metadata, not independent member SHA-256 hashes; no extracted files exist.

## Blockers

- Not the prespecified sample pair; frame/packaging amendment required

- Original ZIP exceeds current 200 MiB engineering budget; no automatic budget increase

- Underlying payload exceeds current 2 GiB expanded-data budget



Task 1 fixed the official sample, at most 1,000 hashed IDs and a pinned layout. This annual bundle is a different input frame. Its orig_/perf_ filenames resemble current naming, but names/timestamps do not prove actual 31/35-column Release 47 compatibility. No positions, events, vintage, sample selection or budgets were changed.

## Empirical gate results

| Gate | Result |
| --- | --- |
| A - provenance/release | ZIP hashed; acquisition attested; release unverified |
| B - schema | STOP: frame/budget incompatible; row layout uninspected |
| C - linkage | Not evaluated |
| D - time | Not evaluated |
| E - outcomes | Not evaluated |
| F - follow-up | Not evaluated |
| G - leakage | Fixture-tested only; not exercised on real records |
| H - exposure | Not evaluated |



## Scientific decision

**STOP — DATA/PROTOCOL INCOMPATIBLE.** This interface mismatch is not evidence of intrinsically unsuitable records. Eligible/selected loans and defaults, payoff rates, gaps, follow-up percentages and loss availability remain unknown.

No Task 2 completion commit or modeling is justified. Next task: a separately approved, annual-bundle frame/parser/resource amendment before inspecting loan rows. Track A and the prespecified event/horizon logic remain unchanged.

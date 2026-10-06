# Track B Task 2A: annual 2010 acquisition/resource amendment

Approved 2026-10-06 by the user before any loan rows were inspected. Version: annual_2010_acquisition_v1. [Machine-readable amendment](annual_bundle_acquisition_amendment.json), [unchanged scientific protocol](mortgage_research_protocol.json), [initial stop](../../reports/track_b/FREDDIE_2010_INITIAL_PREFLIGHT_STOP.json).

The initial stop was correct for the implemented flat-sample interface but did not establish that Freddie data were unsuitable. The approved source is the original historical_data_2010.zip, 1,882,675,775 bytes, SHA-256 a82bc0f1efffccfb494b2b33f61877428bb4a6443c1a73d44bc7ee24d77aa45d. It contains four quarterly ZIPs with origination/performance text pairs and about 14.6 GB of expanded text.

## Changes and unchanged rules

Replace only the flat official-sample input frame and whole-archive 200 MiB/2 GiB rejection assumptions. The annual eligible ID universe is formed across all four origination members, then ranked once using the original SHA-256 salt. No 250-per-quarter quota. Quarter representation is an observed result. No default, loss, payoff, balance, history length or performance result influences sampling. Missing/bad histories never cause replacement.

The entire original protocol remains byte/content anchored, including vintage, at-most-1,000 IDs, t0, default proxy, month-12/month-13 rules, prior-event exclusions, censoring/unknowns/payoff/ambiguity, leakage firewall, borrower/EAD/LGD limitations and Track A preservation. This is not an outcome-protocol amendment. It does not authorize models or ECL.

## Streaming and resource limits

For stored outer quarter members, bounded seekable views open nested central directories directly in the original ZIP. No quarterly copy or extraction is needed. For a deflated outer wrapper, materialize at most one quarter ZIP in a Git-ignored temporary directory and remove it on success/failure. Never extract text members. Reject unexpected, duplicate, absolute/traversal, directory, executable, encrypted and symlink-like names before materialization.

Limits are explicit in JSON: quarter ZIP 1 GiB compressed/declared; text member 1 GiB compressed and 8 GiB declared; annual declared text 32 GiB; compression ratio 128; temporary quota 1 GiB; free-disk reserve 2 GiB; eligible ID metadata cap 2 million; retained performance cap 250,000; process peak memory guard 1.5 GiB; stream chunk 1 MiB; line cap 64 KiB; malformed tolerance zero. These are operational bounds, not banking validity thresholds. A resource breach stops rather than truncates, resamples or switches vintage.

## Ordered empirical gates

1. Verify approved root hash and secure nested preflight.
2. Open only origination streams, verify layout/encoding/IDs and build minimal annual ID metadata.
3. Freeze at most 1,000 IDs, quarter counts and sample fingerprint privately before any performance row access.
4. Verify performance structure, then discard non-selected rows immediately after the ID prefix; validate/retain selected records only.
5. Reuse the unchanged longitudinal/outcome/firewall implementation and assess empirical gates.

The initial STOP evidence is preserved. New run manifests distinguish source, amendment and original scientific protocol hashes. Any actual structural mismatch requires an official, versioned parser mapping and regression checks; no heuristic column shift or sign normalization. Scientific contradictions require another stop. This amendment is not empirical feasibility evidence by itself.

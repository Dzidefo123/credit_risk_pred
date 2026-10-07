# Track B Task 8C — Policy B authorization amendment

Version 1.1.0, 7 October 2026. **SOURCE SEMANTICS UNRESOLVED — RESEARCH ELIGIBILITY POLICY AUTHORIZED.**

The user explicitly authorizes Policy B for exactly A1–A4. Source validity and research eligibility remain separate: the raw source observation is preserved with unresolved semantics; research eligibility becomes false before deterministic ranking. No rate conversion/imputation, identifier rewrite, cohort reassignment, file relocation, or source deletion is permitted.

Audit history: original Task 8 STOP → Task 8A UNRESOLVED → Task 8B ROBUSTNESS POLICY READY WITH MATERIAL LIMITATIONS (`ab871bc`) → Task 8C explicit Policy B authorization and conditional recovery. Previous reports and the Task 8B design remain immutable.

| Reference | Source location | Reason | Counterfactual rank | Selected if retained |
|---|---|---|---:|---|
| A1 | 2006Q3 line 107274 | UNRESOLVED_REQUIRED_FIELD_FORMAT | 474260 | No |
| A2 | 2008Q4 line 206261 | UNRESOLVED_COHORT_MEMBERSHIP | 378990 | No |
| A3 | 2020Q4 line 17103 | UNRESOLVED_QUARTER_MEMBERSHIP | 1413681 | No |
| A4 | 2022Q4 line 98552 | UNRESOLVED_QUARTER_MEMBERSHIP | 646011 | No |

Detection requires the exact authorized archive hash, member, line, immutable private identifier and raw-record hash; it never treats arbitrary dots or mismatches as approved. Raw observations remain in the original archives and an ignored private registry. Public audit references and raw-record hashes expose no licensed identifiers. Each detected exclusion receives eligibility=false and SOURCE_QUARANTINED at the record level, not a manufactured field value or ordinary missingness.

The new [recovery protocol](multi_vintage_recovery_protocol.json) extends frozen version 1.0.0; it does not overwrite it. Eligibility/harmonization advances to 1.1.0; the unchanged numeric/source parser remains `multivintage-r47-v1.0.0`. Event definitions, sample ranking namespace, resource budgets and horizon-support thresholds remain unchanged.

Full origination validation must pass for each recovered vintage before exactly 20,000 IDs are frozen and hashed. Actual IDs must match the identifier-only inclusion counterfactual (symmetric difference zero); counts are recomputed, not substituted from expectations. Any fifth origination anomaly stops. Performance access follows sample freeze, with selected-history typing under the unchanged R47 adapter; unexpected schemas/conventions stop. Source-wide performance delimiter/column-count checks precede selected-history retention. Old 2010/2014/2018 checkpoints require verified source, sample, parser, original protocol/code and output hashes; compatibility is limited to an eligibility amendment affecting only the four stopped vintages.

Fresh recovery output directories preserve all original partial stages. A partial recovery checkpoint is not silently reset; explicit forensic review is required. No modeling or macro acquisition is authorized. Provider clarification is **NOT YET SUBMITTED** according to recorded evidence; no Freddie approval is claimed. Any later authoritative response requires a separately versioned amendment, never retroactive revision of this decision.

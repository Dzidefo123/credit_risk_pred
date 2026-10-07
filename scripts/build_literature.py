"One-shot Task 16 authoring from reviewed public literature and frozen manuscript."

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "45990488607bcc7f0cbdce60fd72a4b12541dcd3"
DECISION = "DISTINCTIVE COMBINATION PLAUSIBLE WITH MATERIAL LIMITATIONS"


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def write(name, value):
    path = ROOT / name
    if path.exists():
        raise ValueError("Additive output already exists: " + name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_references():
    reviewed = read("docs/paper/literature/primary_verification_notes.json")
    fallbacks = {r["reference_id"]: r for r in reviewed["fallback"]}
    refs = []
    for record in read("docs/paper/literature/crossref_metadata.json")["records"]:
        key = record["reference_id"]
        m = record["metadata"]
        if m:
            title = m["title"]
            venue = m["container-title"]
            r = dict(
                reference_id=key,
                title=title[0] if isinstance(title, list) else title,
                authors=[
                    a.get("name") or (a.get("given", "") + " " + a["family"]) for a in m["author"]
                ],
                year=int(re.search(r"\d{4}$", key)[0]),
                venue=venue[0] if isinstance(venue, list) else venue,
                volume=m.get("volume"),
                issue=m.get("issue"),
                pages=m.get("page") or m.get("article-number"),
                doi=m["DOI"],
                url=record["content_source"],
                notes=record["content_review"],
            )
            if key == "Deng2000":
                r["authors"][-1] = "Robert Van Order"
                r["notes"] += " Publisher corrects Crossref surname Order to Van Order."
            if key == "VanCalster2019":
                r["authors"] = r["authors"][1:] + [r["authors"][0]]
            sources = [
                {
                    "url": record["crossref_url"],
                    "kind": "DOI_CONTENT_NEGOTIATION_CROSSREF",
                    "snapshot": "docs/paper/literature/crossref_metadata.json",
                    "reference_id": key,
                },
                {"url": record["content_source"], "kind": "PUBLISHER_OR_AUTHOR_CONTENT"},
            ]
        else:
            r = dict(fallbacks[key])
            r["notes"] += " " + record["content_review"]
            sources = [
                {
                    "url": r["url"],
                    "kind": "PUBLISHER_OR_INSTITUTIONAL_METADATA",
                    "snapshot": "docs/paper/literature/primary_verification_notes.json",
                    "reference_id": key,
                },
                {"url": record["content_source"], "kind": "AUTHOR_CONTENT"},
            ]
        r.update(
            publication_type="PEER_REVIEWED",
            peer_reviewed=True,
            verification_status="VERIFIED_METADATA_CONTENT_SCOPE_LIMITED",
            verification_sources=sources,
            topics=record["buckets"],
        )
        refs.append(r)
    for item in reviewed["additional"]:
        r = dict(item)
        r["topics"] = r.pop("buckets")
        r.update(
            peer_reviewed=r["publication_type"] == "PEER_REVIEWED",
            verification_status="VERIFIED_METADATA_CONTENT_SCOPE_LIMITED",
            verification_sources=[
                {
                    "url": r["url"],
                    "kind": "PRIMARY_RECORD",
                    "snapshot": "docs/paper/literature/primary_verification_notes.json",
                    "reference_id": r["reference_id"],
                }
            ],
        )
        refs.append(r)
    for r in refs:
        r.update(
            citation_key=r["reference_id"],
            relevance=(
                "Context/method support within inspected content; no transfer of p"
                "rior metrics to our experiments"
            ),
            closest_claim_ids=[],
            quality="CORE"
            if r["reference_id"]
            in {
                "Deng1996",
                "Deng2000",
                "Bu2026",
                "Peng2026",
                "Li2023",
                "Heyard2020",
                "Gneiting2007",
                "Croushore2001",
                "Stanton1995",
                "Breeden2020",
            }
            else "SUPPORTING",
            flags=[],
            year_policy=(
                "Issue year when available; online-only year otherwise; dates retained in metadata"
            ),
        )
        if r["reference_id"] == "Bu2026":
            r["content_access"] = "PUBLISHER_INDEXED_PREVIEW_ONLY"
        else:
            r["content_access"] = "SEE_SCOPED_REVIEW_NOTES; full-paper completeness not asserted"
    return refs


def matrix(refs):
    # Unknown means unverified, never an inferred absence from an abstract.
    profiles = {
        "Deng1996": (
            "Mortgage portfolio",
            "YES",
            "UNKNOWN",
            "YES",
            "YES",
            "YES",
            "YES",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "YES",
            "Classical option-based competing hazards",
        ),
        "Deng2000": (
            "Mortgage portfolio",
            "YES",
            "UNKNOWN",
            "YES",
            "YES",
            "YES",
            "YES",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "YES",
            "Dependent hazards and borrower heterogeneity",
        ),
        "Bhattacharya2019": (
            "Freddie Mac",
            "YES",
            "YES",
            "YES",
            "YES",
            "YES",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "Bayesian proportional competing hazards",
        ),
        "Bu2026": (
            "Freddie Mac",
            "YES",
            "YES",
            "YES",
            "YES",
            "YES",
            "YES",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "YES",
            "UNKNOWN",
            "UNKNOWN",
            "Bilinear subdistributions and copula; mortgage insurance",
        ),
        "Sadhwani2021": (
            "CoreLogic",
            "YES",
            "NO",
            "MULTISTATE",
            "YES",
            "YES",
            "YES",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "YES",
            "Nonlinear multistate transition probabilities",
        ),
        "Li2023": (
            "LendingClub",
            "NO",
            "NO",
            "YES",
            "YES",
            "YES",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "PROFIT_FORECAST_CONFIRMED; risk split UNKNOWN",
            "UNKNOWN",
            "YES",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "Competing-event probability and profitability assessment",
        ),
        "Bellotti2009": (
            "Credit cards",
            "NO",
            "NO",
            "UNKNOWN",
            "YES",
            "UNKNOWN",
            "YES",
            "UNKNOWN",
            "UNKNOWN",
            "INDEPENDENT_TEST; chronology UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "Macro-conditioned credit survival",
        ),
        "Breeden2022": (
            "Fannie/Freddie",
            "YES",
            "YES",
            "UNKNOWN",
            "DELINQUENCY",
            "UNKNOWN",
            "YES",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "Multihorizon lagged behavioral survival",
        ),
        "Breeden2023": (
            "Fannie/Freddie",
            "YES",
            "YES",
            "UNKNOWN",
            "YES",
            "DATA_PRESENT; joint fit UNKNOWN",
            "ENVIRONMENT_APC",
            "UNKNOWN",
            "UNKNOWN",
            "YES",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "YES",
            "UNKNOWN",
            "APC inputs stabilize temporal ML forecasts",
        ),
        "Chen2021": (
            "Mortgage; provider not verified",
            "YES",
            "UNKNOWN",
            "UNKNOWN",
            "EARLY_DELINQUENCY",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "YES",
            "BRIER_IN_PREVIEW",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "Mortgage ML metrics and out-of-time comparisons",
        ),
        "Peng2026": (
            "Freddie Mac",
            "YES",
            "YES",
            "NO_IN_INSPECTED_MODEL",
            "YES",
            "FUTURE_EXTENSION",
            "FUTURE_EXTENSION",
            "UNKNOWN",
            "UNKNOWN",
            "SIMULATED_DRIFT; not our natural holdout",
            "BRIER",
            "YES",
            "DEFAULT_SURVIVAL; competing CIF absent",
            "YES",
            "UNKNOWN",
            "Balance-based landmark survival/calibration under drift",
        ),
        "Wang2024": (
            "Freddie Mac",
            "YES",
            "YES",
            "UNKNOWN",
            "YES",
            "UNKNOWN",
            "YES",
            "UNKNOWN",
            "UNKNOWN",
            "NO_FOR_DOCUMENTED_WITHIN_QUARTER_RANDOM_TEST",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "Neural discrete hazards and APC decomposition",
        ),
        "Bianchi2026": (
            "OECD sovereign CDS",
            "NO",
            "NO",
            "NO",
            "CDS_SPREAD; not default event",
            "NO",
            "YES",
            "YES",
            "YES",
            "YES",
            "NOT_EVENT_PROBABILITY_SCORING",
            "UNKNOWN",
            "NO",
            "YES",
            "NO",
            "Real-time macro nonlinear sovereign credit prediction",
        ),
        "Breeden2020": (
            "Fannie/Freddie",
            "YES",
            "YES",
            "UNKNOWN",
            "LOSS_RESERVES",
            "UNKNOWN",
            "YES",
            "HISTORICAL_FORECASTS",
            "OBSERVATION_REVISION_CONTRACT_UNKNOWN",
            "HISTORICAL_FORECAST_CYCLES",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "MODEL_SCENARIO_SENSITIVITY",
            "UNKNOWN",
            "CECL procyclicality depends on model",
        ),
        "Heyard2020": (
            "Clinical",
            "NO",
            "NO",
            "YES",
            "NO",
            "NO",
            "NO",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "INDEPENDENT_VALIDATION",
            "BRIER_ERROR",
            "YES",
            "YES",
            "UNKNOWN",
            "NO",
            "Discrete competing-risk validation methodology",
        ),
        "OPSurv2024": (
            "Multiple; Freddie benchmark",
            "YES",
            "YES",
            "METHOD_SUPPORTS; mortgage task scope limited",
            "DELINQUENCY",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "BRIER_IN_PDF",
            "UNKNOWN",
            "YES",
            "UNKNOWN",
            "UNKNOWN",
            "Functional CIF approximation",
        ),
    }
    fields = (
        "dataset mortgage_data Freddie_data competing_risk default prepaym"
        "ent macro_covariates PIT_macro revision_aware temporal_holdout pr"
        "oper_scores calibration CIF distribution_shift_analysis refinanci"
        "ng_incentive main_contribution"
    ).split()
    by_id = {r["reference_id"]: r for r in refs}
    rows = []
    for key, values in profiles.items():
        r = by_id[key]
        row = dict(zip(fields, values, strict=True))
        row.update(
            paper=key,
            year=r["year"],
            title=r["title"],
            overlap_with_us=(
                "Shared domain/method or validation component as recorded; numeric"
                " results not comparable"
            ),
            difference_from_us=(
                "Our frozen natural-calendar joint-probability evaluation; absence"
                " of full combination in prior work NOT established"
            ),
            threat_to_novelty="HIGH"
            if key in {"Bu2026", "Peng2026", "Li2023", "Heyard2020", "Breeden2020"}
            else "COMPONENT_OVERLAP",
            verification_status="VERIFIED_SCOPED; UNKNOWN cells unresolved",
            evidence_sources=r["verification_sources"],
            content_scope=r["notes"],
        )
        rows.append(row)
    return rows


def claim_map(refs):
    groups = {
        "SURV": [
            "Deng1996",
            "Deng2000",
            "Bhattacharya2019",
            "Austin2016",
            "Aalen1978",
            "Heyard2020",
        ],
        "PIT": ["Croushore2001", "Bianchi2026", "Bellotti2009", "Breeden2020"],
        "MACRO": ["Bu2026", "Bellotti2009", "Sadhwani2021", "Gneiting2007", "Heyard2020"],
        "DIAG": ["Gama2014", "Peng2026", "Ovadia2019", "VanCalster2019"],
        "REFI": ["Stanton1995", "Schwartz1989", "Deng2000"],
        "PD": ["Chen2021", "Peng2026", "Wang2024", "Gneiting2007"],
    }
    claims = read("docs/paper/track_b_claim_evidence_registry.json")["claims"]
    used = {r["claim_id"] for r in read("docs/paper/manuscript_claim_usage.json")["used_claims"]}
    rows = []
    for c in claims:
        if c["claim_id"] not in used:
            continue
        empirical = c["claim_class"] == "SUPPORTED_EMPIRICAL"
        keys = groups.get(c["claim_id"].split("_")[0], [])
        rows.append(
            dict(
                claim_id=c["claim_id"],
                claim_class=c["claim_class"],
                supporting_reference_ids=[] if empirical else keys,
                contradicting_reference_ids=[],
                closest_prior_work=keys,
                citation_needed=bool(keys) and not empirical,
                novelty_relevance=(
                    "Established context/method; not a literature validation of our finding"
                ),
                scope="Own empirical estimates remain supported exclusively by Task14 evidence"
                if empirical
                else (
                    "Contextual support only; exact operational rule and implementatio"
                    "n remain repository-specific"
                ),
            )
        )
    for r in refs:
        r["closest_claim_ids"] = [
            c["claim_id"] for c in rows if r["reference_id"] in c["closest_prior_work"]
        ]
    return rows


def bibliography(refs):
    blocks = []
    for r in refs:
        fields = dict(
            title="{" + r["title"].replace("*", "") + "}",
            author=" and ".join("{" + a + "}" for a in r["authors"]),
            year=str(r["year"]),
            journal=r["venue"],
            url=r["url"],
        )
        for key, target in [
            ("doi", "doi"),
            ("volume", "volume"),
            ("issue", "number"),
            ("pages", "pages"),
        ]:
            if r.get(key):
                fields[target] = r[key]
        if r["publication_type"] == "PREPRINT":
            fields["note"] = "Preprint; peer review not verified"
        lines = ["@article{" + r["reference_id"] + ","]
        lines += ["  " + k + " = {" + v + "}," for k, v in fields.items()]
        blocks.append("\n".join(lines + ["}"]))
    return "\n\n".join(blocks) + "\n"


RELATED = (
    "## 2. Related work\n\n### A. Mortgage default and prepayment as com"
    "peting risks\n\nMortgage competing risks are established. Deng, Qui"
    "gley and Van Order developed proportional competing hazards and l"
    "ater modeled dependent mortgage options and borrower heterogeneit"
    "y [@Deng1996; @Deng2000]. Bhattacharya, Wilson and Soyer applied "
    "Bayesian proportional competing hazards to Freddie mortgage histo"
    "ries [@Bhattacharya2019]. Bu, Wang and Yang's recent Freddie stud"
    "y combines bilinear subdistribution hazards with compatible copul"
    "as and time-varying housing-price and interest-rate covariates [@"
    "Bu2026]. Its indexed publisher preview verifies substantial overl"
    "ap; exact sample, event definitions, temporal testing, scoring, c"
    "alibration and macro release/revision treatment were not establis"
    "hed from the available evidence. These fields remain unknown, rat"
    "her than presumed absent. The present multinomial monthly model i"
    "s not a new competing-risk estimator and does not identify a late"
    "nt dependence copula.\n\n### B. Macroeconomic and refinancing deter"
    "minants of mortgage termination\n\nEconomic conditioning is also es"
    "tablished: Bellotti and Crook use varying macro predictors in cre"
    "dit survival; Sadhwani, Giesecke and Sirignano model mortgage sta"
    "te probabilities with local economic predictors [@Bellotti2009; @"
    "Sadhwani2021]. Breeden and Crook study multihorizon mortgage surv"
    "ival, while Breeden and Leonova evaluate APC inputs for temporal "
    "stability [@Breeden2022; @Breeden2023]. Wang and colleagues combi"
    "ne neural discrete hazards with APC analysis on Freddie data [@Wa"
    "ng2024]. These are distinct target and validation designs, not di"
    "rectly comparable estimates.\n\nRefinancing incentives and heteroge"
    "neous exercise behavior precede this study [@Schwartz1989; @Stant"
    "on1995; @Deng2000]. An original-coupon minus market-rate represen"
    "tation is economically motivated, not a feature invention. Task 1"
    "2 tests that restricted proxy within an already inspected tempora"
    "l evaluation; it does not observe the refinance motive or current"
    " verified contract rate. Burnout and dynamic borrower selection a"
    "re omitted. Their omission constrains interpretation without esta"
    "blishing that they caused the observed failure.\n\n### C. Predictiv"
    "e validation, calibration and temporal shift\n\nLi and colleagues a"
    "ssess default/prepayment discrimination and calibration in online"
    " lending and conduct out-of-time profitability forecasts [@Li2023"
    "]. Chen and colleagues compare mortgage early-delinquency models "
    "across metrics and temporal data [@Chen2021]. Peng and Lessmann e"
    "valuate Freddie default survival under simulated drift with landm"
    "arking and calibration; macro conditioning and competing terminat"
    "ion are described as future extensions [@Peng2026]. Thus neither "
    "forward evaluation nor joint attention to ranking and probability"
    " quality is intrinsically new.\n\nThe scoring framework uses establ"
    "ished foundations: probabilistic forecast verification, strictly "
    "proper scoring, censoring-aware survival error and competing-even"
    "t AUC [@Brier1950; @Gneiting2007; @Graf1999; @Blanche2013; @Gerds"
    "2012]. Heyard and colleagues specifically develop discrimination,"
    " calibration and error measures for discrete competing-risk predi"
    "ctions [@Heyard2020]. Aalen–Johansen and modern competing-risk re"
    "ferences ground the distinction between incidence and net surviva"
    "l risk; Fine–Gray is a related estimand/model alternative, not an"
    " estimator fitted in this study [@Aalen1978; @Austin2016; @Fine19"
    "99]. OPSurv illustrates more flexible CIF estimation, without est"
    "ablishing superiority for our frozen evaluation [@OPSurv2024]. Th"
    "ese references do not certify our implementation: censoring assum"
    "ptions, conditional entry, control definitions and weighting rema"
    "in explicit implementation-review questions.\n\nCalibration and dis"
    "crimination describe different properties. General shift studies "
    "and calibration guidance already establish that probability relia"
    "bility requires separate assessment [@VanCalster2019; @Ovadia2019"
    "; @Roschewitz2025]. Concept-drift terminology does not turn descr"
    "iptive support departures into an identified mechanism [@Gama2014"
    "]. We therefore make no claim that ranking/calibration divergence"
    " itself is new, or that general image-classification results esta"
    "blish mortgage behavior.\n\n### D. Position of this study\n\nReal-tim"
    "e vintage reconstruction has a substantial forecasting literature"
    " [@Croushore2001]. Bianchi and Jiao explicitly use release-aligne"
    "d, revision-aware macro inputs in sovereign-credit prediction [@B"
    "ianchi2026]. Breeden and Vaskouski evaluate GSE loss models with "
    "historical macro scenarios [@Breeden2020]. Thus this paper cannot"
    " claim to introduce historically available macro information to c"
    "redit research. Historical scenario availability and revision-awa"
    "re observed macro inputs are related but different contracts.\n\nWi"
    "thin the inspected literature, we did not identify a verified stu"
    "dy combining mortgage competing events, release/revision-aware ob"
    "served macro inputs, a natural temporal evaluation, proper probab"
    "ility scores, calibration, CIF consequences and support diagnosti"
    "cs in the exact design used here. This is a bounded search findin"
    "g, not proof of priority: full-text access and verification of th"
    "e closest paper remain incomplete. The defensible contribution is"
    " the recorded empirical validation result and its audited informa"
    "tion/estimand boundaries. A distinctive combination is plausible "
    "with material limitations; methodological novelty and an exhausti"
    "ve novelty search are not claimed.\n\n"
)


def manuscript(refs):
    original = (ROOT / "paper/main.md").read_text(encoding="utf-8")
    text = original.replace("**v0.1 INTERNAL DRAFT", "**v0.2 INTERNAL DRAFT", 1)
    start, end = text.index("## 2."), text.index("## 3.")
    text = text[:start] + RELATED + text[end:]
    old = (
        "Novelty relative to the broader literature remains an author-revi"
        "ew and literature-grounding question."
    )
    text = text.replace(
        old,
        (
            "These are empirical validation contributions using established me"
            "thods. The structured literature audit supports a plausible combi"
            "nation with material limitations, rather than methodological inve"
            "ntion or a priority claim [@Deng2000; @Bu2026; @Heyard2020; @Peng"
            "2026]."
        ),
    )
    position = text.index("## 11. Proposed external replication")
    limitation = (
        "### Literature-informed limitations\n\nPrepayment models have long "
        "considered heterogeneous exercise costs and selection among survi"
        "ving loans [@Stanton1995; @Deng2000]. This study omits an explici"
        "t burnout-history representation and does not model dynamic borro"
        "wer exercise behavior. Original-coupon incentives cannot distingu"
        "ish voluntary refinancing from other payoff or maturity. National"
        " macro predictors omit the local economic heterogeneity studied i"
        "n multistate mortgage models [@Sadhwani2021]. These omissions mot"
        "ivate cautious interpretation; they do not demonstrate why macro "
        "augmentation deteriorated here.\n\nThe fitted joint probabilities i"
        "mpose a shared multinomial specification; coherent probabilities "
        "do not identify latent default/prepayment dependence. The copula "
        "approach in Bu and colleagues addresses a different modeling obje"
        "ct [@Bu2026]. Our CIF recursion describes observed competing term"
        "ination under the frozen conditional-entry contract, with no caus"
        "al intervention interpretation. Literature supports the methods, "
        "but independent inspection of implementation assumptions remains "
        "necessary [@Austin2016; @Heyard2020].\n\nThe closest-work audit is "
        "incomplete at the full-text level. In particular, the available B"
        "u preview does not establish its temporal information and evaluat"
        "ion contracts. We cannot assert that it lacks PIT processing or t"
        "he complete validation combination. The literature decision is pr"
        "ovisional and should be challenged in scientific review.\n\n"
    )
    text = text[:position] + limitation + text[position:]
    start, end = text.index("## References"), text.index("## Appendices")
    references = (
        "## References\n\nCitation keys follow FirstAuthorIssueYear; one ver"
        "ified edition per work. The machine-readable source is `paper/ref"
        "erences.bib`. Preprints are explicitly labeled.\n\n"
    )
    for r in refs:
        references += (
            f"- **{r['reference_id']}**. {', '.join(r['authors'])} ({r['year']}). "
            f"[{r['title']}]({r['url']}). {r['venue']}. {r['publication_type']}.\n"
        )
    text = text[:start] + references + "\n" + text[end:]
    (ROOT / "paper/main_v0.2.md").write_text(text, encoding="utf-8")


def main():
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != BASE:
        raise ValueError("Wrong authorized base")
    if (ROOT / "paper/main_v0.2.md").exists():
        raise ValueError("v0.2 already exists")
    refs = build_references()
    mapping = claim_map(refs)
    write(
        "docs/paper/literature/reference_registry.json",
        {"references": refs, "verification_date": "2026-10-07"},
    )
    write(
        "docs/paper/closest_paper_matrix.json",
        {"unknown_policy": "UNKNOWN means not verified, not absent", "papers": matrix(refs)},
    )
    write(
        "docs/paper/claim_literature_map.json",
        {
            "claims": mapping,
            "literature_claim_scope": "Context/method only; own empirical findings unchanged",
        },
    )
    gap_refs = {
        "CG01": ["Chen2021", "Peng2026", "Sadhwani2021"],
        "CG02": ["Stanton1995", "Schwartz1989", "Deng2000"],
        "CG03": ["Aalen1978", "Austin2016", "Heyard2020"],
        "CG04": ["Blanche2013", "Graf1999", "Gerds2012", "Heyard2020"],
        "CG05": ["Peng2026", "Gama2014", "VanCalster2019", "Ovadia2019", "Roschewitz2025"],
        "CG06": ["Croushore2001", "Bianchi2026", "Breeden2020"],
    }
    gaps = []
    for g in read("docs/paper/citation_gaps.json")["gaps"]:
        gaps.append(
            dict(
                gap_id=g["gap_id"],
                topic=g["claim_or_topic"],
                status="PARTIALLY_RESOLVED"
                if g["gap_id"] in {"CG03", "CG06"}
                else "RESOLVED_CONTEXTUAL_SUPPORT",
                reference_ids=gap_refs[g["gap_id"]],
                citation_keys=gap_refs[g["gap_id"]],
                rationale=(
                    "Sources support the named context/method within their inspected s"
                    "cope; see reference-specific notes"
                ),
                residual=(
                    "Exact conditional-entry/IPCW implementation compatibility "
                    "requires author review"
                )
                if g["gap_id"] == "CG03"
                else (
                    "Credit-wide vintage-aware prior exists; exact mortgage observed-m"
                    "acro revision literature remains incomplete"
                )
                if g["gap_id"] == "CG06"
                else "No certification of implementation or transfer of prior empirical results",
            )
        )
    write("docs/paper/literature/citation_gap_resolution.json", {"gaps": gaps})
    names = [
        "competing-risk formulation",
        "Freddie dataset use",
        "macro predictors",
        "PIT macro construction",
        "temporal evaluation",
        "proper-score evaluation",
        "calibration evaluation",
        "ranking/probability divergence",
        "CIF propagation under temporal shift",
        "diagnostic support analysis",
        "refinancing-gap feature",
        "governance/reproducibility",
    ]
    statuses = [
        "ESTABLISHED",
        "ESTABLISHED",
        "ESTABLISHED",
        "INCREMENTAL",
        "ESTABLISHED",
        "ESTABLISHED",
        "ESTABLISHED",
        "ESTABLISHED",
        "POTENTIALLY_DISTINCTIVE",
        "INCREMENTAL",
        "ESTABLISHED",
        "INCREMENTAL",
    ]
    write(
        "docs/paper/literature/novelty_dimensions.json",
        {
            "decision": DECISION,
            "dimensions": [
                dict(
                    id=f"N{i}",
                    dimension=n,
                    classification=s,
                    qualification=(
                        "Individual component not an invention; N9 refers to empirical pat"
                        "tern, not mathematical propagation"
                    ),
                )
                for i, (n, s) in enumerate(zip(names, statuses, strict=True), 1)
            ],
            "combination": {
                "classification": "DISTINCTIVE_COMBINATION",
                "priority_established": False,
                "qualification": (
                    "Plausible within scoped search; Bu full-text overlap and mortgage"
                    " PIT prior unresolved"
                ),
            },
            "no_first_claim": True,
        },
    )
    (ROOT / "paper/references.bib").write_text(bibliography(refs), encoding="utf-8")
    manuscript(refs)
    previous = read("docs/paper/task15_preservation_manifest.json")
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASE], cwd=ROOT, text=True
    ).splitlines()
    public = {
        p: hashlib.sha256(
            subprocess.check_output(["git", "show", BASE + ":" + p], cwd=ROOT).replace(
                b"\r\n", b"\n"
            )
        ).hexdigest()
        for p in paths
    }
    write(
        "docs/paper/literature/task16_preservation_manifest.json",
        {
            "base_commit": BASE,
            "public_lf_hashes": public,
            "private_byte_hashes": previous["private_byte_hashes"],
            "git_refs": previous["git_refs"],
            "authorized_compatibility_change": (
                "scripts/check_manuscript.py: allow only additive references.bib; "
                "v0.1 content validation unchanged"
            ),
        },
    )
    print(
        json.dumps(
            {"references": len(refs), "matrix": len(matrix(refs)), "mapped_claims": len(mapping)}
        )
    )


if __name__ == "__main__":
    main()

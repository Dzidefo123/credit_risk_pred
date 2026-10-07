"""Aggregate public evidence with explicit exploratory language throughout."""

import json

import joblib

from credit_risk.track_b.data.schemas import digest
from credit_risk.track_b.macro_hazard.models import artifact
from credit_risk.track_b.macro_support.study import immutable_json, read_json

from .audit import PRIVATE, verify
from .study import code_hashes

SECTIONS = [
    "Executive Summary",
    "Research Motivation",
    "Post-Validation Boundary",
    "Validation Evidence Audit",
    "PIT Rate Construction",
    "Contract Rate Semantics",
    "Refinancing Incentive Definition",
    "Population",
    "Model Ladder",
    "Development",
    "Independent Validation / Exploratory Evaluation",
    "Payoff Proper Scores",
    "Payoff Calibration",
    "Refinancing-Incentive Stability",
    "Calendar Stability",
    "Vintage Stability",
    "Duration Stability",
    "Competing-Risk CIF",
    "Default Consequences",
    "Sensitivity Analyses",
    "Model-Risk Interpretation",
    "Limitations",
    "Decision",
]
LIMITATIONS = [
    "Hypothesis generated from Tasks10/11 inspected outcomes; no independent validation available.",
    "Freddie nominal-time loan variables lack historically archived knowledge-time verification.",
    (
        "Original coupon remains a proxy after modification; blank flag do"
        "es not prove no modification."
    ),
    (
        "PMMS national 30-year survey is not a personalized refinance offe"
        "r; costs/eligibility unobserved."
    ),
    "Payoff endpoint combines payoff and maturity; refinancing purpose of an exit is unobserved.",
    (
        "Facility bootstrap conditions on realized calendar; only eight co"
        "arse year blocks including partial2026."
    ),
    "Survivor selection and cohort/period/age cannot be causally separated by this experiment.",
    "CIF uses later rolling historical PIT paths, not a prospective entry-date macro forecast.",
    "AJ/IPCW require unverified independent censoring; sparse cells suppressed.",
    (
        "No burnout, threshold search, retuning, outcome recalibration, re"
        "gulatory PD or causal claims."
    ),
]


def table(values):
    lines = [
        "| Model | Joint log loss | Payoff Brier | Default Brier | Payoff AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for n, v in values.items():
        lines.append(
            f"| {n} | {v['joint_log_loss']:.8f} | {v['payoff_brier']:.8f} | "
            f"{v['default_brier']:.8f} | {v['payoff_auc']:.6f} |"
        )
    return "\n".join(lines)


def block(value):
    return "```json\n" + json.dumps(value, indent=2, allow_nan=False) + "\n```"


def make(root):
    private = root / PRIVATE
    result = read_json(private / "results.json")
    audit = read_json(private / "audit.json")
    spec = read_json(root / "docs/track_b/refinancing_incentive_prespecification.json")
    manifest = read_json(private / "models_manifest.json")
    coefficients = {}
    for n, item in manifest["models"].items():
        path = private / (n + ".joblib")
        if digest(path) != item["sha256"]:
            raise ValueError("Task12 model hash mismatch")
        coefficients[n] = artifact(joblib.load(path))
        if coefficients[n] != read_json(private / (n + "_coefficients.json")):
            raise ValueError("Task12 coefficient/preprocessing mismatch")
    evidence = dict(
        **result,
        prespecification=spec,
        prespecification_sha256=digest(
            root / "docs/track_b/refinancing_incentive_prespecification.json"
        ),
        validation_evidence_audit=audit,
        model_specifications=coefficients,
        model_hashes={n: v["sha256"] for n, v in manifest["models"].items()},
        prefit_registration=read_json(private / "prefit_registration.json"),
        code_sha256_lf=code_hashes(root),
        preservation=verify(root),
        limitations=LIMITATIONS,
        external_replication=dict(
            feasible_in_principle=True,
            acquired=False,
            source=(
                "https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-"
                "family-credit-risk-transfer/fannie-mae-single-family-loan-perform"
                "ance-data"
            ),
            requirements=[
                "Accepted terms and authorized access",
                "Separate new population seal",
                "Coupon/current-rate/modification harmonization",
                "Monthly zero-balance/default/administrative event contract",
                "PIT macro overlap and calendar maturity audit",
                "Confirm source/population disjointness, not merely a different vendor",
            ],
            recommendation="Feasibility and harmonization first; no automatic acquisition",
        ),
    )
    report = root / "reports/track_b"
    immutable_json(report / "refinancing_incentive_payoff_research.json", evidence)
    content = {
        "Executive Summary": result["decision"]["verdict"]
        + ". EXPLORATORY_ONLY; no promotion or new validation claim.\n\n"
        + table(result["metrics"]),
        "Research Motivation": (
            "A borrower-relative coupon minus market rate encodes potential re"
            "financing opportunity. This is a predictive association hypothesi"
            "s, not a causal effect or a new default-PD model."
        ),
        "Post-Validation Boundary": (
            "NEW HYPOTHESIS GENERATED BY POST-VALIDATION DIAGNOSIS. Task10 rem"
            "ains NO RELIABLE TEMPORAL MACRO INCREMENT DEMONSTRATED. Task11 re"
            "mains MACRO FAILURE MECHANISMS PARTIALLY IDENTIFIED. All old inpu"
            "ts, predictions, ledgers and conclusions are frozen."
        ),
        "Validation Evidence Audit": block(audit["candidates"])
        + (
            "\n\nNo untouched admissible/sealed evidence in the current assets. "
            "Raw unselected records are a possible future population requiring"
            " a separate audit, not an assumed virgin holdout. No new independ"
            "ent ledger created and no Task10 ledger API called."
        ),
        "PIT Rate Construction": (
            "Frozen Task9 task9_api_v5 MORTGAGE30US; previous-month-end assess"
            "ment. Every used month checked for vintage representation, archiv"
            "e interval, publication/revision/reference bounds and equality to"
            " frozen risk rows. No current-revised history or future weekly ob"
            "servation. Frozen source hash: `"
        )
        + audit["macro_table_sha256"]
        + "`.",
        "Contract Rate Semantics": (
            "orig_interest_rate in percentage units; no missing rates in eithe"
            "r modeling population. Same released nominal origination semantic"
            "s across all seven vintages; historical knowledge time UNVERIFIED"
            ". Current interest rate exists and reflects changed coupons in th"
            "e released panel, but historical modification/report availability"
            " is unverified. ORIGINAL_CONTRACT_RATE_PROXY_GAP retained, includ"
            "ing modified loans. Blank modification flag is not proof of absen"
            "ce.\n\n"
        )
        + block(audit["modification_audit"]),
        "Refinancing Incentive Definition": (
            "REFI_GAP = original coupon minus PIT market rate, percentage poin"
            "ts. 6.50 minus 4.00 = +2.50. Primary: REFI_POS=max(gap,0), REFI_N"
            "EG=min(gap,0). Linear gap only sensitivity; no redundant triple. "
            "Prespecified edges -1,0,1,2 pp, infinite tails. No outcome-driven"
            " threshold search or winsorization. Prespecification SHA256 `"
        )
        + evidence["prespecification_sha256"]
        + "`.",
        "Population": block(result["counts"])
        + (
            "\n\nExact frozen Task10 risk membership, first-event exit and facil"
            "ity-disjoint roles. Development2010-09â€“2017-12, purge2018, explor"
            "atory2019-01â€“2026-02, vintages2006/2008/2010/2014. Previously ins"
            "pected unseen cohorts audited only; no fresh validation claim."
        ),
        "Model Ladder": block(spec["model_ladder"])
        + (
            "\n\nMultinomial logistic, C1 L2, lbfgs, tol1e-8, max3000, seed61010"
            ", one thread. Frozen static predictors, duration and cohort uncha"
            "nged. Development-only preprocessing. No unrestricted calendar ef"
            "fects or absolute-rate terms. P2 versus P1 remains primary regard"
            "less of sensitivity results."
        ),
        "Development": table({n: v["development"] for n, v in result["development"].items()})
        + "\n\nIn-sample diagnostics; not validation.\n\n"
        + block(result["development_refi_bins"]),
        "Independent Validation / Exploratory Evaluation": (
            "EXPLORATORY_ONLY. Outcomes were already inspected when the hypoth"
            "esis was generated. All five fits frozen before this exploratory "
            "scoring session. No evaluation recalibration, refitting or sensit"
            "ivity selection.\n\n"
        )
        + block(result["paired"]),
        "Payoff Proper Scores": table(result["metrics"])
        + (
            "\n\nFacility-paired fixed-model uncertainty and coarse calendar-yea"
            "r uncertainty are descriptive/exploratory. AUC is secondary. Nega"
            "tive P2-P1 score differences favor P2."
        ),
        "Payoff Calibration": block({n: v["payoff"] for n, v in result["calibration"].items()})
        + "\n\nFitted diagnostic intercept/slope never applied to model probabilities.",
        "Refinancing-Incentive Stability": block(result["gap_shift"])
        + "\n\n"
        + block(result["exploratory_refi_bins"])
        + (
            "\n\nObserved bin rates confound composition; linear conditional odd"
            "s slopes are available in the aggregate JSON. No monotonicity was"
            " imposed. Annual bin summaries and small-cell suppression are in "
            "JSON."
        ),
        "Calendar Stability": block(result["calendar"])
        + (
            "\n\n2026 includes Januaryâ€“February only. Pandemic2020â€“21 and post20"
            "22â€“26 pooled cells appear in JSON. No causal pandemic claim."
        ),
        "Vintage Stability": block(
            {n: v for n, v in result["groups"].items() if n.startswith("vintage:")}
        ),
        "Duration Stability": block(
            {n: v for n, v in result["groups"].items() if n.startswith("duration:")}
        ),
        "Competing-Risk CIF": block(
            {
                h: {k: v for k, v in part.items() if k != "models"}
                | dict(
                    models={
                        n: {k: v for k, v in m.items() if k != "mean_curves"}
                        for n, m in part["models"].items()
                    }
                )
                for h, part in result["cif"].items()
            }
        )
        + (
            "\n\n12/24/36/60-month AJ and IPCW reference. Raw default/payoff/sur"
            "vival probabilities conserve mass. Historical rolling PIT paths a"
            "re not known-at-entry forecasts. No cause renormalization."
        ),
        "Default Consequences": (
            "Default remains a joint competing cause. See default Brier/AUC, a"
            "nnual suppression and default CIF above. Any CIF change combines "
            "altered default and payoff hazards; it is not proof of a better d"
            "efault-PD model."
        ),
        "Sensitivity Analyses": (
            "LINEAR is the sole alternative gap representation. P3 adds only f"
            "rozen non-rate context. Neither replaces P2 after observing outco"
            "mes. Burnout excluded before fitting to keep the hypothesis inter"
            "pretable.\n\n"
        )
        + table({n: result["metrics"][n] for n in ["P1", "P2", "LINEAR", "P3"]}),
        "Model-Risk Interpretation": (
            "The primary question is whether borrower-relative rate informatio"
            "n transports more reliably than broad absolute rate coefficients."
            " Proper scores, calibration and accumulated competing CIF jointly"
            " govern interpretation; successful ranking alone is insufficient."
            " All empirical conclusions are exploratory. Prespecified gates:\n\n"
        )
        + block(result["decision"]["gates"]),
        "Limitations": "\n".join("- " + s for s in LIMITATIONS)
        + (
            "\n\nFannie Mae external replication is feasible in principle, subje"
            "ct to access terms, independent population and a separate Freddie"
            "/Fannie harmonization contract. No Fannie data were acquired. Off"
            "icial product information: [Fannie Mae loan performance data]("
        )
        + evidence["external_replication"]["source"]
        + ").",
        "Decision": result["decision"]["verdict"]
        + ".\n\nExactly one next task: "
        + result["next_task"]
        + (
            ". No next-task implementation.\n\n[Machine-readable evidence](refin"
            "ancing_incentive_payoff_research.json) and [prespecification](../"
            "../docs/track_b/refinancing_incentive_prespecification.json)."
        ),
    }
    text = "# Refinancing Incentive Payoff Research\n\nEXPLORATORY HYPOTHESIS DEVELOPMENT\n\n"
    text += "\n\n".join("## " + s + "\n\n" + content[s] for s in SECTIONS) + "\n"
    path = report / "REFINANCING_INCENTIVE_PAYOFF_RESEARCH.md"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError("Public Task12 report already exists and differs")
    path.write_text(text, encoding="utf-8", newline="\n")
    print("Task12 aggregate report written", flush=True)

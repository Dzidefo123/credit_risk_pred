"""Public aggregate evidence and research figures; no row-level licensed disclosure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from credit_risk.track_b.data.schemas import digest
from credit_risk.track_b.macro.information import feature_hash
from credit_risk.track_b.macro_support.study import immutable_json, read_json
from credit_risk.track_b.multivintage.study import lf_hash

from .data import CATEGORICAL, MACRO, NUMERIC, load


def missingness_audit(root):
    arrays = load(root)
    dev = arrays["development"]
    evaluation = arrays["evaluation"]
    from credit_risk.track_b.macro_support.eligibility import ordinal

    groups = {
        "primary_development": dev[dev["month"] >= ordinal("2010-09")],
        "reduced_development": dev,
        "evaluation_seen": evaluation[np.isin(evaluation["vintage"], [2006, 2008, 2010, 2014])],
        "evaluation_unseen": evaluation[np.isin(evaluation["vintage"], [2018, 2020, 2022])],
    }
    result = {}
    for name, data in groups.items():
        result[name] = {}
        for vintage in np.unique(data["vintage"]):
            rows = data[data["vintage"] == vintage]
            result[name][str(vintage)] = dict(
                intervals=len(rows),
                facilities=len(np.unique(rows["facility"])),
                numeric_missing={
                    n: dict(
                        count=int(np.isnan(rows["numeric"][:, i]).sum()),
                        fraction=float(np.isnan(rows["numeric"][:, i]).mean()),
                    )
                    for i, n in enumerate(NUMERIC)
                },
                categorical_missing={
                    n: int(np.count_nonzero(rows["categories"][:, i] == "__MISSING__"))
                    for i, n in enumerate(CATEGORICAL)
                },
                unestimated_duration_181plus_intervals=int(
                    np.count_nonzero(rows["duration"] > 180)
                ),
            )
    return result


def unknown_category_audit(root, development):
    evaluation = load(root)["evaluation"]
    result = {}
    for name in ["M1", "M2", "RATE", "REDUCED_M1", "REDUCED_M2"]:
        vocabulary = development[name]["parameters"]["preprocessing"]["categories"]
        result[name] = {
            str(vintage): {
                category: int(
                    np.count_nonzero(
                        ~np.isin(
                            evaluation["categories"][evaluation["vintage"] == vintage, i],
                            vocabulary[category],
                        )
                    )
                )
                for i, category in enumerate(CATEGORICAL)
            }
            for vintage in np.unique(evaluation["vintage"])
        }
    return dict(fallback="All-zero reference encoding; no evaluation refitting", counts=result)


LIMITATIONS = [
    "Research mortgage hazards, not regulatory/IRB/IFRS9/underwriting or borrower-level PD.",
    (
        "Mortgage operational knowledge time is unverified retrospective disclosure; "
        "macro PIT alone does not cure this."
    ),
    (
        "Delayed entry conditions on surviving facilities; payoff includes maturity "
        "and default is a monthly proxy."
    ),
    (
        "Unseen cohort and unobserved development duration-band effects are "
        "reference/zero restrictions, not estimated effects."
    ),
    (
        "IPCW assumes pooled independent administrative censoring; "
        "informative censoring is not ruled out."
    ),
    (
        "Facility bootstrap conditions on the macro path; calendar uncertainty has "
        "only eight annual blocks, including partial 2026."
    ),
    (
        "Previously inspected support outcomes mean this is locked temporal evaluation "
        "after support design, not a virgin holdout."
    ),
    (
        "Rolling historical PIT CIF paths use subsequently observed macro information; "
        "prospective future paths remain unsolved."
    ),
    (
        "Coefficients are conditional predictive associations; no causal shocks, "
        "fairness assessment or EAD/LGD/ECL claims."
    ),
    (
        "Coefficient uncertainty is represented by prespecified fit sensitivity, "
        "not naive independent-row standard errors."
    ),
]


def make(root):
    private = root / "data/track_b/models/macro_hazard_v1"
    result = read_json(private / "temporal_results.json")
    development = read_json(private / "development_results.json")
    protocol = read_json(root / "docs/track_b/macro_competing_risk_protocol.json")
    ledger = read_json(private / "task10_evaluation_ledger.json")
    if ledger["state"] != "CONSUMED" or ledger["prediction_generation_count"] != 1:
        raise ValueError("Task 10 evaluation is not validly consumed exactly once")
    value = dict(
        **result,
        protocol=protocol,
        development=development,
        leave_vintage_out=read_json(private / "leave_vintage_out.json"),
        data_audit=read_json(private / "data_audit.json"),
        missingness_by_model_split_vintage=missingness_audit(root),
        unknown_evaluation_categories=unknown_category_audit(root, development),
        input_preservation=read_json(
            root / "docs/track_b/macro_competing_risk_preservation_manifest.json"
        ),
        model_manifest=read_json(private / "model_manifest.json"),
        model_specification_sha256={
            n: feature_hash(dict(protocol=protocol, parameters=r["parameters"]))
            for n, r in development.items()
        },
        eligibility_sha256_lf=protocol["eligibility_sha256_lf"],
        protocol_sha256_lf=lf_hash(root / "docs/track_b/macro_competing_risk_protocol.json"),
        ledger=dict(
            state=ledger["state"],
            registration_sha256=ledger["registration_sha256"],
            population_counts=ledger["registration"]["population_counts"],
            primary_counts=ledger["registration"]["primary_counts"],
            code_sha256_lf=ledger["registration"]["code_sha256_lf"],
            prediction_generation_count=1,
            namespace="TASK10_MACRO_HAZARD",
            virgin_holdout=False,
        ),
        reproducibility=read_json(private / "reproducibility.json"),
        preservation=read_json(private / "preservation_after.json"),
        updated_state_sensitivity=dict(
            performed=False, reason=protocol["sensitivities"]["updated_state_reason"]
        ),
        rate_representation_sensitivity=result["primary"]["RATE"],
        reduced_historical_sensitivity={
            n: result["primary"][n] for n in ["REDUCED_M1", "REDUCED_M2"]
        },
        limitations=LIMITATIONS,
        next_task="Track B Task 11 — Scenario-Conditioned Competing-Risk Stress Testing"
        if result["decision"] == "MACRO INFORMATION IMPROVES TEMPORAL COMPETING-RISK PREDICTION"
        else "Track B Task 11 — Macro Signal Attribution and Stability Analysis",
        post_evaluation_retuning=False,
    )
    figures = plot(value, root / "reports/track_b/figures/macro_hazard")
    value["figures"] = {str(p.relative_to(root)).replace("\\", "/"): digest(p) for p in figures}
    immutable_json(root / "reports/track_b/macro_competing_risk_validation.json", value)
    text = markdown(value)
    path = root / "reports/track_b/MACRO_COMPETING_RISK_VALIDATION.md"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError("Final report frozen; explicit amendment required")
    if not path.exists():
        path.write_text(text, encoding="utf-8", newline="\n")
    return value


def table(headers, rows):
    return (
        "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
                *[
                    "| "
                    + " | ".join(format(x, ".7g") if isinstance(x, float) else str(x) for x in r)
                    + " |"
                    for r in rows
                ],
            ]
        )
        + "\n"
    )


def metric_table(models):
    return table(
        ["Model", "Joint log loss", "Default Brier", "Payoff Brier", "Default AUC", "Payoff AUC"],
        [
            [
                n,
                *[
                    r["scores"].get(k)
                    for k in [
                        "joint_log_loss",
                        "default_brier",
                        "payoff_brier",
                        "default_auc",
                        "payoff_auc",
                    ]
                ],
            ]
            for n, r in models.items()
        ],
    )


def markdown(v):
    parts = ["# Track B Task 10 — Macro Competing-Risk Validation\n"]

    def section(title, text):
        parts.extend(["\n## " + title + "\n", "\n" + text + "\n"])

    section(
        "Executive Summary",
        v["decision"] + ". The main comparison is M2 minus M1 on "
        "identical, facility-disjoint, seen-vintage temporal risk intervals. No retuning "
        "followed temporal results.\n\n"
        + metric_table({n: v["primary"][n] for n in ["M0", "M1", "M2"]})
        + "\nPayoff discrimination and probability accuracy must be read separately: "
        "M1/M2 payoff AUC is "
        + format(v["primary"]["M1"]["scores"]["payoff_auc"], ".6f")
        + "/"
        + format(v["primary"]["M2"]["scores"]["payoff_auc"], ".6f")
        + ", while payoff Brier is "
        + format(v["primary"]["M1"]["scores"]["payoff_brier"], ".6f")
        + "/"
        + format(v["primary"]["M2"]["scores"]["payoff_brier"], ".6f")
        + ". Better ranking alone cannot establish a macro increment. "
        "M1 is a research comparator; this study does not establish that it is deployable.",
    )
    section(
        "Research Question",
        "Does frozen PIT macro information improve default/payoff hazards "
        "beyond structural duration/cohort and static mortgage predictors? Predictive association "
        "only; discrimination is secondary to joint multinomial log loss.",
    )
    section(
        "Data Boundary",
        "Frozen seven-vintage, 140,000-facility source sample. Task 9A primary "
        "119,629 contributing facilities / 5,541,179 intervals / 5,617 defaults / 76,729 payoffs "
        "remains unchanged. No new data acquisition, prior holdout access or model artifact "
        "regeneration. New fitted artifacts and row-level predictions remain private.",
    )
    section(
        "PIT Macro Information",
        "Seven primary coefficient terms: "
        + ", ".join(v["protocol"]["primary_macro"])
        + ". Full eight-feature eligibility is preserved. Each source/version, operand and "
        "assessment timestamp passes the frozen PIT join; no current-revised or future values. "
        "Primary interval window 2010-09–2026-02; reduced 2006-02–2026-02.",
    )
    section(
        "Eligibility",
        table(
            ["Split", "Facilities", "Intervals", "Defaults", "Payoffs"],
            [
                [n, c["facilities"], c["intervals"], c["defaults"], c["payoffs"]]
                for n, c in v["split_counts"].items()
            ],
        )
        + "\nCounts exactly match Task 9A. Six known pre-t0 months, unchanged incident prefix, "
        "consecutive months and first-payment proxy age; delayed entry retained. No facility "
        "deletion, redraw, post-endpoint exposure or date changes.",
    )
    section(
        "Competing Events",
        "Monthly 0=no event, 1=research default, 2=payoff/maturity. "
        "Unknown/ambiguous/admin source states censor according to the existing protocol; "
        "they are not no-event labels. Default/payoff remove facilities from the risk set.",
    )
    section(
        "Model Specifications",
        "M0: duration bands + vintage indicators. M1: M0 + "
        "credit score, LTV, DTI, log1p original UPB, original rate, term, purpose, occupancy. "
        "M2: M1 + seven PIT macro terms. Weak-L2 C=1, lbfgs, 3000-iteration limit, tolerance 1e-8, "
        "seed 61010; no class reweighting or hyperparameter search. Training-only medians, "
        "missing indicators, means/scales and categorical mappings are published in JSON. "
        "[Protocol](../../docs/track_b/macro_competing_risk_protocol.json).",
    )
    section(
        "Identification Constraints",
        "No spread alongside both unrestricted rate terms; "
        "no month/year fixed effects or interactions. Six vintage indicators with 2006 reference; "
        "unseen cohorts get explicitly neutral contribution in separate sensitivity. "
        "Frozen duration bands include some absent development levels; their zero effects "
        "are unestimated, not learned seasoning. See zero_design_columns in each artifact. "
        "Macro/intercept ranks are checked on distinct development months before fitting.",
    )
    section(
        "Development",
        metric_table(v["development"]) + "\nIn-sample diagnostic only, "
        "not the final scientific claim. All models passed convergence and probability "
        "conservation before the temporal ledger opened. Leave-vintage-out fits used "
        "development periods exclusively; no temporal preprocessing or tuning.",
    )
    section(
        "Temporal Validation",
        metric_table({n: v["primary"][n] for n in ["M0", "M1", "M2"]})
        + "\nLOCKED TEMPORAL EVALUATION AFTER SUPPORT DESIGN: previous aggregate outcomes "
        "were inspected. 2018 purge; 2019-01–2026-02 evaluation. New Task 10 ledger registered "
        "exact facilities/risk arrays/specification/code before fitting and consumed once. "
        "Unseen 2018/2020/2022 vintages are excluded from the primary temporal estimate.",
    )
    for title, key in [("Paired Macro Increment", "paired_facility")]:
        r = v[key]
        section(
            title,
            table(
                ["Metric", "M2−M1", "95% lower", "95% upper", "Valid replicates"],
                [
                    [n, c["delta"], c["lower"], c["upper"], c["valid_replicates"]]
                    for n, c in r["intervals"].items()
                ],
            )
            + "\n1,000 facility-cluster replicates,seed 61035; all intervals stay together. "
            "Models remain fixed. Negative proper-score differences favor M2; positive "
            "AUC differences favor M2. Facility uncertainty conditions on the realized "
            "macro path. Calendar-year block sensitivity (1,000 draws,seed 61036) is "
            "reported separately:\n\n"
            + table(
                ["Metric", "Delta", "95% lower", "95% upper"],
                [
                    [n, c["delta"], c["lower"], c["upper"]]
                    for n, c in v["paired_calendar"]["intervals"].items()
                ],
            ),
        )
    section(
        "Calibration",
        table(
            ["Model", "Cause", "Observed", "Predicted", "Intercept", "Slope"],
            [
                [n, c, r["observed_rate"], r["mean_predicted"], r["intercept"], r["slope"]]
                for n in ["M1", "M2"]
                for c, r in v["primary"][n]["calibration"].items()
            ],
        )
        + "\nDiagnostic regressions are never applied as recalibration. Decile reliability "
        "and exact support are in JSON; sparse causes (<20) are suppressed.",
    )
    cifrows = []
    for h, r in v["cif"]["horizons"].items():
        for n, m in r["models"].items():
            cifrows.append(
                [
                    h,
                    n,
                    r["facilities"],
                    r["status"],
                    m["default"]["observed_default_cif"],
                    m["default"]["mean_predicted_cif"],
                    m["default"]["ipcw_brier"],
                    m["payoff"]["observed_default_cif"],
                    m["payoff"]["mean_predicted_cif"],
                    m["payoff"]["ipcw_brier"],
                ]
            )
    section(
        "Cumulative Incidence",
        table(
            [
                "Months",
                "Model",
                "Facilities",
                "Support",
                "AJ default",
                "Modeled default",
                "IPCW default Brier",
                "AJ payoff",
                "Modeled payoff",
                "IPCW payoff Brier",
            ],
            cifrows,
        )
        + "\nOne landmark per unique facility; default/payoff counts reported in JSON. "
        "CIFdefault+CIFpayoff+survival=1. AJ handles payoff as a competing event. Pooled "
        "Task6 IPCW assumes independent censoring. Calendar-truncated landmarks are "
        "excluded per horizon without looking at outcomes. These paths use macro "
        "observed at each historical interval, not knowledge available at initial t0. "
        "No future scenarios or unsupported 84/120-month claims.",
    )
    for title, key in [("Calendar Stability", "calendar"), ("Vintage Stability", "vintage")]:
        rows = []
        for cell, models in v["stability"][key].items():
            for n, r in models.items():
                rows.append(
                    [
                        cell,
                        n,
                        r["facilities"],
                        r["defaults"],
                        r["payoffs"],
                        r["joint_log_loss"],
                        r["default_brier"],
                        r["payoff_brier"],
                        r["default_status"],
                    ]
                )
        section(
            title,
            table(
                [
                    "Cell",
                    "Model",
                    "Facilities",
                    "Defaults",
                    "Payoffs",
                    "Joint LL",
                    "Default Brier",
                    "Payoff Brier",
                    "Default status",
                ],
                rows,
            )
            + "\nCause diagnostics with fewer than 20 events are suppressed. No per-cell "
            "models were tuned. 2026 is partial. Exact AUC and event support are in JSON.",
        )
    section(
        "Pandemic Sensitivity",
        "Descriptive 2020–2021 versus adjacent 2019/2022, "
        "using unchanged M1/M2:\n\n"
        + "\n".join(
            "### " + n + "\n\n" + metric_table(r) for n, r in v["period_sensitivity"].items()
        )
        + "\nNo causal pandemic effect. Calendar coefficients can also reflect "
        "policy/intervention changes not identified by these fields.",
    )
    section(
        "Reduced Historical Sensitivity",
        metric_table(v["reduced_historical_sensitivity"])
        + "\nFive frozen terms,development begins 2006-02; evaluated on the identical "
        "primary seen-vintage temporal intervals. It cannot replace primary by performance. "
        "The 2007–2009 behavior below is explicitly in-sample descriptive:\n\n"
        + "\n".join(
            "### "
            + y
            + " — development only\n\n"
            + metric_table({n: dict(scores=r) for n, r in m.items()})
            for y, m in v["gfc_in_sample_descriptive"].items()
        ),
    )
    section(
        "Rate Representation Sensitivity",
        metric_table({n: v["primary"][n] for n in ["M2", "RATE"]})
        + "\nExactly one alternate basis: Treasury+spread replaces Treasury+mortgage rate. "
        "All other terms,rows and regularization are unchanged. L2 is not invariant to "
        "this basis change; differences are sensitivity, not representation selection. "
        "The Treasury coefficient changes its conditioning interpretation when mortgage rate "
        "is replaced by spread; its sign cannot be compared as an identical coefficient.",
    )
    section(
        "Updated-State Sensitivity",
        "Not performed, as frozen before fitting: "
        + v["updated_state_sensitivity"]["reason"]
        + ". No future loan-state trajectories.",
    )
    coeffs = [
        c
        for c in v["development"]["M2"]["parameters"]["cause_vs_none_coefficients"]
        if c["feature"] in v["protocol"]["primary_macro"]
    ]
    section(
        "Coefficient Interpretation",
        table(
            ["Cause", "Macro term", "Log odds per development SD", "Odds ratio"],
            [[c["cause"], c["feature"], c["log_odds"], c["odds_ratio"]] for c in coeffs],
        )
        + "\nConditional cause-versus-no-event odds, not hazard ratios or causal CIF changes. "
        "Signs are reported unchanged, including unexpected directions. Labor stress, "
        "equity and refinancing hypotheses do not justify forcing signs. Correlated "
        "macros, seasoning/cohort restrictions and changing selection can affect associations. "
        "Sensitivity artifacts include all LVO coefficients; coefficient plots use native "
        "macro units to avoid confusing different development SDs. No naive independent-row "
        "coefficient significance claim.\n\nUnseen-vintage extrapolation:\n\n"
        + metric_table(v["unseen_vintage"])
        + "\nNeutral/reference cohort contribution is "
        "explicit and never pooled with primary temporal results. Leave-vintage-out "
        "development-only transport diagnostics follow; no temporal observations enter "
        "these fits. Full coefficient and support details are in JSON:\n\n"
        + table(
            ["Held-out vintage", "Facilities", "Defaults", "Payoffs", "M1 joint LL", "M2 joint LL"],
            [
                [
                    y,
                    r["counts"]["facilities"],
                    r["counts"]["defaults"],
                    r["counts"]["payoffs"],
                    r["models"]["M1"]["scores"]["joint_log_loss"],
                    r["models"]["M2"]["scores"]["joint_log_loss"],
                ]
                for y, r in v["leave_vintage_out"].items()
                if "models" in r
            ],
        ),
    )
    section(
        "Limitations",
        "\n".join("- " + s for s in v["limitations"])
        + "\n\n[Research model card](../../docs/track_b/MACRO_COMPETING_RISK_MODEL_CARD.md). "
        "Method references: [multinomial logistic implementation](https://scikit-learn.org/stable/"
        "modules/generated/sklearn.linear_model.LogisticRegression.html), "
        "[competing-risk reference](https://scikit-survival.readthedocs.io/en/stable/"
        "user_guide/competing-risks.html).",
    )
    section(
        "Decision",
        v["decision"] + ". Decision uses paired temporal proper scores, "
        "facility/calendar uncertainty and prespecified degradation checks; development "
        "fit is not the acceptance gate. Next: " + v["next_task"] + ". Not implemented. "
        "Preservation and deterministic replay passed; no retuning or prior ledger reuse. "
        "Retained Track A AUC 0.868152 / Brier 0.048545 / log loss 0.176030 remain historical "
        "evidence only. Tests/checks are in macro_competing_risk_verification.json.\n\n"
        + "\n".join(
            "![" + Path(n).stem + "](figures/macro_hazard/" + Path(n).name + ")"
            for n in v["figures"]
        ),
    )
    return "\n".join(parts)


def plot(v, folder):
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"figure.dpi": 140, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    files = []

    def save(fig, name):
        p = folder / (name + ".png")
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        files.append(p)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, k in zip(axes, ["joint_log_loss", "default_brier", "payoff_brier"], strict=True):
        for i, key in enumerate(["paired_facility", "paired_calendar"]):
            r = v[key]["intervals"][k]
            ax.hlines(i, r["lower"], r["upper"], color=f"C{i}", linewidth=2)
            ax.plot(r["delta"], i, "o", color=f"C{i}")
        ax.axvline(0, color="gray", ls="--")
        ax.set_yticks([0, 1], ["Facility", "Calendar year"])
        ax.set_title(k.replace("_", " "))
        ax.set_xlabel("M2 − M1; lower is better")
    save(fig, "paired_temporal_scores")
    for cause in ["default", "payoff"]:
        fig, ax = plt.subplots(figsize=(5, 4))
        maximum = 0
        for n in ["M1", "M2"]:
            b = v["primary"][n]["calibration"][cause]["reliability"]
            x = [r["predicted"] for r in b]
            y = [r["observed"] for r in b]
            maximum = max(maximum, *x, *y)
            ax.plot(x, y, "o-", label=n)
        ax.plot([0, maximum], [0, maximum], "--", color="gray")
        ax.legend()
        ax.set(
            xlabel="Mean predicted monthly probability",
            ylabel="Observed event rate",
            title=cause + " calibration",
        )
        save(fig, cause + "_calibration")
    supported = [(int(h), r) for h, r in v["cif"]["horizons"].items() if r["models"]]
    if supported:
        h, r = max(supported, key=lambda x: x[0])
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for ax, cause, col in zip(axes, ["default", "payoff"], [1, 2], strict=True):
            ax.plot(
                range(1, h + 1),
                [e[cause + "_cif"] for e in r["observed"]],
                label="Observed AJ",
                color="black",
            )
            for n in ["M1", "M2"]:
                ax.plot(range(1, h + 1), np.asarray(r["models"][n]["mean_curves"])[:, col], label=n)
            ax.set(
                title=cause + " CIF: historical PIT path",
                xlabel="Months from one landmark",
                ylabel="Cumulative incidence",
            )
            ax.legend()
        save(fig, "rolling_pit_cif")
    for label in ["calendar", "vintage"]:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        cells = v["stability"][label]
        for n in ["M1", "M2"]:
            ax.plot(list(cells), [r[n]["joint_log_loss"] for r in cells.values()], "o-", label=n)
        ax.set(
            xlabel=label, ylabel="Joint monthly log loss", title="Seen-vintage temporal stability"
        )
        ax.legend()
        save(fig, label + "_stability")
    fits = {n: v["development"][n]["parameters"] for n in ["M2", "RATE", "REDUCED_M2"]}
    fits.update(
        {
            "LVO" + y: r["models"]["M2"]["parameters"]
            for y, r in v["leave_vintage_out"].items()
            if "models" in r
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, cause in zip(axes, ["default", "payoff"], strict=True):
        values = np.full((len(MACRO), len(fits)), np.nan)
        for j, p in enumerate(fits.values()):
            for c in p["cause_vs_none_coefficients"]:
                if c["cause"] == cause and c["feature"] in MACRO:
                    index = p["feature_order"].index(c["feature"])
                    values[MACRO.index(c["feature"]), j] = (
                        c["log_odds"] / p["preprocessing"]["scales"][index]
                    )
        limit = max(0.01, float(np.nanmax(abs(values))))
        image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(fits)), list(fits), rotation=45, ha="right")
        ax.set_yticks(range(len(MACRO)), MACRO)
        ax.set_title(cause + ": log odds per native macro unit")
        fig.colorbar(image, ax=ax, fraction=0.04)
    save(fig, "macro_coefficient_stability")
    return files

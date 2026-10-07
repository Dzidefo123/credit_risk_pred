"""Aggregate diagnostic report/figures; never publish licensed identifiers or rows."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from credit_risk.track_b.data.schemas import digest
from credit_risk.track_b.macro_hazard.protocol import PRIMARY
from credit_risk.track_b.macro_support.study import immutable_json, read_json
from credit_risk.track_b.multivintage.study import lf_hash

from .assessment import assess, supplemental
from .protocol import PRIVATE
from .verification import verify

LIMITATIONS = [
    "All findings are post-validation/exploratory; no confirmatory validation or model promotion.",
    "Macro paths are common calendar observations, not millions "
    "of independent macro samples; PSI/KS are descriptive with "
    "no threshold invalidity claims.",
    "Mahalanobis distance uses a development covariance "
    "pseudoinverse; its reference quantile is descriptive, not a "
    "Gaussian coverage guarantee.",
    "Frozen ablations and component substitutions are "
    "mathematical diagnostics, not causal interventions; "
    "correlated contributions and nonlinear responses are "
    "nonadditive.",
    "M1/M2 structural coefficients differ; their full logit gap "
    "cannot be attributed exclusively to macros.",
    "Diagnostic windows have different composition and "
    "identifiability; late defaults number only 43. Coefficient "
    "confidence intervals are not estimated.",
    "APC components, policy changes, borrower clustering and behavioral causes are not identified.",
    "Disjoint facility roles and delayed entry mean composition "
    "comparisons are not matched-facility survival estimates.",
    "Oracle offsets use the same evaluation outcomes and are "
    "optimistically biased; they provide no independently "
    "validated corrected model.",
    "CIF paths use subsequently observed historical PIT macro "
    "information; they are not prospective t0 forecasts. Prior "
    "censoring assumptions remain unverified.",
    "Mortgage operational knowledge timing remains unverified; "
    "fairness, regulatory PD, EAD, LGD and ECL are not "
    "established.",
]


def table(headers, rows):
    def value(x):
        return f"{x:.6g}" if isinstance(x, float) else str(x)

    return (
        "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
                *["| " + " | ".join(value(x) for x in row) + " |" for row in rows],
            ]
        )
        + "\n"
    )


def make(root):
    private = root / PRIVATE
    verify(root)
    result = read_json(private / "diagnostics.json")
    result["diagnostic_refits"] = read_json(private / "diagnostic_refits.json")
    accounting, stability = supplemental(root, result)
    result["score_accounting"] = accounting
    result["coefficient_stability"] = stability
    result["assessment"] = assess(result)
    result["limitations"] = LIMITATIONS
    result["input_preservation_manifest"] = read_json(
        root / "docs/track_b/macro_signal_preservation_manifest.json"
    )
    result["task10_hashes"] = {
        n: h
        for n, h in result["input_preservation_manifest"]["private_byte_hashes"].items()
        if "macro_hazard_v1/" in n
    }
    result["code_sha256_lf"] = {
        str(p.relative_to(root)).replace("\\", "/"): lf_hash(p)
        for p in sorted((root / "src/credit_risk/track_b/macro_diagnostics").glob("*.py"))
    }
    result["preservation_final"] = read_json(private / "preservation_final.json")
    result["technical_serialization_incident"] = read_json(private / "serialization_incident.json")
    result["technical_serialization_incident"]["diagnostic_array_hashes_reproduced"] = True
    result["frozen_task10_report_sha256_lf"] = lf_hash(
        root / "reports/track_b/macro_competing_risk_validation.json"
    )
    result["protocol_sha256_lf"] = lf_hash(
        root / "docs/track_b/macro_signal_diagnostic_protocol.json"
    )
    for name, expected in read_json(private / "pre_serialization_fix_output_hashes.json").items():
        if digest(private / name) != expected:
            raise ValueError("Serialization correction changed diagnostic arrays")
    paths = plot(result, root / "reports/track_b/figures/macro_signal")
    result["figures"] = {str(p.relative_to(root)).replace("\\", "/"): digest(p) for p in paths}
    immutable_json(root / "reports/track_b/macro_signal_attribution_stability.json", result)
    text = markdown(result)
    p = root / "reports/track_b/MACRO_SIGNAL_ATTRIBUTION_STABILITY.md"
    if p.exists() and p.read_text(encoding="utf-8") != text:
        raise ValueError("Published diagnostic report frozen; explicit amendment required")
    if not p.exists():
        p.write_text(text, encoding="utf-8", newline="\n")
    print(result["assessment"]["decision"], flush=True)
    return result


def markdown(v):
    parts = ["# Track B Task 11 — Macro Signal Attribution and Stability Analysis\n"]

    def section(name, content):
        parts.append("\n## " + name + "\n\n" + content + "\n")

    section(
        "Executive Summary",
        "**POST_VALIDATION_DIAGNOSTIC.** "
        + v["assessment"]["decision"]
        + ". Task10 remains frozen. "
        "The failure is not a uniform probability-level offset: "
        "payoff is severely overpredicted in 2020 "
        "and underpredicted after 2022. Macro support shifts, "
        "nonlinear tail amplification and changing "
        "conditional coefficient mappings are the strongest "
        "diagnostic mechanisms. A global intercept "
        f"oracle removes only {v['oracle']['fraction_of_frozen_excess_removed']:.1%} "
        "of excess joint loss. "
        "No corrected model, independent confirmation, causal attribution or promotion is claimed.",
    )
    old = v["frozen_task10"]
    section(
        "Frozen Task 10 Result",
        "**NO RELIABLE TEMPORAL MACRO INCREMENT DEMONSTRATED** remains unchanged.\n\n"
        + table(
            ["Model", "Joint LL", "Default Brier", "Payoff Brier", "Default AUC", "Payoff AUC"],
            [
                [
                    n,
                    *[
                        old["primary"][n]["scores"][k]
                        for k in [
                            "joint_log_loss",
                            "default_brier",
                            "payoff_brier",
                            "default_auc",
                            "payoff_auc",
                        ]
                    ],
                ]
                for n in ["M1", "M2"]
            ],
        )
        + "\nPaired joint-loss difference "
        + f"{old['paired_facility']['intervals']['joint_log_loss']['delta']:.6f}, "
        "with facility95 interval [0.015410,0.016784]. Neither the "
        "decision nor its input/output files were changed.",
    )
    section(
        "Diagnostic Boundary",
        "Every analysis is POST_VALIDATION_DIAGNOSTIC or a "
        "specifically labeled post-hoc diagnostic. "
        "Primary attribution uses frozen Task10 coefficients and "
        "exact preprocessing. Four separate DIAGNOSTIC_REFIT_ONLY "
        "window fits inspect coefficients without replacement "
        "scoring. Evaluation rows enter isolated copies for these "
        "explicitly post-hoc fits; original risk arrays/role flags "
        "remain read-only. No prior ledger API, new macro "
        "series, model family, regularization search, variable selection or Task12 implementation. "
        "[Frozen diagnostic protocol](../../docs/track_b/macro_signal_diagnostic_protocol.json).",
    )
    shifts = v["macro_shift"]["interval_weighted"]
    section(
        "Macro Distribution Shift",
        table(
            ["Feature", "Dev mean", "Eval mean", "Dev SD", "Eval SD", "SMD", "PSI", "KS"],
            [
                [
                    n,
                    r["development"]["mean"],
                    r["evaluation"]["mean"],
                    r["development"]["sd"],
                    r["evaluation"]["sd"],
                    r["standardized_mean_difference"],
                    r["psi"],
                    r["ks_distance"],
                ]
                for n, r in shifts.items()
            ],
        )
        + "\nJSON includes median, minimum, maximum and5/25/50/75/95 "
        "quantiles, plus distinct-month weighting. "
        "PSI uses development-frozen deciles and declared smoothing; "
        "KS reports distance only. Neither uses threshold "
        "labels to prove invalidity. Primary macro histories "
        "contain88 development and86 evaluation months; risk rows "
        "share common macro values and are not independent economic "
        "observations. Calendar-regime summaries include "
        "2010–12,2013–15,2016–17 and each evaluation year.",
    )
    m = v["macro_shift"]["multivariate"]
    section(
        "Support Extrapolation",
        table(
            [
                "Feature",
                "Dev min",
                "Dev max",
                "Eval min",
                "Eval max",
                "Outside dev range %",
                "Outside central90 %",
            ],
            [
                [
                    n,
                    r["development"]["minimum"],
                    r["development"]["maximum"],
                    r["evaluation"]["minimum"],
                    r["evaluation"]["maximum"],
                    100 * r["outside_development_range"]["fraction"],
                    100 * r["outside_development_central90"]["fraction"],
                ]
                for n, r in shifts.items()
            ],
        )
        + "\nOutcome-free development PCA/covariance diagnostics place "
        + f"{m['evaluation_outside_reference_months']}/86 evaluation months and "
        + f"{m['evaluation_outside_reference_intervals']:,} intervals beyond the development95 "
        "Mahalanobis-type squared-distance reference. This is "
        "descriptive geometry, not an inference about Gaussian "
        "coverage or a new exclusion rule. Development support and "
        "evaluation membership remain unchanged.",
    )
    section(
        "Mortgage Composition Shift",
        table(
            [
                "Split weighting",
                "Age median",
                "Age mean",
                "Credit score median",
                "LTV median",
                "Original rate median",
            ],
            [
                [
                    n + " / " + w,
                    r["age"]["median"],
                    r["age"]["mean"],
                    r["numeric"]["orig_credit_score"]["median"],
                    r["numeric"]["orig_ltv"]["median"],
                    r["numeric"]["orig_interest_rate"]["median"],
                ]
                for n in ["development", "evaluation"]
                for w, r in v["composition"][n].items()
            ],
        )
        + "\nJSON reports vintage counts, DTI, term, purpose, "
        "occupancy, missingness, UPB, each calendar regime and "
        "older-vintage survivor composition. Credit-score/LTV "
        "medians barely change; age/vintage mix changes strongly. "
        "These are disjoint-role conditional survivors, not a "
        "matched survival/attrition estimate. Original rate minus "
        "PIT market mortgage rate is descriptive context only, not a "
        "validated borrower refinancing incentive or a fitted new "
        "feature.",
    )
    groups = v["excess_logit"]["evaluation"]["groups"]
    section(
        "Macro Contribution Attribution",
        "Cause-versus-no-event contribution is frozen beta×Task10 standardized feature. "
        "Exact reconstruction includes both cause contrasts and multinomial normalization.\n\n"
        + table(
            ["Component", "Mean default logit gap", "Mean payoff logit gap"],
            [[n, r["default"]["mean"], r["payoff"]["mean"]] for n, r in groups.items()],
        )
        + "\nMortgage/duration/cohort/intercept coefficients also "
        "changed when M2 was fitted. Their allocation cannot "
        "be misattributed to macros. JSON contains contribution "
        "distributions by development/evaluation, calendar year "
        "and represented vintage. One-at-a-time probability "
        "attribution starts from M1, inserts one frozen M2 macro "
        "contrast and holds others at centered development "
        "reference. It is nonadditive and not unique; it is "
        "POST_HOC_PROBABILITY_ATTRIBUTION, not model performance.",
    )
    for cause, title in [("default", "Default Attribution"), ("payoff", "Payoff Attribution")]:
        section(
            title,
            table(
                [
                    "Feature",
                    "Dev contribution mean",
                    "Eval contribution mean",
                    "Eval SD",
                    "Eval5%",
                    "Eval95%",
                ],
                [
                    [
                        n,
                        v["contributions"]["development"]["all"][cause][n]["mean"],
                        r["mean"],
                        r["sd"],
                        r["quantiles"]["0.05"],
                        r["quantiles"]["0.95"],
                    ]
                    for n, r in v["contributions"]["evaluation"]["all"][cause].items()
                ],
            )
            + (
                "\nDefault ranking deterioration is partitioned by calendar, "
                "macro support, vintage and duration in JSON. "
                "The 2020 calibration slope falls near zero; later default "
                "cells are sparse and suppressed. Zeroing the frozen unemployment-change "
                "contribution raises diagnostic default AUC from0.622831 to0.790554, "
                "whereas zeroing unemployment level lowers it to0.521343. The two terms "
                "therefore have different ranking roles; neither change is selected or applied. "
                if cause == "default"
                else "\nMean macro payoff contribution is negative overall even "
                "though mean payoff probability is too high. "
                "The macro payoff logit has a large right tail, amplified "
                "nonlinearly by softmax and interacting with "
                "redistributed structural effects. A mean logit is not a "
                "mean probability. Pandemic unemployment/rates "
                "and later high-rate/inflation suppression have opposite "
                "effects. No coefficient sign was repaired."
            ),
        )
    acc = v["score_accounting"]
    section(
        "Calibration Drift",
        table(
            [
                "Year",
                "Cause",
                "Observed %",
                "M1 predicted %",
                "M2 predicted %",
                "M2 O/E",
                "M1 slope",
                "M2 slope",
            ],
            [
                [
                    y,
                    c,
                    100 * r["models"]["M1"]["calibration"][c]["observed_rate"],
                    100 * r["models"]["M1"]["calibration"][c]["mean_predicted"],
                    100 * r["models"]["M2"]["calibration"][c]["mean_predicted"],
                    r["models"]["M2"]["calibration"][c]["observed_expected_ratio"],
                    r["models"]["M1"]["calibration"][c]["slope"],
                    r["models"]["M2"]["calibration"][c]["slope"],
                ]
                for y, r in v["annual_calibration"]["evaluation"].items()
                for c in ["default", "payoff"]
            ],
        )
        + "\nCounts, diagnostic joint intercept/slope fits and "
        "development-year/subperiod behavior are in JSON. "
        "Sparse cause fits are suppressed;2026 is partial. Severe "
        "payoff divergence is visible in2020, followed by "
        "underprediction after2022; no new training cutoff is chosen.\n\n"
        "Payoff AUC/Brier divergence has exact probability-scale accounting:\n\n"
        + table(
            ["Model", "Var(y)", "Var(p)", "Cov(y,p)", "Mean bias²", "Brier"],
            [
                [
                    n,
                    *[
                        r[k]
                        for k in [
                            "event_variance",
                            "prediction_variance",
                            "covariance",
                            "mean_bias_squared",
                            "brier",
                        ]
                    ],
                ]
                for n, r in acc["brier"]["payoff"].items()
            ],
        )
        + "\nBrier=Var(y)+Var(p)−2Cov(y,p)+mean-bias². This is an exact "
        "identity, not a binned Murphy "
        "reliability decomposition. Increased ranking/covariance "
        "cannot compensate for inflated probability dispersion "
        "and bias. Joint-loss contributions by realized "
        "no-event/default/payoff class are separately reported in JSON: no-event "
        "intervals contribute +0.020298 to excess joint loss, outweighing reductions "
        "on payoff-event intervals (−0.004135) and default-event intervals (−0.000042). "
        "The model assigns excessive event probability to many intervals without exits.",
    )
    rows = []
    for cause, features in v["coefficient_stability"].items():
        for n, r in features.items():
            rows.append([cause, n, *r["common_task10_sd_log_odds"].values(), r["sign_reversal"]])
    section(
        "Coefficient Stability",
        table(
            ["Cause", "Feature", "2010–13", "2014–17", "2019–21", "2022–25", "Sign reversal"], rows
        )
        + "\nAll coefficients are DIAGNOSTIC_REFIT_ONLY. Comparison "
        "uses a common Task10 development SD; "
        "within-window SD, native-unit contrasts, odds ratios, class "
        "counts, convergence, zero columns and geometry "
        "are in JSON. Late default window contains43 events. No "
        "naive independent-row coefficient SE or confidence "
        "interval is asserted. Multiple changes are already visible "
        "between the two development windows and in "
        "Task10 leave-vintage-out fits. Treasury/mortgage/HPI payoff "
        "relationships do not show uniform transport. "
        "Changing window composition/conditioning prevents "
        "identifying pure economic coefficient drift.",
    )
    section(
        "Correlation Stability",
        table(
            ["Weighting/period", "Rank", "Condition number", "Largest VIF"],
            [
                [n, r["rank"], r["condition_number"], max(r["vif"]) if r["vif"] else None]
                for n, r in v["macro_shift"]["correlations"].items()
            ],
        )
        + "\nFull correlation matrices and window geometry are "
        "recorded. Stronger Treasury/mortgage/CPI dependencies "
        "can make coefficient attribution unstable. This does not "
        "establish their separate causal contribution. "
        "Task10 Treasury+spread sensitivity retains its essentially "
        "identical failure; no third representation is tested.",
    )
    section(
        "Pandemic Diagnostics",
        "In2020 observed monthly payoff is2.21% while M2 predicts10.05%; M1 predicts1.28%. "
        "The frozen positive unemployment-payoff association "
        "combines with low Treasury/mortgage rates and extreme "
        "unemployment-change/GDP observations. Annual mean term "
        "contributions, all-feature independent ablations "
        "and class loss accounting quantify the amplification. Their "
        "effects cannot be summed as independent causes. "
        "Default calibration slope in2020 is0.0273, against "
        "M1’s0.4525; rank/support/duration partitions are in JSON. "
        "No causal pandemic effect, policy-intervention effect or feature deletion is established.",
    )
    section(
        "Post-2022 Rate Regime",
        "M2 payoff shifts to underprediction:2023 observed0.874% versus predicted0.179%, "
        "O/E4.87. In2022–25, higher mortgage rates and inflation "
        "generate negative frozen payoff logit contributions. "
        "Window diagnostics show changing payoff associations, "
        "including a positive late-window mortgage-rate "
        "coefficient despite the frozen negative coefficient. "
        "Default event counts are too small for reliable "
        "annual ranking claims. Support and calibration tables retain every difficult year.",
    )
    section(
        "CIF Error Propagation",
        table(
            ["Horizon", "Observed payoff AJ", "M1 payoff", "M2 payoff", "M2 payoff error"],
            [
                [
                    h,
                    old["cif"]["horizons"][h]["models"]["M2"]["payoff"]["observed_default_cif"],
                    r["M1"]["payoff_cif"],
                    r["M2"]["payoff_cif"],
                    r["M2"]["payoff_cif"]
                    - old["cif"]["horizons"][h]["models"]["M2"]["payoff"]["observed_default_cif"],
                ]
                for h, r in v["cif"]["horizons"].items()
            ],
        )
        + "\nOne landmark per5,619 unique facilities. Survival weights "
        "carry monthly errors forward; the2020 exit "
        "overshoot persists even though later monthly hazards "
        "underpredict. All frozen CIFs reconstruct within "
        "declared numerical tolerances. Historical paths use "
        "subsequently observed macro information, not forecasts.",
    )
    section(
        "Component Substitution",
        table(
            ["Horizon", "Diagnostic path", "Default CIF", "Payoff CIF", "Survival"],
            [
                [h, n, r["default_cif"], r["payoff_cif"], r["survival"]]
                for h, m in v["cif"]["horizons"].items()
                for n, r in m.items()
            ],
        )
        + "\nPOST_HOC_COMPONENT_SUBSTITUTION_DIAGNOSTIC. Holding M2 "
        "default hazards fixed and substituting M1 payoff "
        "raises60-month default CIF from1.63% to2.99%: excess payoff "
        "mechanically removes exposure to later "
        "default. These paths are not deployable models or causal "
        "interventions. Raw hazard sums are checked; "
        "invalid substitutions would be suppressed rather than silently renormalized.",
    )
    section(
        "Frozen-Coefficient Ablation",
        table(
            [
                "Term zeroed",
                "Joint LL difference",
                "Default Brier difference",
                "Payoff Brier difference",
                "Default AUC",
            ],
            [
                [
                    n,
                    r["delta"]["joint_log_loss"],
                    r["delta"]["default_brier"],
                    r["delta"]["payoff_brier"],
                    acc["frozen_ablation_ranking"][n]["scores"]["default_auc"],
                ]
                for n, r in v["ablation"].items()
            ],
        )
        + "\nAll differences are diagnostic minus original frozen M2. "
        "FROZEN_COEFFICIENT_DIAGNOSTIC_ABLATION "
        "subtracts one term’s two cause contrasts without fitting. "
        "Unemployment level/change and rates have large "
        "joint-score impacts; correlated/nonlinear effects overlap "
        "and do not sum. No feature is selected or removed "
        "from Task10. Any future deletion hypothesis requires a new prespecified experiment.\n\n"
        "POST_HOC_ORACLE_DIAGNOSTIC jointly fits two multinomial "
        "intercept offsets on the same consumed outcomes. "
        f"Offsets={v['oracle']['offsets']}; apparent same-sample joint loss="
        f"{v['oracle']['diagnostic']['joint_log_loss']:.6f}. It removes "
        f"{v['oracle']['fraction_of_frozen_excess_removed']:.1%} "
        f"of excess loss and leaves {v['oracle']['residual_relative_to_frozen_reference']:.6f}. "
        "This is optimistic error accounting, not independently "
        "validated correction or superiority to M1. "
        "It improves payoff Brier but worsens default Brier. Optional slope oracle was not added.",
    )
    section(
        "Hypothesis Register",
        table(
            ["ID", "Prelisted hypothesis", "Status", "Evidence"],
            [
                [n, r["hypothesis"], r["status"], r["evidence"]]
                for n, r in v["assessment"]["hypothesis_register"].items()
            ],
        ),
    )
    section(
        "Root-Cause Assessment",
        table(
            ["Rank", "Mechanism", "Diagnostic category", "Evidence"],
            [
                [r["rank"], r["mechanism"], r["category"], r["evidence"]]
                for r in v["assessment"]["root_cause_ranking"]
            ],
        )
        + "\nThis ranks explanatory diagnostic evidence, not candidate "
        "features or models. Reduced historical M2 "
        "also worsens Task10 proper scores, so failure is not "
        "exclusive to the full HPI/mortgage feature vector. "
        "Its GFC2007–09 records remain in-sample descriptive context, never new GFC validation.",
    )
    section(
        "Model-Risk Interpretation",
        "Improved discrimination can coexist with worse calibration and joint loss. "
        "Macro-to-probability mappings can fail temporal transport "
        "despite convergence and coherent class probabilities. "
        "Competing payoff calibration changes default CIF "
        "mechanically. Historical support, proper scores, "
        "regime-specific calibration and model-use boundaries matter "
        "together. Neither M1 nor M2 is promoted.",
    )
    section(
        "Future Research Hypotheses",
        "\n".join("- " + r["hypothesis"] for r in v["assessment"]["future_research_hypotheses"])
        + "\n\nExactly one next task: **"
        + v["assessment"]["next_task"]
        + "**. Prespecify its population, incentive "
        "interpretation, benchmark, proper-score gates and fresh "
        "independent validation before fitting. "
        "Do not reuse Task10 evaluation as untouched confirmation. "
        "No Task12 model or stress engine was implemented.",
    )
    section("Limitations", "\n".join("- " + s for s in LIMITATIONS))
    section(
        "Decision",
        "**"
        + v["assessment"]["decision"]
        + "**. "
        + v["assessment"]["reason"]
        + "\n\nTask10 remains **NO RELIABLE TEMPORAL MACRO INCREMENT DEMONSTRATED**. "
        "All prior hashes, models, predictions, ledgers, samples and metrics remain unchanged. "
        "The output-serialization correction changed no diagnostic "
        "arrays; failure evidence is retained privately. "
        "Tests and preservation evidence: [verification](macro_signal_verification.json).\n\n"
        + "\n\n".join(
            "![" + Path(n).stem + "](figures/macro_signal/" + Path(n).name + ")"
            for n in v["figures"]
        ),
    )
    return "\n".join(line.rstrip() for line in "\n".join(parts).splitlines()) + "\n"


def plot(v, folder):
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"figure.dpi": 140, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    paths = []

    def save(fig, name):
        fig.text(
            0.5,
            0.01,
            "POST_VALIDATION_DIAGNOSTIC | Task10 conclusion frozen",
            ha="center",
            fontsize=8,
            color="dimgray",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        p = folder / (name + ".png")
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    for ax, n in zip(axes.flat, PRIMARY, strict=False):
        for split, color in [("development", "C0"), ("evaluation", "C1")]:
            q = v["macro_shift"]["distinct_month_weighted"][n][split]
            x = [
                q["minimum"],
                *[q["quantiles"][str(a)] for a in [0.05, 0.25, 0.5, 0.75, 0.95]],
                q["maximum"],
            ]
            ax.plot(x, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1], "o-", label=split, color=color)
        ax.set(title=n, ylabel="Quantile fraction", xlabel="Native macro units")
        ax.legend()
    axes.flat[-1].axis("off")
    save(fig, "macro_distributions")
    m = v["macro_shift"]["multivariate"]
    d, e = np.array(m["development_projection"]), np.array(m["evaluation_projection"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d[:, 0], d[:, 1], label="Development months", alpha=0.7)
    ax.scatter(e[:, 0], e[:, 1], label="Evaluation months", marker="x", alpha=0.8)
    ax.set(
        xlabel="Development-fitted PC1",
        ylabel="Development-fitted PC2",
        title="Outcome-free macro support projection",
    )
    ax.legend()
    save(fig, "macro_support")
    years = list(v["annual_calibration"]["evaluation"])
    for cause in ["default", "payoff"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        observed = [
            100
            * v["annual_calibration"]["evaluation"][y]["models"]["M1"]["calibration"][cause][
                "observed_rate"
            ]
            for y in years
        ]
        ax.plot(years, observed, "o-", color="black", label="Observed")
        for n in ["M1", "M2"]:
            ax.plot(
                years,
                [
                    100
                    * v["annual_calibration"]["evaluation"][y]["models"][n]["calibration"][cause][
                        "mean_predicted"
                    ]
                    for y in years
                ],
                "o-",
                label=n,
            )
        ax.set(
            title=cause + " monthly rates;2026 partial",
            xlabel="Calendar year",
            ylabel="Probability / event rate (%)",
        )
        ax.legend()
        save(fig, "annual_" + cause)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, cause in zip(axes, ["default", "payoff"], strict=True):
        for name in PRIMARY:
            ax.plot(
                years,
                [v["contributions"]["evaluation"]["year:" + y][cause][name]["mean"] for y in years],
                label=name,
            )
        ax.axhline(0, color="gray", ls="--")
        ax.set(
            title=cause + " frozen macro logit contributions",
            ylabel="Mean cause-vs-none logit contribution",
            xlabel="Calendar year",
        )
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, "macro_logit_contributions")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fits = list(v["diagnostic_refits"])
    for ax, cause in zip(axes, ["default", "payoff"], strict=True):
        values = np.array(
            [
                [v["coefficient_stability"][cause][n]["common_task10_sd_log_odds"][w] for w in fits]
                for n in PRIMARY
            ]
        )
        limit = float(np.max(abs(values)))
        im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_yticks(range(7), PRIMARY)
        ax.set_xticks(range(len(fits)), fits, rotation=45, ha="right")
        ax.set_title(cause + " diagnostic log odds / common SD")
        for i in range(7):
            for j in range(len(fits)):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(values[i, j]) > 0.55 * limit else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.04)
    save(fig, "coefficient_stability")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, cause in zip(axes, ["default", "payoff"], strict=True):
        for n in ["M1", "M2"]:
            slopes = [
                v["annual_calibration"]["evaluation"][y]["models"][n]["calibration"][cause]["slope"]
                for y in years
            ]
            ax.plot(years, [np.nan if s is None else s for s in slopes], "o-", label=n)
        ax.axhline(1, color="gray", ls="--")
        ax.set_xlim(-0.2, len(years) - 0.8)
        ax.set_xticks(range(len(years)), years)
        ax.set(
            title=cause + " diagnostic calibration slope",
            xlabel="Year (2026 partial)",
            ylabel="Slope",
        )
        ax.legend()
    save(fig, "calibration_drift")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    observed = v["frozen_task10"]["cif"]["horizons"]["60"]["observed"]
    for ax, cause, code in zip(axes, ["default", "payoff"], [1, 2], strict=True):
        ax.plot(
            range(1, 61),
            [r[cause + "_cif"] for r in observed],
            label="Observed AJ",
            color="black",
            lw=2,
        )
        for n, q in v["cif"]["mean_curves"].items():
            ax.plot(range(1, 61), np.array(q)[:, code], label=n)
        ax.set(
            title=cause + " CIF: diagnostic substitution",
            xlabel="Months",
            ylabel="Cumulative incidence",
        )
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, "cif_error_accumulation")
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(v["ablation"])
    values = [v["ablation"][n]["delta"]["joint_log_loss"] for n in names]
    ax.barh(names, values)
    ax.axvline(0, color="gray", ls="--")
    ax.set(
        title="Frozen coefficient diagnostic ablation; no refit/selection",
        xlabel="Joint LL change vs frozen M2; lower is smaller diagnostic loss",
    )
    save(fig, "diagnostic_ablation")
    return paths

"""Post-validation diagnostics on sealed Task10 inputs; no old ledger API."""

import joblib
import numpy as np

from credit_risk.track_b.macro_hazard.cif import landmarks
from credit_risk.track_b.macro_hazard.data import macro_join
from credit_risk.track_b.macro_hazard.metrics import calibration, scores
from credit_risk.track_b.macro_hazard.models import predict
from credit_risk.track_b.macro_hazard.protocol import PRIMARY
from credit_risk.track_b.macro_hazard.study import counts, population
from credit_risk.track_b.macro_support.eligibility import monthly_text, ordinal
from credit_risk.track_b.macro_support.study import immutable_json, read_json
from credit_risk.track_b.survival.math import curves

from .distributions import (
    bin_index,
    composition,
    frozen_bins,
    macro_values,
    regimes,
    support_diagnostics,
)
from .math import (
    ablated,
    components,
    one_at_a_time,
    oracle_intercepts,
    probabilities,
    score_difference,
    substitute,
    summary,
)
from .protocol import LABEL, PRIVATE, freeze
from .verification import verify


def contribution_summary(data, values, names):
    groups = dict(all=np.ones(len(data), bool))
    groups.update(
        {"year:" + str(y): data["month"] // 12 == y for y in np.unique(data["month"] // 12)}
    )
    groups.update({"vintage:" + str(v): data["vintage"] == v for v in np.unique(data["vintage"])})
    return {
        key: {
            cause: {name: summary(values[mask, i, code]) for i, name in enumerate(names)}
            for code, cause in enumerate(["default", "payoff"])
        }
        for key, mask in groups.items()
    }


def calendar_diagnostics(data, predictions):
    output = {}
    for year in np.unique(data["month"] // 12):
        mask = data["month"] // 12 == year
        rows = data[mask]
        output[str(year)] = dict(counts=counts(rows), partial=bool(year == 2026), models={})
        for name, p in predictions.items():
            causes = {}
            for code, cause in [(1, "default"), (2, "payoff")]:
                y = (rows["event"] == code).astype(int)
                q = p[mask, code]
                diag = calibration(y, q)
                if output[str(year)]["counts"]["facilities"] < 100:
                    diag.update(
                        status="SPARSE_FACILITIES_SUPPRESSED",
                        intercept=None,
                        slope=None,
                        reliability=[],
                    )
                causes[cause] = dict(
                    **diag,
                    observed_expected_ratio=float(y.mean() / q.mean()) if q.mean() > 0 else None,
                )
            output[str(year)]["models"][name] = dict(
                scores=scores(rows, p[mask], suppress=True), calibration=causes
            )
    return output


def binned_errors(development, evaluation, predictions):
    dv, ev = macro_values(development), macro_values(evaluation)
    result = {}
    for j, name in enumerate(PRIMARY):
        edges = frozen_bins(dv[:, j])
        index = bin_index(ev[:, j], edges)
        cells = []
        for i in range(len(edges) - 1):
            mask = index == i
            if not mask.any():
                continue
            rows = evaluation[mask]
            support = counts(rows)
            causes = {}
            for code, cause in [(1, "default"), (2, "payoff")]:
                supported = (
                    support["facilities"] >= 100 and np.count_nonzero(rows["event"] == code) >= 20
                )
                causes[cause] = dict(
                    status="SUPPORTED" if supported else "SPARSE_CAUSE_SUPPRESSED",
                    observed_rate=float((rows["event"] == code).mean()) if supported else None,
                    predicted={
                        n: float(p[mask, code].mean()) if supported else None
                        for n, p in predictions.items()
                    },
                )
            cells.append(
                dict(
                    bin=i,
                    lower=float(edges[i]) if np.isfinite(edges[i]) else None,
                    upper=float(edges[i + 1]) if np.isfinite(edges[i + 1]) else None,
                    counts=support,
                    outside_development_range_intervals=int(
                        ((ev[mask, j] < dv[:, j].min()) | (ev[mask, j] > dv[:, j].max())).sum()
                    ),
                    causes=causes,
                )
            )
        result[name] = dict(bins_fitted_on="Development predictors only", cells=cells)
    return result


def error_partitions(development, evaluation, predictions):
    dv, ev = macro_values(development), macro_values(evaluation)
    outside = ((ev < dv.min(axis=0)) | (ev > dv.max(axis=0))).any(axis=1)
    groups = {"outside_any_macro_range": outside, "inside_all_macro_ranges": ~outside}
    groups.update(
        {"vintage:" + str(v): evaluation["vintage"] == v for v in np.unique(evaluation["vintage"])}
    )
    bands = np.searchsorted(
        [12, 24, 36, 60, 84, 120, 180, 240], evaluation["duration"], side="left"
    )
    names = ["0-12", "13-24", "25-36", "37-60", "61-84", "85-120", "121-180", "181-240", "241+"]
    groups.update({"duration:" + n: bands == i for i, n in enumerate(names)})
    groups.update(
        {
            "year:" + str(y): evaluation["month"] // 12 == y
            for y in np.unique(evaluation["month"] // 12)
        }
    )
    return {
        key: dict(
            counts=counts(evaluation[mask]),
            models={
                n: scores(evaluation[mask], p[mask], suppress=True) for n, p in predictions.items()
            },
        )
        for key, mask in groups.items()
        if mask.any()
    }


def joint_composition(data):
    return {key: composition(data[mask]) for key, mask in regimes(data).items()}


def cif_diagnostics(root, evaluation, models, private):
    """Same frozen landmarks/path; raw-hazard substitutions, never a corrected model."""
    entries, _ = landmarks(evaluation)
    table = {
        r["reporting_month"]: r
        for r in read_json(
            root / "data/track_b/macro/processed/task9_api_v5/macro_month_table.json"
        )
    }
    old = root / "data/track_b/models/macro_hazard_v1"
    rows = entries.copy()
    first_month = entries["month"].copy()
    first_duration = entries["duration"].copy()
    hazards = {n: [] for n in ["M1", "M2"]}
    for step in range(60):
        rows["month"] = first_month + step
        rows["duration"] = first_duration + step
        if np.any(rows["month"] > ordinal("2026-02")):
            raise ValueError("Task11 CIF must use same calendar-supported landmarks as Task10")
        for month in np.unique(rows["month"]):
            rows["macro"][rows["month"] == month] = macro_join(table, monthly_text(month), PRIMARY)
        for name, model in models.items():
            hazards[name].append(predict(model, rows))
    hazards = {n: np.stack(q, axis=1) for n, q in hazards.items()}
    original = {n: curves(q[:, :, 1], q[:, :, 2]) for n, q in hazards.items()}
    for name in models:
        for h in [12, 24, 36, 60]:
            if not np.allclose(
                original[name][:, :h], np.load(old / f"cif_{name}_{h}.npy"), atol=1e-12, rtol=1e-10
            ):
                raise ValueError("Frozen Task10 CIF reconstruction changed")
    hybrids = {}
    invalid = {}
    for name, d, p in [
        ("M2_default_M1_payoff", hazards["M2"][:, :, 1], hazards["M1"][:, :, 2]),
        ("M1_default_M2_payoff", hazards["M1"][:, :, 1], hazards["M2"][:, :, 2]),
    ]:
        if (d + p > 1 + 1e-12).any():
            invalid[name] = dict(
                status="INVALID_RAW_HAZARD_SUBSTITUTION",
                intervals=int((d + p > 1 + 1e-12).sum()),
                renormalized=False,
            )
        else:
            hybrids[name] = substitute(d, p)
    all_curves = {**original, **hybrids}
    for n, q in all_curves.items():
        np.save(private / (n + "_diagnostic_cif.npy"), q)
    return dict(
        label="POST_HOC_COMPONENT_SUBSTITUTION_DIAGNOSTIC",
        unique_facilities=len(entries),
        landmarks=len(entries),
        path_interpretation=(
            "Historical rolling PIT path; no causal intervention/prospective forecast"
        ),
        invalid_substitutions=invalid,
        original_cif_reconstruction_passed=True,
        maximum_conservation_error=float(
            max(np.max(abs(q.sum(axis=2) - 1)) for q in all_curves.values())
        ),
        monthly_mean_hazards={n: q[:, :, 1:].mean(axis=0).tolist() for n, q in hazards.items()},
        mean_curves={n: q.mean(axis=0).tolist() for n, q in all_curves.items()},
        horizons={
            str(h): {
                n: dict(
                    survival=float(q[:, h - 1, 0].mean()),
                    default_cif=float(q[:, h - 1, 1].mean()),
                    payoff_cif=float(q[:, h - 1, 2].mean()),
                )
                for n, q in all_curves.items()
            }
            for h in [12, 24, 36, 60]
        },
    )


def run(root):
    before = verify(root)
    spec = freeze(root)
    private = root / PRIVATE
    private.mkdir(parents=True, exist_ok=True)
    if (private / "diagnostics.json").exists():
        raise ValueError("Diagnostic results already sealed; explicit amendment required")
    task10 = root / "data/track_b/models/macro_hazard_v1"
    old_spec = read_json(root / "docs/track_b/macro_competing_risk_protocol.json")
    _, development, all_evaluation, seen = population(root, old_spec)
    evaluation = all_evaluation[seen]
    models = {n: joblib.load(task10 / (n + ".joblib")) for n in ["M1", "M2"]}
    predictions = {
        n: np.load(task10 / (n + "_evaluation.npy"), mmap_mode="r")[seen] for n in models
    }
    result = dict(
        label=LABEL,
        protocol=spec,
        preservation_before=before,
        frozen_task10=read_json(task10 / "temporal_results.json"),
        counts=dict(development=counts(development), evaluation=counts(evaluation)),
        task10_prediction_generation_count_unchanged=1,
        no_task10_ledger_api_called=True,
        no_model_promotion=True,
    )
    print("Outcome-free macro support and composition diagnostics", flush=True)
    result["macro_shift"] = support_diagnostics(development, evaluation)
    result["composition"] = {
        n: composition(data)
        for n, data in [
            ("development", development),
            ("evaluation", evaluation),
            ("unseen_vintage_supplement", all_evaluation[~seen]),
        ]
    }
    result["survivor_composition_by_vintage"] = {
        str(v): composition(evaluation[evaluation["vintage"] == v])
        for v in np.unique(evaluation["vintage"])
    }
    result["composition_by_regime"] = {
        n: joint_composition(data)
        for n, data in [("development", development), ("evaluation", evaluation)]
    }
    print("Frozen coefficient attribution and reconstruction", flush=True)
    development_p = {}
    for split, data in [("development", development), ("evaluation", evaluation)]:
        first, first_parts = components(models["M1"], data)
        second, second_parts = components(models["M2"], data)
        p1, p2 = probabilities(first), probabilities(second)
        if split == "evaluation":
            for n, p in [("M1", p1), ("M2", p2)]:
                if not np.allclose(p, predictions[n], atol=1e-12, rtol=1e-10):
                    raise ValueError("Task10 frozen prediction reconstruction failed")
        else:
            development_p = {"M1": p1, "M2": p2}
        macro = second_parts["macro"]
        result.setdefault("contributions", {})[split] = contribution_summary(data, macro, PRIMARY)
        groups = {
            n: second_parts[n] - first_parts[n]
            for n in ["intercept", "mortgage", "duration", "cohort"]
        }
        macro_total = macro.sum(axis=1)
        nonmacro = sum(groups.values())
        residual = float(np.max(abs((second - first) - (macro_total + nonmacro))))
        if residual > 1e-10:
            raise ValueError("Excess logit decomposition failed")
        result.setdefault("excess_logit", {})[split] = dict(
            reconstruction_max_error=residual,
            groups={
                n: {c: summary(q[:, i]) for i, c in enumerate(["default", "payoff"])}
                for n, q in {
                    **groups,
                    "macro_total": macro_total,
                    "nonmacro_total": nonmacro,
                    "total_M2_minus_M1": second - first,
                }.items()
            },
            by_year={
                str(y): {
                    n: {
                        c: summary(q[data["month"] // 12 == y, i])
                        for i, c in enumerate(["default", "payoff"])
                    }
                    for n, q in {
                        **groups,
                        "macro_total": macro_total,
                        "total_M2_minus_M1": second - first,
                    }.items()
                }
                for y in np.unique(data["month"] // 12)
            },
        )
        if split == "evaluation":
            oat = one_at_a_time(first, macro)
            result["probability_attribution"] = dict(
                label="POST_HOC_PROBABILITY_ATTRIBUTION",
                reference=(
                    "Frozen M1 cause logits + one M2 macro contribution; other macros at "
                    "development mean"
                ),
                additive=False,
                unique_attribution=False,
                feature_summaries={
                    n: {
                        cause: summary(oat[:, j, code])
                        for code, cause in [(1, "default"), (2, "payoff")]
                    }
                    for j, n in enumerate(PRIMARY)
                },
            )
            result["ablation"] = {}
            for j, name in enumerate(PRIMARY):
                q = ablated(second, macro, j)
                result["ablation"][name] = dict(
                    label="FROZEN_COEFFICIENT_DIAGNOSTIC_ABLATION",
                    refitted=False,
                    selected_or_promoted=False,
                    **score_difference(data["event"], predictions["M2"], q),
                    by_year={
                        str(y): score_difference(
                            data["event"][mask], predictions["M2"][mask], q[mask]
                        )
                        for y in np.unique(data["month"] // 12)
                        for mask in [data["month"] // 12 == y]
                    },
                )
            reference = probabilities(second - macro_total)
            result["macro_reference_baseline"] = dict(
                label="FROZEN_COEFFICIENT_DIAGNOSTIC_REFERENCE",
                **score_difference(data["event"], predictions["M2"], reference),
            )
            oracle, detail = oracle_intercepts(second, data["event"])
            result["oracle"] = dict(
                **detail, **score_difference(data["event"], predictions["M2"], oracle)
            )
            excess = (
                result["frozen_task10"]["primary"]["M2"]["scores"]["joint_log_loss"]
                - result["frozen_task10"]["primary"]["M1"]["scores"]["joint_log_loss"]
            )
            improvement = (
                result["oracle"]["original"]["joint_log_loss"]
                - result["oracle"]["diagnostic"]["joint_log_loss"]
            )
            result["oracle"].update(
                frozen_excess_joint_loss=excess,
                in_sample_loss_reduction=improvement,
                fraction_of_frozen_excess_removed=improvement / excess,
                residual_relative_to_frozen_reference=excess - improvement,
                reference_comparison_is_error_accounting_not_independent_validation=True,
            )
            np.save(private / "oracle_intercepts_diagnostic_probabilities.npy", oracle)
    result["annual_calibration"] = {
        "development": calendar_diagnostics(development, development_p),
        "evaluation": calendar_diagnostics(evaluation, predictions),
    }
    result["development_subperiod_behavior"] = {
        key: {n: scores(development[mask], p[mask]) for n, p in development_p.items()}
        for key, mask in regimes(development).items()
    }
    result["macro_bin_errors"] = binned_errors(development, evaluation, predictions)
    result["error_partitions"] = error_partitions(development, evaluation, predictions)
    result["cif"] = cif_diagnostics(root, evaluation, models, private)
    result["preservation_after"] = verify(root)
    result["reconstruction"] = dict(
        status="PASSED",
        cause_vs_none_logits=True,
        macro_plus_nonmacro=True,
        probabilities_match_task10=True,
        CIF_matches_task10=True,
        original_arrays_readonly=True,
    )
    immutable_json(private / "diagnostics.json", result)
    print("All primary diagnostic computations complete; Task10 unchanged", flush=True)

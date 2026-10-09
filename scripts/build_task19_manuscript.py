"""Render v0.3 from retained public artifacts only; never run empirical pipelines."""

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "3098da7bf17dffb7f0d0f5b2a0732b6311402864"
MACRO = "reports/track_b/macro_competing_risk_validation.json"
YEAR = "reports/paper/task18_sa01_within_period_auc.json"
CALENDAR = "reports/paper/task18_sa02_calendar_decomposition.json"
RESULTS = "reports/paper/task18_results.json"
LOCAL = "reports/paper/task19_evidence/local_closure_output.json"
FACTS = {}


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def pointer(value, path):
    for part in path.strip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def display(value, form):
    if form == "str":
        return str(value)
    if form == "csv":
        return ", ".join(map(str, value))
    if form == "range":
        return "–".join(value)
    if form == "fraction_delta":
        return f"{value:+.8f}"
    return format(value, form)


def fact(key, artifact, field, form, metric, concern="CH03", analysis="SA01"):
    FACTS[key] = dict(
        source_artifact=artifact,
        source_field=field,
        display_format=form,
        metric=metric,
        task17_change_ids=concern.split(","),
        task18_analysis=analysis,
        frozen_status=(
            "ADMITTED_PREVIOUSLY_EXECUTED_POST_HOC_LOCAL_CLOSURE"
            if artifact == LOCAL
            else (
                "RETAINED_FROZEN_EXPERIMENT_RESULT"
                if artifact == MACRO
                else "RETAINED_POST_HOC_PUBLIC_AGGREGATE_ANALYSIS"
            )
        ),
    )


def definitions():
    FACTS.clear()
    for model in ("M0", "M1", "M2"):
        fact(
            "dev_" + model,
            MACRO,
            f"/development/{model}/scores/joint_log_loss",
            ".5f",
            "development joint log loss",
            "CH09",
            "SA04",
        )
        for key, metric, form in (
            ("ll", "joint_log_loss", ".6f"),
            ("db", "default_brier", ".6f"),
            ("pb", "payoff_brier", ".6f"),
            ("da", "default_auc", ".6f"),
            ("pa", "payoff_auc", ".6f"),
        ):
            fact(
                f"seen_{key}_{model}",
                MACRO,
                f"/primary/{model}/scores/{metric}",
                form,
                "seen " + metric,
                "CH01,CH02,CH04",
                "SA02",
            )
            if model != "M0":
                fact(
                    f"unseen_{key}_{model}",
                    MACRO,
                    f"/unseen_vintage/{model}/scores/{metric}",
                    form,
                    "unseen " + metric,
                    "CH02",
                    "SA03",
                )
    for label, split in (
        ("development", "development"),
        ("seen", "evaluation_seen"),
        ("unseen", "evaluation_unseen"),
    ):
        for metric in ("facilities", "intervals", "defaults", "payoffs"):
            fact(
                label + "_" + metric,
                MACRO,
                f"/split_counts/{split}/{metric}",
                ",d",
                "population " + metric,
                "CH02,CH08",
                "SA03",
            )
    for key, field, form in (
        ("seen_vintages", "primary_vintages", "csv"),
        ("unseen_vintages", "unseen_vintages", "csv"),
        ("development_period", "development", "range"),
        ("purge_period", "purge", "range"),
        ("evaluation_period", "evaluation", "range"),
    ):
        fact(key, MACRO, "/protocol/splits/" + field, form, field, "CH02", "SA03")
    for model in ("M1", "M2"):
        for key, field in (("mean", "mean_predicted"), ("slope", "slope")):
            fact(
                f"seen_{key}_{model}",
                MACRO,
                f"/primary/{model}/calibration/payoff/{field}",
                ".3%" if key == "mean" else ".3f",
                "payoff calibration " + field,
                "CH04",
                "SA02",
            )
    fact(
        "seen_observed_payoff",
        MACRO,
        "/primary/M1/calibration/payoff/observed_rate",
        ".3%",
        "observed payoff frequency",
        "CH04",
        "SA02",
    )
    fact(
        "support_events",
        MACRO,
        "/protocol/cell_suppression/minimum_cause_events",
        "d",
        "support threshold",
    )
    fact(
        "calendar_blocks",
        MACRO,
        "/paired_calendar/clusters",
        "d",
        "calendar blocks",
        "CH04",
        "SA02",
    )
    fact(
        "seen_delta",
        MACRO,
        "/paired_facility/intervals/joint_log_loss/delta",
        "+.5f",
        "joint-loss delta",
        "CH01",
        "SA02",
    )
    fact(
        "payoff_brier_delta",
        MACRO,
        "/paired_facility/intervals/payoff_brier/delta",
        "+.6f",
        "payoff Brier delta",
        "CH04",
        "SA02",
    )
    for metric, prefix in (("joint_log_loss", "ll"), ("payoff_brier", "pb")):
        for unit, short in (("paired_facility", "fac"), ("paired_calendar", "cal")):
            for end in ("lower", "upper"):
                fact(
                    f"{prefix}_{short}_{end}",
                    MACRO,
                    f"/{unit}/intervals/{metric}/{end}",
                    "+.8f",
                    metric + " interval " + end,
                    "CH04",
                    "SA02",
                )
    fact("year2020", CALENDAR, "/per_year/1/year", "str", "calendar label", "CH01", "SA02")
    fact("year2021", CALENDAR, "/per_year/2/year", "str", "calendar label", "CH01", "SA02")
    for key, path, form, metric in (
        (
            "contribution2020",
            "/concentration/calendar_2020_contribution",
            "+.5f",
            "calendar contribution",
        ),
        ("share2020", "/concentration/calendar_2020_share_of_delta", ".2%", "calendar share"),
        (
            "weight2020",
            "/concentration/calendar_2020_interval_weight",
            ".2%",
            "calendar interval weight",
        ),
        (
            "ex2020_delta",
            "/two_ex_2020_quantities/renormalised_ex_2020_evaluation_delta/value",
            "+.5f",
            "renormalized ex-period delta",
        ),
        (
            "ex2020_residual",
            "/two_ex_2020_quantities/contribution_residual/value",
            "+.5f",
            "full-denominator residual",
        ),
        (
            "full_relative",
            "/relative_magnitude/full_period_relative_deterioration",
            ".1%",
            "relative deterioration",
        ),
        (
            "ex2020_relative",
            "/relative_magnitude/ex_2020_relative_deterioration",
            ".1%",
            "ex-period relative deterioration",
        ),
        (
            "ex2020_lower",
            "/year_block_sensitivity/ex_2020/interval/0",
            "+.5f",
            "retained aggregate sensitivity lower",
        ),
        (
            "ex2020_upper",
            "/year_block_sensitivity/ex_2020/interval/1",
            "+.5f",
            "retained aggregate sensitivity upper",
        ),
        (
            "ex2020_blocks",
            "/year_block_sensitivity/ex_2020/blocks",
            "d",
            "retained sensitivity block count",
        ),
    ):
        fact(key, CALENDAR, path, form, metric, "CH01", "SA02")
    # Bind derived counts to an admitted deterministic metadata summary, not a new outcome pass.
    fact(
        "non2020_positive",
        "reports/paper/task19_evidence/derived_metadata.json",
        "/non2020_positive",
        "d",
        "positive non-2020 year count",
        "CH01",
        "SA02",
    )
    fact(
        "non2020_years",
        CALENDAR,
        "/year_block_sensitivity/ex_2020/blocks",
        "d",
        "non-2020 year count",
        "CH01",
        "SA02",
    )
    for model in ("M1", "M2"):
        for key, field in (
            ("pooled", "pooled"),
            ("within_year", "within_year_pair_weighted"),
            ("between_year", "between_year_solved"),
        ):
            fact(f"{key}_{model}", YEAR, f"/aggregates/{model}/{field}", ".5f", key + " AUC")
    for key, path, form in (
        ("pooled_gain", "/gains/pooled", "+.5f"),
        ("within_year_gain", "/gains/within_year_stratum_gain_pair_weighted", "+.5f"),
        ("between_year_gain", "/gains/between_year_stratum_gain_solved", "+.5f"),
        ("within_pair_share", "/pair_structure/within_year_pair_share", ".3%"),
        ("between_pair_share", "/pair_structure/between_year_pair_share", ".3%"),
        ("within_contribution", "/gains/within_year_contribution_to_pooled_gain", "+.6f"),
        ("between_contribution", "/gains/between_year_contribution_to_pooled_gain", "+.6f"),
        ("within_gain_share", "/gains/within_year_contribution_share", ".2%"),
        ("between_gain_share", "/gains/between_year_contribution_share", ".2%"),
    ):
        fact(key, YEAR, path, form, key)
    for key, field in (
        ("supported_months", "/eligible_months"),
        ("supported_intervals", "/eligible_intervals"),
        ("positive_months", "/per_month_gap_signs/positive"),
        ("negative_months", "/per_month_gap_signs/negative"),
    ):
        fact(key, LOCAL, "/SA01_month" + field, ",d", key)
    fact(
        "excluded_months",
        "reports/paper/task19_evidence/derived_metadata.json",
        "/excluded_months",
        "d",
        "excluded month count",
    )
    for model in ("M1", "M2"):
        for key, field in (
            ("month", "within_month_pair_weighted"),
            ("month_equal", "within_month_equal_weighted"),
            ("eligible_pooled", "eligible_population_pooled"),
        ):
            fact(
                f"{key}_{model}",
                LOCAL,
                f"/SA01_month/aggregates/{model}/{field}",
                ".5f",
                key + " AUC",
            )
    for key, field, form in (
        ("month_gain", "within_month_pair_weighted", "+.5f"),
        ("month_contribution", "within_weighted_contribution", "fraction_delta"),
        ("between_month_contribution", "between_weighted_contribution", "fraction_delta"),
        ("eligible_pooled_gain", "eligible_pooled", "+.5f"),
    ):
        fact(key, LOCAL, "/SA01_month/gains/" + field, form, key)
    fact(
        "unseen_delta",
        RESULTS,
        "/headline_findings/SA03/unseen_delta",
        "+.6f",
        "unseen joint-loss delta",
        "CH02",
        "SA03",
    )
    for key, field, form in (
        ("landmarks", "landmarks", ",d"),
        ("modal_entry_share", "modal_month_share", ".2%"),
        ("modal_entry_month", "modal_months/0", "str"),
        ("latest_entry_month", "latest_entry", "str"),
        ("dominant_entry_start", "modal_months/0", "str"),
    ):
        fact(key, LOCAL, "/SA06_entry/" + field, form, key, "CH06", "SA06")
    fact(
        "dominant_path_end",
        "reports/paper/task19_evidence/derived_metadata.json",
        "/dominant_path_end",
        "str",
        "retained endpoint mode",
        "CH06",
        "SA06",
    )
    fact(
        "development_months",
        "reports/track_b/macro_signal_attribution_stability.json",
        "/macro_shift/multivariate/development_distances/count",
        "d",
        "distinct months",
        "CH12",
        "SA08",
    )
    diagnostic = "reports/track_b/macro_signal_attribution_stability.json"
    for key, field, form, metric in (
        (
            "outside_months",
            "/macro_shift/multivariate/evaluation_outside_reference_months",
            "d",
            "months beyond reference",
        ),
        (
            "evaluation_macro_months",
            "/macro_shift/multivariate/evaluation_distances/count",
            "d",
            "distinct evaluation months",
        ),
        (
            "development_condition",
            "/macro_shift/correlations/development_months/condition_number",
            ".1f",
            "correlation condition number",
        ),
        (
            "development_correlation",
            "/macro_shift/correlations/development_months/matrix/0/4",
            "+.3f",
            "development correlation",
        ),
        (
            "evaluation_correlation",
            "/macro_shift/correlations/evaluation_months/matrix/0/4",
            "+.3f",
            "evaluation correlation",
        ),
    ):
        fact(key, diagnostic, field, form, metric, "CH12", "SA08")
    event_source = "docs/track_b/mortgage_research_protocol.json"
    for key, field, form in (
        ("delinquency_min", "numeric_delinquency_min", "d"),
        ("delinquency_max", "numeric_delinquency_max", "d"),
        ("credit_exit_codes", "credit_termination_codes", "csv"),
        ("payoff_code", "competing_payoff_codes/0", "str"),
    ):
        fact(key, event_source, "/event/" + field, form, field, "CH16", "SA12")
    for key, field, form in (
        ("macro_feature_names", "/protocol/primary_macro", "csv"),
        ("penalty", "/protocol/estimator/penalty", "str"),
        ("bootstrap_interval", "/protocol/bootstrap/interval", "str"),
        ("facility_draws", "/paired_facility/draws", "d"),
        ("calendar_draws", "/paired_calendar/draws", "d"),
    ):
        fact(key, MACRO, field, form, key, "CH04", "SA02")
    return FACTS


def table_tokens():
    calendar = read(CALENDAR)["per_year"]
    rows = []
    for index, _row in enumerate(calendar):
        keys = []
        for field, form in (
            ("year", "str"),
            ("intervals", ",d"),
            ("weight", ".2%"),
            ("M1", ".5f"),
            ("M2", ".5f"),
            ("delta", "+.5f"),
            ("contribution", "+.5f"),
        ):
            key = f"cal_{index}_{field}"
            fact(key, CALENDAR, f"/per_year/{index}/{field}", form, field, "CH01", "SA02")
            keys.append("{{" + key + "}}")
        rows.append("| " + " | ".join(keys) + " |")
    calendar_table = (
        "| Year | Intervals | Weight | M1 joint loss | M2 joint loss | "
        "Difference | Contribution |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n" + "\n".join(rows)
    )
    cif_rows = []
    for horizon in (12, 24, 36, 60):
        # Horizon comes from a protocol value, not an unbound table label.
        key = f"cif_{horizon}_horizon"
        fact(
            key,
            MACRO,
            f"/protocol/cif/horizons/{(12, 24, 36, 60).index(horizon)}",
            "d",
            "horizon",
            "CH05",
            "SA06",
        )
        keys = ["{{" + key + "}}"]
        part = read(MACRO)["cif"]["horizons"][str(horizon)]
        for cause in ("payoff", "default"):
            key = f"cif_{horizon}_observed_{cause}"
            fact(
                key,
                MACRO,
                f"/cif/horizons/{horizon}/observed/{len(part['observed']) - 1}/{cause}_cif",
                ".4f",
                cause + " observed CIF",
                "CH05,CH11",
                "SA06",
            )
            keys.append("{{" + key + "}}")
            for model in ("M1", "M2"):
                key = f"cif_{horizon}_{model}_{cause}"
                fact(
                    key,
                    MACRO,
                    f"/cif/horizons/{horizon}/models/{model}/{cause}/mean_predicted_cif",
                    ".4f",
                    cause + " modeled CIF",
                    "CH05,CH11",
                    "SA06",
                )
                keys.append("{{" + key + "}}")
        cif_rows.append("| " + " | ".join(keys) + " |")
    return calendar_table, "\n".join(cif_rows)


def render(text):
    bindings = []
    changes = {
        c["change_id"]: c
        for c in read("reports/paper/review/task17_manuscript_change_register.json")["changes"]
    }
    statuses = read(RESULTS)["statuses"]
    survival = {
        c["claim_id"]: c["verdict"]
        for c in read("reports/paper/review/task17_claim_survival_matrix.json")["claims"]
    }
    old_claims = read("docs/paper/track_b_claim_evidence_registry.json")["claims"]
    cache = {}

    def replacement(match):
        key = match.group(1)
        entry = FACTS[key]
        source = entry["source_artifact"]
        cache.setdefault(source, read(source))
        value = pointer(cache[source], entry["source_field"])
        rendered = display(value, entry["display_format"])
        binding = dict(entry)
        binding.update(
            binding_id=f"Q{len(bindings) + 1:04d}",
            fact_key=key,
            exact_value=value,
            displayed_value=rendered,
            source_sha256_lf=digest(ROOT / source),
            source_commit=BASE
            if source not in {LOCAL, "reports/paper/task19_evidence/derived_metadata.json"}
            else "TASK19_ADMITTED_PRIOR_LOCAL_CLOSURE",
            task17_status={c: changes[c]["status"] for c in entry["task17_change_ids"]},
            task18_status=statuses.get(
                entry["task18_analysis"], "NOT_EXECUTED_STRENGTHENING_ANALYSIS"
            ),
            correction_status=(
                "T18-C01_WEIGHTED_SHARE"
                if "contribution" in key or "gain_share" in key
                else "CORRECTED_SCOPE_OR_RETAINED_VALUE"
            ),
        )
        matched = [
            c["claim_id"]
            for c in old_claims
            if any(
                ref["path"] == source
                and ref["json_pointer"] == entry["source_field"]
                and ref.get("role") == "point_estimate"
                for ref in c["source_artifacts"]
            )
        ]
        binding["original_claim_ids"] = matched
        binding["task17_primary_claim_verdicts"] = {
            cid: survival.get(cid, "NOT_IN_PRIMARY_SURVIVAL_MATRIX") for cid in matched
        }
        binding["local_closure_status"] = (
            "EXECUTED_WITH_ELIGIBLE_MONTH_SUPPORT_QUALIFICATION"
            if source == LOCAL
            else "NOT_APPLICABLE"
        )
        bindings.append(binding)
        return rendered + "<!-- Q19: " + binding["binding_id"] + " -->"

    result = re.sub(r"{{([a-zA-Z0-9_]+)}}", replacement, text).rstrip() + "\n"
    for binding in bindings:
        marker = "<!-- Q19: " + binding["binding_id"] + " -->"
        before = result[: result.index(marker)]
        headings = re.findall(r"^#{2,3} (.+)$", before, re.M)
        binding["manuscript_section"] = headings[-1] if headings else "Abstract"
        binding["claim_text"] = result.split(marker)[0].splitlines()[-1] + marker
    return result, bindings


def references(text):
    keys = {
        k
        for group in re.findall(r"\[@([^\]]+)\]", text)
        for k in re.findall(r"(?:^|;\s*@)([A-Za-z]+\d{4})", group)
    }
    refs = read("docs/paper/literature/reference_registry.json")["references"]
    return (
        "\n".join(
            f"- **{r['reference_id']}**. {', '.join(r['authors'])} ({r['year']}). "
            f"[{r['title']}]({r['url']}). {r['venue']}. {r['publication_type']}."
            for r in refs
            if r["reference_id"] in keys
        )
        + "\n"
    )


def build(refresh=False):
    output = ROOT / "paper/main_v0.3.md"
    if output.exists() and not refresh:
        raise ValueError("v0.3 already exists; explicit --refresh required")
    definitions()
    calendar, cif = table_tokens()
    template = (ROOT / "reports/paper/task19_manuscript_template.md").read_text(encoding="utf-8")
    template = template.replace("{{CALENDAR_TABLE}}", calendar).replace("{{CIF_TABLE}}", cif)
    template = template.replace("{{VERIFIED_REFERENCES}}", references(template))
    main, bindings = render(template)
    output.write_text(main, encoding="utf-8")
    payload = dict(
        version="task19-v03",
        base_commit=BASE,
        quantitative_statements=len(bindings),
        all_mapped=True,
        mathematical_definitions_are_not_empirical_claims=True,
        manuscript_identifiers_and_reference_metadata_are_not_empirical_measurements=True,
        bindings=bindings,
    )
    (ROOT / "reports/paper/task19_claim_evidence_map.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"bindings": len(bindings), "body_words": len(main.split("## References")[0].split())}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    build(args.refresh)

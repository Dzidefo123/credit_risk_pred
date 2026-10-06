"""Phase A accepts only previously published aggregate evidence."""

import hashlib
import json
import math
from pathlib import Path

from scipy.stats import beta, betabinom

ALLOWED = (
    "reports/track_b/freddie_2010_data_audit.json",
    "reports/track_b/pd_baseline_validation.json",
)
CANDIDATES = (5000, 10000, 20000)
OBJECTIVE = {"development": 150, "evaluation": 50}
SOURCE = "a82bc0f1efffccfb494b2b33f61877428bb4a6443c1a73d44bc7ee24d77aa45d"
SAMPLE = "b40de7596b0d4fbb189c1ceec5176f4461f7d290117250c1a7ec15b605ce7b11"


def lf_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def support(k, n, prior=0.5):
    a, b = k + prior, 1000 - k + prior
    return dict(
        observed=k,
        denominator=1000,
        posterior_rate_mean=a / (a + b),
        rate_95=beta.ppf([0.025, 0.975], a, b).tolist(),
        expected=k + (n - 1000) * a / (a + b),
        predictive_95=(k + betabinom.ppf([0.025, 0.975], n - 1000, a, b)).astype(int).tolist(),
    )


def calculations():
    rows = []
    for n in CANDIDATES:
        stats = {
            name: support(k, n)
            for name, k in [("overall", 31), ("development", 13), ("evaluation", 5)]
        }
        for name, k in [("development", 13), ("evaluation", 5)]:
            stats[name]["probability_meeting_target"] = float(
                betabinom.sf(OBJECTIVE[name] - k - 1, n - 1000, k + 0.5, 1000 - k + 0.5)
            )
        bound = max(
            0.0,
            stats["development"]["probability_meeting_target"]
            + stats["evaluation"]["probability_meeting_target"]
            - 1,
        )
        # Uniform-prior sensitivity is analytic, not another sample or outcome access.
        uniform = {
            name: float(betabinom.sf(OBJECTIVE[name] - k - 1, n - 1000, k + 1, 1000 - k + 1))
            for name, k in [("development", 13), ("evaluation", 5)]
        }
        rows.append(
            dict(
                n=n,
                support=stats,
                joint_probability_lower_bound=bound,
                uniform_prior_probabilities=uniform,
            )
        )
    chosen = next((r["n"] for r in rows if r["joint_probability_lower_bound"] >= 0.90), None)
    if chosen is None:
        raise ValueError("No candidate clears the fixed planning objective")
    return rows, chosen


def auc_half_width(a, events, negatives):
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    variance = (a * (1 - a) + (events - 1) * (q1 - a * a) + (negatives - 1) * (q2 - a * a)) / (
        events * negatives
    )
    return 1.96 * math.sqrt(variance)


def plan(root):
    root = Path(root)
    evidence = {name: json.loads((root / name).read_text(encoding="utf-8")) for name in ALLOWED}
    old, new = [evidence[n] for n in ALLOWED]
    if (
        old["sample"]["sample_set_sha256"] != SAMPLE
        or new["sample_set_sha256"] != SAMPLE
        or new["source_archive_sha256"] != SOURCE
    ):
        raise ValueError("Prior evidence identities changed")
    if (
        new["temporal_split"]["development"]["default_loans"] != 13
        or new["temporal_split"]["evaluation"]["default_loans"] != 5
    ):
        raise ValueError("Planning inputs differ from frozen Task 3")
    rows, n = calculations()
    result = dict(
        schema_version="1.0",
        amendment_id="nested_2010_expansion_v1",
        phase="A_FROZEN",
        chosen_n=n,
        authorization=(
            "User explicitly approved Task 4 Phase A and B; Phase B only after Phase A "
            "artifacts pass tests"
        ),
        approval_boundary=(
            "No further expansion, outcome-adaptive resizing, changed cohort or model "
            "refitting authorized"
        ),
        base_protocol_sha256_lf=new["protocol_sha256_lf"],
        source_sha256=SOURCE,
        original_sample_sha256=SAMPLE,
        task3_commit="6fd505a84cabaac82a6835c461fe7ecfa2cc56cd",
        planning_inputs_sha256_lf={name: lf_hash(root / name) for name in ALLOWED},
        candidates=rows,
        event_support_objective=OBJECTIVE,
        selection_criterion=(
            "Smallest candidate with >=0.90 union-bound lower probability of both targets"
        ),
        probability_model=(
            "Nested prediction: observed k stays fixed; new events ~ "
            "BetaBinomial(N-1000,k+0.5,1000-k+0.5); Jeffreys prior"
        ),
        assumptions=[
            (
                "Identifier hash ranking approximates outcome-independent exchangeable "
                "sampling within the fixed 2010 universe"
            ),
            (
                "Facility events treated as independent for planning; unknown borrower "
                "links and finite-universe heterogeneity not captured"
            ),
            (
                "Per-selected-ID temporal yield (5/1000) integrates eligibility and fixed "
                "split; not 5/130 applied to all selected loans"
            ),
            (
                "95% posterior predictive ranges, not frequentist confidence guarantees; "
                "finite population correction omitted (<1.1% sampled)"
            ),
            (
                "No change in calendar windows or original split hash; approximate "
                "planning cannot guarantee calibration or future performance"
            ),
        ],
        conditional_evaluation_rate=dict(
            events=5,
            eligible_loans=130,
            rate=5 / 130,
            rate_95=beta.ppf([0.025, 0.975], 5.5, 125.5).tolist(),
            eligibility_fraction=130 / 1000,
        ),
        precision_planning=dict(
            unit=(
                "one representative observation per facility, illustrative "
                "independent-unit approximation only; actual monthly cluster CI can differ"
            ),
            auc_95_halfwidth=[
                dict(
                    assumed_auc=a,
                    events=e,
                    negatives=25 * e,
                    halfwidth=auc_half_width(a, e, 25 * e),
                )
                for a in [0.7, 0.8, 0.85]
                for e in [5, 50, 100]
            ],
            prevalence_relative_95_halfwidth={str(e): 1.96 / math.sqrt(e) for e in [5, 50, 100]},
            citl_95_halfwidth_rare_event={str(e): 1.96 / math.sqrt(e) for e in [5, 50, 100]},
            bootstrap_no_default_probability={
                str(e): (1 - e / (26 * e)) ** (26 * e) for e in [5, 50, 100]
            },
            brier=(
                "Not forecast: requires future joint score/outcome distribution; no "
                "expanded predictions accessed"
            ),
        ),
        objective_rationale=(
            "50 evaluation defaults gives illustrative rare-event CITL halfwidth ~0.28 "
            "rather than ~0.88 at five; AUC halfwidth at assumed 0.8 ~0.07 rather than "
            "~0.23. Development 150 supplies materially more support for later modest "
            "fixed models. Neither threshold is a validity/regulatory minimum; "
            "dependence and calibration slope still require empirical validation."
        ),
        sampling=dict(
            salt="freddie_mortgage_research_v1",
            algorithm=(
                "SHA256(UTF8 salt + colon + ID); ascending digest then ID; first N; no quotas"
            ),
            annual_universe=1820190,
            original_first_n=1000,
            nested_required=True,
            outcome_fields_allowed=False,
        ),
        cohort=dict(
            reuse="Task 2 build_panel and Task 3 cohort/split without modification",
            development_end="2014-12",
            evaluation_start="2016-01",
            purged_year="2015",
            monthly=True,
        ),
        feasibility_gates=dict(
            adequate=OBJECTIVE,
            marginal=dict(development=100, evaluation=30),
            otherwise="EXPANSION SUPPORT INADEQUATE",
            integrity=(
                "Blocking conflicts or missing selected history prohibit adequate "
                "classification; no replacement"
            ),
        ),
        resources=dict(
            **old["resources"],
            expected_retained_rows=72232 * n / 1000,
            expected_panel_csv_bytes=14820507 * n / 1000,
            strategy=(
                "Bounded SQLite selected-row spool; one facility at a time through "
                "unchanged panel builder; bounded per-loan accounting. Universe IDs are "
                "retained during source scan for linkage, then released. Memory need not "
                "scale with retained history count."
            ),
            expected_disk_range_gib=[0.5, 1.5],
            scan_seconds_planning_range=[465, 1200],
            model_run=(
                "Cached panel avoids repeated population scans; future model memory must "
                "be measured separately"
            ),
        ),
        limits=dict(
            max_peak_memory_bytes=1610612736,
            max_retained_performance_rows=3000000,
            max_cache_bytes=2147483648,
            max_panel_bytes=1073741824,
            free_disk_reserve_bytes=4294967296,
        ),
        models_permitted=False,
        primary_future_pr_summary="Average Precision; trapezoidal PR-AUC secondary diagnostic",
        references=[
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html",
            "https://doi.org/10.1148/radiology.143.1.7063747",
        ],
    )
    path = root / "docs/track_b/sample_expansion_amendment.json"
    if path.exists():
        raise ValueError("Frozen Phase A amendment already exists; no overwrite")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md = (
        "# Track B Task 4 sample-expansion amendment\n\n"
        + result["objective_rationale"]
        + "\n\nChosen N: **"
        + str(n)
        + "**. Frozen before new performance access.\n\n"
    )
    md += (
        "| N | Overall expected (95% predictive) | Development expected (95% "
        "predictive) | Evaluation expected (95% predictive) | Joint lower "
        "probability |\n|---|---|---|---|---|\n"
    )
    for r in rows:
        cells = [
            f"{r['support'][k]['expected']:.1f} {r['support'][k]['predictive_95']}"
            for k in ["overall", "development", "evaluation"]
        ]
        md += (
            "| "
            + " | ".join([str(r["n"]), *cells, f"{r['joint_probability_lower_bound']:.3f}"])
            + " |\n"
        )
    md += (
        "\n"
        + result["probability_model"]
        + (
            ". The nested original event counts remain fixed. A joint union bound "
            "avoids assuming independence between development and evaluation events. "
            "Uniform-prior sensitivity is in JSON.\n\n"
        )
        + "\n".join("- " + s for s in result["assumptions"])
    )
    md += (
        "\n\n50 evaluation / 150 development default-loan targets are research "
        "planning objectives, not guarantees of model validity. Marginal support "
        "is 30 evaluation / 100 development; lower support is inadequate. No "
        "second expansion is permitted here. Historical knowledge time remains "
        "UNVERIFIED. AP will be the primary future PR summary; no model metrics "
        "are computed in Task 4.\n\n"
    )
    md += (
        "The original 1,000 must equal the first 1,000 of the same annual hash "
        "ranking and remain a subset of first N. New IDs are frozen privately "
        "before any performance stream opens. The 2010 universe, default proxy, "
        "t0, months 1-12 horizon, payoff, censoring, ambiguity, minimum history, "
        "firewall and original Task 3 hash/calendar split remain unchanged. This "
        "amendment supersedes only the cap and operational retention budgets for "
        "its own versioned output; original protocols and evidence remain "
        "immutable.\n\n"
    )
    md += (
        "Phase A is aggregate-only and must pass tests with amendment/code hashes "
        "attested before Phase B. Phase B scans each performance member once, "
        "without layout probes that reopen it; selected records are validated and "
        "spooled locally, then one loan at a time is passed through the existing "
        "panel builder. No raw corpus extraction or adaptive replacements.\n\n"
    )
    md += result["resources"]["strategy"] + (
        " Estimated retained rows: 1,444,640; panel CSV about 296 MB, with "
        "selected-row cache/index and disk workspace expected 0.5-1.5 GiB. Scan "
        "runtime planning range 465-1200 seconds is heuristic; retained parsing "
        "adds work, so source scan time alone is not constant. 1.5 GiB process "
        "guard; 2 GiB spool and 1 GiB panel caps; no truncation to fit limits.\n\n"
    )
    md += (
        "Brier precision cannot be predicted without score distributions. AUC "
        "approximations use Hanley-McNeil with assumed AUC scenarios, not expanded "
        "measurements; CITL/prevalence approximations use rare-event "
        "Fisher/Poisson information, not calibration fits. Unknown borrower "
        "dependence, overlapping monthly windows and score heterogeneity can "
        "worsen precision.\n\n"
    )
    md += (
        "Source SHA256: `"
        + SOURCE
        + "`. Original sample SHA256: `"
        + SAMPLE
        + "`. Original protocol SHA256: `"
        + result["base_protocol_sha256_lf"]
        + "`. Input report hashes and complete calculations are in the machine amendment.\n\n"
    )
    md += (
        result["authorization"]
        + ". "
        + result["approval_boundary"]
        + (
            ".\n\nPlanning references: [SciPy "
            "beta-binomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.s"
            "tats.betabinom.html), [Hanley and McNeil "
            "(1982)](https://doi.org/10.1148/radiology.143.1.7063747).\n"
        )
    )
    (root / "docs/track_b/SAMPLE_EXPANSION_AMENDMENT.md").write_text(md, encoding="utf-8")
    return result

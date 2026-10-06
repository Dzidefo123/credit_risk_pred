"""Pinned Release 47 positions and controlled FEATURE/OUTCOME/AUDIT roles."""

import hashlib
import json
from pathlib import Path

import pandas as pd

LAYOUT = "freddie-standard-r47-july-2026"
LAYOUT_URL = "https://www.freddiemac.com/fmac-resources/research/pdf/file_layout_july_2026.xlsx"
LAYOUT_SHA256 = "ce054271c42b7ad5f173a045c73368d997a2ac99253dcb312a45ccef43a4b13e"
PARSER_VERSION = "1.0.0"
PROTOCOL_SHA256_LF = "5fccc658726e339d3fca9c76d300b5e45c0b9fce3189b595a19e12cb425f47c5"
ORIGINATION = (
    "orig_credit_score",
    "first_payment_month",
    "first_time_buyer",
    "maturity_month",
    "msa",
    "mi_percentage",
    "number_units",
    "occupancy_status",
    "orig_cltv",
    "orig_dti",
    "orig_upb",
    "orig_ltv",
    "orig_interest_rate",
    "channel",
    "prepayment_penalty",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_prefix",
    "loan_id",
    "loan_purpose",
    "original_loan_term",
    "number_of_borrowers",
    "seller_name",
    "super_conforming",
    "pre_harp_loan_id",
    "special_program",
    "harp_indicator",
    "valuation_method",
    "interest_only",
    "vantage_score",
)
PERFORMANCE = (
    "loan_id",
    "reporting_month",
    "current_principal_balance",
    "delinquency_state",
    "loan_age",
    "remaining_legal_months",
    "defect_month",
    "modification_flag",
    "termination_code",
    "termination_month",
    "current_interest_rate",
    "non_interest_upb",
    "last_paid_due_month",
    "mi_recoveries",
    "net_sale_proceeds",
    "non_mi_recoveries",
    "total_expenses",
    "legal_costs",
    "preservation_costs",
    "taxes_insurance",
    "misc_expenses",
    "actual_loss",
    "cumulative_modification_costs",
    "rate_step_indicator",
    "payment_deferral_flag",
    "estimated_ltv",
    "removal_upb",
    "delinquent_accrued_interest",
    "disaster_flag",
    "assistance_plan",
    "period_modification_costs",
    "interest_bearing_upb",
    "mi_cancelled",
    "servicer_name",
    "bankruptcy_cramdown_costs",
)
FEATURES = (
    "orig_credit_score",
    "orig_ltv",
    "orig_dti",
    "orig_upb",
    "orig_interest_rate",
    "original_loan_term",
    "number_of_borrowers",
    "occupancy_status",
    "property_state",
    "property_type",
    "loan_purpose",
    "amortization_type",
    "vantage_score",
    "current_principal_balance",
    "current_interest_rate",
    "delinquency_state",
    "loan_age",
    "remaining_legal_months",
    "non_interest_upb",
    "modification_flag",
    "assistance_plan",
    "payment_deferral_flag",
)
OUTCOMES = ("outcome_status", "binary_default_12m", "event_offset", "observed_followup_months")
AUDIT = ("loan_id", "t0", "eligible", "eligibility_reason", "knowledge_time_status")
LOSS_FIELDS = (
    "mi_recoveries",
    "net_sale_proceeds",
    "non_mi_recoveries",
    "total_expenses",
    "legal_costs",
    "preservation_costs",
    "taxes_insurance",
    "misc_expenses",
    "actual_loss",
    "removal_upb",
    "delinquent_accrued_interest",
    "bankruptcy_cramdown_costs",
)
COLUMN_ROLES = {
    **dict.fromkeys(FEATURES, "FEATURE"),
    **dict.fromkeys(OUTCOMES, "OUTCOME"),
    **dict.fromkeys(AUDIT, "AUDIT"),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_protocol(root):
    root = Path(root)
    path = root / "docs/track_b/mortgage_research_protocol.json"
    actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != PROTOCOL_SHA256_LF:
        raise ValueError("Prespecified protocol changed; review amendment before ingestion")
    data = json.loads(path.read_text(encoding="utf-8"))
    # Validate governing policy; do not silently adapt it to observed outcomes.
    if (
        data["protocol_id"] != "freddie_mortgage_research_v1"
        or data["time"]["horizon_months"] != 12
        or data["time"]["minimum_consecutive_pre_t0_months"] != 6
        or data["event"]["credit_termination_codes"] != ["02", "03", "09"]
        or data["event"]["competing_payoff_codes"] != ["01"]
        or data["event"]["administrative_exit_codes"] != ["15", "16", "96"]
    ):
        raise ValueError("Unsupported governing protocol; review amendment before ingestion")
    for key in ("contract", "crosswalk"):
        payload = (root / data["evidence"][key + "_path"]).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(payload).hexdigest() != data["evidence"][key + "_sha256_lf"]:
            raise ValueError("Governing evidence hash mismatch")
    return data, hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def feature_frame(panel, requested=None):
    names = list(FEATURES if requested is None else requested)
    if len(names) != len(set(names)) or any(COLUMN_ROLES.get(n) != "FEATURE" for n in names):
        raise ValueError("Feature firewall rejects OUTCOME, AUDIT or unregistered columns")
    if not all(n in panel.columns for n in names):
        raise ValueError("Missing registered features")
    return panel.loc[:, names].copy()


def empty_panel():
    return pd.DataFrame(columns=[*AUDIT, *FEATURES, *OUTCOMES])

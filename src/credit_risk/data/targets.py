"""Forward-window labels kept separate from observation-time features."""

import pandas as pd
from pydantic import Field, model_validator

from credit_risk.data.validation import validate_history
from credit_risk.utils.config import ConfigModel


class TargetConfig(ConfigModel):
    horizon_months: int = Field(default=12, ge=1, le=120)
    default_dpd_threshold: int = Field(default=90, ge=1)
    indeterminate_dpd_threshold: int | None = Field(default=30, ge=1)

    @model_validator(mode="after")
    def check_thresholds(self) -> "TargetConfig":
        if (
            self.indeterminate_dpd_threshold is not None
            and self.indeterminate_dpd_threshold >= self.default_dpd_threshold
        ):
            raise ValueError("Indeterminate threshold must be below the default threshold")
        return self


def build_forward_targets(
    history: pd.DataFrame,
    config: TargetConfig | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Construct first-default targets over (observation_date, performance_end].

    Bad can be known before window maturity. Good requires every monthly snapshot.
    Prior history must cover origination through the observation date to establish
    eligibility. Future labels are not candidate feature columns.
    """
    config = config or TargetConfig()
    validate_history(history)
    cutoff = pd.Timestamp(as_of) if as_of is not None else history.observation_date.max()
    if cutoff.tzinfo is not None or pd.isna(cutoff):
        raise ValueError("as_of must be a valid timezone-naive date")
    available = history.loc[history.observation_date <= cutoff].sort_values(
        ["account_id", "observation_date"]
    )
    rows = []
    for account_id, group in available.groupby("account_id", sort=False):
        group = group.reset_index(drop=True)
        event = group.default_flag | (group.dpd >= config.default_dpd_threshold)
        for position, observation in group.iterrows():
            date = observation.observation_date
            end = date + pd.offsets.MonthEnd(config.horizon_months)
            future = group.loc[(group.observation_date > date) & (group.observation_date <= end)]
            future_event = event.loc[future.index]
            prior_complete = position + 1 == observation.months_on_book + 1
            mature = len(future) == config.horizon_months
            first_default = future.loc[future_event, "observation_date"].min()
            label = None
            if event.iloc[: position + 1].any():
                status = "preexisting_default"
            elif not prior_complete:
                status = "history_incomplete"
            elif future_event.any():
                status, label = "bad", 1
            elif not mature:
                status = "censored"
            elif (
                config.indeterminate_dpd_threshold is not None
                and (future.dpd >= config.indeterminate_dpd_threshold).any()
            ):
                status = "indeterminate"
            else:
                status, label = "good", 0
            rows.append(
                {
                    "account_id": account_id,
                    "observation_date": date,
                    "performance_end": end,
                    "label": label,
                    "status": status,
                    "observed_months": len(future),
                    "first_default_date": first_default,
                }
            )
    columns = [
        "account_id",
        "observation_date",
        "performance_end",
        "label",
        "status",
        "observed_months",
        "first_default_date",
    ]
    targets = pd.DataFrame(rows, columns=columns)
    targets["label"] = targets.label.astype("Int64")
    return targets

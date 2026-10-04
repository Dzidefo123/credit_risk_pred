"""Controlled historical perturbations; never claim a live monitoring cohort."""

from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.data.loaders import load_origination_csv
from credit_risk.data.validation import ORIGINATION_TARGET
from credit_risk.monitoring.runner import fresh_directory, load_reference
from credit_risk.validation.runner import digest, write_json


def make_monitoring_demo(source_csv, reference_dir, output_dir, seed=101):
    output = fresh_directory(output_dir)
    _, manifest = load_reference(reference_dir)
    if digest(source_csv) != manifest["source_sha256"]:
        raise ValueError("Demo source differs from monitoring reference")
    positions = pd.read_csv(Path(reference_dir) / "reference_positions.csv").row_position.to_numpy()
    frame = load_origination_csv(source_csv).frame.iloc[positions].reset_index(drop=True)
    frame = frame.drop(columns=ORIGINATION_TARGET)
    shifted = frame.copy()
    shifted["RevolvingUtilizationOfUnsecuredLines"] *= 1.5
    shifted["NumberOfTimes90DaysLate"] += 1
    shifted["MonthlyIncome"] *= 0.8
    rng = np.random.default_rng(seed)
    shifted.loc[rng.random(len(shifted)) < 0.25, "MonthlyIncome"] = np.nan
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "unchanged.csv", index=False)
    shifted.to_csv(output / "perturbed.csv", index=False)
    metadata = dict(
        data_kind="controlled perturbation of historical development applicants",
        observation_dates_available=False,
        seed=seed,
        source_sha256=digest(source_csv),
        rows=len(frame),
        labels_removed=True,
        perturbations={
            "utilization_multiplier": 1.5,
            "90_day_late_count_increment": 1,
            "income_multiplier": 0.8,
            "income_mask_probability": 0.25,
        },
        artifacts_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file()},
    )
    write_json(output / "demo_manifest.json", metadata)
    return metadata

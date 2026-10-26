"""Regenerate the entirely synthetic teaching dataset; no patient data are used."""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20261026


def make_example() -> pd.DataFrame:
    """Return 500 simulated people plus four deliberately duplicated records."""
    rng = np.random.default_rng(SEED)
    n = 500
    age = np.clip(rng.normal(57, 13, n), 18, 87).round(1)
    bmi = np.clip(rng.normal(27.5, 4.8, n), 16, 45).round(1)
    sex = rng.choice(["Female", "Male", "Nonbinary"], n, p=[0.51, 0.46, 0.03])
    smoking = rng.choice(["Never", "Former", "Current"], n, p=[0.48, 0.30, 0.22])
    severity = rng.choice(["Mild", "Moderate", "Severe"], n, p=[0.50, 0.35, 0.15])
    smoke_bp = pd.Series(smoking).map({"Never": 0, "Former": 2, "Current": 6}).to_numpy()
    severity_bp = pd.Series(severity).map({"Mild": 0, "Moderate": 4, "Severe": 10}).to_numpy()
    blood_pressure = (
        82 + 0.55 * age + 0.7 * bmi + 2 * (sex == "Male")
        - (sex == "Nonbinary") + smoke_bp + severity_bp + rng.normal(0, 11, n)
    ).round(1)
    log_odds = (
        -1.65 + 0.035 * (age - 55) + 0.055 * (bmi - 27)
        + 0.35 * (smoking == "Former") + 0.75 * (smoking == "Current")
        + 0.45 * (severity == "Moderate") + 0.95 * (severity == "Severe")
        + 0.1 * (sex == "Male")
    )
    probability = 1 / (1 + np.exp(-log_odds))
    outcome = np.where(rng.binomial(1, probability), "Yes", "No")
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    data = pd.DataFrame({
        "participant_id": [f"SYN{i:04d}" for i in range(1, n + 1)],
        "age": age,
        "bmi": bmi,
        "sex": sex,
        "smoking": smoking,
        "severity": severity,
        "blood_pressure": blood_pressure,
        "outcome": outcome,
        "visit_date": dates.strftime("%Y-%m-%d"),
    })
    # Simulated missingness is inserted after generating the outcomes.
    missing_counts = {
        "age": 10, "bmi": 25, "sex": 5, "smoking": 15,
        "severity": 15, "blood_pressure": 5, "outcome": 10,
    }
    for column, count in missing_counts.items():
        data.loc[rng.choice(n, count, replace=False), column] = np.nan
    # Repeated rows are a deliberate cleaning exercise, not additional people.
    return pd.concat([data, data.iloc[[13, 89, 220, 410]]], ignore_index=True)


if __name__ == "__main__":
    destination = Path(__file__).with_name("example_dataset.csv")
    frame = make_example()
    frame.to_csv(destination, index=False)
    print(f"Wrote {len(frame)} entirely synthetic records to {destination}")

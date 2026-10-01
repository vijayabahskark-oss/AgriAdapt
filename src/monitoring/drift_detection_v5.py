from pathlib import Path
import json
import numpy as np
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


ROOT = Path(__file__).resolve().parents[2]

REFERENCE = (
    ROOT
    / "models/production/v5"
    / "reference_training_data.csv"
)

OUTPUT_DIR = (
    ROOT
    / "reports/monitoring/v5"
)

PRODUCTION_DIR = (
    ROOT
    / "data/production/v5"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PRODUCTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CATEGORICAL_FEATURES = [
    "crop",
    "season",
]

NUMERIC_FEATURES = [
    "temperature_mean",
    "rainfall_total",
    "humidity_mean",
    "solar_radiation_mean",
    "wind_speed_mean",
    "latitude",
    "longitude",
]

FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)


def create_stable_production(reference):

    # Reproducible sample from the reference distribution.
    return reference.sample(
        n=min(1000, len(reference)),
        replace=True,
        random_state=42,
    ).reset_index(drop=True)


def create_drifted_production(reference):

    production = create_stable_production(reference).copy()

    # Deliberately introduce climate distribution shifts.
    production["temperature_mean"] += 3.0

    production["rainfall_total"] *= 0.70

    production["humidity_mean"] -= 8.0

    production["solar_radiation_mean"] *= 1.15

    production["wind_speed_mean"] += 1.5

    return production


def run_drift_report(reference, current, name):

    reference_features = reference[FEATURES].copy()
    current_features = current[FEATURES].copy()

    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    snapshot = report.run(
        current_data=current_features,
        reference_data=reference_features,
    )

    output_file = OUTPUT_DIR / f"{name}_drift_report.html"

    snapshot.save_html(str(output_file))

    # Save JSON representation where supported.
    json_file = OUTPUT_DIR / f"{name}_drift_report.json"

    try:
        snapshot.save_json(str(json_file))
    except Exception:
        json_file = None

    print()
    print("=" * 70)
    print(f"DRIFT REPORT: {name.upper()}")
    print("=" * 70)

    print(f"Current rows   : {len(current):,}")
    print(f"Reference rows : {len(reference):,}")
    print(f"HTML report    : {output_file}")

    if json_file:
        print(f"JSON report    : {json_file}")

    return snapshot


def main():

    print("=" * 70)
    print("AGRIADAPT V5.2")
    print("DATA DRIFT DETECTION")
    print("=" * 70)

    if not REFERENCE.exists():
        raise FileNotFoundError(
            f"Reference dataset not found:\n{REFERENCE}"
        )

    reference = pd.read_csv(REFERENCE)

    required = FEATURES

    missing = [
        c for c in required
        if c not in reference.columns
    ]

    if missing:
        raise ValueError(
            f"Missing reference features: {missing}"
        )

    stable = create_stable_production(reference)

    drifted = create_drifted_production(reference)

    stable_path = (
        PRODUCTION_DIR
        / "stable_production_batch.csv"
    )

    drifted_path = (
        PRODUCTION_DIR
        / "drifted_production_batch.csv"
    )

    stable.to_csv(
        stable_path,
        index=False,
    )

    drifted.to_csv(
        drifted_path,
        index=False,
    )

    print()
    print("PRODUCTION BATCHES")
    print("-" * 70)
    print(f"Stable : {stable_path}")
    print(f"Drifted: {drifted_path}")

    run_drift_report(
        reference,
        stable,
        "stable",
    )

    run_drift_report(
        reference,
        drifted,
        "drifted",
    )

    print()
    print("=" * 70)
    print("V5.2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
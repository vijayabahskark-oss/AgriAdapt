from pathlib import Path
import json
import shutil
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SOURCE_MODEL = (
    ROOT
    / "models/v4/v422"
    / "geographic_coordinates_random_forest.joblib"
)

DATASET = (
    ROOT
    / "data/processed/v4"
    / "v4_climate_yield_dataset.csv"
)

OUTPUT_DIR = ROOT / "models/production/v5"

OUTPUT_MODEL = OUTPUT_DIR / "agriadapt_production_model.joblib"
OUTPUT_METADATA = OUTPUT_DIR / "model_metadata.json"
OUTPUT_REFERENCE = OUTPUT_DIR / "reference_training_data.csv"


TARGET = "yield_tonnes_per_ha"

FEATURES = {
    "categorical": [
        "crop",
        "season",
    ],
    "numeric": [
        "temperature_mean",
        "rainfall_total",
        "humidity_mean",
        "solar_radiation_mean",
        "wind_speed_mean",
        "latitude",
        "longitude",
    ],
}


def main():

    print("=" * 70)
    print("AGRIADAPT V5.1")
    print("PRODUCTION MODEL PACKAGING")
    print("=" * 70)

    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(
            f"Production candidate not found:\n{SOURCE_MODEL}"
        )

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(DATASET)

    required = (
        FEATURES["categorical"]
        + FEATURES["numeric"]
        + [TARGET, "year"]
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Production model was trained using the temporal training period.
    df["year_start"] = df["year"].astype(str).str[:4].astype(int)

    training_reference = df[
        df["year_start"] <= 2007
    ].copy()

    if training_reference.empty:
        raise ValueError(
            "Training reference dataset is empty."
        )

    # Copy the already-trained V4.22 pipeline.
    shutil.copy2(
        SOURCE_MODEL,
        OUTPUT_MODEL,
    )

    # Save the exact feature/reference information.
    metadata = {
        "project": "AgriAdapt",
        "version": "V5.1",
        "model_role": "production_baseline",
        "source_model": str(
            SOURCE_MODEL.relative_to(ROOT)
        ),
        "model_type": "RandomForestRegressor",
        "experiment": "geographic_coordinates",
        "target": TARGET,
        "features": FEATURES,
        "training_period": {
            "start": int(
                training_reference["year_start"].min()
            ),
            "end": int(
                training_reference["year_start"].max()
            ),
            "rows": int(len(training_reference)),
        },
        "validation_period": {
            "start": 2008,
            "end": 2010,
        },
        "test_period": {
            "start": 2011,
            "end": 2012,
        },
        "baseline_metrics": {
            "validation_mae": 0.458282,
            "validation_rmse": 0.734992,
            "validation_r2": 0.657583,
            "test_mae": 0.721925,
            "test_rmse": 1.096899,
            "test_r2": 0.498091,
        },
        "reference_data_purpose": (
            "Reference distribution for production monitoring "
            "and data-drift detection."
        ),
    }

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # Reference data contains only model inputs + target.
    reference_columns = (
        FEATURES["categorical"]
        + FEATURES["numeric"]
        + [TARGET, "year"]
    )

    training_reference[
        reference_columns
    ].to_csv(
        OUTPUT_REFERENCE,
        index=False,
    )

    print()
    print("=" * 70)
    print("V5.1 PRODUCTION MODEL PACKAGED")
    print("=" * 70)

    print(f"Model     : {OUTPUT_MODEL}")
    print(f"Metadata  : {OUTPUT_METADATA}")
    print(f"Reference : {OUTPUT_REFERENCE}")

    print()
    print("Reference training data:")
    print(f"Rows      : {len(training_reference):,}")
    print(
        f"Years     : "
        f"{training_reference['year_start'].min()} → "
        f"{training_reference['year_start'].max()}"
    )

    print()
    print("Production features:")
    print("Categorical:", FEATURES["categorical"])
    print("Numeric    :", FEATURES["numeric"])

    print()
    print("V5.1 COMPLETE")


if __name__ == "__main__":
    main()

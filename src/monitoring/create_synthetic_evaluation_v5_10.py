from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "data" / "processed" / "v4" / "v4_climate_yield_dataset.csv"

PRODUCTION_DIR = ROOT / "data" / "production" / "v5"

PREDICTIONS_FILE = PRODUCTION_DIR / "predictions.csv"
GROUND_TRUTH_FILE = PRODUCTION_DIR / "synthetic_ground_truth_v5_10.csv"

SYNTHETIC_PREDICTIONS_FILE = (
    PRODUCTION_DIR / "synthetic_predictions_v5_10.csv"
)

SYNTHETIC_EVALUATED_FILE = (
    PRODUCTION_DIR / "synthetic_evaluated_predictions_v5_10.csv"
)

METADATA_FILE = (
    PRODUCTION_DIR / "synthetic_evaluation_metadata_v5_10.json"
)


def main():

    print("=" * 70)
    print("AGRIADAPT V5.10")
    print("SYNTHETIC PERFORMANCE EVALUATION DATASET")
    print("=" * 70)

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(PREDICTIONS_FILE)

    dataset = pd.read_csv(DATASET)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    # --------------------------------------------------------
    # Select historical observations with real yield labels.
    # These are used ONLY as a simulation of delayed production
    # ground truth.
    # --------------------------------------------------------

    required = {
        "crop",
        "season",
        "temperature_mean",
        "rainfall_total",
        "humidity_mean",
        "solar_radiation_mean",
        "wind_speed_mean",
        "latitude",
        "longitude",
        "yield_tonnes_per_ha",
    }

    missing = required - set(dataset.columns)

    if missing:
        raise RuntimeError(
            "Dataset missing columns:\n"
            + "\n".join(sorted(missing))
        )

    sample = (
        dataset[
            [
                "crop",
                "season",
                "temperature_mean",
                "rainfall_total",
                "humidity_mean",
                "solar_radiation_mean",
                "wind_speed_mean",
                "latitude",
                "longitude",
                "yield_tonnes_per_ha",
            ]
        ]
        .dropna()
        .sample(
            n=12,
            random_state=42,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Use the currently logged prediction as a template.
    # The additional records are synthetic test records.
    # --------------------------------------------------------

    template = predictions.iloc[0]

    rows = []

    for i, row in sample.iterrows():

        prediction_id = (
            f"SYNTHV510_{i + 1:03d}"
        )

        # Deliberately create predictions with controlled
        # error so that the drift detector can be exercised.
        actual = float(row["yield_tonnes_per_ha"])

        predicted = actual * 0.70

        rows.append(
            {
                "prediction_id": prediction_id,
                "timestamp": template["timestamp"],
                "crop": row["crop"],
                "season": row["season"],
                "temperature_mean": row["temperature_mean"],
                "rainfall_total": row["rainfall_total"],
                "humidity_mean": row["humidity_mean"],
                "solar_radiation_mean": row["solar_radiation_mean"],
                "wind_speed_mean": row["wind_speed_mean"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "predicted_yield_tonnes_per_ha": predicted,
                "model_version": template["model_version"],
                "model_name": template["model_name"],
                "actual_yield_tonnes_per_ha": actual,
            }
        )

    synthetic = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Separate prediction and ground-truth files
    # --------------------------------------------------------

    prediction_columns = [
        "prediction_id",
        "timestamp",
        "crop",
        "season",
        "temperature_mean",
        "rainfall_total",
        "humidity_mean",
        "solar_radiation_mean",
        "wind_speed_mean",
        "latitude",
        "longitude",
        "predicted_yield_tonnes_per_ha",
        "model_version",
        "model_name",
    ]

    ground_truth_columns = [
        "prediction_id",
        "actual_yield_tonnes_per_ha",
    ]

    synthetic[prediction_columns].to_csv(
        SYNTHETIC_PREDICTIONS_FILE,
        index=False,
    )

    synthetic[ground_truth_columns].to_csv(
        GROUND_TRUTH_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Evaluated dataset
    # --------------------------------------------------------

    evaluated = synthetic.copy()

    evaluated["absolute_error_tonnes_per_ha"] = (
        evaluated["predicted_yield_tonnes_per_ha"]
        - evaluated["actual_yield_tonnes_per_ha"]
    ).abs()

    evaluated["signed_error_tonnes_per_ha"] = (
        evaluated["predicted_yield_tonnes_per_ha"]
        - evaluated["actual_yield_tonnes_per_ha"]
    )

    evaluated.to_csv(
        SYNTHETIC_EVALUATED_FILE,
        index=False,
    )

    metadata = {
        "project": "AgriAdapt",
        "pipeline": "V5.10 Synthetic Performance Test",
        "synthetic": True,
        "production_state_modified": False,
        "source_dataset": str(
            DATASET.relative_to(ROOT)
        ),
        "records": len(synthetic),
        "purpose": (
            "Testing the performance-drift monitoring branch "
            "without modifying the real production model, "
            "prediction log, or registry."
        ),
        "warning": (
            "These records simulate delayed production "
            "ground truth and must not be presented as "
            "real production observations."
        ),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(f"Synthetic records : {len(synthetic)}")
    print()
    print(f"Predictions       : {SYNTHETIC_PREDICTIONS_FILE}")
    print(f"Ground truth      : {GROUND_TRUTH_FILE}")
    print(f"Evaluated data    : {SYNTHETIC_EVALUATED_FILE}")
    print(f"Metadata          : {METADATA_FILE}")
    print()
    print("REAL PRODUCTION STATE WAS NOT MODIFIED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
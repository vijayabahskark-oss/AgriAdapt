from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_DIR = ROOT / "data" / "production" / "v5"
REPORT_DIR = ROOT / "reports" / "monitoring" / "v5"

PREDICTIONS_FILE = PRODUCTION_DIR / "predictions.csv"
GROUND_TRUTH_FILE = PRODUCTION_DIR / "ground_truth.csv"

EVALUATED_FILE = PRODUCTION_DIR / "evaluated_predictions.csv"
PERFORMANCE_REPORT = REPORT_DIR / "performance_report_v5_9.json"


# ============================================================
# CONFIGURATION
# ============================================================

ID_COLUMN = "prediction_id"
TARGET_COLUMN = "actual_yield_tonnes_per_ha"
PREDICTION_COLUMN = "predicted_yield_tonnes_per_ha"


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    raise RuntimeError(message)


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # R² requires at least two observations with non-constant target.
    if len(y_true) >= 2 and np.std(y_true) > 0:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = None

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": None if r2 is None else float(r2),
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_predictions():
    if not PREDICTIONS_FILE.exists():
        fail(
            f"Prediction log not found:\n"
            f"{PREDICTIONS_FILE}\n\n"
            f"Run the FastAPI prediction endpoint first."
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required_columns = {
        ID_COLUMN,
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
        PREDICTION_COLUMN,
        "model_version",
        "model_name",
    }

    missing = required_columns - set(df.columns)

    if missing:
        fail(
            "Prediction log is missing required columns:\n"
            + "\n".join(sorted(missing))
        )

    return df


def load_ground_truth():
    if not GROUND_TRUTH_FILE.exists():
        fail(
            f"Ground-truth file not found:\n"
            f"{GROUND_TRUTH_FILE}\n\n"
            f"Create it using:\n\n"
            f"prediction_id,actual_yield_tonnes_per_ha\n"
            f"<prediction_id>,<actual_yield>"
        )

    df = pd.read_csv(GROUND_TRUTH_FILE)

    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
    }

    missing = required_columns - set(df.columns)

    if missing:
        fail(
            "Ground-truth file is missing required columns:\n"
            + "\n".join(sorted(missing))
        )

    if df[ID_COLUMN].duplicated().any():
        duplicates = df.loc[
            df[ID_COLUMN].duplicated(keep=False),
            ID_COLUMN
        ].tolist()

        fail(
            "Duplicate prediction IDs found in ground truth:\n"
            + "\n".join(map(str, duplicates))
        )

    return df


# ============================================================
# EVALUATION
# ============================================================

def evaluate():
    print("=" * 70)
    print("AGRIADAPT V5.9")
    print("DELAYED GROUND-TRUTH EVALUATION")
    print("=" * 70)

    predictions = load_predictions()
    ground_truth = load_ground_truth()

    print(f"Prediction records : {len(predictions)}")
    print(f"Ground-truth rows  : {len(ground_truth)}")

    # --------------------------------------------------------
    # Match predictions with later ground truth
    # --------------------------------------------------------

    merged = predictions.merge(
        ground_truth,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    print(f"Matched records    : {len(merged)}")

    if merged.empty:
        fail(
            "No prediction records matched the supplied ground truth.\n"
            "Check prediction_id values."
        )

    # --------------------------------------------------------
    # Validate target values
    # --------------------------------------------------------

    merged[TARGET_COLUMN] = pd.to_numeric(
        merged[TARGET_COLUMN],
        errors="coerce",
    )

    merged[PREDICTION_COLUMN] = pd.to_numeric(
        merged[PREDICTION_COLUMN],
        errors="coerce",
    )

    invalid = merged[
        merged[TARGET_COLUMN].isna()
        | merged[PREDICTION_COLUMN].isna()
    ]

    if not invalid.empty:
        print(
            f"WARNING: Removing {len(invalid)} rows "
            f"with invalid numeric values."
        )

        merged = merged.drop(index=invalid.index)

    if merged.empty:
        fail("No valid records remain after numeric validation.")

    # --------------------------------------------------------
    # Calculate performance
    # --------------------------------------------------------

    y_true = merged[TARGET_COLUMN].to_numpy()
    y_pred = merged[PREDICTION_COLUMN].to_numpy()

    metrics = calculate_metrics(y_true, y_pred)

    # --------------------------------------------------------
    # Per-record error
    # --------------------------------------------------------

    merged["absolute_error_tonnes_per_ha"] = (
        merged[PREDICTION_COLUMN] - merged[TARGET_COLUMN]
    ).abs()

    merged["signed_error_tonnes_per_ha"] = (
        merged[PREDICTION_COLUMN] - merged[TARGET_COLUMN]
    )

    # --------------------------------------------------------
    # Model lineage
    # --------------------------------------------------------

    model_versions = (
        merged["model_version"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    model_names = (
        merged["model_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # --------------------------------------------------------
    # Save evaluated predictions
    # --------------------------------------------------------

    PRODUCTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        EVALUATED_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Performance report
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluated_at = datetime.now(
        timezone.utc
    ).isoformat()

    report = {
        "project": "AgriAdapt",
        "pipeline": "V5.9 Delayed Ground-Truth Evaluation",
        "evaluated_at": evaluated_at,

        "prediction_log": str(
            PREDICTIONS_FILE.relative_to(ROOT)
        ),

        "ground_truth_file": str(
            GROUND_TRUTH_FILE.relative_to(ROOT)
        ),

        "evaluated_predictions_file": str(
            EVALUATED_FILE.relative_to(ROOT)
        ),

        "records": {
            "prediction_records": int(len(predictions)),
            "ground_truth_records": int(len(ground_truth)),
            "matched_records": int(len(merged)),
        },

        "model_lineage": {
            "model_versions": model_versions,
            "model_names": model_names,
        },

        "performance": {
            "mae_tonnes_per_ha": metrics["mae"],
            "rmse_tonnes_per_ha": metrics["rmse"],
            "r2": metrics["r2"],
        },

        "error_summary": {
            "mean_absolute_error_tonnes_per_ha": float(
                merged["absolute_error_tonnes_per_ha"].mean()
            ),
            "maximum_absolute_error_tonnes_per_ha": float(
                merged["absolute_error_tonnes_per_ha"].max()
            ),
            "mean_signed_error_tonnes_per_ha": float(
                merged["signed_error_tonnes_per_ha"].mean()
            ),
        },

        "ground_truth_status": "AVAILABLE",

        "evaluation_note": (
            "Performance is evaluated only for predictions whose "
            "delayed ground-truth yield has been supplied."
        ),
    }

    with open(
        PERFORMANCE_REPORT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("-" * 70)
    print("EVALUATION COMPLETE")
    print("-" * 70)

    print(f"Matched records : {len(merged)}")
    print(f"MAE             : {metrics['mae']:.4f} tonnes/ha")
    print(f"RMSE            : {metrics['rmse']:.4f} tonnes/ha")

    if metrics["r2"] is not None:
        print(f"R²              : {metrics['r2']:.4f}")
    else:
        print("R²              : N/A")

    print()
    print(f"Evaluated data  : {EVALUATED_FILE}")
    print(f"Report          : {PERFORMANCE_REPORT}")
    print()

    print("Model versions evaluated:")
    for version in model_versions:
        print(f"  - {version}")

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    evaluate()
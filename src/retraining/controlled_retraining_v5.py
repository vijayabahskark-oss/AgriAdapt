from pathlib import Path
import json
import shutil
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_MODEL = (
    ROOT / "models" / "production" / "v5"
    / "agriadapt_production_model.joblib"
)

DATASET = (
    ROOT / "data" / "processed" / "v4"
    / "v4_climate_yield_dataset.csv"
)

DRIFT_REPORT = (
    ROOT / "reports" / "monitoring" / "v5"
    / "drifted_drift_report.json"
)

OUTPUT_DIR = ROOT / "models" / "production" / "v5"

REPORT_DIR = ROOT / "reports" / "retraining" / "v5"

CANDIDATE_MODEL = OUTPUT_DIR / "candidate_model.joblib"
PROMOTED_MODEL = OUTPUT_DIR / "agriadapt_production_model_v5_3.joblib"
DECISION_FILE = REPORT_DIR / "retraining_decision_v5_3.json"


TARGET = "yield_tonnes_per_ha"

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

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# A drifted-column count of 5/9 was observed in V5.2.
DRIFTED_COLUMN_THRESHOLD = 3

# Candidate must improve MAE by at least this fraction.
MIN_MAE_IMPROVEMENT = 0.02


def build_pipeline():

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def metrics(model, data):

    predictions = model.predict(data[FEATURES])

    y = data[TARGET]

    mae = mean_absolute_error(y, predictions)

    rmse = np.sqrt(
        mean_squared_error(y, predictions)
    )

    r2 = r2_score(y, predictions)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def read_drift_count():

    if not DRIFT_REPORT.exists():
        raise FileNotFoundError(
            f"Drift report not found:\n{DRIFT_REPORT}"
        )

    with open(
        DRIFT_REPORT,
        "r",
        encoding="utf-8",
    ) as f:
        report = json.load(f)

    for metric in report.get("metrics", []):

        if metric["metric_name"].startswith(
            "DriftedColumnsCount"
        ):
            return int(
                metric["value"]["count"]
            )

    raise ValueError(
        "DriftedColumnsCount metric not found."
    )


def main():

    print("=" * 70)
    print("AGRIADAPT V5.3")
    print("CONTROLLED RETRAINING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    if not PRODUCTION_MODEL.exists():
        raise FileNotFoundError(
            f"Production model not found:\n{PRODUCTION_MODEL}"
        )

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET}"
        )

    production_model = joblib.load(
        PRODUCTION_MODEL
    )

    df = pd.read_csv(DATASET)

    required = FEATURES + [
        TARGET,
        "year",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["year_start"] = (
        df["year"]
        .astype(str)
        .str[:4]
        .astype(int)
    )

    # --------------------------------------------------
    # Drift decision
    # --------------------------------------------------

    drifted_columns = read_drift_count()

    print()
    print("DRIFT STATUS")
    print("-" * 70)
    print(
        f"Drifted columns : {drifted_columns}"
    )
    print(
        f"Trigger threshold: "
        f"{DRIFTED_COLUMN_THRESHOLD}"
    )

    decision = {
        "project": "AgriAdapt",
        "version": "V5.3",
        "drifted_columns": drifted_columns,
        "drift_threshold": DRIFTED_COLUMN_THRESHOLD,
        "action": None,
        "promotion": None,
    }

    if drifted_columns < DRIFTED_COLUMN_THRESHOLD:

        decision["action"] = "NO_RETRAINING"
        decision["promotion"] = "KEEP_PRODUCTION_MODEL"

        with open(
            DECISION_FILE,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                decision,
                f,
                indent=2,
            )

        print()
        print("No significant drift detected.")
        print("Retraining skipped.")
        print(f"Decision: {DECISION_FILE}")
        return

    print("Drift threshold exceeded.")
    print("Proceeding to controlled retraining.")

    # --------------------------------------------------
    # Temporal retraining split
    # --------------------------------------------------

    train = df[
        df["year_start"] <= 2010
    ].copy()

    holdout = df[
        df["year_start"] > 2010
    ].copy()

    if train.empty or holdout.empty:
        raise ValueError(
            "Invalid retraining split."
        )

    print()
    print("RETRAINING DATA")
    print("-" * 70)
    print(f"Training : {len(train):,}")
    print(f"Holdout  : {len(holdout):,}")

    # --------------------------------------------------
    # Current production performance
    # --------------------------------------------------

    current_metrics = metrics(
        production_model,
        holdout,
    )

    print()
    print("CURRENT PRODUCTION MODEL")
    print("-" * 70)
    print(
        f"MAE  : {current_metrics['mae']:.4f}"
    )
    print(
        f"RMSE : {current_metrics['rmse']:.4f}"
    )
    print(
        f"R²   : {current_metrics['r2']:.4f}"
    )

    # --------------------------------------------------
    # Train candidate
    # --------------------------------------------------

    print()
    print("TRAINING CANDIDATE")
    print("-" * 70)

    candidate = build_pipeline()

    candidate.fit(
        train[FEATURES],
        train[TARGET],
    )

    candidate_metrics = metrics(
        candidate,
        holdout,
    )

    print(
        f"MAE  : {candidate_metrics['mae']:.4f}"
    )
    print(
        f"RMSE : {candidate_metrics['rmse']:.4f}"
    )
    print(
        f"R²   : {candidate_metrics['r2']:.4f}"
    )

    joblib.dump(
        candidate,
        CANDIDATE_MODEL,
    )

    # --------------------------------------------------
    # Promotion gate
    # --------------------------------------------------

    mae_improvement = (
        current_metrics["mae"]
        - candidate_metrics["mae"]
    ) / current_metrics["mae"]

    candidate_improves = (
        mae_improvement >= MIN_MAE_IMPROVEMENT
        and candidate_metrics["r2"]
        >= current_metrics["r2"]
    )

    print()
    print("PROMOTION GATE")
    print("-" * 70)
    print(
        f"MAE improvement: "
        f"{mae_improvement * 100:.2f}%"
    )
    print(
        f"Required       : "
        f"{MIN_MAE_IMPROVEMENT * 100:.2f}%"
    )

    if candidate_improves:

        shutil.copy2(
            CANDIDATE_MODEL,
            PROMOTED_MODEL,
        )

        decision["action"] = "RETRAIN"
        decision["promotion"] = "PROMOTED"

        print()
        print("CANDIDATE ACCEPTED.")
        print(
            f"Promoted model: {PROMOTED_MODEL}"
        )

    else:

        decision["action"] = "RETRAIN"
        decision["promotion"] = "REJECTED"

        print()
        print(
            "CANDIDATE REJECTED."
        )
        print(
            "Production model remains unchanged."
        )

    decision["current_metrics"] = current_metrics
    decision["candidate_metrics"] = candidate_metrics
    decision["mae_improvement_fraction"] = float(
        mae_improvement
    )

    decision["artifacts"] = {
        "candidate_model": str(
            CANDIDATE_MODEL
        ),
        "promoted_model": (
            str(PROMOTED_MODEL)
            if candidate_improves
            else None
        ),
    }

    with open(
        DECISION_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            decision,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("V5.3 COMPLETE")
    print("=" * 70)
    print(
        f"Decision report: {DECISION_FILE}"
    )


if __name__ == "__main__":
    main()
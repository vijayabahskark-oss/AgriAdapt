from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# ============================================================
# AGRIADAPT V4.22
# GEOGRAPHIC FEATURE BENCHMARK
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = (
    ROOT
    / "data/processed/v4/"
    "v4_climate_yield_dataset.csv"
)

REPORT_DIR = (
    ROOT
    / "reports/v4/benchmark_v422"
)

MODEL_DIR = (
    ROOT
    / "models/v4/v422"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET = "yield_tonnes_per_ha"


# ============================================================
# BASE CLIMATE FEATURES
# ============================================================

CLIMATE_FEATURES = [
    "temperature_mean",
    "rainfall_total",
    "humidity_mean",
    "solar_radiation_mean",
    "wind_speed_mean",
]

# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

EXPERIMENTS = {

    "baseline": {
        "categorical": [
            "crop",
            "season",
        ],
        "numeric": CLIMATE_FEATURES,
    },

    "geographic_coordinates": {
        "categorical": [
            "crop",
            "season",
        ],
        "numeric": CLIMATE_FEATURES + [
            "latitude",
            "longitude",
        ],
    },

    "district_aware": {
        "categorical": [
            "crop",
            "season",
            "state",
            "district",
        ],
        "numeric": CLIMATE_FEATURES,
    },
}


# ============================================================
# TEMPORAL SPLIT
# ============================================================

TRAIN_END = 2007
VALIDATION_END = 2010


def get_year_start(value):
    return int(str(value)[:4])


# ============================================================
# METRICS
# ============================================================

def evaluate(model, X, y):

    prediction = model.predict(X)

    mae = mean_absolute_error(
        y,
        prediction,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            prediction,
        )
    )

    r2 = r2_score(
        y,
        prediction,
    )

    return mae, rmse, r2


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(
    categorical_features,
    numeric_features,
):

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
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
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
        ]
    )


# ============================================================
# MODELS
# ============================================================

def build_models():

    return {

        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("AGRIADAPT V4.22")
    print("GEOGRAPHIC FEATURE BENCHMARK")
    print("=" * 75)

    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA}"
        )

    df = pd.read_csv(DATA)

    print(
        f"\nDataset: "
        f"{len(df):,} rows × "
        f"{len(df.columns)} columns"
    )

    # --------------------------------------------------------
    # Required validation
    # --------------------------------------------------------

    required_columns = (
        set(
            [
                TARGET,
                "year",
                "crop",
                "season",
                "state",
                "district",
                "latitude",
                "longitude",
            ]
            + CLIMATE_FEATURES
        )
    )

    missing = sorted(
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    df["year_start"] = (
        df["year"]
        .map(get_year_start)
    )

    # --------------------------------------------------------
    # Temporal split
    # --------------------------------------------------------

    train = df[
        df["year_start"] <= TRAIN_END
    ].copy()

    validation = df[
        (df["year_start"] > TRAIN_END)
        & (df["year_start"] <= VALIDATION_END)
    ].copy()

    test = df[
        df["year_start"] > VALIDATION_END
    ].copy()

    print("\nTEMPORAL SPLIT")
    print("-" * 75)

    print(
        f"Train      : {len(train):,} "
        f"({train['year'].min()} → "
        f"{train['year'].max()})"
    )

    print(
        f"Validation : {len(validation):,} "
        f"({validation['year'].min()} → "
        f"{validation['year'].max()})"
    )

    print(
        f"Test       : {len(test):,} "
        f"({test['year'].min()} → "
        f"{test['year'].max()})"
    )

    # --------------------------------------------------------
    # Experiments
    # --------------------------------------------------------

    results = []

    for experiment_name, config in EXPERIMENTS.items():

        categorical = config["categorical"]
        numeric = config["numeric"]

        features = categorical + numeric

        print("\n")
        print("=" * 75)
        print(
            f"EXPERIMENT: "
            f"{experiment_name.upper()}"
        )
        print("=" * 75)

        print(
            f"Categorical features: {categorical}"
        )

        print(
            f"Numeric features: {len(numeric)}"
        )

        X_train = train[features]
        y_train = train[TARGET]

        X_validation = validation[features]
        y_validation = validation[TARGET]

        X_test = test[features]
        y_test = test[TARGET]

        for model_name, estimator in build_models().items():

            print(
                f"\n--- {model_name} ---"
            )

            preprocessor = build_preprocessor(
                categorical,
                numeric,
            )

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "model",
                        estimator,
                    ),
                ]
            )

            print("Training...")

            pipeline.fit(
                X_train,
                y_train,
            )

            val_mae, val_rmse, val_r2 = (
                evaluate(
                    pipeline,
                    X_validation,
                    y_validation,
                )
            )

            test_mae, test_rmse, test_r2 = (
                evaluate(
                    pipeline,
                    X_test,
                    y_test,
                )
            )

            print(
                f"Validation MAE : {val_mae:.4f}"
            )

            print(
                f"Validation RMSE: {val_rmse:.4f}"
            )

            print(
                f"Validation R²  : {val_r2:.4f}"
            )

            print(
                f"Test MAE       : {test_mae:.4f}"
            )

            print(
                f"Test RMSE      : {test_rmse:.4f}"
            )

            print(
                f"Test R²        : {test_r2:.4f}"
            )

            results.append(
                {
                    "Experiment": experiment_name,
                    "Model": model_name,

                    "Train_Rows": len(train),
                    "Validation_Rows": len(validation),
                    "Test_Rows": len(test),

                    "Validation_MAE": val_mae,
                    "Validation_RMSE": val_rmse,
                    "Validation_R2": val_r2,

                    "Test_MAE": test_mae,
                    "Test_RMSE": test_rmse,
                    "Test_R2": test_r2,
                }
            )

            # ------------------------------------------------
            # Save model
            # ------------------------------------------------

            model_filename = (
                experiment_name
                + "_"
                + model_name.lower()
                .replace(" ", "_")
                + ".joblib"
            )

            joblib.dump(
                pipeline,
                MODEL_DIR / model_filename,
            )

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)

    results_file = (
        REPORT_DIR
        / "benchmark_results_v422.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    report_file = (
        REPORT_DIR
        / "benchmark_report_v422.txt"
    )

    report = []

    report.append(
        "AGRIADAPT V4.22 GEOGRAPHIC FEATURE BENCHMARK"
    )

    report.append("=" * 75)

    report.append(
        f"Dataset rows: {len(df):,}"
    )

    report.append(
        f"Train rows: {len(train):,}"
    )

    report.append(
        f"Validation rows: {len(validation):,}"
    )

    report.append(
        f"Test rows: {len(test):,}"
    )

    report.append("")

    report.append(
        results_df.to_string(
            index=False
        )
    )

    report_file.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 75)
    print("V4.22 COMPLETE")
    print("=" * 75)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nResults: {results_file}"
    )

    print(
        f"Report: {report_file}"
    )

    print(
        f"Models: {MODEL_DIR}"
    )


if __name__ == "__main__":
    main()
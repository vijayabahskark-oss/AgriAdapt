from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]

DATA = (
    ROOT
    / "data/processed/v4/"
    "v4_climate_yield_dataset.csv"
)

REPORT_DIR = ROOT / "reports/v4/benchmark"
MODEL_DIR = ROOT / "models/v4"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "yield_tonnes_per_ha"

CATEGORICAL_FEATURES = [
    "crop",
    "season",
]

NUMERIC_FEATURES = [
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "rainfall_total",
    "rainfall_mean",
    "humidity_mean",
    "humidity_min",
    "humidity_max",
    "solar_radiation_mean",
    "solar_radiation_min",
    "solar_radiation_max",
    "wind_speed_mean",
    "wind_speed_max",
]


# Chronological split.
#
# 1998-99 through 2007-08 -> TRAIN
# 2008-09 through 2010-11 -> VALIDATION
# 2011-12 through 2012-13 -> TEST
#
# This is deliberately temporal rather than random.

TRAIN_END = 2007
VALIDATION_END = 2010


def year_start(value):
    return int(str(value)[:4])


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

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def main():

    print("=" * 70)
    print("AGRIADAPT V4.21 — MODEL BENCHMARK")
    print("=" * 70)

    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA}"
        )

    df = pd.read_csv(DATA)

    print(
        f"Dataset: {len(df):,} rows × "
        f"{len(df.columns)} columns"
    )

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if df[TARGET].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    if df[
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    ].isna().any().any():

        raise ValueError(
            "Feature matrix contains missing values."
        )

    df["year_start"] = (
        df["year"]
        .map(year_start)
    )

    # --------------------------------------------------------
    # Chronological split
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
    print("-" * 70)

    print(
        f"Train      : {len(train):,} rows "
        f"({train['year'].min()} → {train['year'].max()})"
    )

    print(
        f"Validation : {len(validation):,} rows "
        f"({validation['year'].min()} → {validation['year'].max()})"
    )

    print(
        f"Test       : {len(test):,} rows "
        f"({test['year'].min()} → {test['year'].max()})"
    )

    if (
        train["year_start"].max()
        >= validation["year_start"].min()
    ):
        raise RuntimeError(
            "Temporal leakage between train and validation."
        )

    if (
        validation["year_start"].max()
        >= test["year_start"].min()
    ):
        raise RuntimeError(
            "Temporal leakage between validation and test."
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X_train = train[
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    ]

    y_train = train[TARGET]

    X_validation = validation[
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    ]

    y_validation = validation[TARGET]

    X_test = test[
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    ]

    y_test = test[TARGET]

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "Linear Regression": LinearRegression(),

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

    results = []

    # --------------------------------------------------------
    # Train and evaluate
    # --------------------------------------------------------

    for name, estimator in models.items():

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

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

        validation_metrics = evaluate(
            pipeline,
            X_validation,
            y_validation,
        )

        test_metrics = evaluate(
            pipeline,
            X_test,
            y_test,
        )

        print("\nValidation:")
        print(
            f"MAE  : {validation_metrics['MAE']:.4f}"
        )
        print(
            f"RMSE : {validation_metrics['RMSE']:.4f}"
        )
        print(
            f"R²   : {validation_metrics['R2']:.4f}"
        )

        print("\nTest:")
        print(
            f"MAE  : {test_metrics['MAE']:.4f}"
        )
        print(
            f"RMSE : {test_metrics['RMSE']:.4f}"
        )
        print(
            f"R²   : {test_metrics['R2']:.4f}"
        )

        results.append(
            {
                "Model": name,

                "Train_Rows": len(train),
                "Validation_Rows": len(validation),
                "Test_Rows": len(test),

                "Validation_MAE":
                    validation_metrics["MAE"],

                "Validation_RMSE":
                    validation_metrics["RMSE"],

                "Validation_R2":
                    validation_metrics["R2"],

                "Test_MAE":
                    test_metrics["MAE"],

                "Test_RMSE":
                    test_metrics["RMSE"],

                "Test_R2":
                    test_metrics["R2"],
            }
        )

        # Save model
        import joblib

        filename = (
            name.lower()
            .replace(" ", "_")
            + "_v4.joblib"
        )

        joblib.dump(
            pipeline,
            MODEL_DIR / filename,
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_file = (
        REPORT_DIR
        / "benchmark_results_v4.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_file = (
        REPORT_DIR
        / "benchmark_report_v4.txt"
    )

    report = []

    report.append(
        "AGRIADAPT V4.21 MODEL BENCHMARK"
    )
    report.append("=" * 70)
    report.append(
        f"Dataset rows: {len(df):,}"
    )
    report.append(
        f"Features: {len(CATEGORICAL_FEATURES) + len(NUMERIC_FEATURES)}"
    )
    report.append("")
    report.append(
        "TEMPORAL SPLIT"
    )
    report.append("-" * 70)
    report.append(
        f"Train: <= {TRAIN_END}"
    )
    report.append(
        f"Validation: {TRAIN_END + 1}–{VALIDATION_END}"
    )
    report.append(
        f"Test: > {VALIDATION_END}"
    )
    report.append("")
    report.append(
        "RESULTS"
    )
    report.append("-" * 70)
    report.append(
        results_df.to_string(index=False)
    )

    report_file.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n")
    print("=" * 70)
    print("V4.21 BENCHMARK COMPLETE")
    print("=" * 70)

    print(
        results_df.to_string(index=False)
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
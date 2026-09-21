from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# AgriAdapt - Stage 3 Model Benchmarking
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "agriadapt_training_dataset.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "benchmark"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

TARGET = "yield_tonnes_per_ha"

# IMPORTANT:
# We intentionally exclude "year" from the model features.
# Year is used to create the chronological train/validation/test
# split and should not be used as a direct predictor in this
# baseline experiment.
FEATURES = [
    "crop",
    "temperature_mean",
    "rainfall_total",
    "humidity_mean",
    "solar_radiation_mean",
    "wind_speed_mean",
]

TRAIN_END_YEAR = 2015
VALIDATION_START_YEAR = 2016
VALIDATION_END_YEAR = 2019
TEST_START_YEAR = 2020

RANDOM_STATE = 42


# ============================================================
# Utility functions
# ============================================================

def calculate_metrics(y_true, predictions):
    """Calculate regression evaluation metrics."""

    return {
        "MAE": mean_absolute_error(
            y_true,
            predictions
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                predictions
            )
        ),
        "R2": r2_score(
            y_true,
            predictions
        ),
    }


def safe_filename(name):
    """Convert model name into a safe filename."""

    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# ============================================================
# 1. Load dataset
# ============================================================

print("=" * 70)
print("AgriAdapt - Stage 3 Model Benchmarking")
print("=" * 70)

if not DATASET_FILE.exists():
    raise FileNotFoundError(
        f"\nTraining dataset not found:\n{DATASET_FILE}\n"
        "Run build_training_dataset.py first."
    )

print("\nLoading dataset:")
print(DATASET_FILE)

df = pd.read_csv(DATASET_FILE)

print(f"\nRows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 2. Validate dataset schema
# ============================================================

required_columns = FEATURES + [TARGET, "year"]

missing_columns = (
    set(required_columns)
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        "Dataset is missing required columns:\n"
        f"{sorted(missing_columns)}"
    )

# ============================================================
# 3. Basic dataset validation
# ============================================================

if df[required_columns].isna().any().any():
    missing = df[required_columns].isna().sum()

    raise ValueError(
        "Missing values detected:\n"
        f"{missing[missing > 0]}"
    )


duplicate_count = df.duplicated(
    subset=["year", "crop"]
).sum()

if duplicate_count > 0:
    raise ValueError(
        f"Found {duplicate_count} duplicate year-crop rows."
    )


df = df.sort_values(
    ["year", "crop"]
).reset_index(drop=True)


print("\nDataset validation:")
print("  ✓ Required columns present")
print("  ✓ No missing values")
print("  ✓ No duplicate year-crop observations")


# ============================================================
# 4. Chronological train/validation/test split
# ============================================================

train = df[
    df["year"] <= TRAIN_END_YEAR
].copy()

validation = df[
    (df["year"] >= VALIDATION_START_YEAR)
    & (df["year"] <= VALIDATION_END_YEAR)
].copy()

test = df[
    df["year"] >= TEST_START_YEAR
].copy()


print("\n" + "-" * 70)
print("Chronological data split")
print("-" * 70)

print(
    f"Training   : {train['year'].min()}-"
    f"{train['year'].max()} "
    f"({len(train)} rows)"
)

print(
    f"Validation : {validation['year'].min()}-"
    f"{validation['year'].max()} "
    f"({len(validation)} rows)"
)

print(
    f"Test       : {test['year'].min()}-"
    f"{test['year'].max()} "
    f"({len(test)} rows)"
)

print("\nNo random shuffling is used.")


# ============================================================
# 5. Prepare X/y
# ============================================================

X_train = train[FEATURES]
y_train = train[TARGET]

X_validation = validation[FEATURES]
y_validation = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# ============================================================
# 6. Preprocessing
# ============================================================

categorical_features = [
    "crop"
]

numeric_features = [
    "temperature_mean",
    "rainfall_total",
    "humidity_mean",
    "solar_radiation_mean",
    "wind_speed_mean",
]


# Use dense output so the same preprocessing works
# cleanly with all three models.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "crop",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features,
        )
    ],
    remainder="passthrough",
)


# ============================================================
# 7. Define models
# ============================================================

models = {
    "Linear Regression": Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor
            ),
            (
                "scale",
                StandardScaler()
            ),
            (
                "model",
                LinearRegression()
            ),
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        ]
    ),
}


# ============================================================
# 8. Add XGBoost
# ============================================================

try:

    from xgboost import XGBRegressor

    models["XGBoost"] = Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        ]
    )

    xgboost_available = True

except ImportError as error:

    xgboost_available = False

    print(
        "\nWARNING: XGBoost is not installed."
    )

    print(
        f"Import error: {error}"
    )

    print(
        "Linear Regression and Random Forest "
        "will still be evaluated."
    )


# ============================================================
# 9. Train and evaluate
# ============================================================

results = []

all_test_predictions = []

trained_models = {}


for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"Training: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    trained_models[model_name] = model


    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    validation_predictions = model.predict(
        X_validation
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions
    )


    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    test_predictions = model.predict(
        X_test
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions
    )


    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    results.append(
        {
            "model": model_name,
            "split": "validation",
            "MAE": validation_metrics["MAE"],
            "RMSE": validation_metrics["RMSE"],
            "R2": validation_metrics["R2"],
        }
    )

    results.append(
        {
            "model": model_name,
            "split": "test",
            "MAE": test_metrics["MAE"],
            "RMSE": test_metrics["RMSE"],
            "R2": test_metrics["R2"],
        }
    )


    # --------------------------------------------------------
    # Store test predictions
    # --------------------------------------------------------

    model_test_predictions = test[
        [
            "year",
            "crop",
            TARGET
        ]
    ].copy()

    model_test_predictions[
        "prediction"
    ] = test_predictions

    model_test_predictions[
        "absolute_error"
    ] = np.abs(
        model_test_predictions[TARGET]
        - model_test_predictions["prediction"]
    )

    model_test_predictions[
        "model"
    ] = model_name

    all_test_predictions.append(
        model_test_predictions
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nValidation:")
    print(
        f"  MAE  : {validation_metrics['MAE']:.4f}"
    )
    print(
        f"  RMSE : {validation_metrics['RMSE']:.4f}"
    )
    print(
        f"  R²   : {validation_metrics['R2']:.4f}"
    )

    print("\nTest:")
    print(
        f"  MAE  : {test_metrics['MAE']:.4f}"
    )
    print(
        f"  RMSE : {test_metrics['RMSE']:.4f}"
    )
    print(
        f"  R²   : {test_metrics['R2']:.4f}"
    )


# ============================================================
# 10. Save benchmark results
# ============================================================

results_df = pd.DataFrame(results)

results_file = (
    REPORT_DIR
    / "benchmark_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# 11. Save test predictions
# ============================================================

predictions_df = pd.concat(
    all_test_predictions,
    ignore_index=True
)

predictions_file = (
    REPORT_DIR
    / "test_predictions.csv"
)

predictions_df.to_csv(
    predictions_file,
    index=False
)


# ============================================================
# 12. Feature importance
# ============================================================

importance_rows = []


for model_name in [
    "Random Forest",
    "XGBoost"
]:

    if model_name not in trained_models:
        continue

    pipeline = trained_models[
        model_name
    ]

    estimator = pipeline.named_steps[
        "model"
    ]

    preprocessing = pipeline.named_steps[
        "preprocess"
    ]

    if not hasattr(
        estimator,
        "feature_importances_"
    ):
        continue

    feature_names = (
        preprocessing
        .get_feature_names_out()
    )

    importances = (
        estimator.feature_importances_
    )

    for feature, importance in zip(
        feature_names,
        importances
    ):

        importance_rows.append(
            {
                "model": model_name,
                "feature": feature,
                "importance": importance,
            }
        )


importance_df = pd.DataFrame(
    importance_rows
)

if not importance_df.empty:

    importance_df = (
        importance_df
        .sort_values(
            ["model", "importance"],
            ascending=[
                True,
                False
            ]
        )
        .reset_index(drop=True)
    )


importance_file = (
    REPORT_DIR
    / "feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)


# ============================================================
# 13. Save trained model artifacts
# ============================================================

for model_name, model in trained_models.items():

    filename = safe_filename(
        model_name
    )

    model_file = (
        MODEL_DIR
        / f"{filename}_baseline.joblib"
    )

    joblib.dump(
        model,
        model_file
    )

    print(
        f"\nSaved model: {model_file}"
    )


# ============================================================
# 14. Save experiment configuration
# ============================================================

config_file = (
    REPORT_DIR
    / "benchmark_configuration.txt"
)

with open(
    config_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AgriAdapt Stage 3 - Model Benchmark\n"
        "====================================\n\n"
    )

    file.write(
        f"Dataset: {DATASET_FILE}\n"
    )

    file.write(
        f"Target: {TARGET}\n\n"
    )

    file.write(
        "Features:\n"
    )

    for feature in FEATURES:

        file.write(
            f"- {feature}\n"
        )

    file.write(
        "\nChronological split:\n"
    )

    file.write(
        f"- Training: <= {TRAIN_END_YEAR}\n"
    )

    file.write(
        f"- Validation: "
        f"{VALIDATION_START_YEAR}-"
        f"{VALIDATION_END_YEAR}\n"
    )

    file.write(
        f"- Test: >= {TEST_START_YEAR}\n"
    )

    file.write(
        "\nRandom shuffling: No\n"
    )

    file.write(
        f"Random state: {RANDOM_STATE}\n"
    )


# ============================================================
# 15. Final summary
# ============================================================

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)

print("\nResults:")
print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nArtifacts:")

print(
    f"  Benchmark results : {results_file}"
)

print(
    f"  Test predictions   : {predictions_file}"
)

print(
    f"  Feature importance : {importance_file}"
)

print(
    f"  Configuration      : {config_file}"
)

print(
    f"  Model artifacts    : {MODEL_DIR}"
)

print("\nStage 3 completed.")
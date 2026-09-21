"""
AgriAdapt - V2 Seasonal Model Benchmark

Purpose
-------
Benchmark crop-yield prediction models using the V2 seasonal
climate dataset.

Dataset
-------
agriadapt_training_dataset_v2.csv

Features
--------
- crop
- 24 seasonal climate features

The year column is intentionally excluded as a direct model feature.

Temporal split
--------------
Training:
    1984-2015

Validation:
    2016-2019

Test:
    2020-2024

Models
------
1. Linear Regression
2. Random Forest
3. XGBoost

Metrics
-------
- MAE
- RMSE
- R2

Important
---------
The test set is not used for model selection.

The purpose of this experiment is to determine whether the
seasonal V2 feature representation improves out-of-time
prediction compared with the V1 annual-climate dataset.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\AgriAdapt")

V2_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "agriadapt_training_dataset_v2.csv"
)

V1_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "benchmark"
    / "benchmark_results.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "benchmark_v2"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TRAIN_START = 1984
TRAIN_END = 2015

VALIDATION_START = 2016
VALIDATION_END = 2019

TEST_START = 2020
TEST_END = 2024

TARGET = "yield_tonnes_per_ha"

CATEGORICAL_FEATURES = [
    "crop",
]


# ============================================================
# 3. V2 SEASONAL FEATURES
# ============================================================

SEASONS = [
    "pre_monsoon",
    "monsoon",
    "post_monsoon",
    "winter",
]

BASE_FEATURES = [
    "temperature_mean",
    "temperature_max",
    "rainfall_total",
    "humidity_mean",
    "solar_mean",
    "wind_mean",
]

SEASONAL_FEATURES = [
    f"{feature}_{season}"
    for season in SEASONS
    for feature in BASE_FEATURES
]

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + SEASONAL_FEATURES
)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    """Calculate regression metrics."""

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def build_preprocessor():
    """
    Build preprocessing pipeline.

    Crop is categorical and is one-hot encoded.

    Climate variables are passed through unchanged.
    """

    return ColumnTransformer(
        transformers=[
            (
                "crop",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "climate",
                "passthrough",
                SEASONAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_models():
    """Create the three benchmark models."""

    models = {}

    # --------------------------------------------------------
    # Linear Regression
    # --------------------------------------------------------

    models["Linear Regression"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    models["Random Forest"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    models["XGBoost"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return models


# ============================================================
# 5. START
# ============================================================

print("=" * 70)
print("AgriAdapt - V2 Seasonal Model Benchmark")
print("=" * 70)

print("\nInput dataset:")
print(V2_DATA_PATH)

print("\nFeature count:")
print(len(FEATURE_COLUMNS))

print("\nFeatures:")

for feature in FEATURE_COLUMNS:
    print(f"  - {feature}")


# ============================================================
# 6. CHECK INPUT FILE
# ============================================================

if not V2_DATA_PATH.exists():

    raise FileNotFoundError(
        f"V2 dataset not found:\n"
        f"{V2_DATA_PATH}"
    )


# ============================================================
# 7. LOAD DATASET
# ============================================================

print("\n" + "-" * 70)
print("Loading V2 dataset")
print("-" * 70)

df = pd.read_csv(
    V2_DATA_PATH
)

print(
    f"Rows loaded: {len(df)}"
)

print(
    f"Columns loaded: {len(df.columns)}"
)

print("\nColumns:")

print(
    df.columns.tolist()
)


# ============================================================
# 8. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "year",
    TARGET,
] + FEATURE_COLUMNS

missing_columns = sorted(
    set(required_columns)
    - set(df.columns)
)

if missing_columns:

    raise ValueError(
        "V2 dataset is missing required columns:\n"
        f"{missing_columns}"
    )


# ============================================================
# 9. NORMALIZE TYPES
# ============================================================

df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce",
)

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce",
)

df["crop"] = (
    df["crop"]
    .astype(str)
    .str.strip()
)

for feature in SEASONAL_FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce",
    )


# ============================================================
# 10. DATA VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("Dataset validation")
print("-" * 70)

missing_counts = (
    df[required_columns]
    .isna()
    .sum()
)

print("\nMissing values:")

print(
    missing_counts.to_string()
)

if missing_counts.sum() > 0:

    raise ValueError(
        "V2 dataset contains missing values."
    )


duplicate_count = (
    df
    .duplicated(
        subset=[
            "crop",
            "year",
        ]
    )
    .sum()
)

print(
    f"\nDuplicate crop-year rows: "
    f"{duplicate_count}"
)

if duplicate_count > 0:

    raise ValueError(
        "V2 dataset contains duplicate "
        "crop-year rows."
    )


print(
    f"\nYear range: "
    f"{int(df['year'].min())}-"
    f"{int(df['year'].max())}"
)

print(
    f"Crops: "
    f"{sorted(df['crop'].unique())}"
)


# ============================================================
# 11. TEMPORAL SPLIT
# ============================================================

print("\n" + "-" * 70)
print("Temporal split")
print("-" * 70)

train_df = df[
    df["year"].between(
        TRAIN_START,
        TRAIN_END,
    )
].copy()

validation_df = df[
    df["year"].between(
        VALIDATION_START,
        VALIDATION_END,
    )
].copy()

test_df = df[
    df["year"].between(
        TEST_START,
        TEST_END,
    )
].copy()


print(
    f"Training:   "
    f"{TRAIN_START}-{TRAIN_END} "
    f"({len(train_df)} rows)"
)

print(
    f"Validation: "
    f"{VALIDATION_START}-{VALIDATION_END} "
    f"({len(validation_df)} rows)"
)

print(
    f"Test:       "
    f"{TEST_START}-{TEST_END} "
    f"({len(test_df)} rows)"
)


# ============================================================
# 12. VERIFY TEMPORAL SPLIT
# ============================================================

if len(train_df) == 0:
    raise ValueError(
        "Training set is empty."
    )

if len(validation_df) == 0:
    raise ValueError(
        "Validation set is empty."
    )

if len(test_df) == 0:
    raise ValueError(
        "Test set is empty."
    )


train_max_year = train_df["year"].max()
validation_min_year = validation_df["year"].min()
validation_max_year = validation_df["year"].max()
test_min_year = test_df["year"].min()

if train_max_year >= validation_min_year:

    raise ValueError(
        "Temporal leakage detected between "
        "training and validation sets."
    )

if validation_max_year >= test_min_year:

    raise ValueError(
        "Temporal leakage detected between "
        "validation and test sets."
    )


# ============================================================
# 13. PREPARE FEATURES / TARGET
# ============================================================

X_train = train_df[
    FEATURE_COLUMNS
].copy()

y_train = train_df[
    TARGET
].copy()

X_validation = validation_df[
    FEATURE_COLUMNS
].copy()

y_validation = validation_df[
    TARGET
].copy()

X_test = test_df[
    FEATURE_COLUMNS
].copy()

y_test = test_df[
    TARGET
].copy()


# ============================================================
# 14. IMPORTANT: YEAR IS NOT A MODEL FEATURE
# ============================================================

if "year" in FEATURE_COLUMNS:

    raise ValueError(
        "Year must not be used as a direct "
        "model feature."
    )

print(
    "\nYear excluded from model features: YES"
)

print(
    f"Number of model features: "
    f"{len(FEATURE_COLUMNS)}"
)


# ============================================================
# 15. BUILD MODELS
# ============================================================

print("\n" + "-" * 70)
print("Building benchmark models")
print("-" * 70)

models = build_models()

for model_name in models:
    print(
        f"  - {model_name}"
    )


# ============================================================
# 16. TRAIN + EVALUATE
# ============================================================

results = []

prediction_records = []

trained_models = {}


for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining...")

    model.fit(
        X_train,
        y_train,
    )

    trained_models[
        model_name
    ] = model

    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    validation_predictions = (
        model.predict(
            X_validation
        )
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
    )

    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    test_predictions = (
        model.predict(
            X_test
        )
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print("\nValidation:")
    print(
        f"  MAE  : "
        f"{validation_metrics['MAE']:.4f}"
    )

    print(
        f"  RMSE : "
        f"{validation_metrics['RMSE']:.4f}"
    )

    print(
        f"  R2   : "
        f"{validation_metrics['R2']:.4f}"
    )

    print("\nTest:")
    print(
        f"  MAE  : "
        f"{test_metrics['MAE']:.4f}"
    )

    print(
        f"  RMSE : "
        f"{test_metrics['RMSE']:.4f}"
    )

    print(
        f"  R2   : "
        f"{test_metrics['R2']:.4f}"
    )

    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    results.append(
        {
            "Model": model_name,

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

    # --------------------------------------------------------
    # Store test predictions
    # --------------------------------------------------------

    for index, row in test_df.iterrows():

        prediction_records.append(
            {
                "year":
                    int(row["year"]),

                "crop":
                    row["crop"],

                "actual_yield":
                    float(
                        y_test.loc[index]
                    ),

                "predicted_yield":
                    float(
                        test_predictions[
                            list(test_df.index)
                            .index(index)
                        ]
                    ),

                "model":
                    model_name,
            }
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    safe_model_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    model_path = (
        MODEL_DIR
        / f"{safe_model_name}_v2_seasonal.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nSaved model: "
        f"{model_path}"
    )


# ============================================================
# 17. SAVE BENCHMARK RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_path = (
    REPORT_DIR
    / "benchmark_results_v2.csv"
)

results_df.to_csv(
    results_path,
    index=False,
)

print("\n" + "-" * 70)
print("V2 benchmark results")
print("-" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print(
    f"\nSaved:\n{results_path}"
)


# ============================================================
# 18. SAVE TEST PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame(
    prediction_records
)

predictions_path = (
    REPORT_DIR
    / "test_predictions_v2.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)

print(
    f"Saved test predictions:\n"
    f"{predictions_path}"
)


# ============================================================
# 19. FEATURE IMPORTANCE
# ============================================================

print("\n" + "-" * 70)
print("Feature importance")
print("-" * 70)

feature_importance_records = []


for model_name in [
    "Random Forest",
    "XGBoost",
]:

    model = trained_models[
        model_name
    ]

    preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    estimator = (
        model.named_steps[
            "model"
        ]
    )

    # Get transformed feature names
    transformed_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = (
        estimator.feature_importances_
    )

    for feature_name, importance in zip(
        transformed_names,
        importances,
    ):

        feature_importance_records.append(
            {
                "model":
                    model_name,

                "feature":
                    feature_name,

                "importance":
                    float(importance),
            }
        )


feature_importance_df = (
    pd.DataFrame(
        feature_importance_records
    )
    .sort_values(
        [
            "model",
            "importance",
        ],
        ascending=[
            True,
            False,
        ],
    )
)


feature_importance_path = (
    REPORT_DIR
    / "feature_importance_v2.csv"
)

feature_importance_df.to_csv(
    feature_importance_path,
    index=False,
)

print(
    f"Saved feature importance:\n"
    f"{feature_importance_path}"
)


# ============================================================
# 20. LOAD V1 RESULTS AND COMPARE WITH V2
# ============================================================

print("\n" + "-" * 70)
print("V1 vs V2 comparison")
print("-" * 70)

if not V1_RESULTS_PATH.exists():

    print(
        "\nWARNING:"
        "\nV1 benchmark results were not found:"
        f"\n{V1_RESULTS_PATH}"
    )

else:

    # --------------------------------------------------------
    # Load V1 results
    # --------------------------------------------------------

    v1_df = pd.read_csv(
        V1_RESULTS_PATH
    )

    print("\nV1 benchmark results:")

    print(
        v1_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Validate V1 schema
    # --------------------------------------------------------

    required_v1_columns = [
        "model",
        "split",
        "MAE",
        "RMSE",
        "R2",
    ]

    missing_v1_columns = (
        set(required_v1_columns)
        - set(v1_df.columns)
    )

    if missing_v1_columns:

        raise ValueError(
            "V1 benchmark results are missing "
            "required columns:\n"
            f"{sorted(missing_v1_columns)}"
        )

    # --------------------------------------------------------
    # Convert V1 long format into one row per model
    # --------------------------------------------------------

    v1_validation = (
        v1_df[
            v1_df["split"]
            .astype(str)
            .str.lower()
            == "validation"
        ]
        .copy()
    )

    v1_test = (
        v1_df[
            v1_df["split"]
            .astype(str)
            .str.lower()
            == "test"
        ]
        .copy()
    )

    if v1_validation.empty:
        raise ValueError(
            "V1 validation results are empty."
        )

    if v1_test.empty:
        raise ValueError(
            "V1 test results are empty."
        )

    v1_comparison = pd.DataFrame(
        {
            "Model": v1_validation["model"].values,

            "V1_Validation_MAE":
                v1_validation["MAE"].values,

            "V1_Validation_RMSE":
                v1_validation["RMSE"].values,

            "V1_Validation_R2":
                v1_validation["R2"].values,

            "V1_Test_MAE":
                v1_test["MAE"].values,

            "V1_Test_RMSE":
                v1_test["RMSE"].values,

            "V1_Test_R2":
                v1_test["R2"].values,
        }
    )

    # --------------------------------------------------------
    # Rename V2 metrics
    # --------------------------------------------------------

    v2_comparison = results_df.rename(
        columns={
            "Validation_MAE":
                "V2_Validation_MAE",

            "Validation_RMSE":
                "V2_Validation_RMSE",

            "Validation_R2":
                "V2_Validation_R2",

            "Test_MAE":
                "V2_Test_MAE",

            "Test_RMSE":
                "V2_Test_RMSE",

            "Test_R2":
                "V2_Test_R2",
        }
    )

    # --------------------------------------------------------
    # Merge V1 and V2
    # --------------------------------------------------------

    comparison_df = v1_comparison.merge(
        v2_comparison,
        on="Model",
        how="inner",
    )

    # --------------------------------------------------------
    # Calculate metric changes
    # --------------------------------------------------------
    #
    # For MAE/RMSE:
    # lower is better.
    #
    # For R2:
    # higher is better.
    #
    # We report numerical differences only.
    # No automatic ranking is assigned.
    # --------------------------------------------------------

    comparison_df[
        "Validation_MAE_Difference_V2_minus_V1"
    ] = (
        comparison_df["V2_Validation_MAE"]
        - comparison_df["V1_Validation_MAE"]
    )

    comparison_df[
        "Validation_RMSE_Difference_V2_minus_V1"
    ] = (
        comparison_df["V2_Validation_RMSE"]
        - comparison_df["V1_Validation_RMSE"]
    )

    comparison_df[
        "Validation_R2_Difference_V2_minus_V1"
    ] = (
        comparison_df["V2_Validation_R2"]
        - comparison_df["V1_Validation_R2"]
    )

    comparison_df[
        "Test_MAE_Difference_V2_minus_V1"
    ] = (
        comparison_df["V2_Test_MAE"]
        - comparison_df["V1_Test_MAE"]
    )

    comparison_df[
        "Test_RMSE_Difference_V2_minus_V1"
    ] = (
        comparison_df["V2_Test_RMSE"]
        - comparison_df["V1_Test_RMSE"]
    )

    comparison_df[
        "Test_R2_Difference_V2_minus_V1"
    ] = (
        comparison_df["V2_Test_R2"]
        - comparison_df["V1_Test_R2"]
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    comparison_path = (
        REPORT_DIR
        / "v1_vs_v2_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    print("\nV1 vs V2 comparison:")

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        f"\nSaved comparison:\n"
        f"{comparison_path}"
    )


# ============================================================
# 21. SAVE CONFIGURATION
# ============================================================

configuration_path = (
    REPORT_DIR
    / "benchmark_configuration_v2.txt"
)

with open(
    configuration_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "AgriAdapt V2 Seasonal Benchmark Configuration\n"
    )

    file.write(
        "=" * 60
        + "\n\n"
    )

    file.write(
        f"Dataset: {V2_DATA_PATH}\n"
    )

    file.write(
        f"Training period: "
        f"{TRAIN_START}-{TRAIN_END}\n"
    )

    file.write(
        f"Validation period: "
        f"{VALIDATION_START}-{VALIDATION_END}\n"
    )

    file.write(
        f"Test period: "
        f"{TEST_START}-{TEST_END}\n"
    )

    file.write(
        "Random shuffling: No\n"
    )

    file.write(
        "Year used as feature: No\n"
    )

    file.write(
        f"Number of seasonal features: "
        f"{len(SEASONAL_FEATURES)}\n"
    )

    file.write(
        "Models:\n"
    )

    for model_name in models:

        file.write(
            f"  - {model_name}\n"
        )

    file.write(
        "\nSeasonal features:\n"
    )

    for feature in SEASONAL_FEATURES:

        file.write(
            f"  - {feature}\n"
        )

    file.write(
        "\nMetrics:\n"
        "  - MAE\n"
        "  - RMSE\n"
        "  - R2\n"
    )


print(
    f"\nSaved configuration:\n"
    f"{configuration_path}"
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("V2 BENCHMARK COMPLETE")
print("=" * 70)

print(
    f"\nTraining rows:   {len(train_df)}"
)

print(
    f"Validation rows: {len(validation_df)}"
)

print(
    f"Test rows:       {len(test_df)}"
)

print(
    f"Features:        {len(FEATURE_COLUMNS)}"
)

print("\nV2 Results:")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nArtifacts saved under:")

print(
    f"Models : {MODEL_DIR}"
)

print(
    f"Reports: {REPORT_DIR}"
)

print("\n" + "=" * 70)
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

from xgboost import XGBRegressor


# ============================================================
# AgriAdapt - V3 Model Benchmark
# ============================================================
#
# Dataset:
#   V3 crop-aware + spatial variability + climate anomalies
#
# Temporal split:
#   Train      : 1984-2015
#   Validation : 2016-2019
#   Test       : 2020-2024
#
# Models:
#   Linear Regression
#   Random Forest
#   XGBoost
#
# IMPORTANT:
#   - No random shuffling.
#   - Year is excluded as a direct model feature.
#   - Test set is used only for final evaluation.
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\AgriAdapt")

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "agriadapt_training_dataset_v3.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "benchmark_v3"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TARGET = "yield_tonnes_per_ha"

YEAR_COLUMN = "year"

CROP_COLUMN = "crop"

TRAIN_END = 2015
VALIDATION_START = 2016
VALIDATION_END = 2019
TEST_START = 2020
TEST_END = 2024


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def rmse(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )


def evaluate_model(
    model,
    X,
    y,
):
    predictions = model.predict(X)

    return {
        "MAE": mean_absolute_error(
            y,
            predictions,
        ),
        "RMSE": rmse(
            y,
            predictions,
        ),
        "R2": r2_score(
            y,
            predictions,
        ),
    }, predictions


# ============================================================
# Load dataset
# ============================================================

print("=" * 70)
print("AgriAdapt - V3 Model Benchmark")
print("=" * 70)

print("\nInput dataset:")
print(DATA_PATH)

print("\nLoading V3 dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"V3 dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"\nRows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# Dataset validation
# ============================================================

print("\n" + "=" * 70)
print("Dataset validation")
print("=" * 70)

required_columns = {
    YEAR_COLUMN,
    CROP_COLUMN,
    TARGET,
}

missing_required = (
    required_columns - set(df.columns)
)

if missing_required:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            f"  - {c}"
            for c in sorted(missing_required)
        )
    )


missing_values = df.isna().sum()

missing_values = missing_values[
    missing_values > 0
]

if len(missing_values) > 0:

    print("\nMissing values:")

    print(missing_values)

    raise ValueError(
        "Dataset contains missing values."
    )

print("\nMissing values: 0")


duplicate_rows = df.duplicated(
    subset=[
        YEAR_COLUMN,
        CROP_COLUMN,
    ]
).sum()

print(
    f"Duplicate crop-year rows: "
    f"{duplicate_rows}"
)

if duplicate_rows > 0:
    raise ValueError(
        "Duplicate crop-year rows detected."
    )


print(
    f"Year range: "
    f"{df[YEAR_COLUMN].min()}-"
    f"{df[YEAR_COLUMN].max()}"
)

print(
    f"Crops: "
    f"{sorted(df[CROP_COLUMN].unique())}"
)


# ============================================================
# Temporal split
# ============================================================

train_df = df[
    df[YEAR_COLUMN] <= TRAIN_END
].copy()

validation_df = df[
    df[YEAR_COLUMN].between(
        VALIDATION_START,
        VALIDATION_END,
    )
].copy()

test_df = df[
    df[YEAR_COLUMN].between(
        TEST_START,
        TEST_END,
    )
].copy()


print("\n" + "=" * 70)
print("Temporal split")
print("=" * 70)

print(
    f"\nTraining:   "
    f"{train_df[YEAR_COLUMN].min()}-"
    f"{train_df[YEAR_COLUMN].max()} "
    f"({len(train_df)} rows)"
)

print(
    f"Validation: "
    f"{validation_df[YEAR_COLUMN].min()}-"
    f"{validation_df[YEAR_COLUMN].max()} "
    f"({len(validation_df)} rows)"
)

print(
    f"Test:       "
    f"{test_df[YEAR_COLUMN].min()}-"
    f"{test_df[YEAR_COLUMN].max()} "
    f"({len(test_df)} rows)"
)


# ============================================================
# Feature / target separation
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column not in {
        YEAR_COLUMN,
        TARGET,
    }
]

print(
    "\nYear excluded from model features: YES"
)

print(
    f"Number of model features: "
    f"{len(feature_columns)}"
)

print("\nModel features:")

for feature in feature_columns:
    print(f"  - {feature}")


X_train = train_df[feature_columns]
y_train = train_df[TARGET]

X_validation = validation_df[feature_columns]
y_validation = validation_df[TARGET]

X_test = test_df[feature_columns]
y_test = test_df[TARGET]


# ============================================================
# Feature preprocessing
# ============================================================

categorical_features = [
    CROP_COLUMN
]

numeric_features = [
    column
    for column in feature_columns
    if column not in categorical_features
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features,
        ),
    ]
)


# ============================================================
# Models
# ============================================================

models = {

    "Linear Regression": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    max_features="sqrt",
                ),
            ),
        ]
    ),

    "XGBoost": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=4,
                ),
            ),
        ]
    ),
}


# ============================================================
# Benchmark
# ============================================================

results = []

test_predictions = []

print("\n" + "=" * 70)
print("Building benchmark models")
print("=" * 70)


for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("\nTraining...")

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_metrics, validation_pred = (
        evaluate_model(
            model,
            X_validation,
            y_validation,
        )
    )

    print("\nValidation:")

    print(
        f"MAE  : "
        f"{validation_metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : "
        f"{validation_metrics['RMSE']:.4f}"
    )

    print(
        f"R2   : "
        f"{validation_metrics['R2']:.4f}"
    )

    results.append(
        {
            "Model": model_name,
            "Split": "validation",
            "MAE": validation_metrics["MAE"],
            "RMSE": validation_metrics["RMSE"],
            "R2": validation_metrics["R2"],
        }
    )


    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_metrics, test_pred = (
        evaluate_model(
            model,
            X_test,
            y_test,
        )
    )

    print("\nTest:")

    print(
        f"MAE  : "
        f"{test_metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : "
        f"{test_metrics['RMSE']:.4f}"
    )

    print(
        f"R2   : "
        f"{test_metrics['R2']:.4f}"
    )

    results.append(
        {
            "Model": model_name,
            "Split": "test",
            "MAE": test_metrics["MAE"],
            "RMSE": test_metrics["RMSE"],
            "R2": test_metrics["R2"],
        }
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    model_path = (
        MODEL_DIR
        / f"{filename}_v3.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nSaved model: {model_path}"
    )


    # --------------------------------------------------------
    # Store predictions
    # --------------------------------------------------------

    for index, row in test_df.iterrows():

        test_predictions.append(
            {
                "year": int(
                    row[YEAR_COLUMN]
                ),
                "crop": row[CROP_COLUMN],
                "actual_yield_tonnes_per_ha": (
                    row[TARGET]
                ),
                "predicted_yield_tonnes_per_ha": (
                    test_pred[
                        test_df.index.get_loc(index)
                    ]
                ),
                "model": model_name,
            }
        )


# ============================================================
# Save benchmark results
# ============================================================

results_df = pd.DataFrame(
    results
)

results_path = (
    REPORT_DIR
    / "benchmark_results_v3.csv"
)

results_df.to_csv(
    results_path,
    index=False,
)


# ============================================================
# Save test predictions
# ============================================================

predictions_df = pd.DataFrame(
    test_predictions
)

predictions_path = (
    REPORT_DIR
    / "test_predictions_v3.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)


# ============================================================
# Feature importance
# ============================================================

importance_rows = []


def extract_feature_importance(
    pipeline,
    model_name,
):

    fitted_preprocessor = (
        pipeline.named_steps[
            "preprocessor"
        ]
    )

    fitted_model = (
        pipeline.named_steps[
            "model"
        ]
    )

    feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )

    if hasattr(
        fitted_model,
        "feature_importances_",
    ):

        importances = (
            fitted_model.feature_importances_
        )

    elif hasattr(
        fitted_model,
        "coef_",
    ):

        importances = np.abs(
            fitted_model.coef_
        )

    else:
        return

    for feature_name, importance in zip(
        feature_names,
        importances,
    ):

        importance_rows.append(
            {
                "Model": model_name,
                "Feature": feature_name,
                "Importance": importance,
            }
        )


for model_name, model in models.items():

    extract_feature_importance(
        model,
        model_name,
    )


importance_df = pd.DataFrame(
    importance_rows
)

importance_path = (
    REPORT_DIR
    / "feature_importance_v3.csv"
)

importance_df.to_csv(
    importance_path,
    index=False,
)


# ============================================================
# Configuration report
# ============================================================

configuration_path = (
    REPORT_DIR
    / "benchmark_configuration_v3.txt"
)

with open(
    configuration_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "AgriAdapt V3 Model Benchmark\n"
    )

    f.write(
        "============================\n\n"
    )

    f.write(
        f"Dataset: {DATA_PATH}\n"
    )

    f.write(
        f"Rows: {len(df)}\n"
    )

    f.write(
        f"Features: {len(feature_columns)}\n"
    )

    f.write(
        "Year excluded: YES\n"
    )

    f.write(
        "Random shuffling: NO\n\n"
    )

    f.write(
        "Temporal split:\n"
    )

    f.write(
        "Training: 1984-2015\n"
    )

    f.write(
        "Validation: 2016-2019\n"
    )

    f.write(
        "Test: 2020-2024\n\n"
    )

    f.write(
        "Models:\n"
    )

    f.write(
        "Linear Regression\n"
    )

    f.write(
        "Random Forest\n"
    )

    f.write(
        "XGBoost\n"
    )


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 70)
print("V3 BENCHMARK RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved:")

print(
    f"Benchmark results:\n"
    f"{results_path}"
)

print(
    f"\nTest predictions:\n"
    f"{predictions_path}"
)

print(
    f"\nFeature importance:\n"
    f"{importance_path}"
)

print(
    f"\nConfiguration:\n"
    f"{configuration_path}"
)

print("\n" + "=" * 70)
print("V3 BENCHMARK COMPLETE")
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
    f"Features:        {len(feature_columns)}"
)
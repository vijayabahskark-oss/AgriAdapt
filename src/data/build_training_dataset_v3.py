from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# AgriAdapt - V3 Training Dataset Builder
# ============================================================
#
# V3 objectives:
#   1. Preserve seasonal climate information.
#   2. Preserve spatial variability across 10 NASA reference locations.
#   3. Add training-period climate anomalies.
#   4. Add crop x climate interaction features.
#   5. Prevent temporal leakage.
#
# Target:
#   yield_tonnes_per_ha
#
# Temporal periods:
#   Training   : 1984-2015
#   Validation : 2016-2019
#   Test       : 2020-2024
#
# Important:
#   - FAOSTAT provides India-level crop yield.
#   - NASA provides climate at 10 reference locations.
#   - We DO NOT create regional yield labels.
#   - Spatial statistics describe climate variability only.
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\AgriAdapt")

FAOSTAT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "faostat_india_crop_yield_processed.csv"
)

NASA_SEASONAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather"
    / "nasa_power_india_reference_locations_seasonal.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "agriadapt_training_dataset_v3.csv"
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

START_YEAR = 1984
END_YEAR = 2024

TRAIN_START_YEAR = 1984
TRAIN_END_YEAR = 2015

EXPECTED_LOCATIONS = 10

CROPS = [
    "Rice",
    "Wheat",
    "Maize (corn)",
]

SEASONS = [
    "pre_monsoon",
    "monsoon",
    "post_monsoon",
    "winter",
]

CLIMATE_VARIABLES = [
    "temperature_mean",
    "temperature_max",
    "rainfall_total",
    "humidity_mean",
    "solar_mean",
    "wind_mean",
]


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names so the script remains robust to
    accidental formatting differences.
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\\_", "_", regex=False)
    )

    return df


def validate_file_exists(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list,
    dataset_name: str,
) -> None:

    missing = sorted(
        set(required_columns) - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


# ------------------------------------------------------------
# Load FAOSTAT
# ------------------------------------------------------------

print("=" * 70)
print("AgriAdapt - V3 Training Dataset Builder")
print("=" * 70)

print("\nLoading FAOSTAT...")

validate_file_exists(
    FAOSTAT_PATH,
    "FAOSTAT processed dataset",
)

yield_df = pd.read_csv(FAOSTAT_PATH)
yield_df = normalize_columns(yield_df)

# Normalize the actual FAOSTAT schema into the AgriAdapt schema.
yield_df = yield_df.rename(
    columns={
        "Item": "crop",
        "Year": "year",
        "Yield_tonnes_per_ha": "yield_tonnes_per_ha",
    }
)


print(f"Rows loaded: {len(yield_df)}")
print(f"Columns: {yield_df.columns.tolist()}")


# ------------------------------------------------------------
# Validate FAOSTAT schema
# ------------------------------------------------------------

validate_required_columns(
    yield_df,
    [
        "year",
        "crop",
        "yield_tonnes_per_ha",
    ],
    "FAOSTAT dataset",
)


# ------------------------------------------------------------
# Filter crops and years
# ------------------------------------------------------------

yield_df["year"] = pd.to_numeric(
    yield_df["year"],
    errors="coerce",
)

yield_df = yield_df[
    yield_df["crop"].isin(CROPS)
].copy()

yield_df = yield_df[
    yield_df["year"].between(
        START_YEAR,
        END_YEAR,
    )
].copy()

yield_df["year"] = yield_df["year"].astype(int)


print("\nFAOSTAT after filtering:")
print(f"Rows: {len(yield_df)}")
print(f"Years: {yield_df['year'].min()}-{yield_df['year'].max()}")
print(f"Crops: {sorted(yield_df['crop'].unique())}")


# ------------------------------------------------------------
# Validate FAOSTAT uniqueness
# ------------------------------------------------------------

duplicate_yield_rows = yield_df.duplicated(
    subset=["crop", "year"]
).sum()

print(
    f"Duplicate crop-year rows: "
    f"{duplicate_yield_rows}"
)

if duplicate_yield_rows > 0:
    raise ValueError(
        "Duplicate crop-year rows detected in FAOSTAT."
    )


# ------------------------------------------------------------
# Load NASA seasonal climate
# ------------------------------------------------------------

print("\nLoading NASA seasonal climate...")

validate_file_exists(
    NASA_SEASONAL_PATH,
    "NASA seasonal dataset",
)

weather_df = pd.read_csv(NASA_SEASONAL_PATH)
weather_df = normalize_columns(weather_df)

print(f"Rows loaded: {len(weather_df)}")
print(f"Columns: {weather_df.columns.tolist()}")


# ------------------------------------------------------------
# Build expected NASA columns
# ------------------------------------------------------------

seasonal_columns = []

for variable in CLIMATE_VARIABLES:
    for season in SEASONS:
        seasonal_columns.append(
            f"{variable}_{season}"
        )

required_weather_columns = [
    "location",
    "year",
] + seasonal_columns

validate_required_columns(
    weather_df,
    required_weather_columns,
    "NASA seasonal dataset",
)


# ------------------------------------------------------------
# Filter NASA years
# ------------------------------------------------------------

weather_df["year"] = pd.to_numeric(
    weather_df["year"],
    errors="coerce",
)

weather_df = weather_df[
    weather_df["year"].between(
        START_YEAR,
        END_YEAR,
    )
].copy()

weather_df["year"] = weather_df["year"].astype(int)

print("\nNASA after filtering:")
print(f"Rows: {len(weather_df)}")
print(f"Years: {weather_df['year'].min()}-{weather_df['year'].max()}")
print(f"Locations: {weather_df['location'].nunique()}")


# ------------------------------------------------------------
# Validate locations
# ------------------------------------------------------------

locations = sorted(
    weather_df["location"].dropna().unique()
)

print("\nReference locations:")
for location in locations:
    print(f"  - {location}")

if len(locations) != EXPECTED_LOCATIONS:
    raise ValueError(
        f"Expected {EXPECTED_LOCATIONS} locations, "
        f"found {len(locations)}."
    )


# ------------------------------------------------------------
# Validate location-year uniqueness
# ------------------------------------------------------------

duplicate_weather_rows = weather_df.duplicated(
    subset=["location", "year"]
).sum()

print(
    f"\nDuplicate location-year rows: "
    f"{duplicate_weather_rows}"
)

if duplicate_weather_rows > 0:
    raise ValueError(
        "Duplicate location-year rows detected."
    )


# ------------------------------------------------------------
# Validate missing values
# ------------------------------------------------------------

weather_missing = weather_df[
    required_weather_columns
].isna().sum()

missing_weather = weather_missing[
    weather_missing > 0
]

if len(missing_weather) > 0:

    print("\nMissing NASA values detected:")

    for column, count in missing_weather.items():
        print(f"  {column}: {count}")

    raise ValueError(
        "NASA seasonal dataset contains missing values."
    )

print("\nNASA missing values: 0")


# ============================================================
# PART 1
# Spatial climate aggregation
# ============================================================

print("\n" + "=" * 70)
print("PART 1 - Spatial climate aggregation")
print("=" * 70)

print(
    "\nAggregating the 10 reference locations "
    "by year..."
)


# ------------------------------------------------------------
# Spatial statistics
# ------------------------------------------------------------
#
# For each year and seasonal climate variable:
#
#   mean
#   std
#
# We deliberately avoid min/max for every variable because
# that would unnecessarily increase dimensionality.
# ------------------------------------------------------------

spatial_features = []

grouped_weather = weather_df.groupby(
    "year"
)

for variable in CLIMATE_VARIABLES:

    for season in SEASONS:

        column = f"{variable}_{season}"

        mean_series = grouped_weather[column].mean()

        std_series = grouped_weather[column].std(
            ddof=0
        )

        mean_series.name = column

        std_series.name = (
            f"{column}_spatial_std"
        )

        spatial_features.append(mean_series)
        spatial_features.append(std_series)


spatial_df = pd.concat(
    spatial_features,
    axis=1,
).reset_index()


print(
    f"Spatially aggregated rows: "
    f"{len(spatial_df)}"
)

print(
    f"Spatial feature count: "
    f"{len(spatial_df.columns) - 1}"
)


# ------------------------------------------------------------
# Validate spatial aggregation
# ------------------------------------------------------------

if spatial_df["year"].duplicated().any():
    raise ValueError(
        "Spatial aggregation produced duplicate years."
    )

if spatial_df.isna().any().any():
    raise ValueError(
        "Spatial aggregation produced missing values."
    )


# ============================================================
# PART 2
# Crop-aware climate interactions
# ============================================================

print("\n" + "=" * 70)
print("PART 2 - Crop-aware climate interactions")
print("=" * 70)

print(
    "\nCreating crop x climate interaction features..."
)


# ------------------------------------------------------------
# Crop-specific seasonal emphasis
# ------------------------------------------------------------
#
# These are feature-engineering assumptions for V3.
# They are NOT presented as authoritative agronomic
# crop calendars.
#
# Rice:
#   pre-monsoon + monsoon
#
# Wheat:
#   post-monsoon + winter
#
# Maize:
#   pre-monsoon + monsoon
#
# The original seasonal features remain in the dataset.
# These interaction features provide additional crop-aware
# signals without deleting the underlying climate information.
# ------------------------------------------------------------

CROP_SEASON_WEIGHTS = {
    "Rice": {
        "pre_monsoon": 0.35,
        "monsoon": 0.65,
        "post_monsoon": 0.00,
        "winter": 0.00,
    },

    "Wheat": {
        "pre_monsoon": 0.00,
        "monsoon": 0.00,
        "post_monsoon": 0.40,
        "winter": 0.60,
    },

    "Maize (corn)": {
        "pre_monsoon": 0.50,
        "monsoon": 0.50,
        "post_monsoon": 0.00,
        "winter": 0.00,
    },
}


# ------------------------------------------------------------
# Create crop-aware features
# ------------------------------------------------------------

crop_rows = []

for _, row in spatial_df.iterrows():

    year = int(row["year"])

    for crop in CROPS:

        output = {
            "year": year,
            "crop": crop,
        }

        weights = CROP_SEASON_WEIGHTS[crop]

        for variable in CLIMATE_VARIABLES:

            weighted_values = []

            for season in SEASONS:

                weight = weights[season]

                if weight == 0:
                    continue

                column = (
                    f"{variable}_{season}"
                )

                weighted_values.append(
                    weight * row[column]
                )

            output[
                f"crop_{variable}_weighted"
            ] = sum(weighted_values)

        crop_rows.append(output)


crop_aware_df = pd.DataFrame(
    crop_rows
)


print(
    f"Crop-aware rows: "
    f"{len(crop_aware_df)}"
)

print(
    "Crop-aware feature count: "
    f"{len(crop_aware_df.columns) - 2}"
)


# ============================================================
# PART 3
# Merge spatial + crop-aware data
# ============================================================

print("\n" + "=" * 70)
print("PART 3 - Building V3 dataset")
print("=" * 70)


# ------------------------------------------------------------
# Remove duplicate year columns before merge
# ------------------------------------------------------------

spatial_for_merge = spatial_df.copy()

merged_df = crop_aware_df.merge(
    spatial_for_merge,
    on="year",
    how="left",
    validate="many_to_one",
)


# ------------------------------------------------------------
# Merge target
# ------------------------------------------------------------

final_df = merged_df.merge(
    yield_df[
        [
            "year",
            "crop",
            "yield_tonnes_per_ha",
        ]
    ],
    on=["year", "crop"],
    how="inner",
    validate="one_to_one",
)


# ============================================================
# PART 4
# Training-period climate anomalies
# ============================================================

print("\n" + "=" * 70)
print("PART 4 - Training-period climate anomalies")
print("=" * 70)

print(
    "\nCalculating baselines using "
    f"{TRAIN_START_YEAR}-{TRAIN_END_YEAR} only..."
)


# ------------------------------------------------------------
# IMPORTANT:
#
# The anomaly baseline is calculated ONLY from the training
# period. Validation and test years are never used to calculate
# the baseline.
# ------------------------------------------------------------

baseline_mask = (
    final_df["year"].between(
        TRAIN_START_YEAR,
        TRAIN_END_YEAR,
    )
)


anomaly_base_features = []

for variable in CLIMATE_VARIABLES:

    for season in SEASONS:

        column = f"{variable}_{season}"

        anomaly_base_features.append(
            column
        )


baseline_values = (
    final_df.loc[
        baseline_mask,
        anomaly_base_features
    ]
    .mean()
)


# ------------------------------------------------------------
# Create anomaly features
# ------------------------------------------------------------

for column in anomaly_base_features:

    anomaly_column = (
        f"{column}_anomaly"
    )

    final_df[anomaly_column] = (
        final_df[column]
        - baseline_values[column]
    )


print(
    f"Anomaly features created: "
    f"{len(anomaly_base_features)}"
)


# ============================================================
# PART 5
# Final validation
# ============================================================

print("\n" + "=" * 70)
print("PART 5 - Final V3 validation")
print("=" * 70)


# ------------------------------------------------------------
# Sort
# ------------------------------------------------------------

final_df = final_df.sort_values(
    ["year", "crop"]
).reset_index(drop=True)


# ------------------------------------------------------------
# Expected structure
# ------------------------------------------------------------

expected_rows = (
    END_YEAR - START_YEAR + 1
) * len(CROPS)

print(f"\nExpected rows: {expected_rows}")
print(f"Actual rows:   {len(final_df)}")

if len(final_df) != expected_rows:
    raise ValueError(
        "Unexpected number of final rows."
    )


# ------------------------------------------------------------
# Duplicate validation
# ------------------------------------------------------------

duplicate_final_rows = final_df.duplicated(
    subset=["year", "crop"]
).sum()

print(
    f"Duplicate crop-year rows: "
    f"{duplicate_final_rows}"
)

if duplicate_final_rows > 0:
    raise ValueError(
        "Duplicate crop-year rows detected."
    )


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

missing_counts = final_df.isna().sum()

missing_counts = missing_counts[
    missing_counts > 0
]

if len(missing_counts) > 0:

    print("\nMissing values detected:")

    for column, count in missing_counts.items():
        print(f"  {column}: {count}")

    raise ValueError(
        "V3 dataset contains missing values."
    )

print("Missing values: 0")


# ------------------------------------------------------------
# Year validation
# ------------------------------------------------------------

actual_years = set(
    final_df["year"].unique()
)

expected_years = set(
    range(
        START_YEAR,
        END_YEAR + 1,
    )
)

missing_years = (
    expected_years - actual_years
)

unexpected_years = (
    actual_years - expected_years
)

if missing_years:
    raise ValueError(
        f"Missing years: {sorted(missing_years)}"
    )

if unexpected_years:
    raise ValueError(
        f"Unexpected years: "
        f"{sorted(unexpected_years)}"
    )


# ------------------------------------------------------------
# Crop validation
# ------------------------------------------------------------

actual_crops = set(
    final_df["crop"].unique()
)

if actual_crops != set(CROPS):
    raise ValueError(
        f"Unexpected crops: {actual_crops}"
    )


# ------------------------------------------------------------
# Feature identification
# ------------------------------------------------------------

target_column = "yield_tonnes_per_ha"

non_feature_columns = {
    "year",
    "crop",
    target_column,
}

feature_columns = [
    column
    for column in final_df.columns
    if column not in non_feature_columns
]


print(
    f"\nFinal feature count: "
    f"{len(feature_columns)}"
)

print("\nFeatures:")

for feature in feature_columns:
    print(f"  - {feature}")


# ------------------------------------------------------------
# Dataset summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("V3 DATASET SUMMARY")
print("=" * 70)

print(
    f"\nRows:             {len(final_df)}"
)

print(
    f"Years:            "
    f"{final_df['year'].min()}-"
    f"{final_df['year'].max()}"
)

print(
    f"Crops:            "
    f"{sorted(final_df['crop'].unique())}"
)

print(
    f"Features:         "
    f"{len(feature_columns)}"
)

print(
    f"Target:           "
    f"{target_column}"
)

print(
    f"Missing values:   0"
)

print(
    f"Duplicate crop-year: 0"
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

final_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\nSaved V3 dataset:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("V3 DATASET BUILD COMPLETE")
print("=" * 70)
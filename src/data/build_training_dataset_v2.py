"""
AgriAdapt - V2 Seasonal Training Dataset Builder

Purpose
-------
Build the V2 AgriAdapt training dataset by combining:

1. FAOSTAT historical crop-yield data
2. NASA POWER seasonal climate data

The NASA POWER data contains 10 reference locations for each year.
Those locations are spatially aggregated into one India-prototype
climate observation per year.

Final structure:

    10 NASA locations
            ↓
    1984-2024 filtering
            ↓
    spatial aggregation by year
            ↓
    41 climate-year observations
            ↓
    merge with 3 crops from FAOSTAT
            ↓
    123 crop-year observations

Crops
-----
- Rice
- Wheat
- Maize (corn)

Target
------
yield_tonnes_per_ha

Climate features
----------------
Seasonal temperature, rainfall, humidity, solar radiation,
and wind speed for:

- pre_monsoon
- monsoon
- post_monsoon
- winter

Important
---------
The NASA aggregation represents a prototype India-level climate
representation based on 10 reference locations. It is not a
population-weighted or authoritative national climate average.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\AgriAdapt")

FAOSTAT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "faostat_india_crop_yield_processed.csv"
)

NASA_PATH = (
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
    / "agriadapt_training_dataset_v2.csv"
)


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

START_YEAR = 1984
END_YEAR = 2024

EXPECTED_YEARS = set(
    range(
        START_YEAR,
        END_YEAR + 1
    )
)

EXPECTED_CROPS = {
    "Rice",
    "Wheat",
    "Maize (corn)",
}

EXPECTED_LOCATION_COUNT = 10

TARGET_COLUMN = "yield_tonnes_per_ha"


# ============================================================
# 3. SEASONAL FEATURES
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


EXPECTED_SEASONAL_FEATURES = [
    f"{feature}_{season}"
    for season in SEASONS
    for feature in BASE_FEATURES
]


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def print_section(title: str) -> None:
    """Print a formatted section heading."""

    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """Validate that all required columns exist."""

    missing_columns = sorted(
        set(required_columns)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns:\n"
            f"{missing_columns}"
        )


def validate_no_missing_values(
    dataframe: pd.DataFrame,
    columns: list[str],
    dataset_name: str,
) -> None:
    """Validate that selected columns contain no missing values."""

    missing_counts = (
        dataframe[columns]
        .isna()
        .sum()
    )

    print("\nMissing values:")

    print(
        missing_counts
        .to_string()
    )

    columns_with_missing = (
        missing_counts[
            missing_counts > 0
        ]
        .index
        .tolist()
    )

    if columns_with_missing:
        raise ValueError(
            f"{dataset_name} contains missing values in:\n"
            f"{columns_with_missing}"
        )


# ============================================================
# 5. START
# ============================================================

print("=" * 70)
print("AgriAdapt - V2 Seasonal Training Dataset Builder")
print("=" * 70)

print("\nInput files:")
print(f"FAOSTAT : {FAOSTAT_PATH}")
print(f"NASA    : {NASA_PATH}")

print("\nOutput file:")
print(f"OUTPUT  : {OUTPUT_PATH}")


# ============================================================
# 6. CHECK INPUT FILES
# ============================================================

if not FAOSTAT_PATH.exists():
    raise FileNotFoundError(
        f"FAOSTAT file not found:\n{FAOSTAT_PATH}"
    )

if not NASA_PATH.exists():
    raise FileNotFoundError(
        f"NASA seasonal file not found:\n{NASA_PATH}"
    )


# ============================================================
# 7. LOAD FAOSTAT
# ============================================================

print_section("Loading FAOSTAT")

faostat = pd.read_csv(
    FAOSTAT_PATH
)

print(
    f"Rows loaded: {len(faostat)}"
)

print("Columns:")
print(
    faostat.columns.tolist()
)


# ============================================================
# 8. NORMALIZE FAOSTAT COLUMN NAMES
# ============================================================

faostat.columns = (
    faostat.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)

# Actual FAOSTAT processed file uses "item".
# Normalize it to "crop".
if "item" in faostat.columns:
    faostat = faostat.rename(
        columns={
            "item": "crop"
        }
    )


# ============================================================
# 9. VALIDATE FAOSTAT SCHEMA
# ============================================================

validate_columns(
    faostat,
    [
        "crop",
        "year",
        TARGET_COLUMN,
    ],
    "FAOSTAT",
)


# ============================================================
# 10. NORMALIZE FAOSTAT TYPES
# ============================================================

faostat["crop"] = (
    faostat["crop"]
    .astype(str)
    .str.strip()
)

faostat["year"] = pd.to_numeric(
    faostat["year"],
    errors="coerce",
)

faostat[TARGET_COLUMN] = pd.to_numeric(
    faostat[TARGET_COLUMN],
    errors="coerce",
)


# ============================================================
# 11. FILTER FAOSTAT YEARS
# ============================================================

faostat = faostat[
    faostat["year"].between(
        START_YEAR,
        END_YEAR,
    )
].copy()

faostat["year"] = (
    faostat["year"]
    .astype(int)
)

print(
    f"FAOSTAT rows after "
    f"{START_YEAR}-{END_YEAR} filter: "
    f"{len(faostat)}"
)

print(
    f"Crops: "
    f"{sorted(faostat['crop'].unique())}"
)


# ============================================================
# 12. FILTER EXPECTED CROPS
# ============================================================

faostat = faostat[
    faostat["crop"].isin(
        EXPECTED_CROPS
    )
].copy()


# ============================================================
# 13. FAOSTAT VALIDATION
# ============================================================

print_section("FAOSTAT validation")

faostat_duplicates = (
    faostat
    .duplicated(
        subset=[
            "crop",
            "year",
        ]
    )
    .sum()
)

print(
    f"Duplicate crop-year rows: "
    f"{faostat_duplicates}"
)

if faostat_duplicates > 0:
    raise ValueError(
        "FAOSTAT contains duplicate "
        "crop-year rows."
    )

validate_no_missing_values(
    faostat,
    [
        "crop",
        "year",
        TARGET_COLUMN,
    ],
    "FAOSTAT",
)


# ============================================================
# 14. VALIDATE FAOSTAT YEAR COVERAGE
# ============================================================

faostat_years = set(
    faostat["year"].unique()
)

missing_faostat_years = (
    EXPECTED_YEARS
    - faostat_years
)

extra_faostat_years = (
    faostat_years
    - EXPECTED_YEARS
)

if missing_faostat_years:
    raise ValueError(
        "FAOSTAT is missing expected years:\n"
        f"{sorted(missing_faostat_years)}"
    )

if extra_faostat_years:
    raise ValueError(
        "FAOSTAT contains unexpected years:\n"
        f"{sorted(extra_faostat_years)}"
    )


# ============================================================
# 15. VALIDATE FAOSTAT CROP COVERAGE
# ============================================================

actual_crops = set(
    faostat["crop"].unique()
)

if actual_crops != EXPECTED_CROPS:
    raise ValueError(
        "Unexpected crop set.\n"
        f"Expected: {sorted(EXPECTED_CROPS)}\n"
        f"Actual:   {sorted(actual_crops)}"
    )


# ============================================================
# 16. LOAD NASA POWER SEASONAL DATA
# ============================================================

print_section(
    "Loading NASA POWER seasonal data"
)

nasa = pd.read_csv(
    NASA_PATH
)

print(
    f"Rows loaded: {len(nasa)}"
)

print("Columns:")
print(
    nasa.columns.tolist()
)


# ============================================================
# 17. NORMALIZE NASA COLUMN NAMES
# ============================================================

nasa.columns = (
    nasa.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nNormalized NASA columns:")
print(
    nasa.columns.tolist()
)


# ============================================================
# 18. BUILD EXPECTED SEASONAL FEATURE LIST
# ============================================================

print("\nExpected seasonal features:")

for feature in EXPECTED_SEASONAL_FEATURES:
    print(f"  - {feature}")


# ============================================================
# 19. VALIDATE NASA SCHEMA
# ============================================================

validate_columns(
    nasa,
    [
        "location",
        "year",
    ] + EXPECTED_SEASONAL_FEATURES,
    "NASA POWER seasonal dataset",
)


# ============================================================
# 20. PRESERVE LOCATION AS STRING
# ============================================================

# IMPORTANT:
# location is a categorical identifier.
# It must NEVER be converted to numeric.

nasa["location"] = (
    nasa["location"]
    .astype("string")
    .str.strip()
)

# Convert only year to numeric here.
nasa["year"] = pd.to_numeric(
    nasa["year"],
    errors="coerce",
)


# ============================================================
# 21. CONVERT ONLY CLIMATE FEATURES TO NUMERIC
# ============================================================

for column in EXPECTED_SEASONAL_FEATURES:

    nasa[column] = pd.to_numeric(
        nasa[column],
        errors="coerce",
    )


# ============================================================
# 22. NASA BASIC VALIDATION
# ============================================================

print_section(
    "NASA POWER seasonal validation"
)

# Check that location was not accidentally converted to NaN.

if nasa["location"].isna().any():
    missing_location_count = (
        nasa["location"]
        .isna()
        .sum()
    )

    raise ValueError(
        "NASA location contains "
        f"{missing_location_count} missing values."
    )

location_count = (
    nasa["location"]
    .nunique()
)

print(
    f"Reference locations found: "
    f"{location_count}"
)

print("Locations:")

for location in sorted(
    nasa["location"].unique()
):
    print(f"  - {location}")


# ============================================================
# 23. VALIDATE LOCATION COUNT
# ============================================================

if location_count != EXPECTED_LOCATION_COUNT:

    raise ValueError(
        "Unexpected number of NASA "
        "reference locations.\n"
        f"Expected: {EXPECTED_LOCATION_COUNT}\n"
        f"Actual:   {location_count}"
    )


# ============================================================
# 24. FILTER NASA YEARS
# ============================================================

nasa = nasa[
    nasa["year"].between(
        START_YEAR,
        END_YEAR,
    )
].copy()

nasa["year"] = (
    nasa["year"]
    .astype(int)
)

print(
    f"\nNASA rows after "
    f"{START_YEAR}-{END_YEAR} filter: "
    f"{len(nasa)}"
)


# ============================================================
# 25. VALIDATE LOCATION-YEAR STRUCTURE
# ============================================================

expected_nasa_rows = (
    EXPECTED_LOCATION_COUNT
    * len(EXPECTED_YEARS)
)

print(
    f"Expected NASA rows: "
    f"{expected_nasa_rows}"
)

print(
    f"Actual NASA rows: "
    f"{len(nasa)}"
)

if len(nasa) != expected_nasa_rows:

    raise ValueError(
        "NASA dataset does not contain "
        "the expected location-year "
        "coverage.\n"
        f"Expected rows: "
        f"{expected_nasa_rows}\n"
        f"Actual rows: "
        f"{len(nasa)}"
    )


# ============================================================
# 26. CHECK DUPLICATE LOCATION-YEAR ROWS
# ============================================================

location_year_duplicates = (
    nasa
    .duplicated(
        subset=[
            "location",
            "year",
        ]
    )
    .sum()
)

print(
    f"Duplicate location-year rows: "
    f"{location_year_duplicates}"
)

if location_year_duplicates > 0:

    raise ValueError(
        "NASA dataset contains duplicate "
        "location-year rows."
    )


# ============================================================
# 27. VALIDATE NASA MISSING VALUES
# ============================================================

validate_no_missing_values(
    nasa,
    [
        "location",
        "year",
    ] + EXPECTED_SEASONAL_FEATURES,
    "NASA POWER seasonal dataset",
)


# ============================================================
# 28. AGGREGATE NASA REFERENCE LOCATIONS
# ============================================================

print_section(
    "Aggregating NASA reference locations"
)

print(
    f"Reference locations found: "
    f"{nasa['location'].nunique()}"
)

print("Locations:")

for location in sorted(
    nasa["location"].unique()
):
    print(f"  - {location}")


# ------------------------------------------------------------
# Spatial aggregation
# ------------------------------------------------------------
#
# We calculate the mean climate value across the 10 reference
# locations for every year.
#
# Result:
#
#   10 locations × 41 years
#                 ↓
#             groupby(year)
#                 ↓
#             41 rows
#
# location is deliberately NOT included in the output because
# the resulting dataset represents one India-prototype climate
# observation per year.
# ------------------------------------------------------------

nasa_annual = (
    nasa
    .groupby(
        "year",
        as_index=False,
    )[EXPECTED_SEASONAL_FEATURES]
    .mean()
)


print(
    f"\nRows after spatial aggregation: "
    f"{len(nasa_annual)}"
)


# ============================================================
# 29. NASA AGGREGATED VALIDATION
# ============================================================

print("\nNASA aggregated-data validation")

nasa_duplicates = (
    nasa_annual
    .duplicated(
        subset=["year"]
    )
    .sum()
)

print(
    f"Duplicate year rows: "
    f"{nasa_duplicates}"
)

if nasa_duplicates > 0:

    raise ValueError(
        "NASA aggregated dataset contains "
        "duplicate year rows."
    )


# ============================================================
# 30. VALIDATE AGGREGATED YEAR COVERAGE
# ============================================================

aggregated_years = set(
    nasa_annual["year"].unique()
)

missing_nasa_years = (
    EXPECTED_YEARS
    - aggregated_years
)

extra_nasa_years = (
    aggregated_years
    - EXPECTED_YEARS
)

if missing_nasa_years:

    raise ValueError(
        "NASA aggregated dataset is "
        "missing expected years:\n"
        f"{sorted(missing_nasa_years)}"
    )

if extra_nasa_years:

    raise ValueError(
        "NASA aggregated dataset contains "
        "unexpected years:\n"
        f"{sorted(extra_nasa_years)}"
    )


# ============================================================
# 31. VALIDATE AGGREGATED MISSING VALUES
# ============================================================

validate_no_missing_values(
    nasa_annual,
    [
        "year",
    ] + EXPECTED_SEASONAL_FEATURES,
    "NASA aggregated dataset",
)


# ============================================================
# 32. VALIDATE EXPECTED AGGREGATED ROW COUNT
# ============================================================

if len(nasa_annual) != len(EXPECTED_YEARS):

    raise ValueError(
        "NASA aggregation produced an "
        "unexpected number of rows.\n"
        f"Expected: {len(EXPECTED_YEARS)}\n"
        f"Actual:   {len(nasa_annual)}"
    )


# ============================================================
# 33. CREATE FINAL NASA DATAFRAME
# ============================================================

nasa = nasa_annual.copy()


# ============================================================
# 34. CREATE V2 UNIFIED SEASONAL DATASET
# ============================================================

print_section(
    "Creating V2 unified seasonal dataset"
)

# Keep only the required FAOSTAT columns.

faostat_final = faostat[
    [
        "year",
        "crop",
        TARGET_COLUMN,
    ]
].copy()


# Merge on year.
#
# Each year has one aggregated NASA climate record.
# Each year has three crop observations.
#
# Therefore:
#
#   41 years × 3 crops = 123 rows
#

training_v2 = faostat_final.merge(
    nasa,
    on="year",
    how="inner",
    validate="many_to_one",
)


# ============================================================
# 35. FINAL COLUMN ORDER
# ============================================================

training_v2 = training_v2[
    [
        "year",
        "crop",
    ]
    + EXPECTED_SEASONAL_FEATURES
    + [
        TARGET_COLUMN,
    ]
]


# ============================================================
# 36. FINAL DATASET VALIDATION
# ============================================================

print_section(
    "Final V2 dataset validation"
)

print(
    f"Rows: {len(training_v2)}"
)

print(
    f"Crops: "
    f"{sorted(training_v2['crop'].unique())}"
)

print(
    f"Years: "
    f"{training_v2['year'].nunique()}"
)

print(
    f"Year range: "
    f"{training_v2['year'].min()}-"
    f"{training_v2['year'].max()}"
)


# ============================================================
# 37. EXPECTED ROW COUNT
# ============================================================

expected_final_rows = (
    len(EXPECTED_CROPS)
    * len(EXPECTED_YEARS)
)

print(
    f"Expected rows: "
    f"{expected_final_rows}"
)

if len(training_v2) != expected_final_rows:

    raise ValueError(
        "Unexpected final dataset row count.\n"
        f"Expected: {expected_final_rows}\n"
        f"Actual:   {len(training_v2)}"
    )


# ============================================================
# 38. FINAL CROP-YEAR DUPLICATE CHECK
# ============================================================

final_duplicates = (
    training_v2
    .duplicated(
        subset=[
            "crop",
            "year",
        ]
    )
    .sum()
)

print(
    f"Duplicate crop-year rows: "
    f"{final_duplicates}"
)

if final_duplicates > 0:

    raise ValueError(
        "Final V2 dataset contains "
        "duplicate crop-year rows."
    )


# ============================================================
# 39. FINAL YEAR COVERAGE
# ============================================================

final_years = set(
    training_v2["year"].unique()
)

if final_years != EXPECTED_YEARS:

    missing_years = (
        EXPECTED_YEARS
        - final_years
    )

    extra_years = (
        final_years
        - EXPECTED_YEARS
    )

    raise ValueError(
        "Final V2 dataset has incorrect "
        "year coverage.\n"
        f"Missing years: {sorted(missing_years)}\n"
        f"Extra years:   {sorted(extra_years)}"
    )


# ============================================================
# 40. FINAL CROP COVERAGE
# ============================================================

final_crops = set(
    training_v2["crop"].unique()
)

if final_crops != EXPECTED_CROPS:

    raise ValueError(
        "Final V2 dataset has incorrect "
        "crop coverage.\n"
        f"Expected: {sorted(EXPECTED_CROPS)}\n"
        f"Actual:   {sorted(final_crops)}"
    )


# ============================================================
# 41. FINAL MISSING VALUE CHECK
# ============================================================

final_required_columns = (
    [
        "year",
        "crop",
    ]
    + EXPECTED_SEASONAL_FEATURES
    + [
        TARGET_COLUMN,
    ]
)

validate_no_missing_values(
    training_v2,
    final_required_columns,
    "Final V2 training dataset",
)


# ============================================================
# 42. FINAL DATA TYPE VALIDATION
# ============================================================

if not pd.api.types.is_integer_dtype(
    training_v2["year"]
):
    raise ValueError(
        "Final year column is not integer."
    )

for column in EXPECTED_SEASONAL_FEATURES:

    if not pd.api.types.is_numeric_dtype(
        training_v2[column]
    ):
        raise ValueError(
            f"Seasonal feature '{column}' "
            "is not numeric."
        )

if not pd.api.types.is_numeric_dtype(
    training_v2[TARGET_COLUMN]
):
    raise ValueError(
        f"Target '{TARGET_COLUMN}' "
        "is not numeric."
    )


# ============================================================
# 43. SORT FINAL DATASET
# ============================================================

training_v2 = (
    training_v2
    .sort_values(
        [
            "year",
            "crop",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 44. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 45. SAVE V2 DATASET
# ============================================================

training_v2.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# 46. FINAL SUMMARY
# ============================================================

print_section(
    "V2 dataset successfully created"
)

print(
    f"Output file:\n{OUTPUT_PATH}"
)

print(
    f"\nRows: "
    f"{len(training_v2)}"
)

print(
    f"Crops: "
    f"{training_v2['crop'].nunique()}"
)

print(
    f"Years: "
    f"{training_v2['year'].nunique()}"
)

print(
    f"Year range: "
    f"{training_v2['year'].min()}-"
    f"{training_v2['year'].max()}"
)

print(
    f"Seasonal climate features: "
    f"{len(EXPECTED_SEASONAL_FEATURES)}"
)

print(
    f"Target: "
    f"{TARGET_COLUMN}"
)

print(
    f"Duplicate crop-year rows: "
    f"{training_v2.duplicated(subset=['crop', 'year']).sum()}"
)

print(
    f"Missing values: "
    f"{training_v2[final_required_columns].isna().sum().sum()}"
)

print("\nFinal columns:")

for column in training_v2.columns:
    print(f"  - {column}")

print("\n" + "=" * 70)
print("V2 BUILD COMPLETE")
print("=" * 70)
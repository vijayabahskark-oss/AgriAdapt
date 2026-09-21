from pathlib import Path
import pandas as pd


# ============================================================
# AgriAdapt - Unified Training Dataset Builder
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FAOSTAT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "faostat_india_crop_yield_processed.csv"
)

NASA_ANNUAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather"
    / "nasa_power_india_annual.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "agriadapt_training_dataset.csv"
)


START_YEAR = 1984
END_YEAR = 2024


# ============================================================
# Helper
# ============================================================

def check_file(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


# ============================================================
# 1. Check input files
# ============================================================

print("=" * 70)
print("AgriAdapt Unified Training Dataset Builder")
print("=" * 70)

check_file(FAOSTAT_FILE, "FAOSTAT processed file")
check_file(NASA_ANNUAL_FILE, "NASA annual weather file")

print("\nInput files:")
print(f"FAOSTAT : {FAOSTAT_FILE}")
print(f"NASA    : {NASA_ANNUAL_FILE}")


# ============================================================
# 2. Load FAOSTAT
# ============================================================

print("\n" + "-" * 70)
print("Loading FAOSTAT yield data")
print("-" * 70)

faostat = pd.read_csv(FAOSTAT_FILE)

print(f"Rows loaded: {len(faostat):,}")
print("Columns:")
print(list(faostat.columns))


# Normalize column names
faostat.columns = (
    faostat.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
# Map actual FAOSTAT column names to the unified schema
faostat = faostat.rename(
    columns={
        "item": "crop",
    }
)

# Expected crop/year/yield columns
required_fao = {
    "crop",
    "year",
    "yield_tonnes_per_ha",
}

missing = required_fao - set(faostat.columns)

if missing:
    raise ValueError(
        f"FAOSTAT is missing required columns: {sorted(missing)}"
    )


# Keep required fields
faostat = faostat[
    ["crop", "year", "yield_tonnes_per_ha"]
].copy()


faostat["year"] = pd.to_numeric(
    faostat["year"],
    errors="coerce"
).astype("Int64")

faostat["yield_tonnes_per_ha"] = pd.to_numeric(
    faostat["yield_tonnes_per_ha"],
    errors="coerce"
)


# Restrict to overlapping period
faostat = faostat[
    (faostat["year"] >= START_YEAR)
    & (faostat["year"] <= END_YEAR)
].copy()


print(
    f"FAOSTAT rows after {START_YEAR}-{END_YEAR} filter: "
    f"{len(faostat):,}"
)

print(
    "Crops:",
    sorted(faostat["crop"].dropna().unique())
)


# ============================================================
# 3. Validate FAOSTAT
# ============================================================

print("\nFAOSTAT validation")

fao_duplicates = faostat.duplicated(
    subset=["crop", "year"]
).sum()

fao_missing = faostat[
    ["crop", "year", "yield_tonnes_per_ha"]
].isna().sum()

print(f"Duplicate crop-year rows : {fao_duplicates}")
print(f"Missing values:")
print(fao_missing.to_string())

if fao_duplicates > 0:
    raise ValueError(
        "FAOSTAT contains duplicate crop-year observations."
    )

if faostat["yield_tonnes_per_ha"].isna().any():
    raise ValueError(
        "FAOSTAT contains missing yield values."
    )


# ============================================================
# 4. Load NASA POWER annual data
# ============================================================

print("\n" + "-" * 70)
print("Loading NASA POWER annual weather data")
print("-" * 70)

nasa = pd.read_csv(NASA_ANNUAL_FILE)

print(f"Rows loaded: {len(nasa):,}")
print("Columns:")
print(list(nasa.columns))


# Normalize names
nasa.columns = (
    nasa.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# Display schema after normalization
print("\nNormalized NASA columns:")
print(list(nasa.columns))


# ============================================================
# 5. Detect expected NASA columns
# ============================================================

# The ingestion script may use slightly different names.
# We resolve the important variables here.

COLUMN_ALIASES = {
    "year": [
        "year"
    ],

    "temperature_mean": [
        "temperature_mean",
        "t2m_mean",
        "temperature"
    ],

    "rainfall_total": [
        "rainfall_total",
        "precipitation_total",
        "precipitation_sum",
        "rainfall"
    ],

    "humidity_mean": [
        "humidity_mean",
        "rh2m_mean",
        "relative_humidity_mean"
    ],

    "solar_radiation_mean": [
        "solar_radiation_mean",
        "allsky_sfc_sw_dwn_mean",
        "solar_mean"
    ],

    "wind_speed_mean": [
        "wind_speed_mean",
        "ws2m_mean",
        "wind_mean"
    ],
}


def resolve_column(dataframe, aliases, variable_name):
    for alias in aliases:
        if alias in dataframe.columns:
            return alias

    raise ValueError(
        f"Could not find NASA column for '{variable_name}'.\n"
        f"Available columns:\n{list(dataframe.columns)}"
    )


resolved = {}

for variable, aliases in COLUMN_ALIASES.items():
    resolved[variable] = resolve_column(
        nasa,
        aliases,
        variable
    )

print("\nResolved NASA columns:")

for variable, column in resolved.items():
    print(f"{variable:25s} -> {column}")


# ============================================================
# 6. Select NASA variables
# ============================================================

nasa = nasa[
    [
        resolved["year"],
        resolved["temperature_mean"],
        resolved["rainfall_total"],
        resolved["humidity_mean"],
        resolved["solar_radiation_mean"],
        resolved["wind_speed_mean"],
    ]
].copy()


nasa.columns = [
    "year",
    "temperature_mean",
    "rainfall_total",
    "humidity_mean",
    "solar_radiation_mean",
    "wind_speed_mean",
]


# Numeric conversion
for column in nasa.columns:
    nasa[column] = pd.to_numeric(
        nasa[column],
        errors="coerce"
    )


nasa["year"] = nasa["year"].astype("Int64")


# Restrict period
nasa = nasa[
    (nasa["year"] >= START_YEAR)
    & (nasa["year"] <= END_YEAR)
].copy()


# ============================================================
# 7. Validate NASA data
# ============================================================

print("\nNASA validation")

nasa_duplicates = nasa.duplicated(
    subset=["year"]
).sum()

nasa_missing = nasa.isna().sum()

print(f"Duplicate year rows : {nasa_duplicates}")

print("Missing values:")
print(nasa_missing.to_string())

if nasa_duplicates > 0:
    raise ValueError(
        "NASA India annual dataset contains duplicate years."
    )


# ============================================================
# 8. Check year coverage
# ============================================================

expected_years = set(
    range(START_YEAR, END_YEAR + 1)
)

fao_years = set(
    faostat["year"].dropna().astype(int)
)

nasa_years = set(
    nasa["year"].dropna().astype(int)
)

print("\nYear coverage")

print(
    f"Expected years : {START_YEAR}-{END_YEAR}"
)

print(
    f"FAOSTAT years  : {min(fao_years)}-{max(fao_years)}"
)

print(
    f"NASA years     : {min(nasa_years)}-{max(nasa_years)}"
)

missing_fao_years = sorted(
    expected_years - fao_years
)

missing_nasa_years = sorted(
    expected_years - nasa_years
)

print(
    f"Missing FAOSTAT years: {missing_fao_years}"
)

print(
    f"Missing NASA years: {missing_nasa_years}"
)


# ============================================================
# 9. Merge FAOSTAT + NASA
# ============================================================

print("\n" + "-" * 70)
print("Creating unified crop-year dataset")
print("-" * 70)

dataset = faostat.merge(
    nasa,
    on="year",
    how="inner",
    validate="many_to_one"
)


# ============================================================
# 10. Final column ordering
# ============================================================

dataset = dataset[
    [
        "year",
        "crop",
        "temperature_mean",
        "rainfall_total",
        "humidity_mean",
        "solar_radiation_mean",
        "wind_speed_mean",
        "yield_tonnes_per_ha",
    ]
].sort_values(
    ["year", "crop"]
).reset_index(drop=True)


# ============================================================
# 11. Final validation
# ============================================================

print("\n" + "-" * 70)
print("Final dataset validation")
print("-" * 70)

expected_rows = len(expected_years) * 3

print(f"Expected crop-year rows : {expected_rows}")
print(f"Actual rows              : {len(dataset)}")

duplicates = dataset.duplicated(
    subset=["year", "crop"]
).sum()

missing = dataset.isna().sum()

print(f"\nDuplicate crop-year rows : {duplicates}")

print("\nMissing values:")
print(missing.to_string())


if duplicates > 0:
    raise ValueError(
        "Final dataset contains duplicate crop-year observations."
    )


if dataset.isna().any().any():
    raise ValueError(
        "Final dataset contains missing values."
    )


# ============================================================
# 12. Basic range checks
# ============================================================

print("\nBasic feature ranges")

numeric_columns = [
    "temperature_mean",
    "rainfall_total",
    "humidity_mean",
    "solar_radiation_mean",
    "wind_speed_mean",
    "yield_tonnes_per_ha",
]

print(
    dataset[numeric_columns]
    .describe()
    .round(3)
    .to_string()
)


# ============================================================
# 13. Save dataset
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 14. Final summary
# ============================================================

print("\n" + "=" * 70)
print("DATASET CREATION SUCCESSFUL")
print("=" * 70)

print(f"Output file: {OUTPUT_FILE}")
print(f"Rows       : {len(dataset):,}")
print(f"Columns    : {len(dataset.columns)}")

print("\nCrops:")
for crop in sorted(dataset["crop"].unique()):
    count = (dataset["crop"] == crop).sum()
    print(f"  {crop}: {count} rows")

print("\nYear range:")
print(
    f"  {dataset['year'].min()} - "
    f"{dataset['year'].max()}"
)

print("\nColumns:")
for column in dataset.columns:
    print(f"  - {column}")

print("\nValidation:")
print("  ✓ No duplicate crop-year rows")
print("  ✓ No missing values")
print("  ✓ FAOSTAT + NASA merged successfully")
print("  ✓ Dataset saved successfully")

print("\nReady for model benchmarking.")
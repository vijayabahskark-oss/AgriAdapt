from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# AgriAdapt V4.1
# Validated district-level crop-yield target dataset
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "external" / "agridata_clean.csv"
OUTPUT = ROOT / "data" / "processed" / "v4" / "agridata_v4_target_valid.csv"
REPORT = ROOT / "reports" / "v4" / "agridata_quality_report.txt"

TARGET_CROPS = ["Rice", "Wheat", "Maize"]

REQUIRED_COLUMNS = [
    "state",
    "district",
    "crop",
    "year",
    "season",
    "area",
    "production",
    "yield",
]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load cleaned source
    # --------------------------------------------------------
    df = pd.read_csv(INPUT, low_memory=False)

    original_rows = len(df)

    # --------------------------------------------------------
    # 2. Schema validation
    # --------------------------------------------------------
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # 3. Keep only target crops
    # --------------------------------------------------------
    target = df[df["crop"].isin(TARGET_CROPS)].copy()

    target_rows = len(target)

    # --------------------------------------------------------
    # 4. Normalize text fields
    # --------------------------------------------------------
    text_columns = [
        "state",
        "district",
        "crop",
        "year",
        "season",
    ]

    for col in text_columns:
        target[col] = target[col].astype("string").str.strip()

    # --------------------------------------------------------
    # 5. Validate year
    #
    # Expected agricultural-year format:
    # YYYY-YY
    # Example: 1998-99
    # --------------------------------------------------------
    valid_year = target["year"].str.match(
        r"^\d{4}-\d{2}$",
        na=False
    )

    invalid_year_count = int((~valid_year).sum())

    if invalid_year_count:
        raise ValueError(
            f"Target dataset contains {invalid_year_count} "
            "invalid agricultural-year values."
        )

    # --------------------------------------------------------
    # 6. Convert numeric fields
    # --------------------------------------------------------
    numeric_columns = [
        "area",
        "production",
        "yield",
    ]

    for col in numeric_columns:
        target[col] = pd.to_numeric(
            target[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # 7. Basic missing-value audit
    # --------------------------------------------------------
    missing_before_filter = (
        target[
            REQUIRED_COLUMNS
        ].isna().sum()
    )

    # --------------------------------------------------------
    # 8. Non-negative value validation
    # --------------------------------------------------------
    negative_area = int((target["area"] < 0).sum())
    negative_production = int(
        (target["production"] < 0).sum()
    )
    negative_yield = int(
        (target["yield"] < 0).sum()
    )

    if negative_area or negative_production or negative_yield:
        raise ValueError(
            "Negative area/production/yield values detected."
        )

    # --------------------------------------------------------
    # 9. Duplicate validation
    # --------------------------------------------------------
    key_columns = [
        "state",
        "district",
        "crop",
        "year",
        "season",
    ]

    duplicate_count = int(
        target.duplicated(key_columns).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "district/crop/year/season records."
        )

    # --------------------------------------------------------
    # 10. Yield consistency validation
    #
    # AGRIDATA relationship:
    #
    # yield ≈ production / area
    # --------------------------------------------------------
    valid_yield_check = (
        target["area"].notna()
        & target["production"].notna()
        & target["yield"].notna()
        & (target["area"] > 0)
    )

    target["calculated_yield_tonnes_per_ha"] = np.nan

    target.loc[
        valid_yield_check,
        "calculated_yield_tonnes_per_ha"
    ] = (
        target.loc[valid_yield_check, "production"]
        / target.loc[valid_yield_check, "area"]
    )

    yield_difference = (
        target["yield"]
        - target["calculated_yield_tonnes_per_ha"]
    ).abs()

    valid_yield_differences = yield_difference[
        valid_yield_check
    ]

    median_yield_error = float(
        valid_yield_differences.median()
    )

    max_yield_error = float(
        valid_yield_differences.max()
    )

    # Tight tolerance because the previous audit showed
    # differences only at floating-point precision.
    YIELD_TOLERANCE = 1e-8

    yield_mismatch_count = int(
        (
            valid_yield_differences
            > YIELD_TOLERANCE
        ).sum()
    )

    if yield_mismatch_count:
        raise ValueError(
            f"{yield_mismatch_count} records failed "
            "the production/area/yield consistency check."
        )

    # --------------------------------------------------------
    # 11. Identify missing supervised targets
    #
    # Do NOT impute them.
    # --------------------------------------------------------
    missing_target = (
        target["production"].isna()
        | target["yield"].isna()
    )

    missing_target_count = int(
        missing_target.sum()
    )

    valid = target.loc[
        ~missing_target
    ].copy()

    # --------------------------------------------------------
    # 12. Construct final target
    # --------------------------------------------------------
    valid["yield_tonnes_per_ha"] = (
        valid["production"]
        / valid["area"]
    )

    final_columns = [
        "state",
        "district",
        "crop",
        "year",
        "season",
        "area",
        "production",
        "yield_tonnes_per_ha",
    ]

    valid = valid[final_columns]

    # --------------------------------------------------------
    # 13. Final validation
    # --------------------------------------------------------
    final_missing = int(
        valid.isna().sum().sum()
    )

    final_duplicates = int(
        valid.duplicated(key_columns).sum()
    )

    if final_missing:
        raise ValueError(
            f"Final dataset contains {final_missing} missing values."
        )

    if final_duplicates:
        raise ValueError(
            f"Final dataset contains {final_duplicates} duplicates."
        )

    # --------------------------------------------------------
    # 14. Save dataset
    # --------------------------------------------------------
    valid.to_csv(
        OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # 15. Build quality report
    # --------------------------------------------------------
    crop_counts = (
        target["crop"]
        .value_counts()
        .reindex(TARGET_CROPS)
        .fillna(0)
        .astype(int)
    )

    valid_crop_counts = (
        valid["crop"]
        .value_counts()
        .reindex(TARGET_CROPS)
        .fillna(0)
        .astype(int)
    )

    missing_crop_counts = (
        target.loc[missing_target, "crop"]
        .value_counts()
        .reindex(TARGET_CROPS)
        .fillna(0)
        .astype(int)
    )

    season_counts = (
        target.loc[missing_target, "season"]
        .value_counts()
    )

    year_counts = (
        target.loc[missing_target, "year"]
        .value_counts()
        .sort_index()
    )

    report_lines = [
        "AgriAdapt V4.1 - AGRIDATA Quality Report",
        "=" * 60,
        "",
        "SOURCE",
        f"Input: {INPUT}",
        "",
        "ROW COUNTS",
        f"Original source rows: {original_rows:,}",
        f"Target crop rows: {target_rows:,}",
        f"Valid supervised observations: {len(valid):,}",
        f"Excluded missing-target observations: "
        f"{missing_target_count:,}",
        "",
        "TARGET CROPS",
        f"Rice:  {crop_counts['Rice']:,}",
        f"Wheat: {crop_counts['Wheat']:,}",
        f"Maize: {crop_counts['Maize']:,}",
        "",
        "VALID SUPERVISED OBSERVATIONS",
        f"Rice:  {valid_crop_counts['Rice']:,}",
        f"Wheat: {valid_crop_counts['Wheat']:,}",
        f"Maize: {valid_crop_counts['Maize']:,}",
        "",
        "EXCLUDED MISSING TARGETS BY CROP",
        f"Rice:  {missing_crop_counts['Rice']:,}",
        f"Wheat: {missing_crop_counts['Wheat']:,}",
        f"Maize: {missing_crop_counts['Maize']:,}",
        "",
        "MISSING TARGETS BY SEASON",
    ]

    for season, count in season_counts.items():
        report_lines.append(
            f"{season}: {int(count):,}"
        )

    report_lines.extend([
        "",
        "MISSING TARGETS BY YEAR",
    ])

    for year, count in year_counts.items():
        report_lines.append(
            f"{year}: {int(count):,}"
        )

    report_lines.extend([
        "",
        "STRUCTURAL VALIDATION",
        f"Invalid year values: {invalid_year_count}",
        f"Duplicate keys: {duplicate_count}",
        f"Negative area values: {negative_area}",
        f"Negative production values: {negative_production}",
        f"Negative yield values: {negative_yield}",
        "",
        "YIELD CONSISTENCY",
        "Expected relationship:",
        "yield_tonnes_per_ha = production / area",
        f"Rows checked: {int(valid_yield_check.sum()):,}",
        f"Median absolute error: {median_yield_error:.15g}",
        f"Maximum absolute error: {max_yield_error:.15g}",
        f"Rows exceeding tolerance ({YIELD_TOLERANCE}): "
        f"{yield_mismatch_count}",
        "",
        "FINAL DATASET",
        f"Rows: {len(valid):,}",
        f"Columns: {len(valid.columns)}",
        f"Missing values: {final_missing}",
        f"Duplicate keys: {final_duplicates}",
        "",
        "PROVENANCE NOTE",
        "The original AGRIDATA source is preserved.",
        "Rows with missing production or yield are retained",
        "in the cleaned source but excluded from supervised",
        "training. No target-value imputation is performed.",
        "",
    ])

    REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print("=" * 60)
    print("AgriAdapt V4.1 completed successfully")
    print("=" * 60)
    print(f"Target rows:       {target_rows:,}")
    print(f"Valid observations:{len(valid):,}")
    print(f"Excluded targets:  {missing_target_count:,}")
    print(f"Output:            {OUTPUT}")
    print(f"Report:            {REPORT}")


if __name__ == "__main__":
    main()
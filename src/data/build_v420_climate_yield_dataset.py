from pathlib import Path
import pandas as pd
import re


ROOT = Path(__file__).resolve().parents[2]

CLIMATE_INPUT = (
    ROOT
    / "data/processed/v4/weather/"
    "nasa_power_district_seasonal_v420.csv"
)

YIELD_INPUT = (
    ROOT
    / "data/processed/v4/"
    "agridata_v4_target_valid.csv"
)

OUTPUT = (
    ROOT
    / "data/processed/v4/"
    "v4_climate_yield_dataset.csv"
)

REPORT = (
    ROOT
    / "reports/v4/"
    "v420_climate_yield_integration_report.txt"
)

UNMATCHED_OUTPUT = (
    ROOT
    / "data/processed/v4/"
    "v4_unmatched_yield_observations.csv"
)


CLIMATE_COLUMNS = [
    "temperature_mean",
    "rainfall_total",
    "humidity_mean",
    "solar_radiation_mean",
    "wind_speed_mean",
]


def normalize_text(value):

    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().upper(),
    )


def normalize_year(value):

    match = re.match(
        r"^(\d{4})-(\d{2})$",
        str(value).strip(),
    )

    if not match:
        return None

    return int(match.group(1))


def main():

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("AGRIADAPT V4.20")
    print("CLIMATEâ€“YIELD INTEGRATION")
    print("=" * 70)

    # ========================================================
    # LOAD
    # ========================================================

    if not CLIMATE_INPUT.exists():
        raise FileNotFoundError(
            f"Climate input not found:\n{CLIMATE_INPUT}"
        )

    if not YIELD_INPUT.exists():
        raise FileNotFoundError(
            f"Yield input not found:\n{YIELD_INPUT}"
        )

    climate = pd.read_csv(CLIMATE_INPUT)
    target = pd.read_csv(YIELD_INPUT)

    print("\nINPUT DATA")
    print("-" * 70)
    print(f"Climate rows : {len(climate):,}")
    print(f"Target rows  : {len(target):,}")

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_climate = [
        "state",
        "district",
        "latitude",
        "longitude",
        "year",
        "season",
        *CLIMATE_COLUMNS,
    ]

    missing_climate = [
        col
        for col in required_climate
        if col not in climate.columns
    ]

    if missing_climate:
        raise ValueError(
            "Missing climate columns:\n"
            + "\n".join(missing_climate)
        )

    required_target = [
        "state",
        "district",
        "crop",
        "year",
        "season",
        "area",
        "production",
        "yield_tonnes_per_ha",
    ]

    missing_target = [
        col
        for col in required_target
        if col not in target.columns
    ]

    if missing_target:
        raise ValueError(
            "Missing target columns:\n"
            + "\n".join(missing_target)
        )

    # ========================================================
    # NORMALIZE KEYS
    # ========================================================

    for df in [climate, target]:

        df["state_key"] = (
            df["state"]
            .map(normalize_text)
        )

        df["district_key"] = (
            df["district"]
            .map(normalize_text)
        )

        df["season_key"] = (
            df["season"]
            .map(normalize_text)
        )

    target["climate_year"] = (
        target["year"]
        .map(normalize_year)
    )

    if target["climate_year"].isna().any():

        raise ValueError(
            "Invalid AGRIDATA agricultural-year values."
        )

    climate["climate_year"] = (
        pd.to_numeric(
            climate["year"],
            errors="raise",
        ).astype(int)
    )

    # ========================================================
    # COORDINATE VALIDATION
    # ========================================================

    climate["latitude"] = pd.to_numeric(
        climate["latitude"],
        errors="coerce",
    )

    climate["longitude"] = pd.to_numeric(
        climate["longitude"],
        errors="coerce",
    )

    if climate["latitude"].isna().any():
        raise ValueError(
            "Missing latitude values in climate data."
        )

    if climate["longitude"].isna().any():
        raise ValueError(
            "Missing longitude values in climate data."
        )

    if not climate["latitude"].between(
        6,
        38,
    ).all():
        raise ValueError(
            "Latitude outside India validation envelope."
        )

    if not climate["longitude"].between(
        68,
        98,
    ).all():
        raise ValueError(
            "Longitude outside India validation envelope."
        )

    # ========================================================
    # CLIMATE KEY
    # ========================================================

    climate_key = [
        "state_key",
        "district_key",
        "climate_year",
        "season_key",
    ]

    climate_duplicates = climate.duplicated(
        climate_key,
        keep=False,
    )

    if climate_duplicates.any():

        print(
            "\nDuplicate climate keys detected:"
        )

        print(
            climate.loc[
                climate_duplicates,
                climate_key,
            ]
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            "Climate dataset contains duplicate "
            "district-year-season keys."
        )

    # ========================================================
    # TARGET KEY
    # ========================================================

    target_key = [
        "state_key",
        "district_key",
        "climate_year",
        "crop",
        "season_key",
    ]

    target_duplicates = target.duplicated(
        target_key,
        keep=False,
    )

    if target_duplicates.any():

        print(
            "\nDuplicate target keys detected:"
        )

        print(
            target.loc[
                target_duplicates,
                target_key,
            ]
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            "Target dataset contains duplicate "
            "district-crop-year-season observations."
        )

    # ========================================================
    # CLIMATE DATA FOR JOIN
    # ========================================================

    climate_for_join = climate[
        climate_key
        + [
            "latitude",
            "longitude",
        ]
        + CLIMATE_COLUMNS
        + [
            "months_available",
            "months_expected",
        ]
    ].copy()

    # ========================================================
    # JOIN
    # ========================================================

    print("\nJOINING CLIMATE + YIELD")
    print("-" * 70)

    merged = target.merge(
        climate_for_join,
        how="left",
        on=climate_key,
        validate="many_to_one",
        indicator=True,
    )

    matched = merged[
        merged["_merge"] == "both"
    ].copy()

    unmatched = merged[
        merged["_merge"] != "both"
    ].copy()

    print(
        f"Matched observations  : {len(matched):,}"
    )

    print(
        f"Unmatched observations: {len(unmatched):,}"
    )

    # ========================================================
    # SAVE UNMATCHED
    # ========================================================

    if not unmatched.empty:

        unmatched[
            [
                "state",
                "district",
                "crop",
                "year",
                "season",
            ]
        ].to_csv(
            UNMATCHED_OUTPUT,
            index=False,
        )

    # ========================================================
    # CLIMATE COMPLETENESS
    # ========================================================

    incomplete = matched[
        matched["months_available"]
        != matched["months_expected"]
    ]

    print(
        f"Incomplete climate windows: "
        f"{len(incomplete):,}"
    )

    if not incomplete.empty:

        raise ValueError(
            "Incomplete climate windows detected."
        )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    missing_climate = (
        matched[CLIMATE_COLUMNS]
        .isna()
        .any(axis=1)
    )

    missing_count = int(
        missing_climate.sum()
    )

    print(
        f"Matched rows with missing climate: "
        f"{missing_count:,}"
    )

    # Keep only complete supervised records.
    final = matched.loc[
        ~missing_climate
    ].copy()

    # ========================================================
    # TARGET VALIDATION
    # ========================================================

    if final[
        "yield_tonnes_per_ha"
    ].isna().any():

        raise ValueError(
            "Missing yield target detected."
        )

    if (
        final["yield_tonnes_per_ha"] < 0
    ).any():

        raise ValueError(
            "Negative yield detected."
        )

    # ========================================================
    # FINAL DUPLICATE CHECK
    # ========================================================

    final_duplicates = final.duplicated(
        target_key,
        keep=False,
    )

    if final_duplicates.any():

        raise ValueError(
            "Duplicate rows detected in final dataset."
        )

    # ========================================================
    # FINAL COLUMN SELECTION
    # ========================================================

    final = final[
        [
            "state",
            "district",
            "crop",
            "year",
            "climate_year",
            "season",

            # Geography
            "latitude",
            "longitude",

            # Original agricultural quantities
            "area",
            "production",
            "yield_tonnes_per_ha",

            # Climate
            *CLIMATE_COLUMNS,

            # Coverage diagnostics
            "months_available",
            "months_expected",
        ]
    ].copy()

    final = final.rename(
        columns={
            "climate_year": "year_climate",
        }
    )

    final = final.sort_values(
        [
            "state",
            "district",
            "crop",
            "year",
            "season",
        ]
    ).reset_index(drop=True)

    # ========================================================
    # FINAL SCHEMA VALIDATION
    # ========================================================

    expected_columns = [
        "state",
        "district",
        "crop",
        "year",
        "year_climate",
        "season",
        "latitude",
        "longitude",
        "area",
        "production",
        "yield_tonnes_per_ha",
        *CLIMATE_COLUMNS,
        "months_available",
        "months_expected",
    ]

    if list(final.columns) != expected_columns:

        raise RuntimeError(
            "Final schema does not match expected schema."
        )

    if final.isna().any().any():

        print(
            "\nMissing values:"
        )

        print(
            final.isna().sum()
        )

        raise RuntimeError(
            "Final V4.20 dataset contains missing values."
        )

    # ========================================================
    # SAVE
    # ========================================================

    final.to_csv(
        OUTPUT,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    match_rate = (
        len(matched)
        / len(target)
        * 100
    )

    report = []

    report.append(
        "AGRIADAPT V4.20 CLIMATE-YIELD INTEGRATION"
    )
    report.append("=" * 70)
    report.append("")
    report.append(
        f"Target input rows: {len(target):,}"
    )
    report.append(
        f"Climate input rows: {len(climate):,}"
    )
    report.append(
        f"Matched rows: {len(matched):,}"
    )
    report.append(
        f"Unmatched rows: {len(unmatched):,}"
    )
    report.append(
        f"Match rate: {match_rate:.2f}%"
    )
    report.append(
        f"Final rows: {len(final):,}"
    )
    report.append("")
    report.append(
        "FINAL COLUMNS"
    )
    report.append("-" * 70)

    for column in final.columns:
        report.append(column)

    report.append("")
    report.append(
        "MISSING VALUES"
    )
    report.append("-" * 70)

    report.append(
        final.isna()
        .sum()
        .to_string()
    )

    report.append("")
    report.append(
        "CROP DISTRIBUTION"
    )
    report.append("-" * 70)

    report.append(
        final["crop"]
        .value_counts()
        .to_string()
    )

    report.append("")
    report.append(
        "SEASON DISTRIBUTION"
    )
    report.append("-" * 70)

    report.append(
        final["season"]
        .value_counts()
        .to_string()
    )

    report.append("")
    report.append(
        f"Output: {OUTPUT}"
    )

    report.append(
        f"Unmatched: {UNMATCHED_OUTPUT}"
    )

    REPORT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("V4.20 FIXED AND REBUILT")
    print("=" * 70)

    print(
        f"Final rows     : {len(final):,}"
    )

    print(
        f"Final columns  : {len(final.columns)}"
    )

    print(
        f"Match rate     : {match_rate:.2f}%"
    )

    print(
        f"Missing values : {int(final.isna().sum().sum())}"
    )

    print(
        "\nCoordinates:"
    )

    print(
        final[
            [
                "latitude",
                "longitude",
            ]
        ].describe().to_string()
    )

    print(
        f"\nDataset: {OUTPUT}"
    )

    print(
        f"Report: {REPORT}"
    )


if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data/processed/v4/weather/"
    "nasa_power_district_monthly_v419.csv"
)

OUTPUT = (
    ROOT
    / "data/processed/v4/weather/"
    "nasa_power_district_seasonal_v420.csv"
)

REPORT = (
    ROOT
    / "reports/v4/weather/"
    "nasa_power_district_seasonal_v420.txt"
)

SEASONS = {
    "Kharif": [6, 7, 8, 9, 10],
    "Rabi": [11, 12, 1, 2, 3],
    "Summer": [4, 5],
    "Winter": [12, 1, 2],
    "Autumn": [10, 11],
}


CLIMATE_COLUMNS = [
    "temperature_mean",
    "rainfall_total",
    "humidity_mean",
    "solar_radiation_mean",
    "wind_speed_mean",
]


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
    print("COORDINATE-PRESERVING CLIMATE AGGREGATION")
    print("=" * 70)

    df = pd.read_csv(INPUT)

    required = [
        "state",
        "district",
        "latitude",
        "longitude",
        "year",
        "month",
        *CLIMATE_COLUMNS,
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in V4.19 climate dataset: {missing}"
        )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="raise",
    ).astype(int)

    df["month"] = pd.to_numeric(
        df["month"],
        errors="raise",
    ).astype(int)

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="raise",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="raise",
    )

    if not df["month"].between(1, 12).all():
        raise ValueError(
            "Invalid month found."
        )

    if df.duplicated(
        ["state", "district", "year", "month"]
    ).any():

        raise ValueError(
            "Duplicate district-year-month rows found."
        )

    records = []

    for (
        state,
        district,
        year,
    ), group in df.groupby(
        ["state", "district", "year"],
        sort=False,
    ):

        latitude_values = (
            group["latitude"].unique()
        )

        longitude_values = (
            group["longitude"].unique()
        )

        if len(latitude_values) != 1:
            raise ValueError(
                f"Multiple latitudes for "
                f"{state}/{district}/{year}"
            )

        if len(longitude_values) != 1:
            raise ValueError(
                f"Multiple longitudes for "
                f"{state}/{district}/{year}"
            )

        latitude = latitude_values[0]
        longitude = longitude_values[0]

        # ----------------------------------------------------
        # Whole year
        # ----------------------------------------------------

        records.append(
            {
                "state": state,
                "district": district,
                "latitude": latitude,
                "longitude": longitude,
                "year": year,
                "season": "Whole year",
                "temperature_mean":
                    group["temperature_mean"].mean(),
                "rainfall_total":
                    group["rainfall_total"].sum(),
                "humidity_mean":
                    group["humidity_mean"].mean(),
                "solar_radiation_mean":
                    group["solar_radiation_mean"].mean(),
                "wind_speed_mean":
                    group["wind_speed_mean"].mean(),
                "months_available":
                    group["month"].nunique(),
                "months_expected": 12,
            }
        )

        # ----------------------------------------------------
        # Agricultural seasons
        # ----------------------------------------------------

        for season, months in SEASONS.items():

            selected = group[
                group["month"].isin(months)
            ].copy()

            if selected.empty:
                continue

            records.append(
                {
                    "state": state,
                    "district": district,
                    "latitude": latitude,
                    "longitude": longitude,
                    "year": year,
                    "season": season,
                    "temperature_mean":
                        selected["temperature_mean"].mean(),
                    "rainfall_total":
                        selected["rainfall_total"].sum(),
                    "humidity_mean":
                        selected["humidity_mean"].mean(),
                    "solar_radiation_mean":
                        selected[
                            "solar_radiation_mean"
                        ].mean(),
                    "wind_speed_mean":
                        selected[
                            "wind_speed_mean"
                        ].mean(),
                    "months_available":
                        selected["month"].nunique(),
                    "months_expected":
                        len(months),
                }
            )

    result = pd.DataFrame(records)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if result.empty:
        raise RuntimeError(
            "No seasonal climate records generated."
        )

    if result[
        [
            "latitude",
            "longitude",
        ]
    ].isna().any().any():

        raise RuntimeError(
            "Missing coordinates in seasonal dataset."
        )

    if result.duplicated(
        [
            "state",
            "district",
            "year",
            "season",
        ]
    ).any():

        raise RuntimeError(
            "Duplicate district-year-season records."
        )

    incomplete = result[
        result["months_available"]
        != result["months_expected"]
    ]

    if not incomplete.empty:

        print(
            "WARNING: incomplete seasonal windows:"
        )

        print(
            incomplete[
                [
                    "state",
                    "district",
                    "year",
                    "season",
                    "months_available",
                    "months_expected",
                ]
            ].head(20).to_string(index=False)
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result = result.sort_values(
        [
            "state",
            "district",
            "year",
            "season",
        ]
    ).reset_index(drop=True)

    result.to_csv(
        OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = [
        "AGRIADAPT V4.20 SEASONAL CLIMATE REPORT",
        "=" * 70,
        f"Input rows: {len(df):,}",
        f"Output rows: {len(result):,}",
        f"Unique districts: "
        f"{result[['state','district']].drop_duplicates().shape[0]:,}",
        f"Latitude missing: "
        f"{result['latitude'].isna().sum()}",
        f"Longitude missing: "
        f"{result['longitude'].isna().sum()}",
        f"Duplicate district-season rows: "
        f"{result.duplicated(['state','district','year','season']).sum()}",
        "",
        "Season definitions:",
    ]

    for season, months in SEASONS.items():
        report.append(
            f"{season}: {months}"
        )

    report.append(
        "Whole year: 1-12"
    )

    report.append("")
    report.append(
        f"Output: {OUTPUT}"
    )

    REPORT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("V4.20 CLIMATE AGGREGATION COMPLETE")
    print("=" * 70)
    print(f"Input rows : {len(df):,}")
    print(f"Output rows: {len(result):,}")
    print(
        f"Coordinates preserved: "
        f"{result['latitude'].notna().all() and result['longitude'].notna().all()}"
    )
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()

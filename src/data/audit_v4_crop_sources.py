from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(r"D:\AgriAdapt")

AGRIDATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "agridata.csv"
)

ICRISAT_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "icrisat"
    / "ICRISAT-District Level Data.csv"
)


def audit_agridata():
    print("\n" + "=" * 70)
    print("AGRIDATA AUDIT")
    print("=" * 70)

    df = pd.read_csv(
        AGRIDATA_PATH,
        low_memory=False,
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nStates:")
    print(df["state"].nunique())

    print("\nDistricts:")
    print(df["district"].nunique())

    print("\nCrops:")
    print(df["crop"].nunique())
    print(sorted(df["crop"].dropna().unique())[:100])

    print("\nYears:")
    print(df["year"].min(), "to", df["year"].max())

    print("\nSeasons:")
    print(df["season"].value_counts(dropna=False))

    target_crops = ["Rice", "Wheat", "Maize"]

    crop_df = df[
        df["crop"].astype(str).str.strip().isin(target_crops)
    ].copy()

    print("\nTarget crop rows:")
    print(
        crop_df["crop"]
        .value_counts()
    )

    print("\nMissing values:")
    print(
        crop_df[
            [
                "state",
                "district",
                "crop",
                "year",
                "season",
                "area",
                "production",
                "yield",
            ]
        ]
        .isna()
        .sum()
    )

    duplicate_cols = [
        "state",
        "district",
        "crop",
        "year",
        "season",
    ]

    duplicates = crop_df.duplicated(
        subset=duplicate_cols,
        keep=False,
    )

    print(
        "\nDuplicate district/crop/year/season records:",
        duplicates.sum(),
    )

    print("\nYield statistics:")
    print(
        crop_df.groupby("crop")["yield"]
        .agg(["count", "min", "max", "mean"])
    )

    print("\nPotential negative values:")
    numeric_cols = [
        "area",
        "production",
        "yield",
    ]

    for col in numeric_cols:
        print(
            f"{col}:",
            (crop_df[col] < 0).sum()
        )

    return df


def audit_icrisat():
    print("\n" + "=" * 70)
    print("ICRISAT AUDIT")
    print("=" * 70)

    df = pd.read_csv(
        ICRISAT_PATH,
        low_memory=False,
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nStates:")
    print(df["State Name"].nunique())

    print("\nDistricts:")
    print(df["Dist Name"].nunique())

    print("\nYears:")
    print(df["Year"].min(), "to", df["Year"].max())

    crop_yield_cols = [
        "RICE YIELD (Kg per ha)",
        "WHEAT YIELD (Kg per ha)",
        "MAIZE YIELD (Kg per ha)",
    ]

    print("\nTarget crop coverage:")

    for col in crop_yield_cols:
        series = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        valid = series.notna() & (series >= 0)

        print(
            f"{col}:"
            f"\n  non-null = {series.notna().sum():,}"
            f"\n  valid >= 0 = {valid.sum():,}"
            f"\n  negative = {(series < 0).sum():,}"
            f"\n  min = {series[valid].min()}"
            f"\n  max = {series[valid].max()}"
            f"\n  mean = {series[valid].mean():.2f}"
        )

    print("\nPotential sentinel -1 counts:")

    for col in crop_yield_cols:
        print(
            f"{col}:",
            (pd.to_numeric(df[col], errors="coerce") == -1).sum()
        )

    return df


def main():
    audit_agridata()
    audit_icrisat()

    print("\n" + "=" * 70)
    print("V4 CROP SOURCE AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
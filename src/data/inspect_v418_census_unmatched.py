"""
AgriAdapt V4.18 — Census Coordinate Matching Diagnostic

Purpose:
Inspect V4.17 matched geographies that could not be matched
literally to the Census 2001 polygon names.

This script DOES NOT create mappings or modify data.
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd


ROOT = Path(__file__).resolve().parents[2]

INPUT_GEOGRAPHY = (
    ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "final_district_geography_v417.csv"
)

CENSUS_SHP = (
    ROOT
    / "data"
    / "external"
    / "geography"
    / "india_district_boundaries"
    / "census-2001"
    / "2001_Dist.shp"
)


def normalize(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()
    text = text.replace("&", " AND ")
    text = " ".join(text.split())

    return text


def main():

    print(
        "AgriAdapt V4.18 — Census Matching Diagnostic"
    )
    print()

    geography = pd.read_csv(
        INPUT_GEOGRAPHY
    )

    census = gpd.read_file(
        CENSUS_SHP
    )

    # --------------------------------------------------------
    # Build Census lookup
    # --------------------------------------------------------

    census["_state_key"] = (
        census["ST_NM"]
        .map(normalize)
    )

    census["_district_key"] = (
        census["DISTRICT"]
        .map(normalize)
    )

    census["_pair_key"] = list(
        zip(
            census["_state_key"],
            census["_district_key"],
        )
    )

    census_pairs = set(
        census["_pair_key"]
    )

    # --------------------------------------------------------
    # Find V4.17 matched geographies without literal match
    # --------------------------------------------------------

    geography["_pair_key"] = list(
        zip(
            geography["matched_state"].map(normalize),
            geography["matched_district"].map(normalize),
        )
    )

    unmatched = geography[
        ~geography["_pair_key"].isin(census_pairs)
    ].copy()

    print(
        f"V4.17 rows: {len(geography):,}"
    )

    print(
        f"Rows without literal Census match: "
        f"{len(unmatched):,}"
    )

    print()

    # --------------------------------------------------------
    # Unique unmatched matched-geography pairs
    # --------------------------------------------------------

    unique_unmatched = (
        unmatched[
            [
                "matched_state",
                "matched_district",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "matched_state",
                "matched_district",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Unique unmatched matched-geography pairs: "
        f"{len(unique_unmatched):,}"
    )

    print()

    # --------------------------------------------------------
    # Display them
    # --------------------------------------------------------

    print(
        unique_unmatched.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save diagnostic
    # --------------------------------------------------------

    output = (
        ROOT
        / "data"
        / "processed"
        / "v4"
        / "district"
        / "v418_unmatched_coordinate_pairs.csv"
    )

    unique_unmatched.to_csv(
        output,
        index=False,
    )

    print()
    print(
        f"Diagnostic output: {output}"
    )


if __name__ == "__main__":
    main()
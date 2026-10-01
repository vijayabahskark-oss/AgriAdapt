"""
AgriAdapt V4.18 — District Coordinates

Purpose
-------
Attach reproducible representative coordinates to the 633
districts resolved in V4.17.

Input
-----
data/processed/v4/district/final_district_geography_v417.csv

Geographic source
-----------------
Census 2001 district boundary shapefile.

Expected files:
    data/external/geography/india_district_boundaries/
        census-2001/
            2001_Dist.shp
            2001_Dist.shx
            2001_Dist.dbf
            2001_Dist.prj

Method
------
1. Load V4.17 resolved geography.
2. Load Census 2001 district polygons.
3. Match matched_state + matched_district.
4. Generate a representative point inside each polygon.
5. Transform point to WGS84 / EPSG:4326.
6. Store latitude and longitude.
7. Validate exactly 633 coordinates.

Important
---------
This script DOES NOT create new district mappings.
It only uses the already-resolved V4.17 geography master.
"""

from pathlib import Path

import pandas as pd
import geopandas as gpd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_GEOGRAPHY = (
    ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "final_district_geography_v417.csv"
)

CENSUS_DIR = (
    ROOT
    / "data"
    / "external"
    / "geography"
    / "india_district_boundaries"
    / "census-2001"
)

CENSUS_SHP = CENSUS_DIR / "2001_Dist.shp"

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "v4"
    / "district"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "district_coordinates_v418.csv"
)

REPORT_FILE = (
    REPORT_DIR
    / "district_coordinates_v418.txt"
)


EXPECTED_ROWS = 633


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize text for matching only.

    Does not modify values written to the final output.
    """

    if pd.isna(value):
        return ""

    text = str(value).strip().upper()

    # Normalize ampersand spacing.
    text = text.replace("&", " AND ")

    # Collapse whitespace.
    text = " ".join(text.split())

    return text


# ============================================================
# MAIN
# ============================================================

def main():

    print("AgriAdapt V4.18 — District Coordinates")
    print()

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    if not INPUT_GEOGRAPHY.exists():
        raise FileNotFoundError(
            "V4.17 geography master not found:\n"
            f"{INPUT_GEOGRAPHY}"
        )

    if not CENSUS_SHP.exists():
        raise FileNotFoundError(
            "Census 2001 shapefile not found:\n"
            f"{CENSUS_SHP}"
        )

    # --------------------------------------------------------
    # Load V4.17 geography master
    # --------------------------------------------------------

    geography_df = pd.read_csv(
        INPUT_GEOGRAPHY
    )

    print(
        f"V4.17 geography rows: "
        f"{len(geography_df):,}"
    )

    print(
        f"V4.17 columns: "
        f"{list(geography_df.columns)}"
    )

    print()

    # --------------------------------------------------------
    # Validate V4.17 schema
    # --------------------------------------------------------

    required_columns = [
        "state_raw",
        "district_raw",
        "matched_state",
        "matched_district",
        "mapping_method",
        "confidence",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in geography_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "V4.17 geography master is missing columns:\n"
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Validate expected row count
    # --------------------------------------------------------

    if len(geography_df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} V4.17 rows, "
            f"found {len(geography_df)}."
        )

    print(
        "V4.17 row-count validation: PASS"
    )

    # --------------------------------------------------------
    # Validate unique source pairs
    # --------------------------------------------------------

    geography_df["_pair_key"] = list(
        zip(
            geography_df["state_raw"]
            .map(normalize_text),

            geography_df["district_raw"]
            .map(normalize_text),
        )
    )

    duplicate_pairs = geography_df[
        geography_df["_pair_key"].duplicated(
            keep=False
        )
    ]

    if not duplicate_pairs.empty:
        raise ValueError(
            "Duplicate state-district pairs found "
            "in V4.17 geography master."
        )

    print(
        "V4.17 duplicate-pair validation: PASS"
    )

    # --------------------------------------------------------
    # Load Census 2001 shapefile
    # --------------------------------------------------------

    print()
    print("Loading Census 2001 district polygons...")

    census_gdf = gpd.read_file(
        CENSUS_SHP
    )

    print(
        f"Census polygon records: "
        f"{len(census_gdf):,}"
    )

    print(
        f"Census columns: "
        f"{list(census_gdf.columns)}"
    )

    print(
        f"Census CRS: "
        f"{census_gdf.crs}"
    )

    print()

    # --------------------------------------------------------
    # Validate required Census fields
    # --------------------------------------------------------

    census_required = [
        "ST_NM",
        "DISTRICT",
        "geometry",
    ]

    missing_census_columns = [
        column
        for column in census_required
        if column not in census_gdf.columns
    ]

    if missing_census_columns:
        raise ValueError(
            "Census shapefile is missing required fields:\n"
            f"{missing_census_columns}"
        )

    print(
        "Census schema validation: PASS"
    )

    # --------------------------------------------------------
    # Validate CRS
    # --------------------------------------------------------

    if census_gdf.crs is None:
        raise ValueError(
            "Census shapefile has no CRS."
        )

    print(
        "Census CRS validation: PASS"
    )

    # --------------------------------------------------------
    # Validate geometries
    # --------------------------------------------------------

    invalid_geometry_count = int(
        (~census_gdf.geometry.is_valid).sum()
    )

    print(
        f"Invalid Census geometries: "
        f"{invalid_geometry_count}"
    )

    if invalid_geometry_count > 0:
        raise ValueError(
            "Census dataset contains invalid geometries."
        )

    print(
        "Census geometry validation: PASS"
    )

    # --------------------------------------------------------
    # Build normalized Census matching keys
    # --------------------------------------------------------

    census_gdf["_state_key"] = (
        census_gdf["ST_NM"]
        .map(normalize_text)
    )

    census_gdf["_district_key"] = (
        census_gdf["DISTRICT"]
        .map(normalize_text)
    )

    census_gdf["_pair_key"] = list(
        zip(
            census_gdf["_state_key"],
            census_gdf["_district_key"],
        )
    )

    # --------------------------------------------------------
    # Validate Census key uniqueness
    # --------------------------------------------------------

    census_duplicate_pairs = census_gdf[
        census_gdf["_pair_key"].duplicated(
            keep=False
        )
    ]

    if not census_duplicate_pairs.empty:

        print(
            "Duplicate Census geography keys detected:"
        )

        print(
            census_duplicate_pairs[
                [
                    "ST_NM",
                    "DISTRICT",
                ]
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

        raise ValueError(
            "Census state-district matching keys "
            "are not unique."
        )

    print(
        "Census geography-key uniqueness: PASS"
    )

    # --------------------------------------------------------
    # Build Census lookup
    # --------------------------------------------------------

    census_lookup = {}

    for _, row in census_gdf.iterrows():

        key = row["_pair_key"]

        census_lookup[key] = row

    # --------------------------------------------------------
    # Match all 633 V4.17 districts
    # --------------------------------------------------------

    matched_records = []
    unmatched_records = []

    for _, row in geography_df.iterrows():

        state_key = normalize_text(
            row["matched_state"]
        )

        district_key = normalize_text(
            row["matched_district"]
        )

        key = (
            state_key,
            district_key,
        )

        if key not in census_lookup:

            unmatched_records.append(
                {
                    "state_raw": row["state_raw"],
                    "district_raw": row["district_raw"],
                    "matched_state": row["matched_state"],
                    "matched_district": row[
                        "matched_district"
                    ],
                }
            )

            continue

        census_row = census_lookup[key]

        matched_records.append(
            {
                "state_raw": row["state_raw"],
                "district_raw": row["district_raw"],
                "matched_state": row["matched_state"],
                "matched_district": row[
                    "matched_district"
                ],
                "mapping_method": row[
                    "mapping_method"
                ],
                "confidence": row[
                    "confidence"
                ],
                "geometry": census_row.geometry,
            }
        )

    # --------------------------------------------------------
    # Validate complete matching
    # --------------------------------------------------------

    print()
    print(
        f"V4.17 districts matched to Census polygons: "
        f"{len(matched_records):,}"
    )

    print(
        f"V4.17 districts without Census polygon match: "
        f"{len(unmatched_records):,}"
    )

    if unmatched_records:

        print()
        print(
            "UNMATCHED RECORDS:"
        )

        for record in unmatched_records:
            print(
                f"  {record['matched_state']} | "
                f"{record['matched_district']}"
            )

        raise ValueError(
            "Not all V4.17 districts matched to "
            "Census 2001 polygons."
        )

    print(
        "V4.17 → Census matching validation: PASS"
    )

    # --------------------------------------------------------
    # Create GeoDataFrame
    # --------------------------------------------------------

    matched_gdf = gpd.GeoDataFrame(
        matched_records,
        geometry="geometry",
        crs=census_gdf.crs,
    )

    # --------------------------------------------------------
    # Generate representative points
    # --------------------------------------------------------
    #
    # representative_point() is intentionally used instead
    # of centroid because it guarantees the resulting point
    # lies inside the polygon.
    #
    # We generate it in the source CRS first, then transform
    # the point to WGS84.
    # --------------------------------------------------------

    matched_gdf["representative_point"] = (
        matched_gdf.geometry.representative_point()
    )

    # --------------------------------------------------------
    # Transform representative points to WGS84
    # --------------------------------------------------------

    points_gdf = matched_gdf[
        [
            "state_raw",
            "district_raw",
            "matched_state",
            "matched_district",
            "mapping_method",
            "confidence",
            "representative_point",
        ]
    ].copy()

    points_gdf = gpd.GeoDataFrame(
        points_gdf,
        geometry="representative_point",
        crs=matched_gdf.crs,
    )

    points_wgs84 = points_gdf.to_crs(
        epsg=4326
    )

    # --------------------------------------------------------
    # Extract longitude / latitude
    # --------------------------------------------------------

    points_wgs84["longitude"] = (
        points_wgs84.geometry.x
    )

    points_wgs84["latitude"] = (
        points_wgs84.geometry.y
    )

    # --------------------------------------------------------
    # Build final table
    # --------------------------------------------------------

    final_df = pd.DataFrame(
        {
            "state_raw": points_wgs84[
                "state_raw"
            ],

            "district_raw": points_wgs84[
                "district_raw"
            ],

            "matched_state": points_wgs84[
                "matched_state"
            ],

            "matched_district": points_wgs84[
                "matched_district"
            ],

            "mapping_method": points_wgs84[
                "mapping_method"
            ],

            "confidence": points_wgs84[
                "confidence"
            ],

            "latitude": points_wgs84[
                "latitude"
            ],

            "longitude": points_wgs84[
                "longitude"
            ],

            "coordinate_source": (
                "Census 2001 district polygon"
            ),

            "coordinate_method": (
                "Representative point inside polygon"
            ),
        }
    )

    # --------------------------------------------------------
    # Sort deterministically
    # --------------------------------------------------------

    final_df = (
        final_df
        .sort_values(
            [
                "state_raw",
                "district_raw",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Coordinate completeness validation
    # --------------------------------------------------------

    missing_latitude = int(
        final_df["latitude"].isna().sum()
    )

    missing_longitude = int(
        final_df["longitude"].isna().sum()
    )

    if (
        missing_latitude > 0
        or missing_longitude > 0
    ):
        raise ValueError(
            "Missing coordinates detected.\n"
            f"Missing latitude: {missing_latitude}\n"
            f"Missing longitude: {missing_longitude}"
        )

    print(
        "Coordinate completeness validation: PASS"
    )

    # --------------------------------------------------------
    # Coordinate range validation
    # --------------------------------------------------------

    invalid_latitude = final_df[
        ~final_df["latitude"].between(
            -90,
            90,
        )
    ]

    invalid_longitude = final_df[
        ~final_df["longitude"].between(
            -180,
            180,
        )
    ]

    if (
        not invalid_latitude.empty
        or not invalid_longitude.empty
    ):
        raise ValueError(
            "Invalid latitude/longitude values detected."
        )

    print(
        "Coordinate-range validation: PASS"
    )

    # --------------------------------------------------------
    # India-region sanity validation
    # --------------------------------------------------------
    #
    # This is NOT used for matching.
    # It only catches gross coordinate errors.
    #
    # India bounding envelope:
    # approximately 6°–38° N
    # approximately 68°–98° E
    # --------------------------------------------------------

    outside_india_envelope = final_df[
        ~final_df["latitude"].between(
            5,
            38,
        )
        |
        ~final_df["longitude"].between(
            67,
            99,
        )
    ]

    if not outside_india_envelope.empty:

        print()
        print(
            "WARNING: coordinates outside broad "
            "India envelope:"
        )

        print(
            outside_india_envelope[
                [
                    "matched_state",
                    "matched_district",
                    "latitude",
                    "longitude",
                ]
            ]
            .to_string(index=False)
        )

        raise ValueError(
            "Coordinate sanity validation failed."
        )

    print(
        "India-envelope sanity validation: PASS"
    )

    # --------------------------------------------------------
    # Row-count validation
    # --------------------------------------------------------

    if len(final_df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} final rows, "
            f"found {len(final_df)}."
        )

    print(
        "Final row-count validation: PASS"
    )

    # --------------------------------------------------------
    # Duplicate validation
    # --------------------------------------------------------

    duplicate_output = final_df[
        final_df[
            [
                "state_raw",
                "district_raw",
            ]
        ].duplicated(
            keep=False
        )
    ]

    if not duplicate_output.empty:
        raise ValueError(
            "Duplicate state-district pairs found "
            "in final coordinate table."
        )

    print(
        "Final duplicate-pair validation: PASS"
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    latitude_min = final_df[
        "latitude"
    ].min()

    latitude_max = final_df[
        "latitude"
    ].max()

    longitude_min = final_df[
        "longitude"
    ].min()

    longitude_max = final_df[
        "longitude"
    ].max()

    # --------------------------------------------------------
    # Write report
    # --------------------------------------------------------

    report_lines = [
        "AgriAdapt V4.18 — District Coordinates",
        "=" * 60,
        "",
        "INPUT",
        str(INPUT_GEOGRAPHY),
        "",
        "GEOGRAPHIC SOURCE",
        str(CENSUS_SHP),
        "Census 2001 district polygon",
        "",
        "METHOD",
        "Representative point inside Census district polygon",
        "Output CRS: EPSG:4326 / WGS84",
        "",
        "COUNTS",
        f"V4.17 input districts: {len(geography_df)}",
        f"Census polygons: {len(census_gdf)}",
        f"Matched districts: {len(matched_records)}",
        f"Final coordinate rows: {len(final_df)}",
        "",
        "VALIDATION",
        "V4.17 row-count: PASS",
        "V4.17 duplicate-pair: PASS",
        "Census schema: PASS",
        "Census CRS: PASS",
        "Census geometry validity: PASS",
        "Census geography-key uniqueness: PASS",
        "V4.17 → Census matching: PASS",
        "Coordinate completeness: PASS",
        "Coordinate range: PASS",
        "India-envelope sanity: PASS",
        "Final row-count: PASS",
        "Final duplicate-pair: PASS",
        "",
        "COORDINATE RANGE",
        f"Latitude minimum: {latitude_min:.6f}",
        f"Latitude maximum: {latitude_max:.6f}",
        f"Longitude minimum: {longitude_min:.6f}",
        f"Longitude maximum: {longitude_max:.6f}",
        "",
        "OUTPUT",
        str(OUTPUT_FILE),
        "",
        "NOTE",
        "Coordinates are representative points derived from "
        "the matched Census 2001 district polygons.",
        "They are intended as representative district locations "
        "for subsequent climate-data extraction.",
        "They are not administrative headquarters coordinates "
        "and should not be interpreted as population-weighted "
        "or agricultural-area-weighted centroids.",
    ]

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("V4.18 COMPLETE")
    print()
    print(
        f"Final coordinate rows: "
        f"{len(final_df):,}"
    )

    print(
        f"Latitude range: "
        f"{latitude_min:.6f} "
        f"to "
        f"{latitude_max:.6f}"
    )

    print(
        f"Longitude range: "
        f"{longitude_min:.6f} "
        f"to "
        f"{longitude_max:.6f}"
    )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()
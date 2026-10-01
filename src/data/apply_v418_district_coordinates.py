"""
AgriAdapt V4.18 — Apply District Coordinates

Purpose
-------
Apply the validated V4.18 coordinate concordance to the
633-row V4.17 geography master and attach representative
coordinates from Census 2001 polygons.

Matching order
--------------
1. Direct match against Census 2001 state + district name.
2. If no direct match, use the validated V4.18 coordinate
   concordance.
3. Otherwise mark SOURCE_REVIEW_REQUIRED.

No new geography mappings are created.

Expected:
    633 total rows
    578 direct Census matches
    47 concordance-based matches
    625 total coordinate-resolved rows
    8 source-review rows
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

INPUT_CONCORDANCE = (
    ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "v418_coordinate_concordance.csv"
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


# ============================================================
# EXPECTED COUNTS
# ============================================================

EXPECTED_INPUT_ROWS = 633
EXPECTED_DIRECT_ROWS = 578
EXPECTED_CONCORDANCE_ROWS = 36
EXPECTED_CONCORDANCE_RESOLVED_ROWS = 47
EXPECTED_TOTAL_RESOLVED = 625
EXPECTED_REVIEW_ROWS = 8


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    text = str(value).strip().upper()

    text = text.replace("&", " AND ")

    text = " ".join(text.split())

    return text


def make_key(state, district):

    return (
        normalize_text(state),
        normalize_text(district),
    )


# ============================================================
# GET REPRESENTATIVE POINT
# ============================================================

def get_wgs84_point(geometry, source_crs):

    representative_point = (
        geometry.representative_point()
    )

    point_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                representative_point
            ]
        },
        geometry="geometry",
        crs=source_crs,
    )

    point_wgs84 = point_gdf.to_crs(
        epsg=4326
    )

    point = point_wgs84.geometry.iloc[0]

    return point.y, point.x


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "AgriAdapt V4.18 — Apply District Coordinates"
    )
    print()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path, label in [
        (
            INPUT_GEOGRAPHY,
            "V4.17 geography",
        ),
        (
            INPUT_CONCORDANCE,
            "V4.18 concordance",
        ),
        (
            CENSUS_SHP,
            "Census 2001 shapefile",
        ),
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"{label} not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    geography = pd.read_csv(
        INPUT_GEOGRAPHY
    )

    concordance = pd.read_csv(
        INPUT_CONCORDANCE
    )

    census = gpd.read_file(
        CENSUS_SHP
    )

    print(
        f"V4.17 geography rows: "
        f"{len(geography):,}"
    )

    print(
        f"Coordinate concordance rows: "
        f"{len(concordance):,}"
    )

    print(
        f"Census polygons: "
        f"{len(census):,}"
    )

    print()

    # --------------------------------------------------------
    # Validate counts
    # --------------------------------------------------------

    if len(geography) != EXPECTED_INPUT_ROWS:

        raise ValueError(
            f"Expected {EXPECTED_INPUT_ROWS} V4.17 rows, "
            f"found {len(geography)}."
        )

    if len(concordance) != EXPECTED_CONCORDANCE_ROWS:

        raise ValueError(
            f"Expected {EXPECTED_CONCORDANCE_ROWS} "
            f"concordance rows, "
            f"found {len(concordance)}."
        )

    print(
        "Input-count validation: PASS"
    )

    # --------------------------------------------------------
    # Validate schemas
    # --------------------------------------------------------

    geography_required = [
        "state_raw",
        "district_raw",
        "matched_state",
        "matched_district",
        "mapping_method",
        "confidence",
    ]

    concordance_required = [
        "matched_state",
        "matched_district",
        "census_state",
        "census_district",
        "coordinate_mapping_method",
        "coordinate_confidence",
    ]

    for column in geography_required:

        if column not in geography.columns:

            raise ValueError(
                f"V4.17 missing column: {column}"
            )

    for column in concordance_required:

        if column not in concordance.columns:

            raise ValueError(
                f"V4.18 concordance missing column: "
                f"{column}"
            )

    print(
        "Schema validation: PASS"
    )

    # --------------------------------------------------------
    # Build Census lookup
    # --------------------------------------------------------

    census["_state_key"] = (
        census["ST_NM"]
        .map(normalize_text)
    )

    census["_district_key"] = (
        census["DISTRICT"]
        .map(normalize_text)
    )

    census["_census_key"] = list(
        zip(
            census["_state_key"],
            census["_district_key"],
        )
    )

    census_lookup = {}

    for _, row in census.iterrows():

        key = row["_census_key"]

        if key in census_lookup:

            raise ValueError(
                "Duplicate Census geography key:\n"
                f"{key}"
            )

        census_lookup[key] = row

    print(
        "Census geography-key validation: PASS"
    )

    # --------------------------------------------------------
    # Build coordinate concordance lookup
    # --------------------------------------------------------

    concordance["_source_key"] = [
        make_key(
            row["matched_state"],
            row["matched_district"],
        )
        for _, row in concordance.iterrows()
    ]

    if concordance[
        "_source_key"
    ].duplicated().any():

        raise ValueError(
            "Duplicate coordinate concordance "
            "source keys."
        )

    concordance_lookup = {}

    for _, row in concordance.iterrows():

        concordance_lookup[
            row["_source_key"]
        ] = row

    print(
        "Concordance uniqueness validation: PASS"
    )

    # --------------------------------------------------------
    # Process V4.17 rows
    # --------------------------------------------------------

    records = []

    direct_count = 0
    concordance_count = 0
    review_count = 0

    # Keep track of which concordance entries were actually used.
    used_concordance_keys = set()

    for _, row in geography.iterrows():

        matched_state = row[
            "matched_state"
        ]

        matched_district = row[
            "matched_district"
        ]

        source_key = make_key(
            matched_state,
            matched_district,
        )

        base_record = {
            "state_raw":
                row["state_raw"],

            "district_raw":
                row["district_raw"],

            "matched_state":
                matched_state,

            "matched_district":
                matched_district,

            "mapping_method":
                row["mapping_method"],

            "confidence":
                row["confidence"],
        }

        # ====================================================
        # STEP 1 — DIRECT CENSUS MATCH
        # ====================================================

        if source_key in census_lookup:

            census_row = census_lookup[
                source_key
            ]

            latitude, longitude = (
                get_wgs84_point(
                    census_row.geometry,
                    census.crs,
                )
            )

            base_record.update(
                {
                    "latitude":
                        latitude,

                    "longitude":
                        longitude,

                    "coordinate_source":
                        "Census 2001 district polygon",

                    "coordinate_method":
                        "Representative point inside polygon",

                    "coordinate_mapping_method":
                        "DIRECT_CENSUS_MATCH",

                    "coordinate_confidence":
                        "HIGH",

                    "coordinate_status":
                        "RESOLVED",
                }
            )

            direct_count += 1

        # ====================================================
        # STEP 2 — APPROVED COORDINATE CONCORDANCE
        # ====================================================

        elif source_key in concordance_lookup:

            concordance_row = concordance_lookup[
                source_key
            ]

            census_key = make_key(
                concordance_row[
                    "census_state"
                ],
                concordance_row[
                    "census_district"
                ],
            )

            if census_key not in census_lookup:

                raise ValueError(
                    "Coordinate concordance points to "
                    "missing Census polygon:\n"
                    f"{concordance_row['census_state']} | "
                    f"{concordance_row['census_district']}"
                )

            census_row = census_lookup[
                census_key
            ]

            latitude, longitude = (
                get_wgs84_point(
                    census_row.geometry,
                    census.crs,
                )
            )

            base_record.update(
                {
                    "latitude":
                        latitude,

                    "longitude":
                        longitude,

                    "coordinate_source":
                        "Census 2001 district polygon",

                    "coordinate_method":
                        "Representative point inside polygon",

                    "coordinate_mapping_method":
                        concordance_row[
                            "coordinate_mapping_method"
                        ],

                    "coordinate_confidence":
                        concordance_row[
                            "coordinate_confidence"
                        ],

                    "coordinate_status":
                        "RESOLVED",
                }
            )

            concordance_count += 1

            used_concordance_keys.add(
                source_key
            )

        # ====================================================
        # STEP 3 — SOURCE REVIEW
        # ====================================================

        else:

            base_record.update(
                {
                    "latitude":
                        pd.NA,

                    "longitude":
                        pd.NA,

                    "coordinate_source":
                        "Census 2001 district polygon",

                    "coordinate_method":
                        pd.NA,

                    "coordinate_mapping_method":
                        pd.NA,

                    "coordinate_confidence":
                        pd.NA,

                    "coordinate_status":
                        "SOURCE_REVIEW_REQUIRED",
                }
            )

            review_count += 1

        records.append(
            base_record
        )

    # --------------------------------------------------------
    # Create final DataFrame
    # --------------------------------------------------------

    final_df = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    print()
    print(
        f"Direct Census matches: "
        f"{direct_count:,}"
    )

    print(
        f"Concordance-based matches: "
        f"{concordance_count:,}"
    )

    print(
        f"Source-review rows: "
        f"{review_count:,}"
    )

    print()

    # --------------------------------------------------------
    # Validate direct count
    # --------------------------------------------------------

    if direct_count != EXPECTED_DIRECT_ROWS:

        raise ValueError(
            f"Expected {EXPECTED_DIRECT_ROWS} "
            f"direct Census matches, "
            f"found {direct_count}."
        )

    print(
        "Direct-match count validation: PASS"
    )

    # --------------------------------------------------------
    # Validate concordance usage
    # --------------------------------------------------------

    if concordance_count != (
        EXPECTED_CONCORDANCE_RESOLVED_ROWS
    ):

        raise ValueError(
            f"Expected {EXPECTED_CONCORDANCE_RESOLVED_ROWS} "
            f"rows resolved through the 36-row concordance, "
            f"found {concordance_count}."
        )

    print(
        "Concordance-resolved count validation: PASS"
    )

    # --------------------------------------------------------
    # Validate total
    # --------------------------------------------------------

    total_resolved = (
        direct_count
        + concordance_count
    )

    if total_resolved != EXPECTED_TOTAL_RESOLVED:

        raise ValueError(
            f"Expected {EXPECTED_TOTAL_RESOLVED} "
            f"total resolved rows, "
            f"found {total_resolved}."
        )

    print(
        "Total-resolved count validation: PASS"
    )

    # --------------------------------------------------------
    # Validate review count
    # --------------------------------------------------------

    if review_count != EXPECTED_REVIEW_ROWS:

        raise ValueError(
            f"Expected {EXPECTED_REVIEW_ROWS} "
            f"source-review rows, "
            f"found {review_count}."
        )

    print(
        "Source-review count validation: PASS"
    )

    # --------------------------------------------------------
    # Validate all input rows retained
    # --------------------------------------------------------

    if len(final_df) != EXPECTED_INPUT_ROWS:

        raise ValueError(
            "Final row count does not equal "
            "V4.17 input row count."
        )

    print(
        "Row-retention validation: PASS"
    )

    # --------------------------------------------------------
    # Validate resolved coordinates
    # --------------------------------------------------------

    resolved = final_df[
        final_df[
            "coordinate_status"
        ]
        == "RESOLVED"
    ]

    if resolved[
        "latitude"
    ].isna().any():

        raise ValueError(
            "Resolved rows contain missing latitude."
        )

    if resolved[
        "longitude"
    ].isna().any():

        raise ValueError(
            "Resolved rows contain missing longitude."
        )

    if not resolved[
        "latitude"
    ].between(
        -90,
        90,
    ).all():

        raise ValueError(
            "Invalid latitude detected."
        )

    if not resolved[
        "longitude"
    ].between(
        -180,
        180,
    ).all():

        raise ValueError(
            "Invalid longitude detected."
        )

    print(
        "Resolved-coordinate validation: PASS"
    )

    # --------------------------------------------------------
    # India envelope
    # --------------------------------------------------------

    outside = resolved[
        ~resolved[
            "latitude"
        ].between(
            5,
            38,
        )
        |
        ~resolved[
            "longitude"
        ].between(
            67,
            99,
        )
    ]

    if not outside.empty:

        print(
            outside[
                [
                    "matched_state",
                    "matched_district",
                    "latitude",
                    "longitude",
                ]
            ].to_string(
                index=False
            )
        )

        raise ValueError(
            "India-envelope validation failed."
        )

    print(
        "India-envelope validation: PASS"
    )

    # --------------------------------------------------------
    # Validate source geography uniqueness
    # --------------------------------------------------------

    duplicate_pairs = final_df[
        final_df[
            [
                "state_raw",
                "district_raw",
            ]
        ].duplicated(
            keep=False
        )
    ]

    if not duplicate_pairs.empty:

        raise ValueError(
            "Duplicate V4.17 source geography pairs."
        )

    print(
        "Source-pair uniqueness validation: PASS"
    )

    # --------------------------------------------------------
    # Sort
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
    # Save
    # --------------------------------------------------------

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Review records
    # --------------------------------------------------------

    review_df = final_df[
        final_df[
            "coordinate_status"
        ]
        == "SOURCE_REVIEW_REQUIRED"
    ]

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_lines = [
        "AgriAdapt V4.18 — District Coordinates",
        "=" * 65,
        "",
        "INPUT",
        str(INPUT_GEOGRAPHY),
        "",
        "COORDINATE CONCORDANCE",
        str(INPUT_CONCORDANCE),
        "",
        "GEOGRAPHIC SOURCE",
        str(CENSUS_SHP),
        "",
        "COUNTS",
        f"V4.17 input rows: {len(geography)}",
        f"Direct Census matches: {direct_count}",
        f"Concordance-based matches: {concordance_count}",
        f"Total resolved: {total_resolved}",
        f"Source-review rows: {review_count}",
        "",
        "VALIDATION",
        "Input-count: PASS",
        "Schema: PASS",
        "Census geography-key: PASS",
        "Concordance uniqueness: PASS",
        "Direct-match count: PASS",
        "Concordance-resolved count: PASS",
        "Total-resolved count: PASS",
        "Source-review count: PASS",
        "Row retention: PASS",
        "Resolved-coordinate validation: PASS",
        "India-envelope validation: PASS",
        "Source-pair uniqueness: PASS",
        "",
        "SOURCE-REVIEW RECORDS",
    ]

    for _, row in review_df.iterrows():

        report_lines.append(
            f"{row['matched_state']} | "
            f"{row['matched_district']}"
        )

    report_lines.extend(
        [
            "",
            "OUTPUT",
            str(OUTPUT_FILE),
            "",
            "NOTE",
            "Direct Census matches use the already-resolved "
            "V4.17 matched geography directly.",
            "The validated V4.18 coordinate concordance is "
            "used only when a direct Census name match is "
            "not available.",
            "The remaining source-review rows are retained "
            "without guessed coordinates.",
        ]
    )

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print(
        "V4.18 COMPLETE"
    )
    print()

    print(
        f"Final rows: {len(final_df):,}"
    )

    print(
        f"Direct Census matches: "
        f"{direct_count:,}"
    )

    print(
        f"Concordance-based matches: "
        f"{concordance_count:,}"
    )

    print(
        f"Total resolved coordinates: "
        f"{total_resolved:,}"
    )

    print(
        f"Source review required: "
        f"{review_count:,}"
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
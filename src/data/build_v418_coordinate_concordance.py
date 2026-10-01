"""
AgriAdapt V4.18 — Coordinate Concordance

Purpose
-------
Create a controlled concordance between the V4.17 matched geography
names and the actual Census 2001 district polygon names.

IMPORTANT
---------
This does NOT modify V4.17 geography mappings.

It only establishes how an already-resolved V4.17 geography name
corresponds to the Census 2001 polygon used for obtaining coordinates.

No fuzzy similarity score is used to create mappings.

Outputs
-------
data/processed/v4/district/v418_coordinate_concordance.csv
reports/v4/district/v418_coordinate_concordance.txt
"""

from pathlib import Path

import pandas as pd
import geopandas as gpd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_CANDIDATES = (
    ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "v418_coordinate_candidate_audit.csv"
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
    / "v418_coordinate_concordance.csv"
)

REPORT_FILE = (
    REPORT_DIR
    / "v418_coordinate_concordance.txt"
)


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


# ============================================================
# APPROVED COORDINATE CONCORDANCE
# ============================================================
#
# These are NOT new geography mappings.
#
# They only resolve naming differences between V4.17's
# already-established matched geography and the Census 2001
# polygon naming.
#
# The 8 questionable cases are intentionally NOT included.
# ============================================================

APPROVED = {

    # --------------------------------------------------------
    # Andaman & Nicobar
    # --------------------------------------------------------

    (
        "A & N ISLANDS",
        "Andamans",
    ): (
        "Andaman & Nicobar Island",
        "Andamans",
        "NAME_EQUIVALENCE",
        "HIGH",
    ),

    (
        "A & N ISLANDS",
        "Nicobars",
    ): (
        "Andaman & Nicobar Island",
        "Nicobars",
        "NAME_EQUIVALENCE",
        "HIGH",
    ),

    # --------------------------------------------------------
    # Andhra Pradesh
    # --------------------------------------------------------

    (
        "ANDHRA PRADESH",
        "Rangareddy",
    ): (
        "Andhra Pradesh",
        "Rangareddi",
        "NAME_VARIANT",
        "HIGH",
    ),

    # --------------------------------------------------------
    # Arunachal Pradesh
    # --------------------------------------------------------

    (
        "ARUNACHAL PRADESH",
        "Changlang",
    ): (
        "Arunanchal Pradesh",
        "Changlang",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "ARUNACHAL PRADESH",
        "East Kameng",
    ): (
        "Arunanchal Pradesh",
        "East Kameng",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "ARUNACHAL PRADESH",
        "East Siang",
    ): (
        "Arunanchal Pradesh",
        "East Siang",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "ARUNACHAL PRADESH",
        "Lohit",
    ): (
        "Arunanchal Pradesh",
        "Lohit",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "ARUNACHAL PRADESH",
        "Lower Subansiri",
    ): (
        "Arunanchal Pradesh",
        "Lower Subansiri",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "ARUNACHAL PRADESH",
        "Papum Pare",
    ): (
        "Arunanchal Pradesh",
        "Papum Pare",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "ARUNACHAL PRADESH",
        "Tawang",
    ): (
        "Arunanchal Pradesh",
        "Tawang",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "ARUNACHAL PRADESH",
        "Tirap",
    ): (
        "Arunanchal Pradesh",
        "Tirap",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "ARUNACHAL PRADESH",
        "Upper Siang",
    ): (
        "Arunanchal Pradesh",
        "Upper Siang",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "ARUNACHAL PRADESH",
        "Upper Subansiri",
    ): (
        "Arunanchal Pradesh",
        "Upper Subansiri",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "ARUNACHAL PRADESH",
        "West Kameng",
    ): (
        "Arunanchal Pradesh",
        "West Kameng",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "ARUNACHAL PRADESH",
        "West Siang",
    ): (
        "Arunanchal Pradesh",
        "West Siang",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    # --------------------------------------------------------
    # Chhattisgarh
    # --------------------------------------------------------

    (
        "CHHATTISGARH",
        "Janjgir-Champa",
    ): (
        "Chhattisgarh",
        "Janjgir - Champa",
        "NAME_FORMAT_VARIANT",
        "HIGH",
    ),

    # --------------------------------------------------------
    # Dadra & Nagar Haveli
    # --------------------------------------------------------

    (
        "Dadra and Nagar Haveli",
        "Dadra & Nagar Haveli",
    ): (
        "Dadara & Nagar Havelli",
        "Dadra & Nagar Haveli",
        "NAME_VARIANT",
        "HIGH",
    ),

    # --------------------------------------------------------
    # Jharkhand
    # --------------------------------------------------------

    (
        "JHARKHAND",
        "Hazaribag",
    ): (
        "Jharkhand",
        "Hazaribagh",
        "NAME_VARIANT",
        "HIGH",
    ),

    # --------------------------------------------------------
    # Puducherry / Pondicherry
    # --------------------------------------------------------

    (
        "PONDICHERRY",
        "Karaikal",
    ): (
        "Puducherry",
        "Karaikal",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "PONDICHERRY",
        "Mahe",
    ): (
        "Puducherry",
        "Mahe",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "PONDICHERRY",
        "Pondicherry",
    ): (
        "Puducherry",
        "Pondicherry",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "PONDICHERRY",
        "Yanam",
    ): (
        "Puducherry",
        "Yanam",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    # --------------------------------------------------------
    # Uttar Pradesh
    # --------------------------------------------------------

    (
        "UTTAR PRADESH",
        "Maharajganj",
    ): (
        "Uttar Pradesh",
        "Mahrajganj",
        "NAME_VARIANT",
        "HIGH",
    ),

    # --------------------------------------------------------
    # Uttarakhand / Uttaranchal
    # --------------------------------------------------------

    (
        "UTTARANCHAL",
        "Almora",
    ): (
        "Uttarakhand",
        "Almora",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "UTTARANCHAL",
        "Bageshwar",
    ): (
        "Uttarakhand",
        "Bageshwar",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "UTTARANCHAL",
        "Chamoli",
    ): (
        "Uttarakhand",
        "Chamoli",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "UTTARANCHAL",
        "Champawat",
    ): (
        "Uttarakhand",
        "Champawat",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "UTTARANCHAL",
        "Dehradun",
    ): (
        "Uttarakhand",
        "Dehradun",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "UTTARANCHAL",
        "Garhwal",
    ): (
        "Uttarakhand",
        "Garhwal",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "UTTARANCHAL",
        "Hardwar",
    ): (
        "Uttarakhand",
        "Hardwar",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "UTTARANCHAL",
        "Nainital",
    ): (
        "Uttarakhand",
        "Nainital",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "UTTARANCHAL",
        "Pithoragarh",
    ): (
        "Uttarakhand",
        "Pithoragarh",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "UTTARANCHAL",
        "Rudraprayag",
    ): (
        "Uttarakhand",
        "Rudraprayag",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "UTTARANCHAL",
        "Tehri Garhwal",
    ): (
        "Uttarakhand",
        "Tehri Garhwal",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "UTTARANCHAL",
        "Udham Singh Nagar",
    ): (
        "Uttarakhand",
        "Udham Singh Nagar",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),

    (
        "UTTARANCHAL",
        "Uttarkashi",
    ): (
        "Uttarakhand",
        "Uttarkashi",
        "STATE_NAME_VARIANT",
        "HIGH",
    ),
}


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "AgriAdapt V4.18 — Coordinate Concordance"
    )
    print()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not INPUT_CANDIDATES.exists():
        raise FileNotFoundError(
            f"Candidate audit not found:\n"
            f"{INPUT_CANDIDATES}"
        )

    if not CENSUS_SHP.exists():
        raise FileNotFoundError(
            f"Census shapefile not found:\n"
            f"{CENSUS_SHP}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    candidates = pd.read_csv(
        INPUT_CANDIDATES
    )

    census = gpd.read_file(
        CENSUS_SHP
    )

    print(
        f"Candidate audit rows: "
        f"{len(candidates):,}"
    )

    print(
        f"Census polygons: "
        f"{len(census):,}"
    )

    print()

    # --------------------------------------------------------
    # Build Census key set
    # --------------------------------------------------------

    census_keys = set()

    for _, row in census.iterrows():

        key = (
            normalize_text(row["ST_NM"]),
            normalize_text(row["DISTRICT"]),
        )

        census_keys.add(key)

    # --------------------------------------------------------
    # Validate approved mappings
    # --------------------------------------------------------

    records = []

    for (
        source_key,
        target_info,
    ) in APPROVED.items():

        (
            census_state,
            census_district,
            method,
            confidence,
        ) = target_info

        source_state, source_district = source_key

        census_key = (
            normalize_text(census_state),
            normalize_text(census_district),
        )

        if census_key not in census_keys:
            raise ValueError(
                "Approved Census geography does not exist "
                "in the Census shapefile:\n"
                f"  {census_state} | {census_district}"
            )

        records.append(
            {
                "matched_state": source_state,
                "matched_district": source_district,
                "census_state": census_state,
                "census_district": census_district,
                "coordinate_mapping_method": method,
                "coordinate_confidence": confidence,
            }
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    concordance = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Validate uniqueness
    # --------------------------------------------------------

    source_duplicates = concordance[
        concordance[
            [
                "matched_state",
                "matched_district",
            ]
        ].duplicated(
            keep=False
        )
    ]

    if not source_duplicates.empty:
        raise ValueError(
            "Duplicate V4.17 geography keys in "
            "coordinate concordance."
        )

    print(
        "Source-key uniqueness validation: PASS"
    )

    # --------------------------------------------------------
    # Validate Census target uniqueness
    # --------------------------------------------------------

    target_duplicates = concordance[
        concordance[
            [
                "census_state",
                "census_district",
            ]
        ].duplicated(
            keep=False
        )
    ]

    if not target_duplicates.empty:
        raise ValueError(
            "Duplicate Census target geometries "
            "in coordinate concordance."
        )

    print(
        "Census-target uniqueness validation: PASS"
    )

    # --------------------------------------------------------
    # Validate candidate audit coverage
    # --------------------------------------------------------

    candidate_pairs = set(
        zip(
            candidates[
                "matched_state"
            ].map(normalize_text),

            candidates[
                "matched_district"
            ].map(normalize_text),
        )
    )

    concordance_pairs = set(
        zip(
            concordance[
                "matched_state"
            ].map(normalize_text),

            concordance[
                "matched_district"
            ].map(normalize_text),
        )
    )

    not_in_candidates = (
        concordance_pairs - candidate_pairs
    )

    if not_in_candidates:
        raise ValueError(
            "Coordinate concordance contains source "
            "pairs absent from candidate audit:\n"
            f"{sorted(not_in_candidates)}"
        )

    print(
        "Candidate-audit coverage validation: PASS"
    )

    # --------------------------------------------------------
    # Expected approved count
    # --------------------------------------------------------

    expected_approved = 36

    if len(concordance) != expected_approved:
        raise ValueError(
            f"Expected {expected_approved} approved "
            f"coordinate mappings, found "
            f"{len(concordance)}."
        )

    print(
        f"Approved coordinate mappings: "
        f"{len(concordance)}"
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

    concordance = concordance.sort_values(
        [
            "matched_state",
            "matched_district",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    concordance.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_lines = [
        "AgriAdapt V4.18 — Coordinate Concordance",
        "=" * 65,
        "",
        "SOURCE",
        str(INPUT_CANDIDATES),
        "",
        "GEOGRAPHIC SOURCE",
        str(CENSUS_SHP),
        "",
        "APPROVED COORDINATE CONCORDANCES",
        f"{len(concordance)}",
        "",
        "VALIDATION",
        "Source-key uniqueness: PASS",
        "Census-target uniqueness: PASS",
        "Candidate-audit coverage: PASS",
        "Census target existence: PASS",
        f"Approved mapping count ({expected_approved}): PASS",
        "",
        "INTENTIONALLY UNRESOLVED",
        "BIHAR | Arwal",
        "DELHI | Delhi",
        "JHARKHAND | Jamtara",
        "JHARKHAND | Latehar",
        "JHARKHAND | Saraikela-Kharsawan",
        "JHARKHAND | Simdega",
        "MANIPUR | Senapati",
        "TAMIL NADU | Krishnagiri",
        "",
        "IMPORTANT",
        "No fuzzy similarity score was used as sufficient "
        "evidence for a coordinate mapping.",
        "V4.17 geography mappings are unchanged.",
        "This concordance only resolves naming differences "
        "against the Census 2001 polygon source.",
        "",
        "OUTPUT",
        str(OUTPUT_FILE),
    ]

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print(
        "V4.18 COORDINATE CONCORDANCE COMPLETE"
    )

    print()
    print(
        f"Approved mappings: "
        f"{len(concordance)}"
    )

    print(
        "Intentionally unresolved: 8"
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
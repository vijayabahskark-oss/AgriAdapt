"""
AgriAdapt V4.18.1 — Coordinate Candidate Audit

Purpose
-------
For the V4.17 geography pairs that do not literally match
Census 2001 polygon names, generate candidate Census district
names from the SAME state using string similarity.

IMPORTANT
---------
This script DOES NOT create mappings.

It only generates candidates for human/research validation.

Outputs
-------
data/processed/v4/district/v418_coordinate_candidate_audit.csv
reports/v4/district/v418_coordinate_candidate_audit.txt
"""

from pathlib import Path
import re
import pandas as pd
import geopandas as gpd
from difflib import SequenceMatcher


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_UNMATCHED = (
    ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "v418_unmatched_coordinate_pairs.csv"
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
    / "v418_coordinate_candidate_audit.csv"
)

REPORT_FILE = (
    REPORT_DIR
    / "v418_coordinate_candidate_audit.txt"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()

    text = text.replace("&", " AND ")

    # Remove punctuation.
    text = re.sub(
        r"[^A-Z0-9\s]",
        " ",
        text,
    )

    text = " ".join(text.split())

    return text


def similarity(a, b):

    a = normalize_text(a)
    b = normalize_text(b)

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "AgriAdapt V4.18.1 — Coordinate Candidate Audit"
    )
    print()

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not INPUT_UNMATCHED.exists():
        raise FileNotFoundError(
            f"Unmatched-pair file not found:\n"
            f"{INPUT_UNMATCHED}"
        )

    if not CENSUS_SHP.exists():
        raise FileNotFoundError(
            f"Census shapefile not found:\n"
            f"{CENSUS_SHP}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    unmatched = pd.read_csv(
        INPUT_UNMATCHED
    )

    census = gpd.read_file(
        CENSUS_SHP
    )

    print(
        f"Unmatched unique pairs: "
        f"{len(unmatched):,}"
    )

    print(
        f"Census polygons: "
        f"{len(census):,}"
    )

    print()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    required_unmatched = [
        "matched_state",
        "matched_district",
    ]

    missing = [
        c
        for c in required_unmatched
        if c not in unmatched.columns
    ]

    if missing:
        raise ValueError(
            "Unmatched file missing columns:\n"
            f"{missing}"
        )

    required_census = [
        "ST_NM",
        "DISTRICT",
        "geometry",
    ]

    missing = [
        c
        for c in required_census
        if c not in census.columns
    ]

    if missing:
        raise ValueError(
            "Census shapefile missing columns:\n"
            f"{missing}"
        )

    # --------------------------------------------------------
    # Normalize state names
    # --------------------------------------------------------

    unmatched["_state_key"] = (
        unmatched["matched_state"]
        .map(normalize_text)
    )

    unmatched["_district_key"] = (
        unmatched["matched_district"]
        .map(normalize_text)
    )

    census["_state_key"] = (
        census["ST_NM"]
        .map(normalize_text)
    )

    census["_district_key"] = (
        census["DISTRICT"]
        .map(normalize_text)
    )

    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    records = []

    for _, source in unmatched.iterrows():

        source_state = source["matched_state"]
        source_district = source["matched_district"]

        state_key = source["_state_key"]

        # ----------------------------------------------------
        # Only compare against Census districts in same state
        # after normalization.
        # ----------------------------------------------------

        state_candidates = census[
            census["_state_key"] == state_key
        ].copy()

        # ----------------------------------------------------
        # If exact state name does not exist, also search
        # Census states using similarity.
        # ----------------------------------------------------

        if state_candidates.empty:

            state_scores = []

            for census_state in census[
                "ST_NM"
            ].drop_duplicates():

                score = similarity(
                    source_state,
                    census_state,
                )

                state_scores.append(
                    (
                        census_state,
                        score,
                    )
                )

            state_scores.sort(
                key=lambda x: x[1],
                reverse=True,
            )

            best_states = [
                state
                for state, score
                in state_scores[:3]
            ]

            state_candidates = census[
                census["ST_NM"].isin(
                    best_states
                )
            ].copy()

        # ----------------------------------------------------
        # Score district-name similarity
        # ----------------------------------------------------

        for _, candidate in (
            state_candidates.iterrows()
        ):

            score = similarity(
                source_district,
                candidate["DISTRICT"],
            )

            records.append(
                {
                    "matched_state":
                        source_state,

                    "matched_district":
                        source_district,

                    "candidate_state":
                        candidate["ST_NM"],

                    "candidate_district":
                        candidate["DISTRICT"],

                    "similarity_score":
                        round(score, 6),

                    "candidate_geometry_exists":
                        True,
                }
            )

    # --------------------------------------------------------
    # Create candidate table
    # --------------------------------------------------------

    candidates = pd.DataFrame(
        records
    )

    if candidates.empty:
        raise ValueError(
            "No Census candidates were generated."
        )

    # --------------------------------------------------------
    # Rank candidates
    # --------------------------------------------------------

    candidates = candidates.sort_values(
        [
            "matched_state",
            "matched_district",
            "similarity_score",
        ],
        ascending=[
            True,
            True,
            False,
        ],
        kind="stable",
    )

    candidates[
        "candidate_rank"
    ] = (
        candidates
        .groupby(
            [
                "matched_state",
                "matched_district",
            ]
        )
        .cumcount()
        + 1
    )

    # --------------------------------------------------------
    # Candidate confidence band
    # --------------------------------------------------------

    def classify_score(score):

        if score >= 0.90:
            return "HIGH_SIMILARITY"

        if score >= 0.75:
            return "MEDIUM_SIMILARITY"

        if score >= 0.60:
            return "LOW_SIMILARITY"

        return "VERY_LOW_SIMILARITY"

    candidates[
        "similarity_band"
    ] = (
        candidates[
            "similarity_score"
        ]
        .map(classify_score)
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Create report
    # --------------------------------------------------------

    report_lines = []

    report_lines.append(
        "AgriAdapt V4.18.1 — Coordinate Candidate Audit"
    )

    report_lines.append(
        "=" * 65
    )

    report_lines.append("")

    report_lines.append(
        f"Unmatched unique geography pairs: "
        f"{len(unmatched)}"
    )

    report_lines.append(
        f"Candidate rows generated: "
        f"{len(candidates)}"
    )

    report_lines.append("")

    report_lines.append(
        "IMPORTANT:"
    )

    report_lines.append(
        "This file contains CANDIDATES only."
    )

    report_lines.append(
        "No candidate is automatically accepted as a geography mapping."
    )

    report_lines.append("")

    # --------------------------------------------------------
    # Top candidates
    # --------------------------------------------------------

    report_lines.append(
        "TOP CANDIDATE FOR EACH UNMATCHED PAIR"
    )

    report_lines.append(
        "-" * 65
    )

    top_candidates = candidates[
        candidates["candidate_rank"] == 1
    ]

    for _, row in top_candidates.iterrows():

        report_lines.append(
            f"{row['matched_state']} | "
            f"{row['matched_district']}"
        )

        report_lines.append(
            f"  -> "
            f"{row['candidate_state']} | "
            f"{row['candidate_district']} "
            f"(similarity="
            f"{row['similarity_score']:.3f}, "
            f"{row['similarity_band']})"
        )

    report_lines.append("")

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    report_lines.append(
        "OUTPUT"
    )

    report_lines.append(
        str(OUTPUT_FILE)
    )

    report_lines.append("")

    report_lines.append(
        "NEXT STEP"
    )

    report_lines.append(
        "Review the candidate table before creating "
        "any coordinate concordance."
    )

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final terminal output
    # --------------------------------------------------------

    print(
        "Candidate generation complete."
    )

    print(
        f"Candidate rows: "
        f"{len(candidates):,}"
    )

    print()

    print(
        "Top candidates:"
    )

    print(
        top_candidates[
            [
                "matched_state",
                "matched_district",
                "candidate_state",
                "candidate_district",
                "similarity_score",
                "similarity_band",
            ]
        ].to_string(
            index=False
        )
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
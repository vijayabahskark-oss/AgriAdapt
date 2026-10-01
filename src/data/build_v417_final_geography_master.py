"""
AgriAdapt V4.17 — Final Geography Master

Builds the final resolved district geography table from:
    V4.15 mapping audit
    V4.16 unresolved geography audit

No new mappings are created.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_MAPPING = (
    ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_v415_audit.csv"
)

INPUT_UNRESOLVED = (
    ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "unresolved_geography_v416.csv"
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

OUTPUT_FILE = OUTPUT_DIR / "final_district_geography_v417.csv"
REPORT_FILE = REPORT_DIR / "final_district_geography_v417.txt"


EXPECTED_TOTAL = 665
EXPECTED_UNRESOLVED = 32
EXPECTED_RESOLVED = 633


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()
    text = " ".join(text.split())

    return text


def pair_key(state, district):
    return (
        normalize_text(state),
        normalize_text(district),
    )


def find_column(df, candidates, label):
    """
    Find the first matching column from candidates.
    """
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find {label} column.\n"
        f"Tried: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("AgriAdapt V4.17 — Final Geography Master")
    print()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not INPUT_MAPPING.exists():
        raise FileNotFoundError(
            f"V4.15 mapping audit not found:\n{INPUT_MAPPING}"
        )

    if not INPUT_UNRESOLVED.exists():
        raise FileNotFoundError(
            f"V4.16 unresolved audit not found:\n{INPUT_UNRESOLVED}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    mapping_df = pd.read_csv(INPUT_MAPPING)
    unresolved_df = pd.read_csv(INPUT_UNRESOLVED)

    print(f"V4.15 mapping audit rows: {len(mapping_df):,}")
    print(f"V4.16 unresolved rows:    {len(unresolved_df):,}")
    print()

    # --------------------------------------------------------
    # Show actual schemas
    # --------------------------------------------------------

    print("V4.15 columns:")
    print(list(mapping_df.columns))
    print()

    print("V4.16 columns:")
    print(list(unresolved_df.columns))
    print()

    # --------------------------------------------------------
    # Resolve V4.15 column names
    # --------------------------------------------------------

    state_col = find_column(
        mapping_df,
        [
            "state_raw",
            "state",
            "source_state",
        ],
        "V4.15 state",
    )

    district_col = find_column(
        mapping_df,
        [
            "district_raw",
            "district",
            "source_district",
        ],
        "V4.15 district",
    )

    matched_state_col = find_column(
        mapping_df,
        [
            "matched_state",
            "state_census",
            "census_state",
        ],
        "V4.15 matched state",
    )

    matched_district_col = find_column(
        mapping_df,
        [
            "matched_district",
            "district_census",
            "census_district",
        ],
        "V4.15 matched district",
    )

    mapping_method_col = find_column(
        mapping_df,
        [
            "mapping_method",
            "mapping_status",
            "method",
        ],
        "V4.15 mapping method",
    )

    confidence_col = find_column(
        mapping_df,
        [
            "confidence",
        ],
        "V4.15 confidence",
    )

    # --------------------------------------------------------
    # Resolve V4.16 columns
    # --------------------------------------------------------

    unresolved_state_col = find_column(
        unresolved_df,
        [
            "state",
            "state_raw",
        ],
        "V4.16 state",
    )

    unresolved_district_col = find_column(
        unresolved_df,
        [
            "district_raw",
            "district",
        ],
        "V4.16 district",
    )

    # --------------------------------------------------------
    # Build pair keys
    # --------------------------------------------------------

    mapping_df["_pair_key"] = [
        pair_key(
            row[state_col],
            row[district_col],
        )
        for _, row in mapping_df.iterrows()
    ]

    unresolved_df["_pair_key"] = [
        pair_key(
            row[unresolved_state_col],
            row[unresolved_district_col],
        )
        for _, row in unresolved_df.iterrows()
    ]

    # --------------------------------------------------------
    # Validation: total pairs
    # --------------------------------------------------------

    total_pairs = mapping_df["_pair_key"].nunique()

    print(f"Total unique V4.15 pairs: {total_pairs:,}")

    if total_pairs != EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL} total pairs, "
            f"found {total_pairs}."
        )

    print("Total-pair validation: PASS")

    # --------------------------------------------------------
    # Validation: unresolved count
    # --------------------------------------------------------

    unresolved_pairs = unresolved_df["_pair_key"].nunique()

    print(
        f"V4.16 unresolved unique pairs: "
        f"{unresolved_pairs:,}"
    )

    if unresolved_pairs != EXPECTED_UNRESOLVED:
        raise ValueError(
            f"Expected {EXPECTED_UNRESOLVED} unresolved pairs, "
            f"found {unresolved_pairs}."
        )

    print("Unresolved-count validation: PASS")

    # --------------------------------------------------------
    # Make sets
    # --------------------------------------------------------

    mapping_pair_set = set(
        mapping_df["_pair_key"]
    )

    unresolved_pair_set = set(
        unresolved_df["_pair_key"]
    )

    # --------------------------------------------------------
    # Ensure all unresolved pairs exist in mapping audit
    # --------------------------------------------------------

    missing_unresolved = (
        unresolved_pair_set - mapping_pair_set
    )

    if missing_unresolved:
        print("Missing unresolved pairs:")

        for state, district in sorted(missing_unresolved):
            print(
                f"  {state} | {district}"
            )

        raise ValueError(
            "Some V4.16 unresolved pairs are not present "
            "in the V4.15 mapping audit."
        )

    print(
        "Unresolved-pair existence validation: PASS"
    )

    # --------------------------------------------------------
    # Remove unresolved pairs
    # --------------------------------------------------------

    final_df = mapping_df[
        ~mapping_df["_pair_key"].isin(
            unresolved_pair_set
        )
    ].copy()

    # --------------------------------------------------------
    # Final count
    # --------------------------------------------------------

    final_pairs = final_df["_pair_key"].nunique()

    print(
        f"Final resolved unique pairs: "
        f"{final_pairs:,}"
    )

    if final_pairs != EXPECTED_RESOLVED:
        raise ValueError(
            f"Expected {EXPECTED_RESOLVED} resolved pairs, "
            f"found {final_pairs}."
        )

    print("Final-resolved-count validation: PASS")

    # --------------------------------------------------------
    # Duplicate validation
    # --------------------------------------------------------

    duplicates = final_df[
        final_df["_pair_key"].duplicated(
            keep=False
        )
    ]

    if not duplicates.empty:

        print(
            "Duplicate pairs detected:"
        )

        print(
            duplicates[
                [
                    state_col,
                    district_col,
                ]
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

        raise ValueError(
            "Duplicate state-district pairs found."
        )

    print("Duplicate-pair validation: PASS")

    # --------------------------------------------------------
    # Ensure unresolved records were excluded
    # --------------------------------------------------------

    remaining_unresolved = (
        set(final_df["_pair_key"])
        & unresolved_pair_set
    )

    if remaining_unresolved:
        raise ValueError(
            "V4.16 unresolved pairs remain in final dataset."
        )

    print(
        "Unresolved-exclusion validation: PASS"
    )

    # --------------------------------------------------------
    # Validate matched geography
    # --------------------------------------------------------

    invalid_matched = final_df[
        final_df[matched_state_col].isna()
        | final_df[matched_district_col].isna()
        | (
            final_df[matched_state_col]
            .astype(str)
            .str.strip()
            == ""
        )
        | (
            final_df[matched_district_col]
            .astype(str)
            .str.strip()
            == ""
        )
    ]

    if not invalid_matched.empty:
        raise ValueError(
            "Final dataset contains rows without "
            "matched geography."
        )

    print(
        "Matched-geography validation: PASS"
    )

    # --------------------------------------------------------
    # Build standardized output
    # --------------------------------------------------------

    final_output = pd.DataFrame(
        {
            "state_raw": final_df[state_col],
            "district_raw": final_df[district_col],
            "matched_state": final_df[matched_state_col],
            "matched_district": final_df[matched_district_col],
            "mapping_method": final_df[mapping_method_col],
            "confidence": final_df[confidence_col],
        }
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    final_output = (
        final_output
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
    # Final row count
    # --------------------------------------------------------

    if len(final_output) != EXPECTED_RESOLVED:
        raise ValueError(
            f"Final row count expected "
            f"{EXPECTED_RESOLVED}, "
            f"found {len(final_output)}."
        )

    print("Row-count validation: PASS")

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

    final_output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    method_summary = (
        final_output[
            "mapping_method"
        ]
        .value_counts()
        .sort_index()
    )

    confidence_summary = (
        final_output[
            "confidence"
        ]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = []

    report.append(
        "AgriAdapt V4.17 — Final Geography Master"
    )
    report.append("=" * 60)
    report.append("")

    report.append("INPUTS")
    report.append(
        f"V4.15: {INPUT_MAPPING}"
    )
    report.append(
        f"V4.16: {INPUT_UNRESOLVED}"
    )
    report.append("")

    report.append("PAIR COUNTS")
    report.append(
        f"V4.15 total pairs: {total_pairs}"
    )
    report.append(
        f"V4.16 unresolved: {unresolved_pairs}"
    )
    report.append(
        f"V4.17 resolved: {final_pairs}"
    )
    report.append("")

    report.append("VALIDATION")
    report.append(
        "Total-pair validation: PASS"
    )
    report.append(
        "Unresolved-count validation: PASS"
    )
    report.append(
        "Unresolved-pair existence validation: PASS"
    )
    report.append(
        "Final-resolved-count validation: PASS"
    )
    report.append(
        "Duplicate-pair validation: PASS"
    )
    report.append(
        "Unresolved-exclusion validation: PASS"
    )
    report.append(
        "Matched-geography validation: PASS"
    )
    report.append(
        "Row-count validation: PASS"
    )
    report.append("")

    report.append("MAPPING METHOD SUMMARY")

    for method, count in method_summary.items():
        report.append(
            f"{method}: {count}"
        )

    report.append("")

    report.append("CONFIDENCE SUMMARY")

    for confidence, count in confidence_summary.items():
        report.append(
            f"{confidence}: {count}"
        )

    report.append("")

    report.append("OUTPUT")
    report.append(
        str(OUTPUT_FILE)
    )

    report.append(
        str(REPORT_FILE)
    )

    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("V4.17 COMPLETE")
    print()
    print(
        f"Final geography rows: "
        f"{len(final_output):,}"
    )
    print(
        f"Final geography columns: "
        f"{len(final_output.columns)}"
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
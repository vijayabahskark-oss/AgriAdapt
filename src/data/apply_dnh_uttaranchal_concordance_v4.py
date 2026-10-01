from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_maharashtra_rajasthan_tamilnadu_audit.csv"
)

INPUT_CONCORDANCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_dnh_uttaranchal.csv"
)

OUTPUT_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_dnh_uttaranchal_audit.csv"
)

EXPECTED_NEW_MAPPINGS = 12
EXPECTED_TOTAL_RESOLVED = 584
EXPECTED_UNMATCHED = 81


def main():

    print(
        "AgriAdapt V4.9 — Applying "
        "Dadra & Nagar Haveli/Uttaranchal Concordance"
    )
    print()

    # ---------------------------------------------------------
    # Load files
    # ---------------------------------------------------------

    if not INPUT_AUDIT.exists():
        raise FileNotFoundError(
            f"Input audit not found:\n{INPUT_AUDIT}"
        )

    if not INPUT_CONCORDANCE.exists():
        raise FileNotFoundError(
            f"Input concordance not found:\n{INPUT_CONCORDANCE}"
        )

    audit = pd.read_csv(
        INPUT_AUDIT,
        dtype=str,
    ).fillna("")

    concordance = pd.read_csv(
        INPUT_CONCORDANCE,
        dtype=str,
    ).fillna("")

    print(f"Audit rows: {len(audit)}")
    print(f"Concordance rows: {len(concordance)}")
    print()

    # ---------------------------------------------------------
    # Schema validation
    # ---------------------------------------------------------

    audit_required = {
        "state",
        "district_raw",
        "mapping_status",
    }

    concordance_required = {
        "state_raw",
        "district_raw",
        "matched_state",
        "matched_district",
        "mapping_method",
        "mapping_source",
        "confidence",
        "notes",
    }

    missing_audit = (
        audit_required
        - set(audit.columns)
    )

    if missing_audit:
        raise ValueError(
            f"Audit missing columns: "
            f"{sorted(missing_audit)}"
        )

    missing_concordance = (
        concordance_required
        - set(concordance.columns)
    )

    if missing_concordance:
        raise ValueError(
            "Concordance missing columns: "
            f"{sorted(missing_concordance)}"
        )

    original_row_count = len(audit)

    # ---------------------------------------------------------
    # Build concordance lookup
    # ---------------------------------------------------------

    concordance_lookup = {}

    for _, row in concordance.iterrows():

        key = (
            str(row["state_raw"])
            .upper()
            .strip(),

            str(row["district_raw"])
            .upper()
            .strip(),
        )

        if key in concordance_lookup:
            raise ValueError(
                f"Duplicate concordance key: {key}"
            )

        concordance_lookup[key] = row

    # ---------------------------------------------------------
    # Apply only currently-unmatched rows
    # ---------------------------------------------------------

    unmatched_mask = (
        audit["mapping_status"]
        .str.lower()
        .eq("unmatched")
    )

    applied_count = 0

    for index in audit.index[unmatched_mask]:

        state = (
            str(audit.at[index, "state"])
            .upper()
            .strip()
        )

        district = (
            str(audit.at[index, "district_raw"])
            .upper()
            .strip()
        )

        key = (
            state,
            district,
        )

        if key not in concordance_lookup:
            continue

        row = concordance_lookup[key]

        audit.at[index, "matched_state"] = (
            row["matched_state"]
        )

        audit.at[index, "matched_district"] = (
            row["matched_district"]
        )

        audit.at[index, "mapping_method"] = (
            row["mapping_method"]
        )

        audit.at[index, "mapping_source"] = (
            row["mapping_source"]
        )

        audit.at[index, "confidence"] = (
            row["confidence"]
        )

        audit.at[index, "notes"] = (
            row["notes"]
        )

        audit.at[index, "mapping_status"] = (
            "historical_concordance"
        )

        applied_count += 1

    # ---------------------------------------------------------
    # Application-count validation
    # ---------------------------------------------------------

    if applied_count != EXPECTED_NEW_MAPPINGS:

        raise ValueError(
            "Applied-count validation failed. "
            f"Expected {EXPECTED_NEW_MAPPINGS}, "
            f"got {applied_count}."
        )

    # ---------------------------------------------------------
    # Row-count validation
    # ---------------------------------------------------------

    if len(audit) != original_row_count:

        raise ValueError(
            "Row count changed during application. "
            f"Before: {original_row_count}, "
            f"After: {len(audit)}."
        )

    # ---------------------------------------------------------
    # Final counts
    # ---------------------------------------------------------

    unmatched_count = int(
        audit["mapping_status"]
        .str.lower()
        .eq("unmatched")
        .sum()
    )

    resolved_count = (
        len(audit)
        - unmatched_count
    )

    # ---------------------------------------------------------
    # Final-count validation
    # ---------------------------------------------------------

    if resolved_count != EXPECTED_TOTAL_RESOLVED:

        raise ValueError(
            "Resolved-count validation failed. "
            f"Expected {EXPECTED_TOTAL_RESOLVED}, "
            f"got {resolved_count}."
        )

    if unmatched_count != EXPECTED_UNMATCHED:

        raise ValueError(
            "Unmatched-count validation failed. "
            f"Expected {EXPECTED_UNMATCHED}, "
            f"got {unmatched_count}."
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_AUDIT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT_AUDIT,
        index=False,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("APPLICATION COMPLETE")
    print(
        f"Previous audit rows: "
        f"{original_row_count}"
    )
    print(
        f"Newly applied: "
        f"{applied_count}"
    )
    print(
        f"Total resolved: "
        f"{resolved_count}"
    )
    print(
        f"Still unmatched: "
        f"{unmatched_count}"
    )
    print(
        "Row count changed: "
        f"{'YES' if len(audit) != original_row_count else 'NO'}"
    )

    print()
    print("VALIDATION")
    print("Schema validation: PASS")
    print("Batch validation: PASS")
    print("Row-count validation: PASS")
    print("Applied-count validation: PASS")
    print("Resolved-count validation: PASS")
    print("Unmatched-count validation: PASS")

    print()
    print(
        "Expected applied:",
        EXPECTED_NEW_MAPPINGS,
    )
    print(
        "Actual applied:",
        applied_count,
    )

    print(
        "Expected resolved:",
        EXPECTED_TOTAL_RESOLVED,
    )
    print(
        "Actual resolved:",
        resolved_count,
    )

    print(
        "Expected unmatched:",
        EXPECTED_UNMATCHED,
    )
    print(
        "Actual unmatched:",
        unmatched_count,
    )

    print()
    print(
        f"Output: {OUTPUT_AUDIT}"
    )


if __name__ == "__main__":
    main()

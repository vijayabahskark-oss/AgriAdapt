from pathlib import Path
import pandas as pd


# ============================================================
# AgriAdapt V4.4
# Apply Arunachal Pradesh + Assam Concordance
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

AUDIT_INPUT = (
    BASE_DIR
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_south_north_audit.csv"
)

CONCORDANCE_INPUT = (
    BASE_DIR
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_arunachal_assam.csv"
)

AUDIT_OUTPUT = (
    BASE_DIR
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_arunachal_assam_audit.csv"
)

REPORT_OUTPUT = (
    BASE_DIR
    / "reports"
    / "v4"
    / "district_mapping_arunachal_assam_audit.txt"
)


EXPECTED_AUDIT_COLUMNS = [
    "state",
    "district_raw",
    "matched_state",
    "matched_district",
    "state_code",
    "district_code",
    "mapping_status",
    "alias_reason",
    "confidence",
    "mapping_method",
    "mapping_source",
    "mapping_confidence",
    "mapping_notes",
    "notes",
]

EXPECTED_CONCORDANCE_COLUMNS = [
    "state_raw",
    "district_raw",
    "matched_state",
    "matched_district",
    "mapping_method",
    "mapping_source",
    "confidence",
    "notes",
]


EXPECTED_MAPPINGS = {
    ("ARUNACHAL PRADESH", "E/KAMENG"),
    ("ARUNACHAL PRADESH", "E/SIANG"),
    ("ARUNACHAL PRADESH", "L/SUBABSIRI"),
    ("ARUNACHAL PRADESH", "LOHIT"),
    ("ARUNACHAL PRADESH", "PAPUMPARE"),
    ("ARUNACHAL PRADESH", "TAWANG"),
    ("ARUNACHAL PRADESH", "TIRAP"),
    ("ARUNACHAL PRADESH", "U/SIANG"),
    ("ARUNACHAL PRADESH", "U/SUBANSIRI"),
    ("ARUNACHAL PRADESH", "W/KAMENG"),
    ("ARUNACHAL PRADESH", "W/SIANG"),
    ("ARUNACHAL PRADESH", "CHANGLANG"),
    ("ASSAM", "N C HILLS"),
}


def main():

    print("AgriAdapt V4.4 — Applying Arunachal/Assam Concordance")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    if not AUDIT_INPUT.exists():
        raise FileNotFoundError(
            f"Audit input not found:\n{AUDIT_INPUT}"
        )

    if not CONCORDANCE_INPUT.exists():
        raise FileNotFoundError(
            f"Concordance input not found:\n{CONCORDANCE_INPUT}"
        )

    audit = pd.read_csv(AUDIT_INPUT)
    concordance = pd.read_csv(CONCORDANCE_INPUT)

    print(f"Audit rows:        {len(audit)}")
    print(f"Concordance rows:  {len(concordance)}")

    # --------------------------------------------------------
    # 2. Schema validation
    # --------------------------------------------------------

    missing_audit = [
        col for col in EXPECTED_AUDIT_COLUMNS
        if col not in audit.columns
    ]

    if missing_audit:
        raise ValueError(
            "Audit schema mismatch:\n"
            + "\n".join(missing_audit)
        )

    missing_concordance = [
        col for col in EXPECTED_CONCORDANCE_COLUMNS
        if col not in concordance.columns
    ]

    if missing_concordance:
        raise ValueError(
            "Concordance schema mismatch:\n"
            + "\n".join(missing_concordance)
        )

    # --------------------------------------------------------
    # 3. Duplicate validation
    # --------------------------------------------------------

    duplicate_mask = concordance.duplicated(
        subset=["state_raw", "district_raw"],
        keep=False,
    )

    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count:
        raise ValueError(
            "Duplicate concordance keys detected:\n"
            + concordance.loc[
                duplicate_mask,
                ["state_raw", "district_raw"],
            ].drop_duplicates().to_string(index=False)
        )

    # --------------------------------------------------------
    # 4. Extract expected batch
    # --------------------------------------------------------

    new_records = concordance[
        concordance.apply(
            lambda row:
            (
                str(row["state_raw"]).strip(),
                str(row["district_raw"]).strip()
            ) in EXPECTED_MAPPINGS,
            axis=1,
        )
    ].copy()

    if len(new_records) != 13:
        raise ValueError(
            f"Expected 13 mappings, found {len(new_records)}."
        )

    # --------------------------------------------------------
    # 5. Validate batch
    # --------------------------------------------------------

    if not (
        new_records["mapping_method"].astype(str).str.strip()
        == "NAME_ALIAS"
    ).all():
        raise ValueError(
            "All Arunachal/Assam mappings must be NAME_ALIAS."
        )

    if not (
        new_records["confidence"].astype(str).str.strip()
        == "high"
    ).all():
        raise ValueError(
            "All Arunachal/Assam mappings must have high confidence."
        )

    # --------------------------------------------------------
    # 6. Preserve row count
    # --------------------------------------------------------

    original_row_count = len(audit)

    # --------------------------------------------------------
    # 7. Apply
    # --------------------------------------------------------

    applied = 0

    for _, row in new_records.iterrows():

        state_raw = str(row["state_raw"]).strip()
        district_raw = str(row["district_raw"]).strip()

        mask = (
            (audit["state"].astype(str).str.strip() == state_raw)
            & (
                audit["district_raw"]
                .astype(str)
                .str.strip()
                == district_raw
            )
            & (
                audit["mapping_status"]
                .astype(str)
                .str.strip()
                == "unmatched"
            )
        )

        matches = int(mask.sum())

        if matches != 1:
            raise ValueError(
                f"Expected exactly one unmatched record for "
                f"{state_raw} | {district_raw}; "
                f"found {matches}."
            )

        audit.loc[mask, "matched_state"] = row["matched_state"]
        audit.loc[mask, "matched_district"] = row["matched_district"]

        audit.loc[mask, "mapping_status"] = "historical"
        audit.loc[mask, "alias_reason"] = "NAME_CONCORDANCE"
        audit.loc[mask, "confidence"] = row["confidence"]

        audit.loc[mask, "mapping_method"] = row["mapping_method"]
        audit.loc[mask, "mapping_source"] = row["mapping_source"]
        audit.loc[mask, "mapping_confidence"] = row["confidence"]
        audit.loc[mask, "mapping_notes"] = row["notes"]

        if "notes" in audit.columns:
            audit.loc[mask, "notes"] = row["notes"]

        applied += matches

    # --------------------------------------------------------
    # 8. Validate row count
    # --------------------------------------------------------

    final_row_count = len(audit)

    if final_row_count != original_row_count:
        raise ValueError(
            "Row count changed during application."
        )

    # --------------------------------------------------------
    # 9. Final counts
    # --------------------------------------------------------

    unmatched_count = int(
        (
            audit["mapping_status"].astype(str).str.strip()
            == "unmatched"
        ).sum()
    )

    resolved_count = original_row_count - unmatched_count

    historical_count = int(
        (
            audit["mapping_status"].astype(str).str.strip()
            == "historical"
        ).sum()
    )

    alias_count = int(
        (
            audit["mapping_method"].astype(str).str.strip()
            == "NAME_ALIAS"
        ).sum()
    )

    # --------------------------------------------------------
    # 10. Expected results
    # --------------------------------------------------------

    expected_applied = 13
    expected_resolved = 523
    expected_unmatched = 142

    if applied != expected_applied:
        raise ValueError(
            f"Expected {expected_applied} applied mappings, "
            f"got {applied}."
        )

    if resolved_count != expected_resolved:
        raise ValueError(
            f"Expected {expected_resolved} resolved records, "
            f"got {resolved_count}."
        )

    if unmatched_count != expected_unmatched:
        raise ValueError(
            f"Expected {expected_unmatched} unmatched records, "
            f"got {unmatched_count}."
        )

    # --------------------------------------------------------
    # 11. Individual mapping validation
    # --------------------------------------------------------

    for _, row in new_records.iterrows():

        state_raw = str(row["state_raw"]).strip()
        district_raw = str(row["district_raw"]).strip()

        mask = (
            (audit["state"].astype(str).str.strip() == state_raw)
            & (
                audit["district_raw"]
                .astype(str)
                .str.strip()
                == district_raw
            )
        )

        result = audit.loc[mask]

        if len(result) != 1:
            raise ValueError(
                f"Individual validation failed for "
                f"{state_raw} | {district_raw}"
            )

        result_row = result.iloc[0]

        if str(result_row["mapping_status"]).strip() != "historical":
            raise ValueError(
                f"Status validation failed for "
                f"{state_raw} | {district_raw}"
            )

        if (
            str(result_row["matched_district"]).strip()
            != str(row["matched_district"]).strip()
        ):
            raise ValueError(
                f"District validation failed for "
                f"{state_raw} | {district_raw}"
            )

    # --------------------------------------------------------
    # 12. Save
    # --------------------------------------------------------

    AUDIT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        AUDIT_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 13. Report
    # --------------------------------------------------------

    report = f"""AgriAdapt V4.4 — Arunachal/Assam Concordance Application
=================================================================

Previous audit rows:        {original_row_count}
Newly applied:              {applied}
Total resolved:             {resolved_count}
Still unmatched:            {unmatched_count}
Historical mappings:        {historical_count}
NAME_ALIAS mappings:        {alias_count}
Duplicate concordance:      {duplicate_count}
Row count changed:          {"YES" if final_row_count != original_row_count else "NO"}

VALIDATION
-----------------------------------------------------------------
Schema validation:          PASS
Batch validation:           PASS
Row-count validation:       PASS
Applied-count validation:   PASS
Resolved-count validation:  PASS
Unmatched-count validation: PASS
Individual mappings:        PASS

Expected applied:           {expected_applied}
Actual applied:             {applied}

Expected resolved:          {expected_resolved}
Actual resolved:            {resolved_count}

Expected unmatched:         {expected_unmatched}
Actual unmatched:           {unmatched_count}

Input audit:
{AUDIT_INPUT}

Input concordance:
{CONCORDANCE_INPUT}

Output audit:
{AUDIT_OUTPUT}
"""

    REPORT_OUTPUT.write_text(
        report,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 14. Console
    # --------------------------------------------------------

    print()
    print("APPLICATION COMPLETE")
    print("-" * 65)
    print(f"Previous audit rows:        {original_row_count}")
    print(f"Newly applied:              {applied}")
    print(f"Total resolved:             {resolved_count}")
    print(f"Still unmatched:            {unmatched_count}")
    print(f"Historical mappings:        {historical_count}")
    print(f"NAME_ALIAS mappings:        {alias_count}")
    print(f"Duplicate concordance:      {duplicate_count}")
    print(
        "Row count changed:          "
        f"{'YES' if final_row_count != original_row_count else 'NO'}"
    )

    print()
    print("VALIDATION")
    print("-" * 65)
    print("Schema validation:          PASS")
    print("Batch validation:           PASS")
    print("Row-count validation:       PASS")
    print("Applied-count validation:   PASS")
    print("Resolved-count validation:  PASS")
    print("Unmatched-count validation: PASS")
    print("Individual mappings:        PASS")

    print()
    print(f"Expected applied:           {expected_applied}")
    print(f"Actual applied:             {applied}")
    print(f"Expected resolved:          {expected_resolved}")
    print(f"Actual resolved:            {resolved_count}")
    print(f"Expected unmatched:         {expected_unmatched}")
    print(f"Actual unmatched:           {unmatched_count}")

    print()
    print(f"Output: {AUDIT_OUTPUT}")
    print(f"Report: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd


# ============================================================
# AgriAdapt V4.2 — ORISSA Name Concordance Application
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

AUDIT_INPUT = (
    BASE_DIR
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_up_uttaranchal_audit.csv"
)

CONCORDANCE_INPUT = (
    BASE_DIR
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_orissa.csv"
)

AUDIT_OUTPUT = (
    BASE_DIR
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_orissa_audit.csv"
)

REPORT_OUTPUT = (
    BASE_DIR
    / "reports"
    / "v4"
    / "district_mapping_orissa_audit.txt"
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


def main():

    print("AgriAdapt V4.2 — Applying ORISSA Name Concordance")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load files
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
    # 2. Validate schemas
    # --------------------------------------------------------

    missing_audit = [
        col for col in EXPECTED_AUDIT_COLUMNS
        if col not in audit.columns
    ]

    if missing_audit:
        raise ValueError(
            "Audit schema mismatch. Missing columns:\n"
            + "\n".join(missing_audit)
        )

    missing_concordance = [
        col for col in EXPECTED_CONCORDANCE_COLUMNS
        if col not in concordance.columns
    ]

    if missing_concordance:
        raise ValueError(
            "Concordance schema mismatch. Missing columns:\n"
            + "\n".join(missing_concordance)
        )

    # --------------------------------------------------------
    # 3. Validate concordance contents
    # --------------------------------------------------------

    expected_new = {
        ("ORISSA", "ANGUL"),
        ("ORISSA", "BALASORE"),
        ("ORISSA", "BOLANGIR"),
        ("ORISSA", "BOUDH"),
        ("ORISSA", "BURAGARH"),
        ("ORISSA", "DEOGARH"),
        ("ORISSA", "GAJAPATTI"),
        ("ORISSA", "JAGATSINGPUR"),
        ("ORISSA", "JAJPUR"),
        ("ORISSA", "JHARSUGDA"),
        ("ORISSA", "KEDRAPARA"),
        ("ORISSA", "KEONJHAR"),
        ("ORISSA", "KHURDA"),
        ("ORISSA", "NAWAPARA"),
        ("ORISSA", "NAWORANGPUR"),
        ("ORISSA", "PHULBANI"),
        ("ORISSA", "SONEPUR"),
    }

    concordance_keys = set(
        zip(
            concordance["state_raw"].astype(str).str.strip(),
            concordance["district_raw"].astype(str).str.strip(),
        )
    )

    missing_expected = expected_new - concordance_keys

    if missing_expected:
        raise ValueError(
            "Expected ORISSA concordance records are missing:\n"
            + "\n".join(
                f"{state} | {district}"
                for state, district in sorted(missing_expected)
            )
        )

    # Verify all new records are ORISSA → ORISSA
    new_records = concordance[
        concordance.apply(
            lambda row:
            (
                str(row["state_raw"]).strip(),
                str(row["district_raw"]).strip()
            ) in expected_new,
            axis=1,
        )
    ].copy()

    if len(new_records) != 17:
        raise ValueError(
            f"Expected 17 ORISSA records, found {len(new_records)}."
        )

    invalid_state_mapping = new_records[
        new_records["matched_state"].astype(str).str.strip() != "ORISSA"
    ]

    if not invalid_state_mapping.empty:
        raise ValueError(
            "Invalid ORISSA state mapping detected."
        )

    invalid_methods = new_records[
        new_records["mapping_method"].astype(str).str.strip()
        != "NAME_ALIAS"
    ]

    if not invalid_methods.empty:
        raise ValueError(
            "Unexpected mapping method detected in ORISSA batch."
        )

    # --------------------------------------------------------
    # 4. Check duplicate concordance keys
    # --------------------------------------------------------

    duplicate_mask = concordance.duplicated(
        subset=["state_raw", "district_raw"],
        keep=False,
    )

    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count:
        duplicate_rows = concordance.loc[
            duplicate_mask,
            ["state_raw", "district_raw"],
        ].drop_duplicates()

        raise ValueError(
            "Duplicate concordance keys detected:\n"
            + duplicate_rows.to_string(index=False)
        )

    # --------------------------------------------------------
    # 5. Preserve original row count
    # --------------------------------------------------------

    original_row_count = len(audit)

    # --------------------------------------------------------
    # 6. Apply ORISSA mappings
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
                f"{state_raw} | {district_raw}, found {matches}."
            )

        audit.loc[mask, "matched_state"] = row["matched_state"]
        audit.loc[mask, "matched_district"] = row["matched_district"]

        audit.loc[mask, "mapping_status"] = "historical"
        audit.loc[mask, "alias_reason"] = "ORISSA_NAME_CONCORDANCE"
        audit.loc[mask, "confidence"] = row["confidence"]

        audit.loc[mask, "mapping_method"] = row["mapping_method"]
        audit.loc[mask, "mapping_source"] = row["mapping_source"]
        audit.loc[mask, "mapping_confidence"] = row["confidence"]

        audit.loc[mask, "mapping_notes"] = row["notes"]

        # Preserve compatibility with the additional notes field
        if "notes" in audit.columns:
            audit.loc[mask, "notes"] = row["notes"]

        applied += matches

    # --------------------------------------------------------
    # 7. Validation
    # --------------------------------------------------------

    final_row_count = len(audit)

    if final_row_count != original_row_count:
        raise ValueError(
            "Row count changed during mapping application."
        )

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
    # 8. Validate expected result
    # --------------------------------------------------------

    expected_resolved = 494
    expected_unmatched = 171

    if resolved_count != expected_resolved:
        raise ValueError(
            f"Unexpected resolved count: {resolved_count}. "
            f"Expected {expected_resolved}."
        )

    if unmatched_count != expected_unmatched:
        raise ValueError(
            f"Unexpected unmatched count: {unmatched_count}. "
            f"Expected {expected_unmatched}."
        )

    # --------------------------------------------------------
    # 9. Validate newly applied mappings
    # --------------------------------------------------------

    for _, row in new_records.iterrows():

        mask = (
            (audit["state"].astype(str).str.strip()
             == row["state_raw"].strip())
            & (
                audit["district_raw"]
                .astype(str)
                .str.strip()
                == row["district_raw"].strip()
            )
        )

        result = audit.loc[mask]

        if len(result) != 1:
            raise ValueError(
                f"Post-application validation failed for "
                f"{row['state_raw']} | {row['district_raw']}"
            )

        result_row = result.iloc[0]

        if str(result_row["mapping_status"]).strip() != "historical":
            raise ValueError(
                f"Mapping status validation failed for "
                f"{row['state_raw']} | {row['district_raw']}"
            )

        if (
            str(result_row["matched_district"]).strip()
            != str(row["matched_district"]).strip()
        ):
            raise ValueError(
                f"Matched district validation failed for "
                f"{row['state_raw']} | {row['district_raw']}"
            )

    # --------------------------------------------------------
    # 10. Save output
    # --------------------------------------------------------

    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    audit.to_csv(
        AUDIT_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 11. Write report
    # --------------------------------------------------------

    report_lines = [
        "AgriAdapt V4.2 — ORISSA Name Concordance Application",
        "=" * 60,
        "",
        f"Previous audit rows:       {original_row_count}",
        f"Newly applied:             {applied}",
        f"Total resolved:            {resolved_count}",
        f"Still unmatched:           {unmatched_count}",
        f"Historical mappings:       {historical_count}",
        f"NAME_ALIAS mappings:       {alias_count}",
        f"Duplicate concordance:     {duplicate_count}",
        f"Row count changed:         {'YES' if final_row_count != original_row_count else 'NO'}",
        "",
        "Validation:",
        f"Expected resolved:         {expected_resolved}",
        f"Actual resolved:           {resolved_count}",
        f"Expected unmatched:        {expected_unmatched}",
        f"Actual unmatched:          {unmatched_count}",
        f"Schema validation:         PASS",
        f"ORISSA batch validation:   PASS",
        f"Row-count validation:      PASS",
        "",
        f"Input audit:",
        f"  {AUDIT_INPUT}",
        "",
        f"Input concordance:",
        f"  {CONCORDANCE_INPUT}",
        "",
        f"Output audit:",
        f"  {AUDIT_OUTPUT}",
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 12. Console summary
    # --------------------------------------------------------

    print()
    print("APPLICATION COMPLETE")
    print("-" * 60)
    print(f"Previous audit rows:       {original_row_count}")
    print(f"Newly applied:             {applied}")
    print(f"Total resolved:            {resolved_count}")
    print(f"Still unmatched:           {unmatched_count}")
    print(f"Historical mappings:       {historical_count}")
    print(f"NAME_ALIAS mappings:       {alias_count}")
    print(f"Duplicate concordance:     {duplicate_count}")
    print(
        "Row count changed:         "
        f"{'YES' if final_row_count != original_row_count else 'NO'}"
    )

    print()
    print("VALIDATION")
    print("-" * 60)
    print("Schema validation:         PASS")
    print("ORISSA batch validation:   PASS")
    print("Row-count validation:      PASS")
    print(f"Expected resolved:         {expected_resolved}")
    print(f"Actual resolved:           {resolved_count}")
    print(f"Expected unmatched:        {expected_unmatched}")
    print(f"Actual unmatched:          {unmatched_count}")

    print()
    print(f"Output: {AUDIT_OUTPUT}")
    print(f"Report: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
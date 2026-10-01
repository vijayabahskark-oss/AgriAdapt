from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_v414_audit.csv"
)

INPUT_CONCORDANCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_v415.csv"
)

OUTPUT_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_v415_audit.csv"
)

EXPECTED_NEW = 2
EXPECTED_PREVIOUS_RESOLVED = 631
EXPECTED_PREVIOUS_UNMATCHED = 34
EXPECTED_RESOLVED = 633
EXPECTED_UNMATCHED = 32

EXPECTED_NEW_KEYS = {
    "CHHATTISGARH|||DANTEWARA",
    "MANIPUR|||SENAPATI",
}


def norm(value):
    text = str(value).strip().upper()
    text = text.replace("&", "AND")
    text = " ".join(text.split())
    return text


def make_key(state, district):
    return f"{norm(state)}|||{norm(district)}"


def main():

    print(
        "AgriAdapt V4.15 — Applying "
        "District Concordance"
    )
    print()

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    audit = pd.read_csv(
        INPUT_AUDIT,
        dtype=str
    ).fillna("")

    concordance = pd.read_csv(
        INPUT_CONCORDANCE,
        dtype=str
    ).fillna("")

    print(f"Audit rows: {len(audit)}")
    print(f"Concordance rows: {len(concordance)}")
    print()

    # ---------------------------------------------------------
    # Schema validation
    # ---------------------------------------------------------

    required_audit = {
        "state",
        "district_raw",
        "mapping_status",
        "matched_state",
        "matched_district",
    }

    required_concordance = {
        "state_raw",
        "district_raw",
        "matched_state",
        "matched_district",
        "mapping_method",
        "confidence",
    }

    missing_audit = (
        required_audit - set(audit.columns)
    )

    missing_concordance = (
        required_concordance
        - set(concordance.columns)
    )

    if missing_audit:
        raise ValueError(
            f"Audit missing columns: "
            f"{sorted(missing_audit)}"
        )

    if missing_concordance:
        raise ValueError(
            "Concordance missing columns: "
            f"{sorted(missing_concordance)}"
        )

    # ---------------------------------------------------------
    # Build deterministic keys
    # ---------------------------------------------------------

    audit["_mapping_key"] = [
        make_key(state, district)
        for state, district in zip(
            audit["state"],
            audit["district_raw"]
        )
    ]

    concordance["_mapping_key"] = [
        make_key(state, district)
        for state, district in zip(
            concordance["state_raw"],
            concordance["district_raw"]
        )
    ]

    # ---------------------------------------------------------
    # Select ONLY V4.15 mappings
    # ---------------------------------------------------------

    batch = concordance[
        concordance["_mapping_key"].isin(
            EXPECTED_NEW_KEYS
        )
    ].copy()

    if len(batch) != EXPECTED_NEW:
        raise ValueError(
            f"Expected {EXPECTED_NEW} V4.15 mappings, "
            f"found {len(batch)}."
        )

    actual_keys = set(
        batch["_mapping_key"]
    )

    if actual_keys != EXPECTED_NEW_KEYS:

        missing = EXPECTED_NEW_KEYS - actual_keys
        extra = actual_keys - EXPECTED_NEW_KEYS

        raise ValueError(
            "V4.15 key mismatch.\n"
            f"Missing: {sorted(missing)}\n"
            f"Unexpected: {sorted(extra)}"
        )

    # ---------------------------------------------------------
    # Duplicate validation
    # ---------------------------------------------------------

    if audit["_mapping_key"].duplicated().any():

        duplicates = audit[
            audit["_mapping_key"].duplicated(
                keep=False
            )
        ]

        raise ValueError(
            "Duplicate audit mapping keys detected:\n"
            + duplicates[
                [
                    "state",
                    "district_raw",
                    "_mapping_key",
                ]
            ].to_string(index=False)
        )

    if batch["_mapping_key"].duplicated().any():
        raise ValueError(
            "Duplicate keys detected inside V4.15 batch."
        )

    # ---------------------------------------------------------
    # Verify both are currently unmatched
    # ---------------------------------------------------------

    print("VERIFYING V4.15 SOURCE KEYS")
    print()

    for _, row in batch.iterrows():

        key = row["_mapping_key"]

        matches = audit[
            audit["_mapping_key"] == key
        ]

        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one audit row for:\n"
                f"{row['state_raw']} | "
                f"{row['district_raw']}\n"
                f"Found: {len(matches)}"
            )

        current_status = (
            matches.iloc[0]["mapping_status"]
        )

        if current_status != "unmatched":
            raise ValueError(
                "V4.15 mapping is not currently unmatched:\n"
                f"{row['state_raw']} | "
                f"{row['district_raw']}\n"
                f"Current status: {current_status}"
            )

        print(
            f"  PASS  {row['state_raw']} | "
            f"{row['district_raw']} -> "
            f"{row['matched_district']}"
        )

    print()

    # ---------------------------------------------------------
    # Verify starting counts
    # ---------------------------------------------------------

    previous_resolved = int(
        (
            audit["mapping_status"]
            != "unmatched"
        ).sum()
    )

    previous_unmatched = int(
        (
            audit["mapping_status"]
            == "unmatched"
        ).sum()
    )

    if previous_resolved != EXPECTED_PREVIOUS_RESOLVED:
        raise ValueError(
            f"Expected previous resolved count "
            f"{EXPECTED_PREVIOUS_RESOLVED}, "
            f"found {previous_resolved}."
        )

    if previous_unmatched != EXPECTED_PREVIOUS_UNMATCHED:
        raise ValueError(
            f"Expected previous unmatched count "
            f"{EXPECTED_PREVIOUS_UNMATCHED}, "
            f"found {previous_unmatched}."
        )

    # ---------------------------------------------------------
    # Apply mappings
    # ---------------------------------------------------------

    applied = 0

    for _, row in batch.iterrows():

        mask = (
            audit["_mapping_key"]
            == row["_mapping_key"]
        )

        audit.loc[
            mask,
            "matched_state"
        ] = row["matched_state"]

        audit.loc[
            mask,
            "matched_district"
        ] = row["matched_district"]

        audit.loc[
            mask,
            "mapping_status"
        ] = "historical_concordance"

        applied += int(mask.sum())

    # ---------------------------------------------------------
    # Final counts
    # ---------------------------------------------------------

    resolved = int(
        (
            audit["mapping_status"]
            != "unmatched"
        ).sum()
    )

    unmatched = int(
        (
            audit["mapping_status"]
            == "unmatched"
        ).sum()
    )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    if len(audit) != 665:
        raise ValueError(
            f"Row count changed: {len(audit)}"
        )

    if applied != EXPECTED_NEW:
        raise ValueError(
            f"Expected {EXPECTED_NEW} mappings "
            f"to be applied, got {applied}."
        )

    if resolved != EXPECTED_RESOLVED:
        raise ValueError(
            f"Expected {EXPECTED_RESOLVED} resolved, "
            f"got {resolved}."
        )

    if unmatched != EXPECTED_UNMATCHED:
        raise ValueError(
            f"Expected {EXPECTED_UNMATCHED} unmatched, "
            f"got {unmatched}."
        )

    # ---------------------------------------------------------
    # Remove temporary key
    # ---------------------------------------------------------

    audit.drop(
        columns=["_mapping_key"],
        inplace=True
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_AUDIT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    audit.to_csv(
        OUTPUT_AUDIT,
        index=False
    )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print("APPLICATION COMPLETE")
    print()
    print(
        f"Previous resolved: {previous_resolved}"
    )
    print(
        f"Previous unmatched: {previous_unmatched}"
    )
    print(
        f"Newly applied:      {applied}"
    )
    print(
        f"Total resolved:     {resolved}"
    )
    print(
        f"Still unmatched:    {unmatched}"
    )
    print(
        "Row count changed:  NO"
    )
    print()

    print("VALIDATION")
    print("Schema validation: PASS")
    print("Batch key validation: PASS")
    print("Duplicate validation: PASS")
    print("Starting-count validation: PASS")
    print("Applied-count validation: PASS")
    print("Final-count validation: PASS")
    print("Row-count validation: PASS")
    print()

    print(
        f"Output: {OUTPUT_AUDIT}"
    )


if __name__ == "__main__":
    main()
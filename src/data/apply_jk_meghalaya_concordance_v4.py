from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_west_bengal_audit.csv"
)

INPUT_CONCORDANCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_jk_meghalaya.csv"
)

OUTPUT_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_jk_meghalaya_audit.csv"
)

EXPECTED_NEW = 13
EXPECTED_PREVIOUS_RESOLVED = 595
EXPECTED_PREVIOUS_UNMATCHED = 70
EXPECTED_RESOLVED = 608
EXPECTED_UNMATCHED = 57


def norm(value):
    return (
        str(value)
        .strip()
        .upper()
        .replace("&", "AND")
        .replace("  ", " ")
    )


def make_key(state, district):
    return f"{norm(state)}|||{norm(district)}"


def main():

    print(
        "AgriAdapt V4.11 — Applying "
        "Jammu & Kashmir/Meghalaya Concordance"
    )
    print()

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

    missing_audit = required_audit - set(audit.columns)

    missing_concordance = (
        required_concordance
        - set(concordance.columns)
    )

    if missing_audit:
        raise ValueError(
            f"Audit missing columns: {sorted(missing_audit)}"
        )

    if missing_concordance:
        raise ValueError(
            "Concordance missing columns: "
            f"{sorted(missing_concordance)}"
        )

    # ---------------------------------------------------------
    # Select J&K + Meghalaya mappings
    # ---------------------------------------------------------

    target_states = {
        "JAMMU AND KASHMIR",
        "MEGHALYA",
    }

    batch = concordance[
        concordance["state_raw"]
        .map(norm)
        .isin(target_states)
    ].copy()

    if len(batch) != EXPECTED_NEW:
        raise ValueError(
            f"Expected {EXPECTED_NEW} mappings, "
            f"found {len(batch)}."
        )

    # ---------------------------------------------------------
    # Build deterministic composite keys
    # ---------------------------------------------------------

    audit["_mapping_key"] = [
        make_key(state, district)
        for state, district in zip(
            audit["state"],
            audit["district_raw"],
        )
    ]

    batch["_mapping_key"] = [
        make_key(state, district)
        for state, district in zip(
            batch["state_raw"],
            batch["district_raw"],
        )
    ]

    # ---------------------------------------------------------
    # Validate audit keys
    # ---------------------------------------------------------

    duplicate_audit_keys = audit[
        audit["_mapping_key"].duplicated(
            keep=False
        )
    ]

    if not duplicate_audit_keys.empty:
        raise ValueError(
            "Duplicate audit mapping keys detected:\n"
            f"{duplicate_audit_keys[['state', 'district_raw', '_mapping_key']].to_string(index=False)}"
        )

    # ---------------------------------------------------------
    # Validate batch keys
    # ---------------------------------------------------------

    if batch["_mapping_key"].duplicated().any():
        raise ValueError(
            "Duplicate mapping keys detected inside "
            "concordance batch."
        )

    # ---------------------------------------------------------
    # Verify every mapping exists
    # ---------------------------------------------------------

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
                f"Normalized key: {key}\n"
                f"Found: {len(matches)}"
            )

        if (
            matches.iloc[0]["mapping_status"]
            != "unmatched"
        ):
            raise ValueError(
                "Mapping is not currently unmatched:\n"
                f"{row['state_raw']} | "
                f"{row['district_raw']}"
            )

    # ---------------------------------------------------------
    # Starting counts
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

        key = row["_mapping_key"]

        mask = (
            audit["_mapping_key"] == key
        )

        audit.loc[
            mask,
            "matched_state",
        ] = row["matched_state"]

        audit.loc[
            mask,
            "matched_district",
        ] = row["matched_district"]

        audit.loc[
            mask,
            "mapping_status",
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
            f"Row-count changed: {len(audit)}"
        )

    if applied != EXPECTED_NEW:
        raise ValueError(
            f"Expected applied {EXPECTED_NEW}, "
            f"got {applied}"
        )

    if resolved != EXPECTED_RESOLVED:
        raise ValueError(
            f"Expected resolved {EXPECTED_RESOLVED}, "
            f"got {resolved}"
        )

    if unmatched != EXPECTED_UNMATCHED:
        raise ValueError(
            f"Expected unmatched {EXPECTED_UNMATCHED}, "
            f"got {unmatched}"
        )

    # Remove temporary column
    audit.drop(
        columns=["_mapping_key"],
        inplace=True,
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
    print(f"Previous audit rows: {len(audit)}")
    print(f"Newly applied: {applied}")
    print(f"Total resolved: {resolved}")
    print(f"Still unmatched: {unmatched}")
    print("Row count changed: NO")
    print()

    print("VALIDATION")
    print("Schema validation: PASS")
    print("Batch validation: PASS")
    print("Row-count validation: PASS")
    print("Applied-count validation: PASS")
    print("Resolved-count validation: PASS")
    print("Unmatched-count validation: PASS")
    print()

    print(f"Expected applied: {EXPECTED_NEW}")
    print(f"Actual applied: {applied}")
    print(f"Expected resolved: {EXPECTED_RESOLVED}")
    print(f"Actual resolved: {resolved}")
    print(f"Expected unmatched: {EXPECTED_UNMATCHED}")
    print(f"Actual unmatched: {unmatched}")
    print()

    print(f"Output: {OUTPUT_AUDIT}")


if __name__ == "__main__":
    main()
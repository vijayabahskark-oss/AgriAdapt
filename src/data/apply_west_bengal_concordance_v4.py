from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_AUDIT = (
    PROJECT_ROOT / "data" / "processed" / "v4" / "district"
    / "district_mapping_dnh_uttaranchal_audit.csv"
)

INPUT_CONCORDANCE = (
    PROJECT_ROOT / "data" / "processed" / "v4" / "district"
    / "historical_district_concordance_west_bengal.csv"
)

OUTPUT_AUDIT = (
    PROJECT_ROOT / "data" / "processed" / "v4" / "district"
    / "district_mapping_west_bengal_audit.csv"
)

TARGET_STATE = "WEST BENGAL"

EXPECTED_NEW = 11
EXPECTED_PREVIOUS_RESOLVED = 584
EXPECTED_PREVIOUS_UNMATCHED = 81
EXPECTED_RESOLVED = 595
EXPECTED_UNMATCHED = 70


def norm(value):
    return str(value).strip().upper()


def main():

    print(
        "AgriAdapt V4.10 — Applying West Bengal Concordance"
    )
    print()

    audit = pd.read_csv(INPUT_AUDIT, dtype=str).fillna("")
    concordance = pd.read_csv(
        INPUT_CONCORDANCE,
        dtype=str,
    ).fillna("")

    print(f"Audit rows: {len(audit)}")
    print(f"Concordance rows: {len(concordance)}")
    print()

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
        required_concordance - set(concordance.columns)
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

    # Only use the 11 new West Bengal mappings.
    batch = concordance[
        concordance["state_raw"].map(norm) == TARGET_STATE
    ].copy()

    if len(batch) != EXPECTED_NEW:
        raise ValueError(
            f"Expected {EXPECTED_NEW} West Bengal mappings, "
            f"found {len(batch)}."
        )

    # Ensure every batch row is currently unmatched.
    for _, row in batch.iterrows():

        mask = (
            audit["state"].map(norm)
            == norm(row["state_raw"])
        ) & (
            audit["district_raw"].map(norm)
            == norm(row["district_raw"])
        )

        matches = audit.loc[mask]

        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one audit row for:\n"
                f"{row['state_raw']} | "
                f"{row['district_raw']}\n"
                f"Found: {len(matches)}"
            )

        if matches.iloc[0]["mapping_status"] != "unmatched":
            raise ValueError(
                "Mapping is not currently unmatched:\n"
                f"{row['state_raw']} | "
                f"{row['district_raw']}"
            )

    previous_resolved = int(
        (audit["mapping_status"] != "unmatched").sum()
    )

    previous_unmatched = int(
        (audit["mapping_status"] == "unmatched").sum()
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

    # Apply mappings.
    applied = 0

    for _, row in batch.iterrows():

        mask = (
            audit["state"].map(norm)
            == norm(row["state_raw"])
        ) & (
            audit["district_raw"].map(norm)
            == norm(row["district_raw"])
        )

        audit.loc[mask, "matched_state"] = (
            row["matched_state"]
        )

        audit.loc[mask, "matched_district"] = (
            row["matched_district"]
        )

        audit.loc[mask, "mapping_status"] = (
            "historical_concordance"
        )

        applied += int(mask.sum())

    resolved = int(
        (audit["mapping_status"] != "unmatched").sum()
    )

    unmatched = int(
        (audit["mapping_status"] == "unmatched").sum()
    )

    # Validation.
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

    OUTPUT_AUDIT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT_AUDIT,
        index=False,
    )

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
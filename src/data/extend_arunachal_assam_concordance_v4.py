from pathlib import Path
import pandas as pd


# ============================================================
# AgriAdapt V4.4
# Arunachal Pradesh + Assam
# District Name Concordance Extension
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_CONCORDANCE = (
    BASE_DIR
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_south_north.csv"
)

OUTPUT_CONCORDANCE = (
    BASE_DIR
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_arunachal_assam.csv"
)


EXPECTED_COLUMNS = [
    "state_raw",
    "district_raw",
    "matched_state",
    "matched_district",
    "mapping_method",
    "mapping_source",
    "confidence",
    "notes",
]


NEW_MAPPINGS = [

    # --------------------------------------------------------
    # Arunachal Pradesh
    # --------------------------------------------------------

    {
        "state_raw": "ARUNACHAL PRADESH",
        "district_raw": "E/KAMENG",
        "matched_state": "ARUNACHAL PRADESH",
        "matched_district": "East Kameng",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Abbreviated form of East Kameng.",
    },

    {
        "state_raw": "ARUNACHAL PRADESH",
        "district_raw": "E/SIANG",
        "matched_state": "ARUNACHAL PRADESH",
        "matched_district": "East Siang",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Abbreviated form of East Siang.",
    },

    {
        "state_raw": "ARUNACHAL PRADESH",
        "district_raw": "L/SUBABSIRI",
        "matched_state": "ARUNACHAL PRADESH",
        "matched_district": "Lower Subansiri",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Abbreviated legacy spelling of Lower Subansiri.",
    },

    {
        "state_raw": "ARUNACHAL PRADESH",
        "district_raw": "LOHIT",
        "matched_state": "ARUNACHAL PRADESH",
        "matched_district": "Lohit",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Exact historical district name; unmatched because of source-name normalization differences.",
    },

    {
        "state_raw": "ARUNACHAL PRADESH",
        "district_raw": "PAPUMPARE",
        "matched_state": "ARUNACHAL PRADESH",
        "matched_district": "Papum Pare",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Spacing-normalized form of Papum Pare.",
    },

    {
        "state_raw": "ARUNACHAL PRADESH",
        "district_raw": "TAWANG",
        "matched_state": "ARUNACHAL PRADESH",
        "matched_district": "Tawang",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Exact historical district name.",
    },

    {
        "state_raw": "ARUNACHAL PRADESH",
        "district_raw": "TIRAP",
        "matched_state": "ARUNACHAL PRADESH",
        "matched_district": "Tirap",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Exact historical district name.",
    },

    {
        "state_raw": "ARUNACHAL PRADESH",
        "district_raw": "U/SIANG",
        "matched_state": "ARUNACHAL PRADESH",
        "matched_district": "Upper Siang",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Abbreviated form of Upper Siang.",
    },

    {
        "state_raw": "ARUNACHAL PRADESH",
        "district_raw": "U/SUBANSIRI",
        "matched_state": "ARUNACHAL PRADESH",
        "matched_district": "Upper Subansiri",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Abbreviated form of Upper Subansiri.",
    },

    {
        "state_raw": "ARUNACHAL PRADESH",
        "district_raw": "W/KAMENG",
        "matched_state": "ARUNACHAL PRADESH",
        "matched_district": "West Kameng",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Abbreviated form of West Kameng.",
    },

    {
        "state_raw": "ARUNACHAL PRADESH",
        "district_raw": "W/SIANG",
        "matched_state": "ARUNACHAL PRADESH",
        "matched_district": "West Siang",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Abbreviated form of West Siang.",
    },

    {
        "state_raw": "ARUNACHAL PRADESH",
        "district_raw": "CHANGLANG",
        "matched_state": "ARUNACHAL PRADESH",
        "matched_district": "Changlang",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Source spelling variant of Changlang.",
    },

    # --------------------------------------------------------
    # Assam
    # --------------------------------------------------------

    {
        "state_raw": "ASSAM",
        "district_raw": "N C HILLS",
        "matched_state": "ASSAM",
        "matched_district": "North Cachar Hills",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Abbreviated form of North Cachar Hills.",
    },
]


def main():

    print("AgriAdapt V4.4 — Arunachal/Assam Concordance Extension")
    print("=" * 65)

    if not INPUT_CONCORDANCE.exists():
        raise FileNotFoundError(
            f"Input concordance not found:\n{INPUT_CONCORDANCE}"
        )

    concordance = pd.read_csv(INPUT_CONCORDANCE)

    print(f"Existing rows:     {len(concordance)}")

    # --------------------------------------------------------
    # Schema validation
    # --------------------------------------------------------

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in concordance.columns
    ]

    if missing_columns:
        raise ValueError(
            "Input concordance schema mismatch:\n"
            + "\n".join(missing_columns)
        )

    # --------------------------------------------------------
    # Existing duplicate validation
    # --------------------------------------------------------

    duplicate_mask = concordance.duplicated(
        subset=["state_raw", "district_raw"],
        keep=False,
    )

    if duplicate_mask.any():

        duplicates = concordance.loc[
            duplicate_mask,
            ["state_raw", "district_raw"],
        ].drop_duplicates()

        raise ValueError(
            "Existing duplicate keys detected:\n"
            + duplicates.to_string(index=False)
        )

    # --------------------------------------------------------
    # Build new mapping DataFrame
    # --------------------------------------------------------

    new_df = pd.DataFrame(NEW_MAPPINGS)

    if len(new_df) != 13:
        raise ValueError(
            f"Expected 13 new mappings, found {len(new_df)}."
        )

    # --------------------------------------------------------
    # Validate new mappings
    # --------------------------------------------------------

    if list(new_df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "New mapping schema does not match expected schema."
        )

    new_keys = list(
        zip(
            new_df["state_raw"],
            new_df["district_raw"],
        )
    )

    if len(set(new_keys)) != len(new_keys):
        raise ValueError(
            "Duplicate keys found within new mappings."
        )

    existing_keys = set(
        zip(
            concordance["state_raw"].astype(str).str.strip(),
            concordance["district_raw"].astype(str).str.strip(),
        )
    )

    overlapping = [
        key for key in new_keys
        if key in existing_keys
    ]

    if overlapping:
        raise ValueError(
            "Mappings already exist in concordance:\n"
            + "\n".join(
                f"{state} | {district}"
                for state, district in overlapping
            )
        )

    # --------------------------------------------------------
    # Validate state set
    # --------------------------------------------------------

    allowed_states = {
        "ARUNACHAL PRADESH",
        "ASSAM",
    }

    invalid_states = new_df[
        ~new_df["state_raw"].isin(allowed_states)
    ]

    if not invalid_states.empty:
        raise ValueError(
            "Unexpected states detected:\n"
            + invalid_states.to_string(index=False)
        )

    # --------------------------------------------------------
    # Validate method/confidence
    # --------------------------------------------------------

    if not (new_df["mapping_method"] == "NAME_ALIAS").all():
        raise ValueError(
            "All mappings must use NAME_ALIAS."
        )

    if not (new_df["confidence"] == "high").all():
        raise ValueError(
            "All mappings must have high confidence."
        )

    # --------------------------------------------------------
    # Append
    # --------------------------------------------------------

    result = pd.concat(
        [concordance, new_df],
        ignore_index=True,
    )

    expected_final = len(concordance) + 13

    if len(result) != expected_final:
        raise ValueError(
            f"Unexpected final row count: {len(result)}. "
            f"Expected {expected_final}."
        )

    # --------------------------------------------------------
    # Final duplicate validation
    # --------------------------------------------------------

    final_duplicates = result.duplicated(
        subset=["state_raw", "district_raw"],
        keep=False,
    )

    duplicate_count = int(final_duplicates.sum())

    if duplicate_count:
        duplicates = result.loc[
            final_duplicates,
            ["state_raw", "district_raw"],
        ].drop_duplicates()

        raise ValueError(
            "Duplicate keys after extension:\n"
            + duplicates.to_string(index=False)
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_CONCORDANCE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_CONCORDANCE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print(f"New rows:          {len(new_df)}")
    print(f"Final rows:        {len(result)}")
    print(f"Duplicate keys:    {duplicate_count}")

    print()
    print("New mappings:")
    print(
        new_df[
            [
                "state_raw",
                "district_raw",
                "matched_state",
                "matched_district",
                "mapping_method",
                "confidence",
            ]
        ].to_string(index=False)
    )

    print()
    print("VALIDATION")
    print("-" * 65)
    print("Schema validation:       PASS")
    print("New mapping count:       PASS")
    print("Duplicate validation:    PASS")
    print("State validation:        PASS")
    print("Method validation:       PASS")
    print("Confidence validation:   PASS")
    print("Final row count:         PASS")

    print()
    print(f"Output: {OUTPUT_CONCORDANCE}")


if __name__ == "__main__":
    main()
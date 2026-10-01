from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_arunachal_assam.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_andaman_chhattisgarh_haryana.csv"
)


REQUIRED_COLUMNS = [
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
    # =========================================================
    # A & N ISLANDS
    # =========================================================

    {
        "state_raw": "A & N ISLANDS",
        "district_raw": "ANDAMAN",
        "matched_state": "A & N ISLANDS",
        "matched_district": "Andamans",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA district-name variant; Census 2001 uses "
            "Andamans as the district name."
        ),
    },

    {
        "state_raw": "A & N ISLANDS",
        "district_raw": "NICOBAR",
        "matched_state": "A & N ISLANDS",
        "matched_district": "Nicobars",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA district-name variant; Census 2001 uses "
            "Nicobars as the district name."
        ),
    },

    # =========================================================
    # CHHATTISGARH
    # =========================================================

    {
        "state_raw": "CHHATTISGARH",
        "district_raw": "KAWARDHA (KABIRDHAM)",
        "matched_state": "CHHATTISGARH",
        "matched_district": "Kawardha",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA uses Kawardha (Kabirdham); Census 2001 "
            "uses Kawardha."
        ),
    },

    {
        "state_raw": "CHHATTISGARH",
        "district_raw": "MAHASMUND",
        "matched_state": "CHHATTISGARH",
        "matched_district": "Mahasamund",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling variant of Mahasamund."
        ),
    },

    {
        "state_raw": "CHHATTISGARH",
        "district_raw": "RAJ NANDGAON",
        "matched_state": "CHHATTISGARH",
        "matched_district": "Rajnandgaon",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spacing variant of Rajnandgaon."
        ),
    },

    {
        "state_raw": "CHHATTISGARH",
        "district_raw": "SARGUJA",
        "matched_state": "CHHATTISGARH",
        "matched_district": "Surguja",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling variant of Surguja."
        ),
    },

    # =========================================================
    # HARYANA
    # =========================================================

    {
        "state_raw": "HARYANA",
        "district_raw": "MAHENDRA GARH",
        "matched_state": "HARYANA",
        "matched_district": "Mahendragarh",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spacing variant of Mahendragarh."
        ),
    },

    {
        "state_raw": "HARYANA",
        "district_raw": "SONEPAT",
        "matched_state": "HARYANA",
        "matched_district": "Sonipat",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling variant; Census 2001 uses Sonipat."
        ),
    },

    {
        "state_raw": "HARYANA",
        "district_raw": "YAMUNA NAGAR",
        "matched_state": "HARYANA",
        "matched_district": "Yamunanagar",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spacing variant of Yamunanagar."
        ),
    },
]


def validate_schema(df: pd.DataFrame, name: str) -> None:
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


def main() -> None:

    print("AgriAdapt V4.5 — Extending District Concordance")
    print()

    # ---------------------------------------------------------
    # Load existing concordance
    # ---------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input concordance not found:\n{INPUT_PATH}"
        )

    base = pd.read_csv(
        INPUT_PATH,
        dtype=str,
    ).fillna("")

    new_df = pd.DataFrame(
        NEW_MAPPINGS,
        columns=REQUIRED_COLUMNS,
    )

    print(f"Existing rows: {len(base)}")
    print(f"New mappings:  {len(new_df)}")
    print()

    # ---------------------------------------------------------
    # Schema validation
    # ---------------------------------------------------------

    validate_schema(
        base,
        "Existing concordance",
    )

    validate_schema(
        new_df,
        "New mappings",
    )

    # ---------------------------------------------------------
    # Blank-field validation
    # ---------------------------------------------------------

    if (new_df[REQUIRED_COLUMNS] == "").any().any():

        bad_rows = new_df[
            (new_df[REQUIRED_COLUMNS] == "").any(axis=1)
        ]

        raise ValueError(
            "Blank fields detected in new mappings:\n"
            f"{bad_rows.to_string(index=False)}"
        )

    # ---------------------------------------------------------
    # Duplicate validation inside new batch
    # ---------------------------------------------------------

    key_columns = [
        "state_raw",
        "district_raw",
    ]

    if new_df[key_columns].duplicated().any():

        duplicates = new_df[
            new_df[key_columns].duplicated(
                keep=False
            )
        ]

        raise ValueError(
            "Duplicate raw keys inside new batch:\n"
            f"{duplicates.to_string(index=False)}"
        )

    # ---------------------------------------------------------
    # Existing-key validation
    # ---------------------------------------------------------

    existing_keys = set(
        zip(
            base["state_raw"]
            .str.upper()
            .str.strip(),

            base["district_raw"]
            .str.upper()
            .str.strip(),
        )
    )

    overlapping_rows = []

    for _, row in new_df.iterrows():

        key = (
            str(row["state_raw"])
            .upper()
            .strip(),

            str(row["district_raw"])
            .upper()
            .strip(),
        )

        if key in existing_keys:

            overlapping_rows.append(
                {
                    "state_raw": row["state_raw"],
                    "district_raw": row["district_raw"],
                }
            )

    if overlapping_rows:

        raise ValueError(
            "New mappings already exist "
            "in the concordance:\n"
            f"{pd.DataFrame(overlapping_rows).to_string(index=False)}"
        )

    # ---------------------------------------------------------
    # Append new mappings
    # ---------------------------------------------------------

    combined = pd.concat(
        [
            base[REQUIRED_COLUMNS],
            new_df[REQUIRED_COLUMNS],
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Final duplicate validation
    # ---------------------------------------------------------

    duplicate_mask = combined[
        key_columns
    ].duplicated(
        keep=False
    )

    if duplicate_mask.any():

        duplicates = combined[
            duplicate_mask
        ]

        raise ValueError(
            "Duplicate raw keys detected after merge:\n"
            f"{duplicates.to_string(index=False)}"
        )

    # ---------------------------------------------------------
    # Row-count validation
    # ---------------------------------------------------------

    expected_rows = (
        len(base) +
        len(new_df)
    )

    if len(combined) != expected_rows:

        raise ValueError(
            "Row-count validation failed. "
            f"Expected {expected_rows}, "
            f"got {len(combined)}."
        )

    # ---------------------------------------------------------
    # Mapping-method validation
    # ---------------------------------------------------------

    allowed_methods = {
        "NAME_ALIAS",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
    }

    invalid_methods = sorted(
        set(new_df["mapping_method"])
        - allowed_methods
    )

    if invalid_methods:

        raise ValueError(
            f"Invalid mapping methods: "
            f"{invalid_methods}"
        )

    # ---------------------------------------------------------
    # Confidence validation
    # ---------------------------------------------------------

    allowed_confidence = {
        "high",
        "medium",
        "low",
    }

    invalid_confidence = sorted(
        set(new_df["confidence"])
        - allowed_confidence
    )

    if invalid_confidence:

        raise ValueError(
            f"Invalid confidence values: "
            f"{invalid_confidence}"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("EXTENSION COMPLETE")
    print(
        f"Existing rows: {len(base)}"
    )
    print(
        f"New rows:      {len(new_df)}"
    )
    print(
        f"Final rows:    {len(combined)}"
    )
    print(
        "Duplicate keys: 0"
    )
    print()

    print("New mapping methods:")

    for method, count in (
        new_df["mapping_method"]
        .value_counts()
        .items()
    ):
        print(
            f"  {method}: {count}"
        )

    print()

    print("NEW MAPPINGS")

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
    print("Schema validation: PASS")
    print("Blank-field validation: PASS")
    print("Batch duplicate validation: PASS")
    print("Existing-key validation: PASS")
    print("Final duplicate validation: PASS")
    print("Row-count validation: PASS")
    print("Mapping-method validation: PASS")
    print("Confidence validation: PASS")

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
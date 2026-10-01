from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_dnh_uttaranchal.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_west_bengal.csv"
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
    # WEST BENGAL
    # =========================================================

    {
        "state_raw": "WEST BENGAL",
        "district_raw": "24 PARGANAS (NORTH)",
        "matched_state": "WEST BENGAL",
        "matched_district": "North Twenty Four Parganas",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA abbreviated district label corresponds "
            "to Census 2001 North Twenty Four Parganas."
        ),
    },

    {
        "state_raw": "WEST BENGAL",
        "district_raw": "24 PARGANAS (SOUTH)",
        "matched_state": "WEST BENGAL",
        "matched_district": "South Twenty Four Parganas",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA abbreviated district label corresponds "
            "to Census 2001 South Twenty Four Parganas."
        ),
    },

    {
        "state_raw": "WEST BENGAL",
        "district_raw": "BURDWAN",
        "matched_state": "WEST BENGAL",
        "matched_district": "Barddhaman",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA common spelling corresponds to Census "
            "2001 Barddhaman."
        ),
    },

    {
        "state_raw": "WEST BENGAL",
        "district_raw": "COOCH-BEHAR",
        "matched_state": "WEST BENGAL",
        "matched_district": "Koch Bihar",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling variant corresponds to Census "
            "2001 Koch Bihar."
        ),
    },

    {
        "state_raw": "WEST BENGAL",
        "district_raw": "DARJEELING",
        "matched_state": "WEST BENGAL",
        "matched_district": "Darjiling",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling variant corresponds to Census "
            "2001 Darjiling."
        ),
    },

    {
        "state_raw": "WEST BENGAL",
        "district_raw": "DINAJPUR(SOUTH)",
        "matched_state": "WEST BENGAL",
        "matched_district": "Dakshin Dinajpur",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA directional label corresponds to Census "
            "2001 Dakshin Dinajpur."
        ),
    },

    {
        "state_raw": "WEST BENGAL",
        "district_raw": "DINAJUR(NORTH)",
        "matched_state": "WEST BENGAL",
        "matched_district": "Uttar Dinajpur",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling/label variant corresponds to "
            "Census 2001 Uttar Dinajpur."
        ),
    },

    {
        "state_raw": "WEST BENGAL",
        "district_raw": "HOOGHLY",
        "matched_state": "WEST BENGAL",
        "matched_district": "Hugli",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling variant corresponds to Census "
            "2001 Hugli."
        ),
    },

    {
        "state_raw": "WEST BENGAL",
        "district_raw": "HOWRAH",
        "matched_state": "WEST BENGAL",
        "matched_district": "Haora",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling variant corresponds to Census "
            "2001 Haora."
        ),
    },

    {
        "state_raw": "WEST BENGAL",
        "district_raw": "MALDA",
        "matched_state": "WEST BENGAL",
        "matched_district": "Maldah",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling variant corresponds to Census "
            "2001 Maldah."
        ),
    },

    {
        "state_raw": "WEST BENGAL",
        "district_raw": "PURULIA",
        "matched_state": "WEST BENGAL",
        "matched_district": "Puruliya",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling variant corresponds to Census "
            "2001 Puruliya."
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


def main():

    print(
        "AgriAdapt V4.10 — Extending "
        "West Bengal Concordance"
    )
    print()

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
    # Blank validation
    # ---------------------------------------------------------

    if (new_df[REQUIRED_COLUMNS] == "").any().any():

        bad_rows = new_df[
            (new_df[REQUIRED_COLUMNS] == "").any(axis=1)
        ]

        raise ValueError(
            "Blank fields detected:\n"
            f"{bad_rows.to_string(index=False)}"
        )

    # ---------------------------------------------------------
    # Duplicate validation inside batch
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
            base["state_raw"].str.upper().str.strip(),
            base["district_raw"].str.upper().str.strip(),
        )
    )

    overlapping_rows = []

    for _, row in new_df.iterrows():

        key = (
            str(row["state_raw"]).upper().strip(),
            str(row["district_raw"]).upper().strip(),
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
            "New mappings already exist:\n"
            f"{pd.DataFrame(overlapping_rows).to_string(index=False)}"
        )

    # ---------------------------------------------------------
    # Append
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

    if combined[key_columns].duplicated().any():

        duplicates = combined[
            combined[key_columns].duplicated(
                keep=False
            )
        ]

        raise ValueError(
            "Duplicate raw keys after merge:\n"
            f"{duplicates.to_string(index=False)}"
        )

    # ---------------------------------------------------------
    # Row-count validation
    # ---------------------------------------------------------

    expected_rows = len(base) + len(new_df)

    if len(combined) != expected_rows:

        raise ValueError(
            f"Row-count validation failed. "
            f"Expected {expected_rows}, got {len(combined)}."
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
            f"Invalid mapping methods: {invalid_methods}"
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
            f"Invalid confidence values: {invalid_confidence}"
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
    print(f"Existing rows: {len(base)}")
    print(f"New rows:      {len(new_df)}")
    print(f"Final rows:    {len(combined)}")
    print("Duplicate keys: 0")
    print()

    print("New mapping methods:")

    for method, count in (
        new_df["mapping_method"]
        .value_counts()
        .items()
    ):
        print(f"  {method}: {count}")

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
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
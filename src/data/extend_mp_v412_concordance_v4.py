from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_jk_meghalaya.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_mp_v412.csv"
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

    {
        "state_raw": "MADHYA PRADESH",
        "district_raw": "CHINDWARA",
        "matched_state": "MADHYA PRADESH",
        "matched_district": "Chhindwara",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "Spelling variant of Census 2001 Chhindwara."
        ),
    },

    {
        "state_raw": "MADHYA PRADESH",
        "district_raw": "KHANDWA",
        "matched_state": "MADHYA PRADESH",
        "matched_district": "East Nimar",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA Khandwa corresponds to the Census 2001 "
            "East Nimar district label."
        ),
    },

    {
        "state_raw": "MADHYA PRADESH",
        "district_raw": "KHARGAON",
        "matched_state": "MADHYA PRADESH",
        "matched_district": "West Nimar",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA Khargaon corresponds to the Census 2001 "
            "West Nimar district label."
        ),
    },

    {
        "state_raw": "MADHYA PRADESH",
        "district_raw": "NARSIMPUR",
        "matched_state": "MADHYA PRADESH",
        "matched_district": "Narsimhapur",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "Spelling variant of Census 2001 Narsimhapur."
        ),
    },

    {
        "state_raw": "MADHYA PRADESH",
        "district_raw": "NEEMACH",
        "matched_state": "MADHYA PRADESH",
        "matched_district": "Neemuch",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "Spelling variant of Census 2001 Neemuch."
        ),
    },

    {
        "state_raw": "MADHYA PRADESH",
        "district_raw": "SHEOPUR KALA",
        "matched_state": "MADHYA PRADESH",
        "matched_district": "Sheopur",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling/name variant corresponding to "
            "Census 2001 Sheopur."
        ),
    },
]


def validate_schema(df, name):
    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


def main():

    print(
        "AgriAdapt V4.12 — Extending "
        "Madhya Pradesh Concordance"
    )
    print()

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

    validate_schema(
        base,
        "Existing concordance",
    )

    validate_schema(
        new_df,
        "New mappings",
    )

    if (new_df[REQUIRED_COLUMNS] == "").any().any():
        raise ValueError(
            "Blank fields detected."
        )

    key_columns = [
        "state_raw",
        "district_raw",
    ]

    if new_df[key_columns].duplicated().any():
        raise ValueError(
            "Duplicate raw keys inside new batch."
        )

    existing_keys = set(
        zip(
            base["state_raw"].str.upper().str.strip(),
            base["district_raw"].str.upper().str.strip(),
        )
    )

    for _, row in new_df.iterrows():

        key = (
            row["state_raw"].upper().strip(),
            row["district_raw"].upper().strip(),
        )

        if key in existing_keys:
            raise ValueError(
                f"Mapping already exists: {key}"
            )

    combined = pd.concat(
        [
            base[REQUIRED_COLUMNS],
            new_df[REQUIRED_COLUMNS],
        ],
        ignore_index=True,
    )

    if combined[key_columns].duplicated().any():
        raise ValueError(
            "Duplicate raw keys after merge."
        )

    if len(combined) != len(base) + len(new_df):
        raise ValueError(
            "Row-count validation failed."
        )

    allowed_methods = {
        "NAME_ALIAS",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
    }

    if not set(new_df["mapping_method"]).issubset(
        allowed_methods
    ):
        raise ValueError(
            "Invalid mapping method."
        )

    allowed_confidence = {
        "high",
        "medium",
        "low",
    }

    if not set(new_df["confidence"]).issubset(
        allowed_confidence
    ):
        raise ValueError(
            "Invalid confidence value."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("EXTENSION COMPLETE")
    print(f"Existing rows: {len(base)}")
    print(f"New rows:      {len(new_df)}")
    print(f"Final rows:    {len(combined)}")
    print("Duplicate keys: 0")
    print()

    print("New mapping methods:")
    print("  NAME_ALIAS: 6")
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
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
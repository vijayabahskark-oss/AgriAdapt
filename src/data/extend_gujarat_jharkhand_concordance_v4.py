from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_andaman_chhattisgarh_haryana.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_gujarat_jharkhand.csv"
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
    # GUJARAT
    # =========================================================

    {
        "state_raw": "GUJARAT",
        "district_raw": "AHMEDABAD",
        "matched_state": "GUJARAT",
        "matched_district": "Ahmadabad",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling variant; Census 2001 uses "
            "Ahmadabad."
        ),
    },

    {
        "state_raw": "GUJARAT",
        "district_raw": "BROACH",
        "matched_state": "GUJARAT",
        "matched_district": "Bharuch",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "Historical/common district name Broach corresponds "
            "to Census 2001 Bharuch."
        ),
    },

    {
        "state_raw": "GUJARAT",
        "district_raw": "DANGS",
        "matched_state": "GUJARAT",
        "matched_district": "The Dangs",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA abbreviated district name; Census 2001 "
            "uses The Dangs."
        ),
    },

    {
        "state_raw": "GUJARAT",
        "district_raw": "JUNAGARH",
        "matched_state": "GUJARAT",
        "matched_district": "Junagadh",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling variant of Junagadh."
        ),
    },

    {
        "state_raw": "GUJARAT",
        "district_raw": "KUTCH",
        "matched_state": "GUJARAT",
        "matched_district": "Kachchh",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "Common name Kutch corresponds to Census 2001 "
            "district name Kachchh."
        ),
    },

    {
        "state_raw": "GUJARAT",
        "district_raw": "MEHSANA",
        "matched_state": "GUJARAT",
        "matched_district": "Mahesana",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling variant of Mahesana."
        ),
    },

    {
        "state_raw": "GUJARAT",
        "district_raw": "SABARKANTHA",
        "matched_state": "GUJARAT",
        "matched_district": "Sabar Kantha",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spacing variant of Sabar Kantha."
        ),
    },


    # =========================================================
    # JHARKHAND
    # =========================================================

    {
        "state_raw": "JHARKHAND",
        "district_raw": "EAST SINGHBHUM",
        "matched_state": "JHARKHAND",
        "matched_district": "Purbi Singhbhum",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA English directional name corresponds to "
            "Census 2001 Purbi Singhbhum."
        ),
    },

    {
        "state_raw": "JHARKHAND",
        "district_raw": "JAMTARA",
        "matched_state": "JHARKHAND",
        "matched_district": "Jamtara",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "Direct district-name concordance."
        ),
    },

    {
        "state_raw": "JHARKHAND",
        "district_raw": "KODERMA",
        "matched_state": "JHARKHAND",
        "matched_district": "Kodarma",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling variant of Census 2001 Kodarma."
        ),
    },

    {
        "state_raw": "JHARKHAND",
        "district_raw": "LATEHAR",
        "matched_state": "JHARKHAND",
        "matched_district": "Latehar",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "Direct district-name concordance."
        ),
    },

    {
        "state_raw": "JHARKHAND",
        "district_raw": "PAKUR",
        "matched_state": "JHARKHAND",
        "matched_district": "Pakaur",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA spelling variant of Census 2001 Pakaur."
        ),
    },

    {
        "state_raw": "JHARKHAND",
        "district_raw": "SERAIKELA",
        "matched_state": "JHARKHAND",
        "matched_district": "Saraikela-Kharsawan",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA shortened district name corresponding "
            "to Census 2001 Saraikela-Kharsawan."
        ),
    },

    {
        "state_raw": "JHARKHAND",
        "district_raw": "SIMDEGA",
        "matched_state": "JHARKHAND",
        "matched_district": "Simdega",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "Direct district-name concordance."
        ),
    },

    {
        "state_raw": "JHARKHAND",
        "district_raw": "WEST SINGHBHUM",
        "matched_state": "JHARKHAND",
        "matched_district": "Pashchimi Singhbhum",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district attributes",
        "confidence": "high",
        "notes": (
            "AGRIDATA English directional name corresponds to "
            "Census 2001 Pashchimi Singhbhum."
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
        "AgriAdapt V4.6 — Extending Gujarat/Jharkhand Concordance"
    )
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
    # Append mappings
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
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
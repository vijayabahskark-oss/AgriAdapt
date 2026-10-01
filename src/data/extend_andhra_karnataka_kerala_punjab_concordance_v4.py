from pathlib import Path
import pandas as pd


# ============================================================
# AgriAdapt V4.3
# Andhra Pradesh + Karnataka + Kerala + Punjab
# District Name Concordance Extension
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_CONCORDANCE = (
    BASE_DIR
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_orissa.csv"
)

OUTPUT_CONCORDANCE = (
    BASE_DIR
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_south_north.csv"
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


# ------------------------------------------------------------
# 16 controlled mappings
# ------------------------------------------------------------

NEW_MAPPINGS = [
    # Andhra Pradesh
    {
        "state_raw": "ANDHRA PRADESH",
        "district_raw": "MAHABOOBNAGAR",
        "matched_state": "ANDHRA PRADESH",
        "matched_district": "Mahbubnagar",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Legacy spelling variant of Mahbubnagar.",
    },
    {
        "state_raw": "ANDHRA PRADESH",
        "district_raw": "RANGAREDDY",
        "matched_state": "ANDHRA PRADESH",
        "matched_district": "Rangareddy",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "District name spelling standardized to Census polygon name.",
    },

    # Karnataka
    {
        "state_raw": "KARNATAKA",
        "district_raw": "CHAMARAJANNAGAR",
        "matched_state": "KARNATAKA",
        "matched_district": "Chamarajanagar",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Legacy spelling variant of Chamarajanagar.",
    },
    {
        "state_raw": "KARNATAKA",
        "district_raw": "DAKSHINAKANNADA",
        "matched_state": "KARNATAKA",
        "matched_district": "Dakshina Kannada",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Spacing-normalized district name.",
    },
    {
        "state_raw": "KARNATAKA",
        "district_raw": "DAVANGERE",
        "matched_state": "KARNATAKA",
        "matched_district": "Davanagere",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Legacy spelling variant of Davanagere.",
    },
    {
        "state_raw": "KARNATAKA",
        "district_raw": "KODAGU(COORG)",
        "matched_state": "KARNATAKA",
        "matched_district": "Kodagu",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Coorg is the historical/common name for Kodagu.",
    },
    {
        "state_raw": "KARNATAKA",
        "district_raw": "UTTARAKANNADA",
        "matched_state": "KARNATAKA",
        "matched_district": "Uttara Kannada",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Spacing-normalized district name.",
    },

    # Kerala
    {
        "state_raw": "KERALA",
        "district_raw": "KASARGOD",
        "matched_state": "KERALA",
        "matched_district": "Kasaragod",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Legacy spelling variant of Kasaragod.",
    },
    {
        "state_raw": "KERALA",
        "district_raw": "QUILON",
        "matched_state": "KERALA",
        "matched_district": "Kollam",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Quilon is the historical English name corresponding to Kollam.",
    },
    {
        "state_raw": "KERALA",
        "district_raw": "TRIVANDRUM",
        "matched_state": "KERALA",
        "matched_district": "Thiruvananthapuram",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Trivandrum is the common historical English name corresponding to Thiruvananthapuram.",
    },
    {
        "state_raw": "KERALA",
        "district_raw": "WYNAD",
        "matched_state": "KERALA",
        "matched_district": "Wayanad",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Legacy spelling variant of Wayanad.",
    },

    # Punjab
    {
        "state_raw": "PUNJAB",
        "district_raw": "BHATINDA",
        "matched_state": "PUNJAB",
        "matched_district": "Bathinda",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Legacy spelling variant of Bathinda.",
    },
    {
        "state_raw": "PUNJAB",
        "district_raw": "MANSHA",
        "matched_state": "PUNJAB",
        "matched_district": "Mansa",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Legacy spelling variant of Mansa.",
    },
    {
        "state_raw": "PUNJAB",
        "district_raw": "MUKATSAR",
        "matched_state": "PUNJAB",
        "matched_district": "Muktsar",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Legacy spelling variant of Muktsar.",
    },
    {
        "state_raw": "PUNJAB",
        "district_raw": "N.SHAHAR",
        "matched_state": "PUNJAB",
        "matched_district": "Nawanshahr",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Abbreviated legacy district name corresponding to Nawanshahr.",
    },
    {
        "state_raw": "PUNJAB",
        "district_raw": "ROPAR(RUPNAGAR)",
        "matched_state": "PUNJAB",
        "matched_district": "Rupnagar",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 district name concordance",
        "confidence": "high",
        "notes": "Ropar is the historical/common name corresponding to Rupnagar.",
    },
]


def main():

    print("AgriAdapt V4.3 — South/North Name Concordance Extension")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Check input
    # --------------------------------------------------------

    if not INPUT_CONCORDANCE.exists():
        raise FileNotFoundError(
            f"Input concordance not found:\n{INPUT_CONCORDANCE}"
        )

    concordance = pd.read_csv(INPUT_CONCORDANCE)

    print(f"Existing rows:     {len(concordance)}")

    # --------------------------------------------------------
    # 2. Validate schema
    # --------------------------------------------------------

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in concordance.columns
    ]

    if missing_columns:
        raise ValueError(
            "Input concordance schema mismatch.\n"
            "Missing columns:\n"
            + "\n".join(missing_columns)
        )

    # --------------------------------------------------------
    # 3. Validate existing duplicate keys
    # --------------------------------------------------------

    duplicate_existing = concordance.duplicated(
        subset=["state_raw", "district_raw"],
        keep=False,
    )

    if duplicate_existing.any():
        duplicates = concordance.loc[
            duplicate_existing,
            ["state_raw", "district_raw"],
        ].drop_duplicates()

        raise ValueError(
            "Duplicate keys already exist in input concordance:\n"
            + duplicates.to_string(index=False)
        )

    # --------------------------------------------------------
    # 4. Convert new mappings to DataFrame
    # --------------------------------------------------------

    new_df = pd.DataFrame(NEW_MAPPINGS)

    if len(new_df) != 16:
        raise ValueError(
            f"Expected 16 new mappings, found {len(new_df)}."
        )

    # --------------------------------------------------------
    # 5. Validate new mapping schema
    # --------------------------------------------------------

    missing_new_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in new_df.columns
    ]

    if missing_new_columns:
        raise ValueError(
            "New mapping schema mismatch:\n"
            + "\n".join(missing_new_columns)
        )

    # --------------------------------------------------------
    # 6. Validate new mapping keys
    # --------------------------------------------------------

    existing_keys = set(
        zip(
            concordance["state_raw"].astype(str).str.strip(),
            concordance["district_raw"].astype(str).str.strip(),
        )
    )

    new_keys = list(
        zip(
            new_df["state_raw"].astype(str).str.strip(),
            new_df["district_raw"].astype(str).str.strip(),
        )
    )

    if len(set(new_keys)) != len(new_keys):
        raise ValueError(
            "Duplicate keys detected within new mappings."
        )

    already_existing = [
        key for key in new_keys
        if key in existing_keys
    ]

    if already_existing:
        raise ValueError(
            "Some new mappings already exist in the concordance:\n"
            + "\n".join(
                f"{state} | {district}"
                for state, district in already_existing
            )
        )

    # --------------------------------------------------------
    # 7. Validate all mappings
    # --------------------------------------------------------

    allowed_states = {
        "ANDHRA PRADESH",
        "KARNATAKA",
        "KERALA",
        "PUNJAB",
    }

    invalid_states = new_df[
        ~new_df["state_raw"].isin(allowed_states)
    ]

    if not invalid_states.empty:
        raise ValueError(
            "Unexpected state found in new mappings:\n"
            + invalid_states.to_string(index=False)
        )

    invalid_methods = new_df[
        new_df["mapping_method"] != "NAME_ALIAS"
    ]

    if not invalid_methods.empty:
        raise ValueError(
            "All mappings in this batch must use NAME_ALIAS."
        )

    invalid_confidence = new_df[
        new_df["confidence"] != "high"
    ]

    if not invalid_confidence.empty:
        raise ValueError(
            "All mappings in this batch must have high confidence."
        )

    # --------------------------------------------------------
    # 8. Append
    # --------------------------------------------------------

    result = pd.concat(
        [concordance, new_df],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 9. Validate final result
    # --------------------------------------------------------

    expected_final_rows = len(concordance) + 16

    if len(result) != expected_final_rows:
        raise ValueError(
            f"Unexpected final row count: {len(result)}. "
            f"Expected {expected_final_rows}."
        )

    duplicate_final = result.duplicated(
        subset=["state_raw", "district_raw"],
        keep=False,
    )

    duplicate_count = int(duplicate_final.sum())

    if duplicate_count:
        duplicates = result.loc[
            duplicate_final,
            ["state_raw", "district_raw"],
        ].drop_duplicates()

        raise ValueError(
            "Duplicate keys detected after extension:\n"
            + duplicates.to_string(index=False)
        )

    # --------------------------------------------------------
    # 10. Save
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
    # 11. Print result
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
    print("Final row count:          PASS")

    print()
    print(f"Output: {OUTPUT_CONCORDANCE}")


if __name__ == "__main__":
    main()
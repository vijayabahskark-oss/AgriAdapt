from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_v415_audit.csv"
)

OUTPUT_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "unresolved_geography_v416.csv"
)

EXPECTED_UNMATCHED = 32


# -------------------------------------------------------------
# Conservative classification.
#
# IMPORTANT:
# These are classification labels, NOT new geographic mappings.
# No Census polygon is assigned here.
# -------------------------------------------------------------

CLASSIFICATIONS = {

    # Assam: later administrative structure / subdivision
    ("ASSAM", "BASKA"):
        (
            "SUBDISTRICT_OR_SUBDIVISION",
            "Requires historical administrative concordance "
            "before assignment to a 2001 district polygon."
        ),

    ("ASSAM", "CHIRANG"):
        (
            "LATER_CREATED_DISTRICT",
            "Requires historical administrative concordance "
            "before assignment to a 2001 district polygon."
        ),

    ("ASSAM", "KAMRUP (M)"):
        (
            "LATER_CREATED_DISTRICT",
            "Kamrup Metropolitan requires historical "
            "boundary treatment."
        ),

    ("ASSAM", "UDALGURI"):
        (
            "SUBDISTRICT_OR_SUBDIVISION",
            "Requires historical administrative concordance "
            "before assignment to a 2001 district polygon."
        ),

    # Bihar / Jharkhand historical state structure
    ("BIHAR", "SINGHBHUM"):
        (
            "HISTORICAL_ADMINISTRATIVE_MAPPING",
            "Historical Bihar/Jharkhand district structure; "
            "source does not distinguish East/West Singhbhum."
        ),

    # Gujarat
    ("GUJARAT", "TAPI"):
        (
            "LATER_CREATED_DISTRICT",
            "Requires historical boundary concordance."
        ),

    # Haryana
    ("HARYANA", "MEWAT"):
        (
            "LATER_CREATED_DISTRICT",
            "Requires historical boundary concordance."
        ),

    ("HARYANA", "PALWAL"):
        (
            "LATER_CREATED_DISTRICT",
            "Requires historical boundary concordance."
        ),

    # Karnataka
    ("KARNATAKA", "CHICKBALLAPUR"):
        (
            "LATER_CREATED_DISTRICT",
            "Requires historical boundary concordance."
        ),

    ("KARNATAKA", "RAMANAGAR"):
        (
            "LATER_CREATED_DISTRICT",
            "Requires historical boundary concordance."
        ),

    # Madhya Pradesh
    ("MADHYA PRADESH", "ANUPPUR"):
        (
            "LATER_CREATED_DISTRICT",
            "Requires historical boundary concordance."
        ),

    ("MADHYA PRADESH", "ASHOK NAGAR"):
        (
            "LATER_CREATED_DISTRICT",
            "Requires historical boundary concordance."
        ),

    ("MADHYA PRADESH", "BURHANPUR"):
        (
            "HISTORICAL_ADMINISTRATIVE_MAPPING",
            "Burhanpur requires historical district-level "
            "boundary treatment; do not equate it with "
            "the 2001 East Nimar polygon without evidence."
        ),

    # Uttar Pradesh historical names
    ("UTTAR PRADESH",
     "AMETHI (CHHATRAPTI SAHUJI MAHARAJ NGR)"):
        (
            "HISTORICAL_ADMINISTRATIVE_MAPPING",
            "Historical district-name / administrative change "
            "requires explicit source concordance."
        ),

    ("UTTAR PRADESH",
     "AMROHA (J.B.PHULE NGR.)"):
        (
            "HISTORICAL_ADMINISTRATIVE_MAPPING",
            "Historical district-name change requires "
            "explicit source concordance."
        ),

    ("UTTAR PRADESH",
     "KASGANJ (KASHIRAM NAGAR)"):
        (
            "HISTORICAL_ADMINISTRATIVE_MAPPING",
            "Historical district-name change requires "
            "explicit source concordance."
        ),

    ("UTTAR PRADESH", "MAHAMAYA NAGAR"):
        (
            "HISTORICAL_ADMINISTRATIVE_MAPPING",
            "Historical district-name change requires "
            "explicit source concordance."
        ),

    ("UTTAR PRADESH", "RAMABAI NAGAR"):
        (
            "HISTORICAL_ADMINISTRATIVE_MAPPING",
            "Historical district-name change requires "
            "explicit source concordance."
        ),

    # West Bengal
    ("WEST BENGAL", "MIDNAPUR ( EAST)"):
        (
            "AMBIGUOUS_GEOGRAPHY",
            "Census-era boundary relationship requires "
            "historical concordance."
        ),

    ("WEST BENGAL", "MIDNAPUR (WEST)"):
        (
            "AMBIGUOUS_GEOGRAPHY",
            "Census-era boundary relationship requires "
            "historical concordance."
        ),

    ("WEST BENGAL", "WEST DINAJPUR"):
        (
            "HISTORICAL_ADMINISTRATIVE_MAPPING",
            "Legacy Dinajpur label requires explicit "
            "historical boundary concordance."
        ),

    # Andaman & Nicobar
    ("A & N ISLANDS", "A & N ISLANDS"):
        (
            "GENERIC_STATE_LEVEL_LABEL",
            "Generic island-level label does not identify "
            "a single district polygon."
        ),

    ("A & N ISLANDS", "NORTH AND MIDDLE ANDAMAN"):
        (
            "SUBDISTRICT_OR_SUBDIVISION",
            "Requires historical administrative concordance."
        ),

    ("A & N ISLANDS", "SOUTH ANDAMAN"):
        (
            "SUBDISTRICT_OR_SUBDIVISION",
            "Requires historical administrative concordance."
        ),

    # Arunachal Pradesh
    ("ARUNACHAL PRADESH", "ANJAW"):
        (
            "LATER_CREATED_DISTRICT",
            "Requires historical boundary concordance."
        ),

    ("ARUNACHAL PRADESH", "D/VALLEY"):
        (
            "LATER_CREATED_DISTRICT",
            "Requires historical boundary concordance."
        ),

    ("ARUNACHAL PRADESH", "KURUNG KAMEY"):
        (
            "LATER_CREATED_DISTRICT",
            "Requires historical boundary concordance."
        ),

    ("ARUNACHAL PRADESH", "LOWER DIBANG VALLEY"):
        (
            "LATER_CREATED_DISTRICT",
            "Requires historical boundary concordance."
        ),

    # Nagaland
    ("NAGALAND", "NAGALAND"):
        (
            "GENERIC_STATE_LEVEL_LABEL",
            "State-level label does not identify a district."
        ),

    # Mizoram
    ("MIZORAM", "EAST AIZAWL"):
        (
            "SUBDISTRICT_OR_SUBDIVISION",
            "Does not directly identify a Census 2001 "
            "district polygon."
        ),

    ("MIZORAM", "WEST AIZAWL"):
        (
            "SUBDISTRICT_OR_SUBDIVISION",
            "Does not directly identify a Census 2001 "
            "district polygon."
        ),

    # Goa
    ("GOA", "Goa"):
        (
            "AMBIGUOUS_GEOGRAPHY",
            "Raw value does not distinguish North Goa "
            "from South Goa."
        ),
}


def norm(value):
    text = str(value).strip().upper()
    text = text.replace("&", "AND")
    text = " ".join(text.split())
    return text


def main():

    print(
        "AgriAdapt V4.16 — Unresolved Geography Audit"
    )
    print()

    df = pd.read_csv(
        INPUT_AUDIT,
        dtype=str
    ).fillna("")

    unmatched = df[
        df["mapping_status"] == "unmatched"
    ].copy()

    print(
        f"Unmatched records found: {len(unmatched)}"
    )

    if len(unmatched) != EXPECTED_UNMATCHED:
        raise ValueError(
            f"Expected {EXPECTED_UNMATCHED} unmatched "
            f"records, found {len(unmatched)}."
        )

    classifications = []

    for _, row in unmatched.iterrows():

        state = norm(row["state"])
        district = norm(row["district_raw"])

        key = (state, district)

        # CLASSIFICATIONS contains source-form keys.
        # Build a normalized lookup once so punctuation such as
        # "&" cannot cause false "classification not defined"
        # errors.

        normalized_classifications = {
            (
                norm(class_state),
                norm(class_district),
            ): value
            for (class_state, class_district), value
            in CLASSIFICATIONS.items()
        }

        if key not in normalized_classifications:
            raise ValueError(
                "No V4.16 classification defined for:\n"
                f"{row['state']} | "
                f"{row['district_raw']}\n"
                f"Normalized key: {key}"
            )

        category, rationale = normalized_classifications[key]

        classifications.append(
            {
                "state": row["state"],
                "district_raw": row["district_raw"],
                "classification": category,
                "rationale": rationale,
                "mapping_status": "unresolved",
                "matched_state": "",
                "matched_district": "",
            }
        )

    result = pd.DataFrame(classifications)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if len(result) != EXPECTED_UNMATCHED:
        raise ValueError(
            "Classification row count mismatch."
        )

    if result[
        ["state", "district_raw"]
    ].duplicated().any():
        raise ValueError(
            "Duplicate state/district pairs detected."
        )

    valid_categories = {
        "SAFE_TO_MAP",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "LATER_CREATED_DISTRICT",
        "SUBDISTRICT_OR_SUBDIVISION",
        "AMBIGUOUS_GEOGRAPHY",
        "GENERIC_STATE_LEVEL_LABEL",
        "REQUIRES_SOURCE_REVIEW",
    }

    invalid = set(
        result["classification"]
    ) - valid_categories

    if invalid:
        raise ValueError(
            f"Invalid classification values: {invalid}"
        )

    if result[
        ["matched_state", "matched_district"]
    ].astype(str).apply(
        lambda col: col.str.strip().eq("")
    ).all(axis=None) is False:
        raise ValueError(
            "V4.16 must not contain geographic mappings."
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_AUDIT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_AUDIT,
        index=False
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("CLASSIFICATION COMPLETE")
    print(
        f"Records classified: {len(result)}"
    )
    print()

    print("CLASSIFICATION SUMMARY")

    counts = (
        result["classification"]
        .value_counts()
        .sort_index()
    )

    for category, count in counts.items():
        print(
            f"  {category}: {count}"
        )

    print()
    print("VALIDATION")
    print("Input unmatched-count validation: PASS")
    print("Classification coverage validation: PASS")
    print("Duplicate-pair validation: PASS")
    print("Category validation: PASS")
    print("No-new-mapping validation: PASS")
    print("Row-count validation: PASS")
    print()

    print(
        f"Output: {OUTPUT_AUDIT}"
    )


if __name__ == "__main__":
    main()
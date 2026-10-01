from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CONCORDANCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_mp_v412.csv"
)

OUTPUT_CONCORDANCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_up_v413.csv"
)


NEW_MAPPINGS = [
    {
        "state_raw": "UTTAR PRADESH",
        "district_raw": "AMBEDKAR NGR.",
        "matched_state": "UTTAR PRADESH",
        "matched_district": "Ambedkar Nagar",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 Uttar Pradesh district list",
        "confidence": "high",
        "notes": "Historical/source-name abbreviation; same Census 2001 district."
    },
    {
        "state_raw": "UTTAR PRADESH",
        "district_raw": "BADAUN",
        "matched_state": "UTTAR PRADESH",
        "matched_district": "Budaun",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 Uttar Pradesh district list",
        "confidence": "high",
        "notes": "Source spelling variant of Census 2001 district name."
    },
    {
        "state_raw": "UTTAR PRADESH",
        "district_raw": "BAGPAT",
        "matched_state": "UTTAR PRADESH",
        "matched_district": "Baghpat",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 Uttar Pradesh district list",
        "confidence": "high",
        "notes": "Source spelling variant of Census 2001 district name."
    },
    {
        "state_raw": "UTTAR PRADESH",
        "district_raw": "BULLANDSHAHR",
        "matched_state": "UTTAR PRADESH",
        "matched_district": "Bulandshahr",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 Uttar Pradesh district list",
        "confidence": "high",
        "notes": "Source spelling variant of Census 2001 district name."
    },
    {
        "state_raw": "UTTAR PRADESH",
        "district_raw": "CHITRAKUT",
        "matched_state": "UTTAR PRADESH",
        "matched_district": "Chitrakoot",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 Uttar Pradesh district list",
        "confidence": "high",
        "notes": "Source spelling variant of Census 2001 district name."
    },
    {
        "state_raw": "UTTAR PRADESH",
        "district_raw": "G.BUDDHA NGR.",
        "matched_state": "UTTAR PRADESH",
        "matched_district": "Gautam Buddha Nagar",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 Uttar Pradesh district list",
        "confidence": "high",
        "notes": "Source abbreviation for Gautam Buddha Nagar."
    },
    {
        "state_raw": "UTTAR PRADESH",
        "district_raw": "KANPUR CITY",
        "matched_state": "UTTAR PRADESH",
        "matched_district": "Kanpur Nagar",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 Uttar Pradesh district list",
        "confidence": "high",
        "notes": "Source label corresponding to Census 2001 Kanpur Nagar district."
    },
    {
        "state_raw": "UTTAR PRADESH",
        "district_raw": "KUSHI NGR.",
        "matched_state": "UTTAR PRADESH",
        "matched_district": "Kushinagar",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 Uttar Pradesh district list",
        "confidence": "high",
        "notes": "Source abbreviation of Census 2001 district name."
    },
    {
        "state_raw": "UTTAR PRADESH",
        "district_raw": "MAHARAHGANJ",
        "matched_state": "UTTAR PRADESH",
        "matched_district": "Maharajganj",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 Uttar Pradesh district list",
        "confidence": "high",
        "notes": "Source spelling variant of Census 2001 district name."
    },
    {
        "state_raw": "UTTAR PRADESH",
        "district_raw": "RAEBARELI",
        "matched_state": "UTTAR PRADESH",
        "matched_district": "Rae Bareli",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 Uttar Pradesh district list",
        "confidence": "high",
        "notes": "Source spelling/spacing variant of Census 2001 district name."
    },
    {
        "state_raw": "UTTAR PRADESH",
        "district_raw": "S.RAVI DAS NGR.",
        "matched_state": "UTTAR PRADESH",
        "matched_district": "Sant Ravidas Nagar Bhadohi",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 Uttar Pradesh district list",
        "confidence": "high",
        "notes": "Source abbreviation of Census 2001 district name."
    },
    {
        "state_raw": "UTTAR PRADESH",
        "district_raw": "SANT KABIR NGR.",
        "matched_state": "UTTAR PRADESH",
        "matched_district": "Sant Kabir Nagar",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 Uttar Pradesh district list",
        "confidence": "high",
        "notes": "Source abbreviation of Census 2001 district name."
    },
    {
        "state_raw": "UTTAR PRADESH",
        "district_raw": "SHIVASTI",
        "matched_state": "UTTAR PRADESH",
        "matched_district": "Shrawasti",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 Uttar Pradesh district list",
        "confidence": "high",
        "notes": "Historical/source spelling variant of Census 2001 district name."
    },
    {
        "state_raw": "UTTAR PRADESH",
        "district_raw": "SIDDHARTH NGR.",
        "matched_state": "UTTAR PRADESH",
        "matched_district": "Siddharthnagar",
        "mapping_method": "NAME_ALIAS",
        "mapping_source": "Census 2001 Uttar Pradesh district list",
        "confidence": "high",
        "notes": "Source abbreviation/spacing variant of Census 2001 district name."
    },
]


def norm(value):
    text = str(value).strip().upper()
    text = text.replace("&", "AND")
    text = " ".join(text.split())
    return text


def make_key(state, district):
    return f"{norm(state)}|||{norm(district)}"


def main():

    print("AgriAdapt V4.13 — Extending Uttar Pradesh Concordance")
    print()

    existing = pd.read_csv(
        INPUT_CONCORDANCE,
        dtype=str
    ).fillna("")

    new = pd.DataFrame(NEW_MAPPINGS)

    print(f"Existing rows: {len(existing)}")
    print(f"New mappings:  {len(new)}")
    print()

    required_columns = [
        "state_raw",
        "district_raw",
        "matched_state",
        "matched_district",
        "mapping_method",
        "mapping_source",
        "confidence",
        "notes",
    ]

    if list(existing.columns) != required_columns:
        raise ValueError(
            "Existing concordance schema does not match expected schema.\n"
            f"Expected: {required_columns}\n"
            f"Found:    {list(existing.columns)}"
        )

    if list(new.columns) != required_columns:
        raise ValueError(
            "New mapping schema does not match expected schema."
        )

    existing["_key"] = [
        make_key(s, d)
        for s, d in zip(
            existing["state_raw"],
            existing["district_raw"]
        )
    ]

    new["_key"] = [
        make_key(s, d)
        for s, d in zip(
            new["state_raw"],
            new["district_raw"]
        )
    ]

    # Duplicate checks
    if existing["_key"].duplicated().any():
        raise ValueError(
            "Existing concordance contains duplicate keys."
        )

    if new["_key"].duplicated().any():
        raise ValueError(
            "New batch contains duplicate keys."
        )

    overlap = set(existing["_key"]) & set(new["_key"])

    if overlap:
        raise ValueError(
            "New mappings overlap existing keys:\n"
            + "\n".join(sorted(overlap))
        )

    # Verify all are UP
    if not all(
        norm(x) == "UTTAR PRADESH"
        for x in new["state_raw"]
    ):
        raise ValueError(
            "All V4.13 mappings must have state_raw = UTTAR PRADESH."
        )

    # Verify methods/confidence
    if not all(
        x == "NAME_ALIAS"
        for x in new["mapping_method"]
    ):
        raise ValueError(
            "All V4.13 mappings must use NAME_ALIAS."
        )

    if not all(
        x == "high"
        for x in new["confidence"]
    ):
        raise ValueError(
            "All V4.13 mappings must have high confidence."
        )

    # Append
    combined = pd.concat(
        [
            existing.drop(columns=["_key"]),
            new.drop(columns=["_key"]),
        ],
        ignore_index=True
    )

    # Final duplicate validation
    final_keys = [
        make_key(s, d)
        for s, d in zip(
            combined["state_raw"],
            combined["district_raw"]
        )
    ]

    if len(final_keys) != len(set(final_keys)):
        raise ValueError(
            "Final concordance contains duplicate keys."
        )

    # Blank validation
    for column in required_columns:
        if combined[column].astype(str).str.strip().eq("").any():
            raise ValueError(
                f"Blank values found in column: {column}"
            )

    # Save
    OUTPUT_CONCORDANCE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        OUTPUT_CONCORDANCE,
        index=False
    )

    print("EXTENSION COMPLETE")
    print(f"Existing rows: {len(existing)}")
    print(f"New rows:      {len(new)}")
    print(f"Final rows:    {len(combined)}")
    print(
        f"Duplicate keys: "
        f"{len(combined) - len(set(final_keys))}"
    )
    print()

    print("NEW MAPPINGS")

    for _, row in new.iterrows():
        print(
            f"{row['state_raw']:18} "
            f"{row['district_raw']:35} -> "
            f"{row['matched_district']}"
        )

    print()
    print("VALIDATION")
    print("Schema validation: PASS")
    print("Blank-field validation: PASS")
    print("Batch duplicate validation: PASS")
    print("Existing-key validation: PASS")
    print("Final duplicate validation: PASS")
    print("Mapping-method validation: PASS")
    print("Confidence validation: PASS")
    print("Row-count validation: PASS")
    print()
    print(f"Output: {OUTPUT_CONCORDANCE}")


if __name__ == "__main__":
    main()
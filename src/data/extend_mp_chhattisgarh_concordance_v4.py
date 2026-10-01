from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data" / "processed" / "v4" / "district"
    / "historical_district_concordance_expanded.csv"
)

OUTPUT = (
    ROOT / "data" / "processed" / "v4" / "district"
    / "historical_district_concordance_mp_chhattisgarh.csv"
)

existing = pd.read_csv(INPUT)

expected_columns = [
    "state_raw",
    "district_raw",
    "matched_state",
    "matched_district",
    "mapping_method",
    "mapping_source",
    "confidence",
    "notes",
]

if existing.columns.tolist() != expected_columns:
    raise ValueError(
        "Unexpected concordance schema.\n"
        f"Expected: {expected_columns}\n"
        f"Found:    {existing.columns.tolist()}"
    )

new_rows = [
    (
        "MADHYA PRADESH", "BASTAR",
        "CHHATTISGARH", "Bastar",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical state-label relationship; Bastar was represented under Chhattisgarh in the Census 2001 reference geography.",
    ),
    (
        "MADHYA PRADESH", "BILASPUR",
        "CHHATTISGARH", "Bilaspur",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical state-label relationship; Bilaspur was represented under Chhattisgarh in the Census 2001 reference geography.",
    ),
    (
        "MADHYA PRADESH", "DANTEWARA",
        "CHHATTISGARH", "Dantewada",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical state-label relationship; AGRIDATA spelling corresponds to Census 2001 Dantewada.",
    ),
    (
        "MADHYA PRADESH", "DHAMANTARI",
        "CHHATTISGARH", "Dhamtari",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical state-label relationship; AGRIDATA spelling corresponds to Census 2001 Dhamtari.",
    ),
    (
        "MADHYA PRADESH", "DURG",
        "CHHATTISGARH", "Durg",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical state-label relationship; Durg was represented under Chhattisgarh in the Census 2001 reference geography.",
    ),
    (
        "MADHYA PRADESH", "JANGIR",
        "CHHATTISGARH", "Janjgir-Champa",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical district-name relationship; AGRIDATA JANGIR corresponds to Janjgir-Champa.",
    ),
    (
        "MADHYA PRADESH", "JUSHPUR",
        "CHHATTISGARH", "Jashpur",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical district-name relationship; AGRIDATA spelling corresponds to Jashpur.",
    ),
    (
        "MADHYA PRADESH", "KANKAR",
        "CHHATTISGARH", "Kanker",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical district-name relationship; AGRIDATA spelling corresponds to Kanker.",
    ),
    (
        "MADHYA PRADESH", "KORBA",
        "CHHATTISGARH", "Korba",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical state-label relationship; Korba was represented under Chhattisgarh in the Census 2001 reference geography.",
    ),
    (
        "MADHYA PRADESH", "KORIYA",
        "CHHATTISGARH", "Koriya",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical state-label relationship; Koriya was represented under Chhattisgarh in the Census 2001 reference geography.",
    ),
    (
        "MADHYA PRADESH", "KWARDHA",
        "CHHATTISGARH", "Kawardha",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical district-name relationship; AGRIDATA spelling corresponds to Kawardha.",
    ),
    (
        "MADHYA PRADESH", "MAHASUMUNDRA",
        "CHHATTISGARH", "Mahasamund",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical district-name relationship; AGRIDATA spelling corresponds to Mahasamund.",
    ),
    (
        "MADHYA PRADESH", "RAIGARH",
        "CHHATTISGARH", "Raigarh",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical state-label relationship; Raigarh was represented under Chhattisgarh in the Census 2001 reference geography.",
    ),
    (
        "MADHYA PRADESH", "RAIPUR",
        "CHHATTISGARH", "Raipur",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical state-label relationship; Raipur was represented under Chhattisgarh in the Census 2001 reference geography.",
    ),
    (
        "MADHYA PRADESH", "RAJNANDGAON",
        "CHHATTISGARH", "Rajnandgaon",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical state-label relationship; Rajnandgaon was represented under Chhattisgarh in the Census 2001 reference geography.",
    ),
    (
        "MADHYA PRADESH", "SARGUJA",
        "CHHATTISGARH", "Surguja",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Chhattisgarh district records",
        "high",
        "Historical district-name relationship; AGRIDATA spelling corresponds to Surguja.",
    ),
]

new_df = pd.DataFrame(new_rows, columns=expected_columns)

combined = pd.concat(
    [existing, new_df],
    ignore_index=True,
)

duplicate_mask = combined.duplicated(
    subset=["state_raw", "district_raw"],
    keep=False,
)

if duplicate_mask.any():
    print("Duplicate concordance keys found:")
    print(
        combined.loc[
            duplicate_mask,
            ["state_raw", "district_raw"]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )
    raise ValueError("Duplicate concordance keys detected.")

combined.to_csv(OUTPUT, index=False)

print("AgriAdapt V4.2 — MP → Chhattisgarh Concordance Extension")
print()
print(f"Existing rows:     {len(existing)}")
print(f"New rows:          {len(new_df)}")
print(f"Final rows:        {len(combined)}")
print(
    "Duplicate keys:    "
    f"{combined.duplicated(['state_raw', 'district_raw']).sum()}"
)
print()
print("New mappings:")
print(
    new_df[
        [
            "state_raw",
            "district_raw",
            "matched_state",
            "matched_district",
            "confidence",
        ]
    ].to_string(index=False)
)
print()
print(f"Output: {OUTPUT}")
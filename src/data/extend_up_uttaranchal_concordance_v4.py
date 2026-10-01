from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data" / "processed" / "v4" / "district"
    / "historical_district_concordance_mp_chhattisgarh.csv"
)

OUTPUT = (
    ROOT / "data" / "processed" / "v4" / "district"
    / "historical_district_concordance_up_uttaranchal.csv"
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
        "UTTAR PRADESH",
        "ALMORAH",
        "UTTARANCHAL",
        "Almora",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Uttar Pradesh Reorganisation Act 2000; Census 2001",
        "high",
        "Former Uttar Pradesh district transferred to Uttaranchal under the Uttar Pradesh Reorganisation Act, 2000; Census 2001 lists Almora as district 05/09.",
    ),
    (
        "UTTAR PRADESH",
        "BAGESHWAR",
        "UTTARANCHAL",
        "Bageshwar",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Uttar Pradesh Reorganisation Act 2000; Census 2001",
        "high",
        "Former Uttar Pradesh district transferred to Uttaranchal; Census 2001 lists Bageshwar as district 05/08.",
    ),
    (
        "UTTAR PRADESH",
        "CHAMOLI",
        "UTTARANCHAL",
        "Chamoli",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Uttar Pradesh Reorganisation Act 2000; Census 2001",
        "high",
        "Former Uttar Pradesh district transferred to Uttaranchal; Census 2001 lists Chamoli as district 05/02.",
    ),
    (
        "UTTAR PRADESH",
        "CHAMPAVAT",
        "UTTARANCHAL",
        "Champawat",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Uttar Pradesh Reorganisation Act 2000; Census 2001",
        "high",
        "Former Uttar Pradesh district transferred to Uttaranchal; Census 2001 lists Champawat as district 05/10.",
    ),
    (
        "UTTAR PRADESH",
        "DEHRADUN",
        "UTTARANCHAL",
        "Dehradun",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Uttar Pradesh Reorganisation Act 2000; Census 2001",
        "high",
        "Former Uttar Pradesh district transferred to Uttaranchal; Census 2001 lists Dehradun as district 05/05.",
    ),
    (
        "UTTAR PRADESH",
        "GARHWAL",
        "UTTARANCHAL",
        "Garhwal",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Uttar Pradesh Reorganisation Act 2000; Census 2001",
        "high",
        "Former Uttar Pradesh district transferred to Uttaranchal; Census 2001 lists Garhwal as district 05/06.",
    ),
    (
        "UTTAR PRADESH",
        "HARIDWAR",
        "UTTARANCHAL",
        "Hardwar",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Uttar Pradesh Reorganisation Act 2000; Census 2001",
        "high",
        "Former Uttar Pradesh district transferred to Uttaranchal; Census 2001 uses the historical spelling Hardwar, district 05/13.",
    ),
    (
        "UTTAR PRADESH",
        "NAINITAL",
        "UTTARANCHAL",
        "Nainital",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Uttar Pradesh Reorganisation Act 2000; Census 2001",
        "high",
        "Former Uttar Pradesh district transferred to Uttaranchal; Census 2001 lists Nainital as district 05/11.",
    ),
    (
        "UTTAR PRADESH",
        "PITHORAGARH",
        "UTTARANCHAL",
        "Pithoragarh",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Uttar Pradesh Reorganisation Act 2000; Census 2001",
        "high",
        "Former Uttar Pradesh district transferred to Uttaranchal; Census 2001 lists Pithoragarh as district 05/07.",
    ),
    (
        "UTTAR PRADESH",
        "RUDRA PRAYAG",
        "UTTARANCHAL",
        "Rudraprayag",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Uttar Pradesh Reorganisation Act 2000; Census 2001",
        "high",
        "Former Uttar Pradesh district transferred to Uttaranchal; AGRIDATA spacing variant corresponds to Census 2001 Rudraprayag, district 05/03.",
    ),
    (
        "UTTAR PRADESH",
        "TEHRIGARHWAL",
        "UTTARANCHAL",
        "Tehri Garhwal",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Uttar Pradesh Reorganisation Act 2000; Census 2001",
        "high",
        "Former Uttar Pradesh district transferred to Uttaranchal; AGRIDATA compact spelling corresponds to Census 2001 Tehri Garhwal, district 05/04.",
    ),
    (
        "UTTAR PRADESH",
        "U.S.NAGAR",
        "UTTARANCHAL",
        "Udham Singh Nagar",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Uttar Pradesh Reorganisation Act 2000; Census 2001",
        "high",
        "Former Uttar Pradesh district transferred to Uttaranchal; abbreviated AGRIDATA label corresponds to Census 2001 Udham Singh Nagar, district 05/12.",
    ),
    (
        "UTTAR PRADESH",
        "UTTAR KASHI",
        "UTTARANCHAL",
        "Uttarkashi",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Uttar Pradesh Reorganisation Act 2000; Census 2001",
        "high",
        "Former Uttar Pradesh district transferred to Uttaranchal; AGRIDATA spacing variant corresponds to Census 2001 Uttarkashi, district 05/01.",
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
        ].drop_duplicates().to_string(index=False)
    )
    raise ValueError("Duplicate concordance keys detected.")

if len(new_df) != 13:
    raise ValueError(
        f"Expected 13 new rows, got {len(new_df)}"
    )

combined.to_csv(OUTPUT, index=False)

print("AgriAdapt V4.2 — UP → Uttaranchal Concordance Extension")
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
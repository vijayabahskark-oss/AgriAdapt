from pathlib import Path

import pandas as pd


# ============================================================
# AgriAdapt V4.2 — Extend Historical District Concordance
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONCORDANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance.csv"
)

AUDIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_historical_audit.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_expanded.csv"
)


# ------------------------------------------------------------
# New Bihar / Jharkhand concordance records
# ------------------------------------------------------------

new_rows = [
    [
        "BIHAR", "BOKARO",
        "JHARKHAND", "Bokaro",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Jharkhand district records",
        "high",
        "AGRIDATA uses Bihar as the state label; Census 2001 identifies Bokaro as a Jharkhand district."
    ],
    [
        "BIHAR", "DEVGHAR",
        "JHARKHAND", "Deoghar",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Jharkhand district records",
        "high",
        "Legacy spelling; Census 2001 identifies Deoghar as a Jharkhand district."
    ],
    [
        "BIHAR", "DHANBAD",
        "JHARKHAND", "Dhanbad",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Jharkhand district records",
        "high",
        "Historical state-label relationship."
    ],
    [
        "BIHAR", "DUMKA",
        "JHARKHAND", "Dumka",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Jharkhand district records",
        "high",
        "Historical state-label relationship."
    ],
    [
        "BIHAR", "GADHWA",
        "JHARKHAND", "Garhwa",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Jharkhand district records",
        "high",
        "Legacy spelling; Census 2001 uses Garhwa."
    ],
    [
        "BIHAR", "GIRIDIH",
        "JHARKHAND", "Giridih",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Jharkhand district records",
        "high",
        "Historical state-label relationship."
    ],
    [
        "BIHAR", "GODDA",
        "JHARKHAND", "Godda",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Jharkhand district records",
        "high",
        "Historical state-label relationship."
    ],
    [
        "BIHAR", "HAZARIBAGH",
        "JHARKHAND", "Hazaribag",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Jharkhand district records",
        "high",
        "Spelling normalization; Census 2001 uses Hazaribag."
    ],
    [
        "BIHAR", "PAKUR",
        "JHARKHAND", "Pakaur",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Jharkhand district records",
        "high",
        "Spelling normalization; Census 2001 uses Pakaur."
    ],
    [
        "BIHAR", "PALAMAU",
        "JHARKHAND", "Palamu",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Jharkhand district records",
        "high",
        "Spelling normalization; Census 2001 uses Palamu."
    ],
    [
        "BIHAR", "RANCHI",
        "JHARKHAND", "Ranchi",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Jharkhand district records",
        "high",
        "Historical state-label relationship."
    ],
    [
        "BIHAR", "SAHEBGANJ",
        "JHARKHAND", "Sahibganj",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Jharkhand district records",
        "high",
        "Spelling normalization; Census 2001 uses Sahibganj."
    ],
    [
        "BIHAR", "SINGHBHUR(WEST)",
        "JHARKHAND", "Pashchimi Singhbhum",
        "HISTORICAL_ADMINISTRATIVE_MAPPING",
        "Census 2001 Jharkhand district records",
        "high",
        "Legacy district name corresponding to West Singhbhum/Pashchimi Singhbhum."
    ],

    # --------------------------------------------------------
    # Bihar name variants
    # --------------------------------------------------------

    [
        "BIHAR", "BHABHA",
        "BIHAR", "Kaimur (Bhabua)",
        "NAME_ALIAS",
        "Census 2001 Bihar district records",
        "high",
        "AGRIDATA legacy district label."
    ],
    [
        "BIHAR", "BHANKA",
        "BIHAR", "Banka",
        "NAME_ALIAS",
        "Census 2001 Bihar district records",
        "high",
        "AGRIDATA spelling variant."
    ],
    [
        "BIHAR", "CHAMPARAN(EAST)",
        "BIHAR", "Purba Champaran",
        "NAME_ALIAS",
        "Census 2001 Bihar district records",
        "high",
        "Legacy East Champaran district label."
    ],
    [
        "BIHAR", "CHAMPARAN(WEST)",
        "BIHAR", "Pashchim Champaran",
        "NAME_ALIAS",
        "Census 2001 Bihar district records",
        "high",
        "Legacy West Champaran district label."
    ],
    [
        "BIHAR", "JAHANABAD",
        "BIHAR", "Jehanabad",
        "NAME_ALIAS",
        "Census 2001 Bihar district records",
        "high",
        "AGRIDATA spelling variant."
    ],
    [
        "BIHAR", "LAKHISARIA",
        "BIHAR", "Lakhisarai",
        "NAME_ALIAS",
        "Census 2001 Bihar district records",
        "high",
        "AGRIDATA spelling variant."
    ],
    [
        "BIHAR", "MADHUPURA",
        "BIHAR", "Madhepura",
        "NAME_ALIAS",
        "Census 2001 Bihar district records",
        "high",
        "AGRIDATA spelling variant."
    ],
    [
        "BIHAR", "NAWADHA",
        "BIHAR", "Nawada",
        "NAME_ALIAS",
        "Census 2001 Bihar district records",
        "high",
        "AGRIDATA spelling variant."
    ],
    [
        "BIHAR", "SHIVHAR",
        "BIHAR", "Sheohar",
        "NAME_ALIAS",
        "Census 2001 Bihar district records",
        "high",
        "AGRIDATA spelling variant."
    ],
    [
        "BIHAR", "SHKHPURA",
        "BIHAR", "Sheikhpura",
        "NAME_ALIAS",
        "Census 2001 Bihar district records",
        "high",
        "AGRIDATA spelling variant."
    ],
    [
        "BIHAR", "SUMAL",
        "BIHAR", "Supaul",
        "NAME_ALIAS",
        "Census 2001 Bihar district records",
        "high",
        "AGRIDATA legacy spelling variant."
    ],
]


columns = [
    "state_raw",
    "district_raw",
    "matched_state",
    "matched_district",
    "mapping_method",
    "mapping_source",
    "confidence",
    "notes",
]


new_df = pd.DataFrame(new_rows, columns=columns)


# ------------------------------------------------------------
# Load existing 9-row concordance
# ------------------------------------------------------------

existing_df = pd.read_csv(CONCORDANCE_PATH)


# ------------------------------------------------------------
# Validate new records
# ------------------------------------------------------------

if new_df[["state_raw", "district_raw"]].duplicated().any():
    raise ValueError("Duplicate keys found in new concordance records.")


if new_df[columns].isna().any().any():
    raise ValueError("Blank fields found in new concordance records.")


# ------------------------------------------------------------
# Make sure new records actually exist in the 241 unmatched set
# ------------------------------------------------------------

audit_df = pd.read_csv(AUDIT_PATH)

unmatched_keys = set(
    zip(
        audit_df.loc[
            audit_df["mapping_status"] == "unmatched",
            "state",
        ],
        audit_df.loc[
            audit_df["mapping_status"] == "unmatched",
            "district_raw",
        ],
    )
)

new_keys = set(
    zip(
        new_df["state_raw"],
        new_df["district_raw"],
    )
)

missing_from_unmatched = new_keys - unmatched_keys

if missing_from_unmatched:
    raise ValueError(
        "These new concordance records are not present in "
        "the current unmatched set:\n"
        + "\n".join(
            f"{state} | {district}"
            for state, district in sorted(missing_from_unmatched)
        )
    )


# ------------------------------------------------------------
# Prevent duplicate keys against existing concordance
# ------------------------------------------------------------

existing_keys = set(
    zip(
        existing_df["state_raw"],
        existing_df["district_raw"],
    )
)

duplicate_existing = new_keys & existing_keys

if duplicate_existing:
    raise ValueError(
        "New records duplicate existing concordance keys:\n"
        + "\n".join(
            f"{state} | {district}"
            for state, district in sorted(duplicate_existing)
        )
    )


# ------------------------------------------------------------
# Combine
# ------------------------------------------------------------

expanded_df = pd.concat(
    [existing_df, new_df],
    ignore_index=True,
)


# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

if expanded_df[["state_raw", "district_raw"]].duplicated().any():
    raise ValueError(
        "Duplicate raw keys detected in expanded concordance."
    )


if expanded_df[columns].isna().any().any():
    raise ValueError(
        "Blank fields detected in expanded concordance."
    )


expected_rows = len(existing_df) + len(new_df)

if len(expanded_df) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} rows, got {len(expanded_df)}."
    )


# ------------------------------------------------------------
# Save expanded concordance
# ------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

expanded_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
)


# ------------------------------------------------------------
# Console report
# ------------------------------------------------------------

print("=" * 72)
print("AgriAdapt V4.2 — Expanded Historical District Concordance")
print("=" * 72)

print()
print("EXISTING CONCORDANCE")
print(f"Existing rows:          {len(existing_df)}")

print()
print("NEW RECORDS")
print(f"New rows:               {len(new_df)}")

print()
print("RESULT")
print(f"Expanded rows:          {len(expanded_df)}")
print(
    "Duplicate raw keys:     ",
    int(
        expanded_df[
            ["state_raw", "district_raw"]
        ].duplicated().sum()
    ),
)
print(
    "Blank fields:           ",
    int(expanded_df[columns].isna().sum().sum()),
)
print(
    "New keys found in "
    "unmatched set:         ",
    len(new_keys - missing_from_unmatched),
)

print()
print("MAPPING METHODS")
print(
    expanded_df["mapping_method"]
    .value_counts()
    .to_string()
)

print()
print("OUTPUT")
print(OUTPUT_PATH)

print()
print("Concordance expansion complete.")
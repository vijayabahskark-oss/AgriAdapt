from pathlib import Path
import pandas as pd


# ============================================================
# AgriAdapt V4.2
# ORISSA → Census 2001 District Name Concordance
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_up_uttaranchal.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_orissa.csv"
)


# ============================================================
# EXPECTED SCHEMA
# ============================================================

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


# ============================================================
# LOAD EXISTING CONCORDANCE
# ============================================================

existing = pd.read_csv(INPUT)


if existing.columns.tolist() != EXPECTED_COLUMNS:
    print("ERROR: Unexpected concordance schema.")
    print()
    print("Expected:")
    print(EXPECTED_COLUMNS)
    print()
    print("Found:")
    print(existing.columns.tolist())
    raise ValueError("Concordance schema mismatch.")


# ============================================================
# 17 ORISSA NAME CONCORDANCES
# ============================================================

new_rows = [

    (
        "ORISSA",
        "ANGUL",
        "ORISSA",
        "Anugul",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses ANGUL; Census 2001 uses Anugul, district code 21/15.",
    ),

    (
        "ORISSA",
        "BALASORE",
        "ORISSA",
        "Baleshwar",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses BALASORE; Census 2001 uses Baleshwar, district code 21/08.",
    ),

    (
        "ORISSA",
        "BOLANGIR",
        "ORISSA",
        "Balangir",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses BOLANGIR; Census 2001 uses Balangir, district code 21/24.",
    ),

    (
        "ORISSA",
        "BOUDH",
        "ORISSA",
        "Baudh",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses BOUDH; Census 2001 uses Baudh, district code 21/22.",
    ),

    (
        "ORISSA",
        "BURAGARH",
        "ORISSA",
        "Bargarh",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses BURAGARH; Census 2001 uses Bargarh, district code 21/01.",
    ),

    (
        "ORISSA",
        "DEOGARH",
        "ORISSA",
        "Debagarh",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses DEOGARH; Census 2001 uses Debagarh, district code 21/04.",
    ),

    (
        "ORISSA",
        "GAJAPATTI",
        "ORISSA",
        "Gajapati",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses GAJAPATTI; Census 2001 uses Gajapati, district code 21/20.",
    ),

    (
        "ORISSA",
        "JAGATSINGPUR",
        "ORISSA",
        "Jagatsinghapur",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses JAGATSINGPUR; Census 2001 uses Jagatsinghapur, district code 21/11.",
    ),

    (
        "ORISSA",
        "JAJPUR",
        "ORISSA",
        "Jajapur",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses JAJPUR; Census 2001 uses Jajapur, district code 21/13.",
    ),

    (
        "ORISSA",
        "JHARSUGDA",
        "ORISSA",
        "Jharsuguda",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses JHARSUGDA; Census 2001 uses Jharsuguda, district code 21/02.",
    ),

    (
        "ORISSA",
        "KEDRAPARA",
        "ORISSA",
        "Kendrapara",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses KEDRAPARA; Census 2001 uses Kendrapara, district code 21/10.",
    ),

    (
        "ORISSA",
        "KEONJHAR",
        "ORISSA",
        "Kendujhar",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses KEONJHAR; Census 2001 uses Kendujhar, district code 21/06.",
    ),

    (
        "ORISSA",
        "KHURDA",
        "ORISSA",
        "Khordha",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses KHURDA; Census 2001 uses Khordha, district code 21/17.",
    ),

    (
        "ORISSA",
        "NAWAPARA",
        "ORISSA",
        "Nuapada",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses NAWAPARA; Census 2001 uses Nuapada, district code 21/25.",
    ),

    (
        "ORISSA",
        "NAWORANGPUR",
        "ORISSA",
        "Nabarangapur",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses NAWORANGPUR; Census 2001 uses Nabarangapur, district code 21/28.",
    ),

    (
        "ORISSA",
        "PHULBANI",
        "ORISSA",
        "Kandhamal",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses the legacy district name PHULBANI; Census 2001 uses Kandhamal, district code 21/21.",
    ),

    (
        "ORISSA",
        "SONEPUR",
        "ORISSA",
        "Sonapur",
        "NAME_ALIAS",
        "Census 2001 Orissa district records",
        "high",
        "AGRIDATA uses SONEPUR; Census 2001 uses Sonapur, district code 21/23.",
    ),
]


new_df = pd.DataFrame(
    new_rows,
    columns=EXPECTED_COLUMNS,
)


# ============================================================
# VALIDATION
# ============================================================

if len(new_df) != 17:
    raise ValueError(
        f"Expected 17 new mappings, got {len(new_df)}."
    )


# Ensure all mappings remain within ORISSA.
if not (
    (new_df["state_raw"] == "ORISSA")
    & (new_df["matched_state"] == "ORISSA")
).all():
    raise ValueError(
        "Found mapping outside ORISSA → ORISSA."
    )


# Ensure all mappings use NAME_ALIAS.
if not (
    new_df["mapping_method"] == "NAME_ALIAS"
).all():
    raise ValueError(
        "Unexpected mapping method found."
    )


# Check duplicate keys inside new mappings.
if new_df.duplicated(
    subset=["state_raw", "district_raw"]
).any():

    raise ValueError(
        "Duplicate keys found within new mappings."
    )


# Check against existing concordance.
combined = pd.concat(
    [existing, new_df],
    ignore_index=True,
)


duplicate_mask = combined.duplicated(
    subset=["state_raw", "district_raw"],
    keep=False,
)


if duplicate_mask.any():

    print("ERROR: Duplicate concordance keys detected.")
    print()

    print(
        combined.loc[
            duplicate_mask,
            ["state_raw", "district_raw"]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

    raise ValueError(
        "Duplicate concordance keys detected."
    )


# ============================================================
# WRITE OUTPUT
# ============================================================

combined.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "AgriAdapt V4.2 — ORISSA Name Concordance Extension"
)
print()

print(
    f"Existing rows:     {len(existing)}"
)

print(
    f"New rows:          {len(new_df)}"
)

print(
    f"Final rows:        {len(combined)}"
)

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
            "mapping_method",
            "confidence",
        ]
    ].to_string(index=False)
)

print()
print(f"Output: {OUTPUT}")
from pathlib import Path
import re
import unicodedata

import pandas as pd
import shapefile


from geography_normalization import normalize_state, normalize_district


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

AGRIDATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "v4"
    / "agridata_v4_target_valid.csv"
)

CENSUS_SHP = (
    ROOT
    / "data"
    / "external"
    / "geography"
    / "india_district_boundaries"
    / "census-2001"
    / "2001_Dist.shp"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "v4"
REPORT_DIR = ROOT / "reports" / "v4"

OUTPUT_MAPPING = OUTPUT_DIR / "district_mapping_audit_v4.csv"
OUTPUT_REPORT = REPORT_DIR / "district_mapping_audit_v4.txt"


# ============================================================
# Normalization
# ============================================================

def normalize_text(value):
    """
    Conservative text normalization.

    This does NOT attempt to resolve historical district
    relationships. It only standardizes formatting.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    # Uppercase
    text = text.upper()

    # Common punctuation
    text = text.replace("&", " AND ")

    # Remove punctuation
    text = re.sub(r"[^A-Z0-9]+", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_state(value):
    """
    Conservative state normalization for obvious historical
    naming differences present in the two datasets.
    """
    value = normalize_text(value)

    aliases = {
        "ORISSA": "ORISSA",
        "ODISHA": "ORISSA",

        "UTTARANCHAL": "UTTARANCHAL",
        "UTTARAKHAND": "UTTARANCHAL",

        "JAMMU AND KASHMIR": "JAMMU AND KASHMIR",

        "PONDICHERRY": "PONDICHERRY",
        "PUDUCHERRY": "PONDICHERRY",

        "ANDAMAN AND NICOBAR ISLANDS":
            "ANDAMAN AND NICOBAR ISLANDS",

        "DADRA AND NAGAR HAVELI":
            "DADRA AND NAGAR HAVELI",

        "DAMAN AND DIU":
            "DAMAN AND DIU",
    }

    return aliases.get(value, value)


# ============================================================
# Validation
# ============================================================

if not AGRIDATA_PATH.exists():
    raise FileNotFoundError(
        f"AGRIDATA file not found:\n{AGRIDATA_PATH}"
    )

if not CENSUS_SHP.exists():
    raise FileNotFoundError(
        f"Census shapefile not found:\n{CENSUS_SHP}"
    )


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load AGRIDATA
# ============================================================

print("=" * 80)
print("AgriAdapt V4.2 — District Mapping Audit")
print("=" * 80)

agridata = pd.read_csv(AGRIDATA_PATH)

required_columns = {
    "state",
    "district",
    "crop",
    "year",
    "season",
    "yield_tonnes_per_ha",
}

missing = required_columns - set(agridata.columns)

if missing:
    raise ValueError(
        f"AGRIDATA is missing required columns: {sorted(missing)}"
    )


district_pairs = (
    agridata[
        ["state", "district"]
    ]
    .drop_duplicates()
    .sort_values(["state", "district"])
    .reset_index(drop=True)
)

print(f"\nAGRIDATA rows: {len(agridata):,}")
print(
    f"Unique state-district pairs: "
    f"{len(district_pairs):,}"
)


# ============================================================
# Load Census 2001
# ============================================================

reader = shapefile.Reader(str(CENSUS_SHP))

census_records = []

for record in reader.iterRecords():
    census_records.append(
        {
            "census_state": record["ST_NM"],
            "census_state_code": record["ST_CEN_CD"],
            "census_district_code": record["DT_CEN_CD"],
            "census_district": record["DISTRICT"],
        }
    )

census = pd.DataFrame(census_records)

print(
    f"Census 2001 district polygons: "
    f"{len(census):,}"
)


# ============================================================
# Normalize both datasets
# ============================================================

district_pairs["state_normalized"] = (
    district_pairs["state"]
    .map(normalize_state)
)

district_pairs["district_normalized"] = (
    district_pairs["district"]
    .map(normalize_text)
)

census["state_normalized"] = (
    census["census_state"]
    .map(normalize_state)
)

census["district_normalized"] = (
    census["census_district"]
    .map(normalize_text)
)


# ============================================================
# Build exact lookup
# ============================================================

lookup = {}

for _, row in census.iterrows():

    key = (
        row["state_normalized"],
        row["district_normalized"],
    )

    lookup.setdefault(key, []).append(row)


# ============================================================
# First-pass matching
# ============================================================

results = []

for _, row in district_pairs.iterrows():

    state_raw = row["state"]
    district_raw = row["district"]

    state_norm = row["state_normalized"]
    district_norm = row["district_normalized"]

    key = (state_norm, district_norm)

    candidates = lookup.get(key, [])

    if len(candidates) == 1:

        match = candidates[0]

        status = "exact_or_normalized"
        method = "state_district_normalized"

        matched_state = match["census_state"]
        matched_district = match["census_district"]

        state_code = match["census_state_code"]
        district_code = match["census_district_code"]

        notes = ""

    elif len(candidates) > 1:

        status = "ambiguous"
        method = "multiple_census_candidates"

        matched_state = ""
        matched_district = ""

        state_code = ""
        district_code = ""

        notes = (
            f"{len(candidates)} Census candidates "
            f"for normalized state/district"
        )

    else:

        status = "unmatched"
        method = "no_normalized_match"

        matched_state = ""
        matched_district = ""

        state_code = ""
        district_code = ""

        notes = ""

    results.append(
        {
            "state": state_raw,
            "district_raw": district_raw,
            "state_normalized": state_norm,
            "district_normalized": district_norm,
            "matched_state": matched_state,
            "matched_district": matched_district,
            "census_state_code": state_code,
            "census_district_code": district_code,
            "mapping_status": status,
            "mapping_method": method,
            "notes": notes,
        }
    )


mapping = pd.DataFrame(results)


# ============================================================
# Save mapping
# ============================================================

mapping.to_csv(
    OUTPUT_MAPPING,
    index=False,
    encoding="utf-8",
)


# ============================================================
# Audit statistics
# ============================================================

status_counts = (
    mapping["mapping_status"]
    .value_counts()
)

exact_or_normalized = int(
    (mapping["mapping_status"] == "exact_or_normalized").sum()
)

ambiguous = int(
    (mapping["mapping_status"] == "ambiguous").sum()
)

unmatched = int(
    (mapping["mapping_status"] == "unmatched").sum()
)


# ============================================================
# Report
# ============================================================

report_lines = []

report_lines.append(
    "AgriAdapt V4.2 — Census 2001 District Mapping Audit"
)

report_lines.append("=" * 80)

report_lines.append(
    f"AGRIDATA rows: {len(agridata):,}"
)

report_lines.append(
    f"Unique AGRIDATA state-district pairs: "
    f"{len(district_pairs):,}"
)

report_lines.append(
    f"Census 2001 district polygons: {len(census):,}"
)

report_lines.append("")

report_lines.append("FIRST-PASS MAPPING RESULTS")
report_lines.append("-" * 80)

for status, count in status_counts.items():
    report_lines.append(
        f"{status}: {count}"
    )

report_lines.append("")

report_lines.append(
    f"Matched:   {exact_or_normalized}"
)

report_lines.append(
    f"Ambiguous: {ambiguous}"
)

report_lines.append(
    f"Unmatched: {unmatched}"
)

report_lines.append("")

report_lines.append(
    "IMPORTANT:"
)

report_lines.append(
    "This is only a conservative first-pass name mapping."
)

report_lines.append(
    "No nearest-neighbour or geographic guessing was performed."
)

report_lines.append(
    "Historical district splits/mergers are intentionally left "
    "unresolved for the next audit stage."
)

report_lines.append("")

if unmatched > 0:

    report_lines.append("UNMATCHED DISTRICTS")
    report_lines.append("-" * 80)

    unmatched_rows = mapping[
        mapping["mapping_status"] == "unmatched"
    ]

    for _, row in unmatched_rows.iterrows():

        report_lines.append(
            f"{row['state']} | {row['district_raw']}"
        )

report_lines.append("")

if ambiguous > 0:

    report_lines.append("AMBIGUOUS DISTRICTS")
    report_lines.append("-" * 80)

    ambiguous_rows = mapping[
        mapping["mapping_status"] == "ambiguous"
    ]

    for _, row in ambiguous_rows.iterrows():

        report_lines.append(
            f"{row['state']} | {row['district_raw']}"
        )

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(report_lines)
    )


# ============================================================
# Console summary
# ============================================================

print("\nFIRST-PASS RESULTS")
print("-" * 80)

print(
    f"Matched:   {exact_or_normalized}"
)

print(
    f"Ambiguous: {ambiguous}"
)

print(
    f"Unmatched: {unmatched}"
)

print("\nOutputs:")

print(OUTPUT_MAPPING)

print(OUTPUT_REPORT)

print("\nAudit complete.")
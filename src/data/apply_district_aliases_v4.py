from pathlib import Path
import pandas as pd
import shapefile
import re
import unicodedata

from geography_normalization import normalize_state, normalize_district

# rest of your existing code...


ROOT = Path(__file__).resolve().parents[2]

AGRIDATA_PATH = (
    ROOT / "data" / "processed" / "v4"
    / "agridata_v4_target_valid.csv"
)

ALIAS_PATH = (
    ROOT / "data" / "processed" / "v4"
    / "district" / "district_name_aliases.csv"
)

CENSUS_SHP = (
    ROOT / "data" / "external" / "geography"
    / "india_district_boundaries" / "census-2001"
    / "2001_Dist.shp"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed" / "v4"
    / "district" / "district_mapping_alias_audit.csv"
)

REPORT_PATH = (
    ROOT / "reports" / "v4"
    / "district_mapping_alias_audit.txt"
)


def normalize(value):
    """
    Must remain consistent with the first-pass mapping script.
    Only standardizes formatting; it does not infer geography.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    # Uppercase
    text = text.upper()

    # Treat ampersand as a word separator
    text = text.replace("&", " AND ")

    # Remove punctuation consistently
    text = re.sub(
        r"[^A-Z0-9]+",
        " ",
        text
    )

    # Collapse whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ------------------------------------------------------------
# Load AGRIDATA
# ------------------------------------------------------------

df = pd.read_csv(AGRIDATA_PATH)

pairs = (
    df[["state", "district"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# Load alias table
# ------------------------------------------------------------

aliases = pd.read_csv(ALIAS_PATH)

required_alias_columns = {
    "state",
    "district_raw",
    "district_census2001",
    "reason",
    "confidence",
}

missing = required_alias_columns - set(aliases.columns)

if missing:
    raise ValueError(
        f"Alias file missing columns: {sorted(missing)}"
    )

aliases["state_key"] = aliases["state"].map(normalize)
aliases["district_key"] = aliases["district_raw"].map(normalize)

alias_lookup = {}

for _, row in aliases.iterrows():

    key = (
        row["state_key"],
        row["district_key"],
    )

    if key in alias_lookup:
        raise ValueError(
            f"Duplicate alias definition: {key}"
        )

    alias_lookup[key] = row


# ------------------------------------------------------------
# Load Census 2001
# ------------------------------------------------------------

reader = shapefile.Reader(str(CENSUS_SHP))

census_rows = []

for record in reader.iterRecords():

    census_rows.append(
        {
            "state": record["ST_NM"],
            "state_code": record["ST_CEN_CD"],
            "district_code": record["DT_CEN_CD"],
            "district": record["DISTRICT"],
        }
    )

census = pd.DataFrame(census_rows)

census["state_key"] = census["state"].map(normalize)
census["district_key"] = census["district"].map(normalize)

census_lookup = {}

for _, row in census.iterrows():

    key = (
        row["state_key"],
        row["district_key"],
    )

    census_lookup.setdefault(key, []).append(row)


# ------------------------------------------------------------
# Apply aliases
# ------------------------------------------------------------

results = []

alias_used = 0
alias_failed = 0
already_matched = 0

for _, pair in pairs.iterrows():

    state = pair["state"]
    district = pair["district"]

    key = (
        normalize(state),
        normalize(district),
    )

    # First check direct Census match.
    direct_candidates = census_lookup.get(key, [])

    if len(direct_candidates) == 1:

        match = direct_candidates[0]

        already_matched += 1

        results.append(
            {
                "state": state,
                "district_raw": district,
                "matched_state": match["state"],
                "matched_district": match["district"],
                "state_code": match["state_code"],
                "district_code": match["district_code"],
                "mapping_status": "direct",
                "mapping_method": "census_2001_normalized",
                "alias_reason": "",
                "confidence": "high",
            }
        )

        continue

    # Then check controlled alias.
    alias = alias_lookup.get(key)

    if alias is None:

        results.append(
            {
                "state": state,
                "district_raw": district,
                "matched_state": "",
                "matched_district": "",
                "state_code": "",
                "district_code": "",
                "mapping_status": "unmatched",
                "mapping_method": "",
                "alias_reason": "",
                "confidence": "",
            }
        )

        continue

    target_key = (
        normalize(state),
        normalize(alias["district_census2001"]),
    )

    alias_candidates = census_lookup.get(target_key, [])

    if len(alias_candidates) == 1:

        match = alias_candidates[0]

        alias_used += 1

        results.append(
            {
                "state": state,
                "district_raw": district,
                "matched_state": match["state"],
                "matched_district": match["district"],
                "state_code": match["state_code"],
                "district_code": match["district_code"],
                "mapping_status": "alias",
                "mapping_method": "controlled_alias",
                "alias_reason": alias["reason"],
                "confidence": alias["confidence"],
            }
        )

    else:

        alias_failed += 1

        results.append(
            {
                "state": state,
                "district_raw": district,
                "matched_state": "",
                "matched_district": "",
                "state_code": "",
                "district_code": "",
                "mapping_status": "alias_unresolved",
                "mapping_method": "controlled_alias",
                "alias_reason": alias["reason"],
                "confidence": alias["confidence"],
            }
        )


result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

status_counts = result_df["mapping_status"].value_counts()

matched = int(
    result_df["mapping_status"].isin(
        ["direct", "alias"]
    ).sum()
)

unmatched = int(
    (result_df["mapping_status"] == "unmatched").sum()
)

alias_unresolved = int(
    (result_df["mapping_status"] == "alias_unresolved").sum()
)

report = []

report.append(
    "AgriAdapt V4.2 — Controlled District Alias Audit"
)

report.append("=" * 70)

report.append(
    f"Unique AGRIDATA pairs: {len(pairs)}"
)

report.append(
    f"Census 2001 districts: {len(census)}"
)

report.append("")

report.append("RESULTS")
report.append("-" * 70)

report.append(
    f"Direct matches:       {already_matched}"
)

report.append(
    f"Alias matches:        {alias_used}"
)

report.append(
    f"Alias unresolved:     {alias_failed}"
)

report.append(
    f"Still unmatched:      {unmatched}"
)

report.append(
    f"Total resolved:       {matched}"
)

report.append("")

report.append("STATUS COUNTS")
report.append("-" * 70)

for status, count in status_counts.items():

    report.append(
        f"{status}: {count}"
    )

report.append("")

report.append(
    "No fuzzy matching or geographic guessing was performed."
)

report.append(
    "Aliases were applied only when explicitly defined in "
    "district_name_aliases.csv."
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write("\n".join(report))


print("=" * 70)
print("AgriAdapt V4.2 — Controlled District Alias Audit")
print("=" * 70)

print(f"\nUnique AGRIDATA pairs: {len(pairs)}")
print(f"Census 2001 districts: {len(census)}")

print("\nRESULTS")
print("-" * 70)
print(f"Direct matches:   {already_matched}")
print(f"Alias matches:    {alias_used}")
print(f"Alias unresolved: {alias_failed}")
print(f"Still unmatched:  {unmatched}")
print(f"Total resolved:   {matched}")

print("\nOutputs:")
print(OUTPUT_PATH)
print(REPORT_PATH)
from pathlib import Path

import pandas as pd


# ============================================================
# AgriAdapt V4.2 — Apply Expanded Historical Concordance
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_AUDIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_historical_audit.csv"
)

CONCORDANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_expanded.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_expanded_historical_audit.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "v4"
    / "district_mapping_expanded_historical_audit.txt"
)


# ------------------------------------------------------------
# Load inputs
# ------------------------------------------------------------

audit_df = pd.read_csv(INPUT_AUDIT_PATH)
concordance_df = pd.read_csv(CONCORDANCE_PATH)


# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

required_audit_columns = {
    "state",
    "district_raw",
    "mapping_status",
}

required_concordance_columns = {
    "state_raw",
    "district_raw",
    "matched_state",
    "matched_district",
    "mapping_method",
    "mapping_source",
    "confidence",
    "notes",
}

missing_audit = required_audit_columns - set(audit_df.columns)
missing_concordance = (
    required_concordance_columns - set(concordance_df.columns)
)

if missing_audit:
    raise ValueError(
        f"Missing audit columns: {sorted(missing_audit)}"
    )

if missing_concordance:
    raise ValueError(
        f"Missing concordance columns: "
        f"{sorted(missing_concordance)}"
    )


# ------------------------------------------------------------
# Validate concordance
# ------------------------------------------------------------

duplicate_keys = concordance_df.duplicated(
    subset=["state_raw", "district_raw"],
    keep=False,
)

if duplicate_keys.any():
    duplicates = concordance_df.loc[
        duplicate_keys,
        ["state_raw", "district_raw"],
    ]

    raise ValueError(
        "Duplicate keys in expanded concordance:\n"
        f"{duplicates.to_string(index=False)}"
    )


if concordance_df[list(required_concordance_columns)].isna().any().any():
    raise ValueError(
        "Blank fields found in expanded concordance."
    )


# ------------------------------------------------------------
# Preserve original audit
# ------------------------------------------------------------

result_df = audit_df.copy()

# Remove old historical metadata if rerun against an already
# enriched file. This keeps the output deterministic.
for column in [
    "mapping_method",
    "mapping_source",
    "mapping_confidence",
    "mapping_notes",
]:
    if column in result_df.columns:
        result_df.drop(columns=[column], inplace=True)

result_df["mapping_method"] = ""
result_df["mapping_source"] = ""
result_df["mapping_confidence"] = ""
result_df["mapping_notes"] = ""


# ------------------------------------------------------------
# Build lookup
# ------------------------------------------------------------

concordance_lookup = {
    (
        str(row["state_raw"]).strip(),
        str(row["district_raw"]).strip(),
    ): row
    for _, row in concordance_df.iterrows()
}


# ------------------------------------------------------------
# Apply expanded concordance ONLY to currently unresolved
# records.
#
# Existing direct/alias mappings remain untouched.
# Historical mappings from the previous 9-row concordance are
# also preserved because this input contains the previous
# historical audit.
# ------------------------------------------------------------

historical_applied = 0
already_historical = 0
not_found = []


for index, row in result_df.iterrows():

    current_status = str(row["mapping_status"]).strip()

    key = (
        str(row["state"]).strip(),
        str(row["district_raw"]).strip(),
    )

    concordance = concordance_lookup.get(key)

    if concordance is None:
        continue

    # Existing historical mappings are allowed to remain.
    if current_status == "historical":
        already_historical += 1
        continue

    # Only unresolved records can be newly changed.
    if current_status != "unmatched":
        continue

    result_df.at[index, "mapping_status"] = "historical"

    result_df.at[index, "mapping_method"] = concordance[
        "mapping_method"
    ]

    result_df.at[index, "mapping_source"] = concordance[
        "mapping_source"
    ]

    result_df.at[index, "mapping_confidence"] = concordance[
        "confidence"
    ]

    result_df.at[index, "mapping_notes"] = concordance[
        "notes"
    ]

    result_df.at[index, "matched_state"] = concordance[
        "matched_state"
    ]

    result_df.at[index, "matched_district"] = concordance[
        "matched_district"
    ]

    historical_applied += 1


# ------------------------------------------------------------
# Safety check
# ------------------------------------------------------------

expected_new_mappings = 24

if historical_applied != expected_new_mappings:
    raise ValueError(
        "Unexpected number of newly applied historical mappings: "
        f"{historical_applied}; expected {expected_new_mappings}."
    )


# ------------------------------------------------------------
# Validate row count
# ------------------------------------------------------------

if len(result_df) != len(audit_df):
    raise ValueError(
        "Row count changed unexpectedly."
    )


# ------------------------------------------------------------
# Final counts
# ------------------------------------------------------------

status_counts = result_df["mapping_status"].value_counts()

direct_count = int(
    status_counts.get("direct", 0)
)

alias_count = int(
    status_counts.get("alias", 0)
)

historical_count = int(
    status_counts.get("historical", 0)
)

unmatched_count = int(
    status_counts.get("unmatched", 0)
)

resolved_count = (
    direct_count
    + alias_count
    + historical_count
)


# ------------------------------------------------------------
# Expected totals
# ------------------------------------------------------------

expected_direct = 398
expected_alias = 17
expected_historical = 33
expected_unmatched = 217
expected_resolved = 448

if direct_count != expected_direct:
    raise ValueError(
        f"Direct count changed unexpectedly: "
        f"{direct_count} != {expected_direct}"
    )

if alias_count != expected_alias:
    raise ValueError(
        f"Alias count changed unexpectedly: "
        f"{alias_count} != {expected_alias}"
    )

if historical_count != expected_historical:
    raise ValueError(
        f"Historical count mismatch: "
        f"{historical_count} != {expected_historical}"
    )

if unmatched_count != expected_unmatched:
    raise ValueError(
        f"Unmatched count mismatch: "
        f"{unmatched_count} != {expected_unmatched}"
    )

if resolved_count != expected_resolved:
    raise ValueError(
        f"Resolved count mismatch: "
        f"{resolved_count} != {expected_resolved}"
    )


# ------------------------------------------------------------
# Save output
# ------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

report = f"""
AgriAdapt V4.2 — Expanded Historical District Concordance Audit
===============================================================

INPUT
-----
Previous audit rows:                 {len(audit_df)}
Expanded concordance rows:           {len(concordance_df)}


RESULTS
-------
Direct matches:                      {direct_count}
Alias matches:                       {alias_count}
Historical matches:                  {historical_count}
Still unmatched:                     {unmatched_count}

Total resolved:                      {resolved_count}
Total district-state pairs:          {len(result_df)}


HISTORICAL CONCORDANCE
----------------------
Existing historical mappings retained: {already_historical}
New historical mappings applied:       {historical_applied}
Total historical mappings:             {historical_count}


VALIDATION
----------
Duplicate concordance keys:          {
    int(
        concordance_df[
            ["state_raw", "district_raw"]
        ].duplicated().sum()
    )
}

Row count changed:                   NO


OUTPUTS
-------
{OUTPUT_PATH}
{REPORT_PATH}
"""

REPORT_PATH.write_text(
    report.strip() + "\n",
    encoding="utf-8",
)


# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------

print("=" * 72)
print("AgriAdapt V4.2 — Expanded Historical District Concordance")
print("=" * 72)

print()
print("INPUTS")
print(f"Previous audit rows:       {len(audit_df)}")
print(f"Concordance rows:          {len(concordance_df)}")

print()
print("RESULTS")
print(f"Direct matches:            {direct_count}")
print(f"Alias matches:             {alias_count}")
print(f"Historical matches:        {historical_count}")
print(f"Still unmatched:           {unmatched_count}")
print(f"Total resolved:            {resolved_count}")

print()
print("HISTORICAL CONCORDANCE")
print(f"Previously historical:     {already_historical}")
print(f"Newly applied:             {historical_applied}")
print(f"Total historical:          {historical_count}")

print()
print("VALIDATION")
print(
    "Duplicate concordance keys:",
    int(
        concordance_df[
            ["state_raw", "district_raw"]
        ].duplicated().sum()
    ),
)
print("Row count changed:         NO")

print()
print("OUTPUTS")
print(OUTPUT_PATH)
print(REPORT_PATH)

print()
print("Audit complete.")
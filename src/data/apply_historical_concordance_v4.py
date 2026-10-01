from pathlib import Path

import pandas as pd


# ============================================================
# AgriAdapt V4.2 — Historical District Concordance Application
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALIAS_AUDIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_alias_audit.csv"
)

CONCORDANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_historical_audit.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "v4"
    / "district_mapping_historical_audit.txt"
)


# ------------------------------------------------------------
# Load inputs
# ------------------------------------------------------------

audit_df = pd.read_csv(ALIAS_AUDIT_PATH)
concordance_df = pd.read_csv(CONCORDANCE_PATH)


# ------------------------------------------------------------
# Validate required columns
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
missing_concordance = required_concordance_columns - set(concordance_df.columns)

if missing_audit:
    raise ValueError(
        f"Missing required audit columns: {sorted(missing_audit)}"
    )

if missing_concordance:
    raise ValueError(
        f"Missing required concordance columns: "
        f"{sorted(missing_concordance)}"
    )


# ------------------------------------------------------------
# Validate concordance uniqueness
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
        "Duplicate keys found in historical concordance:\n"
        f"{duplicates.to_string(index=False)}"
    )


# ------------------------------------------------------------
# Preserve original mapping information
# ------------------------------------------------------------

result_df = audit_df.copy()

result_df["mapping_method"] = ""
result_df["mapping_source"] = ""
result_df["mapping_confidence"] = ""
result_df["mapping_notes"] = ""


# ------------------------------------------------------------
# Apply historical concordance ONLY to alias_unresolved rows
# ------------------------------------------------------------

concordance_lookup = {
    (row["state_raw"], row["district_raw"]): row
    for _, row in concordance_df.iterrows()
}


historical_applied = 0
historical_missing = []


for index, row in result_df.iterrows():

    if row["mapping_status"] != "alias_unresolved":
        continue

    key = (
        str(row["state"]).strip(),
        str(row["district_raw"]).strip(),
    )

    concordance = concordance_lookup.get(key)

    if concordance is None:
        historical_missing.append(key)
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

    # Store the historically matched geography.
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

if historical_missing:
    raise ValueError(
        "Some alias_unresolved records did not have a "
        "historical concordance:\n"
        + "\n".join(
            f"{state} | {district}"
            for state, district in historical_missing
        )
    )


if historical_applied != len(concordance_df):
    raise ValueError(
        "Historical concordance count mismatch: "
        f"applied {historical_applied}, "
        f"expected {len(concordance_df)}."
    )


# ------------------------------------------------------------
# Validate total records
# ------------------------------------------------------------

if len(result_df) != len(audit_df):
    raise ValueError(
        "Row count changed unexpectedly during concordance application."
    )


# ------------------------------------------------------------
# Save output
# ------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

result_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
)


# ------------------------------------------------------------
# Generate report
# ------------------------------------------------------------

status_counts = result_df["mapping_status"].value_counts()

direct_count = int(status_counts.get("direct", 0))
alias_count = int(status_counts.get("alias", 0))
historical_count = int(status_counts.get("historical", 0))
unresolved_count = int(status_counts.get("unmatched", 0))

resolved_count = (
    direct_count
    + alias_count
    + historical_count
)

report = f"""
AgriAdapt V4.2 — Historical District Concordance Audit
=======================================================

INPUTS
------
Alias audit:
{ALIAS_AUDIT_PATH}

Historical concordance:
{CONCORDANCE_PATH}


RESULTS
-------
Total AGRIDATA district-state pairs: {len(result_df)}

Direct matches:                     {direct_count}
Alias matches:                      {alias_count}
Historical concordance matches:     {historical_count}
Still unmatched:                    {unresolved_count}

Total resolved:                     {resolved_count}


CONCORDANCE VALIDATION
----------------------
Concordance rows:                   {len(concordance_df)}
Historical mappings applied:        {historical_applied}
Duplicate concordance keys:         {
    int(
        concordance_df[
            ["state_raw", "district_raw"]
        ].duplicated().sum()
    )
}


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
print("AgriAdapt V4.2 — Historical District Concordance")
print("=" * 72)

print()
print("INPUTS")
print(f"Alias audit rows:       {len(audit_df)}")
print(f"Concordance rows:       {len(concordance_df)}")

print()
print("RESULTS")
print(f"Direct matches:         {direct_count}")
print(f"Alias matches:          {alias_count}")
print(f"Historical matches:     {historical_count}")
print(f"Still unmatched:        {unresolved_count}")
print(f"Total resolved:         {resolved_count}")

print()
print("VALIDATION")
print(f"Historical applied:     {historical_applied}")
print(
    "Duplicate concordance keys:",
    int(
        concordance_df[
            ["state_raw", "district_raw"]
        ].duplicated().sum()
    ),
)

print()
print("Outputs:")
print(OUTPUT_PATH)
print(REPORT_PATH)

print()
print("Audit complete.")
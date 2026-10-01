from pathlib import Path
import pandas as pd


# ============================================================
# AgriAdapt V4.2
# Apply UP → Uttaranchal Historical District Concordance
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

AUDIT_INPUT = (
    ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_mp_chhattisgarh_audit.csv"
)

CONCORDANCE_INPUT = (
    ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "historical_district_concordance_up_uttaranchal.csv"
)

# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

AUDIT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "v4"
    / "district"
    / "district_mapping_up_uttaranchal_audit.csv"
)

REPORT_OUTPUT = (
    ROOT
    / "reports"
    / "v4"
    / "district_mapping_up_uttaranchal_audit.txt"
)


# ============================================================
# EXPECTED SCHEMAS
# ============================================================

EXPECTED_AUDIT_COLUMNS = [
    "state",
    "district_raw",
    "matched_state",
    "matched_district",
    "state_code",
    "district_code",
    "mapping_status",
    "alias_reason",
    "confidence",
    "mapping_method",
    "mapping_source",
    "mapping_confidence",
    "mapping_notes",
    "notes",
]

EXPECTED_CONCORDANCE_COLUMNS = [
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
# EXPECTED RESULTS
# ============================================================

EXPECTED_AUDIT_ROWS = 665

EXPECTED_NEW_MAPPINGS = 13

EXPECTED_HISTORICAL_TOTAL = 62

EXPECTED_TOTAL_RESOLVED = 477

EXPECTED_UNMATCHED = 188


# ============================================================
# LOAD INPUTS
# ============================================================

print("AgriAdapt V4.2 — UP → Uttaranchal Audit")
print()

audit = pd.read_csv(AUDIT_INPUT)
concordance = pd.read_csv(CONCORDANCE_INPUT)


# ============================================================
# SCHEMA VALIDATION
# ============================================================

if audit.columns.tolist() != EXPECTED_AUDIT_COLUMNS:
    print("ERROR: Unexpected audit schema.")
    print()
    print("Expected:")
    print(EXPECTED_AUDIT_COLUMNS)
    print()
    print("Found:")
    print(audit.columns.tolist())
    raise ValueError("Audit schema mismatch.")


if concordance.columns.tolist() != EXPECTED_CONCORDANCE_COLUMNS:
    print("ERROR: Unexpected concordance schema.")
    print()
    print("Expected:")
    print(EXPECTED_CONCORDANCE_COLUMNS)
    print()
    print("Found:")
    print(concordance.columns.tolist())
    raise ValueError("Concordance schema mismatch.")


# ============================================================
# ROW COUNT VALIDATION
# ============================================================

if len(audit) != EXPECTED_AUDIT_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_AUDIT_ROWS} audit rows, "
        f"found {len(audit)}."
    )


# ============================================================
# VALIDATE CONCORDANCE
# ============================================================

# We expect the complete 62-row concordance.
if len(concordance) != EXPECTED_HISTORICAL_TOTAL:
    raise ValueError(
        f"Expected {EXPECTED_HISTORICAL_TOTAL} concordance rows, "
        f"found {len(concordance)}."
    )


# No duplicate raw state + district keys.
duplicate_mask = concordance.duplicated(
    subset=["state_raw", "district_raw"],
    keep=False,
)

if duplicate_mask.any():

    print("ERROR: Duplicate concordance keys detected.")
    print()

    print(
        concordance.loc[
            duplicate_mask,
            ["state_raw", "district_raw"]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

    raise ValueError("Duplicate concordance keys detected.")


# ============================================================
# ISOLATE ONLY UP → UTTARANCHAL MAPPINGS
# ============================================================

up_lookup = concordance[
    (concordance["state_raw"] == "UTTAR PRADESH")
    & (concordance["matched_state"] == "UTTARANCHAL")
    & (
        concordance["mapping_method"]
        == "HISTORICAL_ADMINISTRATIVE_MAPPING"
    )
].copy()


# We expect exactly 13.
if len(up_lookup) != EXPECTED_NEW_MAPPINGS:
    raise ValueError(
        "Expected exactly "
        f"{EXPECTED_NEW_MAPPINGS} UP → Uttaranchal mappings, "
        f"found {len(up_lookup)}."
    )


# ============================================================
# VERIFY THE 13 EXPECTED KEYS
# ============================================================

EXPECTED_UP_DISTRICTS = {
    "ALMORAH",
    "BAGESHWAR",
    "CHAMOLI",
    "CHAMPAVAT",
    "DEHRADUN",
    "GARHWAL",
    "HARIDWAR",
    "NAINITAL",
    "PITHORAGARH",
    "RUDRA PRAYAG",
    "TEHRIGARHWAL",
    "U.S.NAGAR",
    "UTTAR KASHI",
}

actual_up_districts = set(
    up_lookup["district_raw"].astype(str).str.strip()
)

if actual_up_districts != EXPECTED_UP_DISTRICTS:

    print("ERROR: UP district key mismatch.")
    print()

    print("Expected:")
    print(sorted(EXPECTED_UP_DISTRICTS))
    print()

    print("Found:")
    print(sorted(actual_up_districts))
    print()

    raise ValueError("UP district key mismatch.")


# ============================================================
# FIND CURRENTLY UNMATCHED RECORDS
# ============================================================

unmatched_mask = audit["mapping_status"].eq("unmatched")

unmatched = audit.loc[unmatched_mask].copy()

print(f"Previous audit rows:       {len(audit)}")
print(f"Currently unmatched:       {len(unmatched)}")
print()


# ============================================================
# BUILD EXACT LOOKUP
# ============================================================

lookup = {}

for _, row in up_lookup.iterrows():

    key = (
        str(row["state_raw"]).strip(),
        str(row["district_raw"]).strip(),
    )

    if key in lookup:
        raise ValueError(
            f"Duplicate lookup key detected: {key}"
        )

    lookup[key] = row


# ============================================================
# APPLY MAPPINGS
# ============================================================

newly_applied = 0

applied_records = []


for idx in unmatched.index:

    state_value = str(
        audit.at[idx, "state"]
    ).strip()

    district_value = str(
        audit.at[idx, "district_raw"]
    ).strip()

    key = (
        state_value,
        district_value,
    )

    if key not in lookup:
        continue

    row = lookup[key]

    # --------------------------------------------------------
    # Apply matched geography
    # --------------------------------------------------------

    audit.at[idx, "matched_state"] = (
        row["matched_state"]
    )

    audit.at[idx, "matched_district"] = (
        row["matched_district"]
    )

    # --------------------------------------------------------
    # Preserve existing code fields.
    #
    # These historical mappings do not have Census codes
    # supplied by the concordance itself.
    # --------------------------------------------------------

    # state_code and district_code intentionally unchanged.

    # --------------------------------------------------------
    # Mapping metadata
    # --------------------------------------------------------

    audit.at[idx, "mapping_status"] = "historical"

    audit.at[idx, "alias_reason"] = pd.NA

    audit.at[idx, "confidence"] = row["confidence"]

    audit.at[idx, "mapping_method"] = (
        row["mapping_method"]
    )

    audit.at[idx, "mapping_source"] = (
        row["mapping_source"]
    )

    audit.at[idx, "mapping_confidence"] = (
        row["confidence"]
    )

    audit.at[idx, "mapping_notes"] = (
        row["notes"]
    )

    newly_applied += 1

    applied_records.append(
        {
            "state_raw": state_value,
            "district_raw": district_value,
            "matched_state": row["matched_state"],
            "matched_district": row["matched_district"],
        }
    )


# ============================================================
# POST-APPLICATION VALIDATION
# ============================================================

historical_count = int(
    audit["mapping_status"]
    .eq("historical")
    .sum()
)

unmatched_count = int(
    audit["mapping_status"]
    .eq("unmatched")
    .sum()
)

resolved_count = len(audit) - unmatched_count

duplicate_concordance_keys = int(
    concordance.duplicated(
        subset=["state_raw", "district_raw"]
    ).sum()
)


# ============================================================
# STRICT EXPECTATION CHECKS
# ============================================================

if newly_applied != EXPECTED_NEW_MAPPINGS:

    raise ValueError(
        f"Expected {EXPECTED_NEW_MAPPINGS} newly applied mappings, "
        f"got {newly_applied}."
    )


if historical_count != EXPECTED_HISTORICAL_TOTAL:

    raise ValueError(
        f"Expected {EXPECTED_HISTORICAL_TOTAL} historical mappings, "
        f"got {historical_count}."
    )


if resolved_count != EXPECTED_TOTAL_RESOLVED:

    raise ValueError(
        f"Expected {EXPECTED_TOTAL_RESOLVED} resolved records, "
        f"got {resolved_count}."
    )


if unmatched_count != EXPECTED_UNMATCHED:

    raise ValueError(
        f"Expected {EXPECTED_UNMATCHED} unmatched records, "
        f"got {unmatched_count}."
    )


if duplicate_concordance_keys != 0:

    raise ValueError(
        "Concordance contains duplicate state + district keys."
    )


if len(audit) != EXPECTED_AUDIT_ROWS:

    raise ValueError(
        "Audit row count changed unexpectedly."
    )


# ============================================================
# VERIFY ALL 13 WERE APPLIED TO THE CORRECT TARGET
# ============================================================

for record in applied_records:

    state = record["state_raw"]
    district = record["district_raw"]

    rows = audit[
        (audit["state"].astype(str).str.strip() == state)
        & (
            audit["district_raw"]
            .astype(str)
            .str.strip()
            == district
        )
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one audit row for "
            f"{state} | {district}, found {len(rows)}."
        )

    result = rows.iloc[0]

    if result["mapping_status"] != "historical":
        raise ValueError(
            f"{state} | {district} was not marked historical."
        )

    if result["matched_state"] != "UTTARANCHAL":
        raise ValueError(
            f"{state} | {district} mapped to unexpected state: "
            f"{result['matched_state']}"
        )


# ============================================================
# WRITE OUTPUT
# ============================================================

AUDIT_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

audit.to_csv(
    AUDIT_OUTPUT,
    index=False,
)


# ============================================================
# WRITE REPORT
# ============================================================

with open(
    REPORT_OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "AgriAdapt V4.2 — UP → Uttaranchal Historical Audit\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Previous audit rows:       {len(audit)}\n"
    )

    f.write(
        f"Previously historical:     49\n"
    )

    f.write(
        f"Newly applied:             {newly_applied}\n"
    )

    f.write(
        f"Historical total:          {historical_count}\n"
    )

    f.write(
        f"Total resolved:            {resolved_count}\n"
    )

    f.write(
        f"Still unmatched:           {unmatched_count}\n"
    )

    f.write(
        f"Duplicate concordance:     "
        f"{duplicate_concordance_keys}\n"
    )

    f.write("\n")
    f.write("NEWLY APPLIED MAPPINGS\n")
    f.write("-" * 60 + "\n")

    for record in applied_records:

        f.write(
            f"{record['state_raw']} | "
            f"{record['district_raw']} -> "
            f"{record['matched_state']} | "
            f"{record['matched_district']}\n"
        )

    f.write("\n")
    f.write("VALIDATION\n")
    f.write("-" * 60 + "\n")
    f.write("Schema validation:         PASS\n")
    f.write("Duplicate concordance:     PASS\n")
    f.write("Row count unchanged:       PASS\n")
    f.write("13 mappings applied:       PASS\n")


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("AgriAdapt V4.2 — UP → Uttaranchal Audit")
print()
print(f"Previous audit rows:       {len(audit)}")
print(f"Newly applied:             {newly_applied}")
print(f"Historical total:          {historical_count}")
print(f"Total resolved:            {resolved_count}")
print(f"Still unmatched:           {unmatched_count}")
print()
print("VALIDATION")
print(f"Duplicate concordance:     {duplicate_concordance_keys}")
print("Schema validation:         PASS")
print("Row count changed:         NO")
print()
print("OUTPUTS")
print(AUDIT_OUTPUT)
print(REPORT_OUTPUT)
print()
print("Audit complete.")
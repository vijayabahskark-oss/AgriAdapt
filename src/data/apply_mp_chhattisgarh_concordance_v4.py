from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

AUDIT_INPUT = (
    ROOT / "data" / "processed" / "v4" / "district"
    / "district_mapping_expanded_historical_audit.csv"
)

CONCORDANCE_INPUT = (
    ROOT / "data" / "processed" / "v4" / "district"
    / "historical_district_concordance_mp_chhattisgarh.csv"
)

AUDIT_OUTPUT = (
    ROOT / "data" / "processed" / "v4" / "district"
    / "district_mapping_mp_chhattisgarh_audit.csv"
)

REPORT_OUTPUT = (
    ROOT / "reports" / "v4"
    / "district_mapping_mp_chhattisgarh_audit.txt"
)

audit = pd.read_csv(AUDIT_INPUT)
concordance = pd.read_csv(CONCORDANCE_INPUT)

required_concordance_columns = [
    "state_raw",
    "district_raw",
    "matched_state",
    "matched_district",
    "mapping_method",
    "mapping_source",
    "confidence",
    "notes",
]

missing = set(required_concordance_columns) - set(concordance.columns)

if missing:
    raise ValueError(
        f"Missing concordance columns: {sorted(missing)}"
    )

# Only apply mappings to records that are currently unmatched.
unmatched_mask = audit["mapping_status"].eq("unmatched")

unmatched = audit.loc[unmatched_mask].copy()

# Build lookup from the 16 new MP -> Chhattisgarh mappings.
lookup = concordance[
    concordance["mapping_method"]
    == "HISTORICAL_ADMINISTRATIVE_MAPPING"
].copy()

# We only want the newly introduced MP -> Chhattisgarh records.
lookup = lookup[
    (lookup["state_raw"] == "MADHYA PRADESH")
    & (lookup["matched_state"] == "CHHATTISGARH")
].copy()

if len(lookup) != 16:
    raise ValueError(
        f"Expected 16 MP -> Chhattisgarh mappings, found {len(lookup)}"
    )

lookup_keys = set(
    zip(
        lookup["state_raw"],
        lookup["district_raw"],
    )
)

# Apply only when the raw state/district pair is currently unmatched.
newly_applied = 0

for idx in unmatched.index:
    key = (
        str(audit.at[idx, "state"]),
        str(audit.at[idx, "district_raw"]),
    )

    if key in lookup_keys:
        row = lookup[
            (lookup["state_raw"] == key[0])
            & (lookup["district_raw"] == key[1])
        ].iloc[0]

        audit.at[idx, "matched_state"] = row["matched_state"]
        audit.at[idx, "matched_district"] = row["matched_district"]
        audit.at[idx, "mapping_status"] = "historical"
        audit.at[idx, "mapping_method"] = row["mapping_method"]
        audit.at[idx, "mapping_source"] = row["mapping_source"]
        audit.at[idx, "confidence"] = row["confidence"]
        audit.at[idx, "notes"] = row["notes"]

        newly_applied += 1

# Validation
historical_count = int(
    audit["mapping_status"].eq("historical").sum()
)

unmatched_count = int(
    audit["mapping_status"].eq("unmatched").sum()
)

resolved_count = len(audit) - unmatched_count

duplicate_keys = int(
    concordance.duplicated(
        subset=["state_raw", "district_raw"]
    ).sum()
)

if len(audit) != 665:
    raise ValueError(
        f"Audit row count changed: {len(audit)}"
    )

if newly_applied != 16:
    raise ValueError(
        f"Expected 16 newly applied mappings, got {newly_applied}"
    )

if historical_count != 49:
    raise ValueError(
        f"Expected 49 historical mappings, got {historical_count}"
    )

if unmatched_count != 201:
    raise ValueError(
        f"Expected 201 unmatched records, got {unmatched_count}"
    )

if resolved_count != 464:
    raise ValueError(
        f"Expected 464 resolved records, got {resolved_count}"
    )

if duplicate_keys != 0:
    raise ValueError(
        f"Concordance contains {duplicate_keys} duplicate keys"
    )

audit.to_csv(AUDIT_OUTPUT, index=False)

REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
    f.write("AgriAdapt V4.2 — MP → Chhattisgarh Audit\n\n")
    f.write(f"Previous audit rows:       {len(audit)}\n")
    f.write(f"Newly applied:             {newly_applied}\n")
    f.write(f"Historical total:          {historical_count}\n")
    f.write(f"Total resolved:            {resolved_count}\n")
    f.write(f"Still unmatched:           {unmatched_count}\n")
    f.write(f"Duplicate concordance:     {duplicate_keys}\n")

print("AgriAdapt V4.2 — MP → Chhattisgarh Audit")
print()
print(f"Previous audit rows:       665")
print(f"Newly applied:             {newly_applied}")
print(f"Historical total:          {historical_count}")
print(f"Total resolved:            {resolved_count}")
print(f"Still unmatched:           {unmatched_count}")
print()
print("VALIDATION")
print(f"Duplicate concordance:     {duplicate_keys}")
print(f"Row count changed:         {'YES' if len(audit) != 665 else 'NO'}")
print()
print("OUTPUTS")
print(AUDIT_OUTPUT)
print(REPORT_OUTPUT)
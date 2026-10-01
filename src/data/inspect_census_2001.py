from pathlib import Path
import shapefile

ROOT = Path(__file__).resolve().parents[2]

shp_path = (
    ROOT
    / "data"
    / "external"
    / "geography"
    / "india_district_boundaries"
    / "census-2001"
    / "2001_Dist.shp"
)

print("=" * 70)
print("AgriAdapt — Census 2001 District Boundary Inspection")
print("=" * 70)

if not shp_path.exists():
    raise FileNotFoundError(f"Shapefile not found: {shp_path}")

reader = shapefile.Reader(str(shp_path))

print(f"\nShapefile: {shp_path}")
print(f"Number of records: {len(reader)}")
print(f"Number of shapes:  {len(reader.shapes())}")

print("\nATTRIBUTE FIELDS")
print("-" * 70)

fields = reader.fields[1:]  # skip DeletionFlag

for i, field in enumerate(fields):
    name, field_type, size, decimal = field
    print(
        f"{i:>3} | "
        f"{name:<30} | "
        f"type={field_type:<3} | "
        f"size={size:<4} | "
        f"decimal={decimal}"
    )

print("\nFIRST 5 RECORDS")
print("-" * 70)

field_names = [field[0] for field in fields]

for row_number, record in enumerate(reader.iterRecords(), start=1):
    print(f"\nRecord {row_number}")

    for name, value in zip(field_names, record):
        print(f"  {name}: {value}")

    if row_number >= 5:
        break

print("\nBOUNDING BOX")
print("-" * 70)
print(reader.bbox)

print("\nPRJ FILE")
print("-" * 70)

prj_path = shp_path.with_suffix(".prj")

if prj_path.exists():
    print(prj_path.read_text(encoding="utf-8").strip())
else:
    print("WARNING: .prj file not found.")

print("\nInspection complete.")
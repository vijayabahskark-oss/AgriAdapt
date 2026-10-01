from pathlib import Path
import hashlib
import json
import math
import re
import time

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[3]

INPUT = ROOT / "data/processed/v4/district/district_coordinates_v418.csv"
CACHE = ROOT / "data/raw/weather/nasa_power_v419"
OUTPUT = ROOT / "data/processed/v4/weather/nasa_power_district_monthly_v419.csv"
REPORT = ROOT / "reports/v4/weather/nasa_power_v419_ingestion_report.txt"
FAILED = ROOT / "reports/v4/weather/nasa_power_v419_failed_districts.csv"
DUPLICATES = ROOT / "reports/v4/weather/nasa_power_v419_duplicate_identities.csv"

START_YEAR = 1984
END_YEAR = 2024
EXPECTED_MONTHS = (END_YEAR - START_YEAR + 1) * 12

URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"

PARAMETERS = [
    "T2M",
    "PRECTOTCORR_SUM",
    "RH2M",
    "ALLSKY_SFC_SW_DWN",
    "WS2M",
]

CLIMATE_COLUMNS = {
    "T2M": "temperature_mean",
    "PRECTOTCORR_SUM": "rainfall_total",
    "RH2M": "humidity_mean",
    "ALLSKY_SFC_SW_DWN": "solar_radiation_mean",
    "WS2M": "wind_speed_mean",
}

OUTPUT_COLUMNS = [
    "state",
    "district",
    "latitude",
    "longitude",
    "year",
    "month",
    *CLIMATE_COLUMNS.values(),
]

SESSION = requests.Session()


def normalize(value):
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def detect_column(df, candidates):
    lookup = {
        re.sub(r"[^a-z0-9]", "", str(col).lower()): col
        for col in df.columns
    }

    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in lookup:
            return lookup[key]

    return None


def load_coordinates():
    if not INPUT.exists():
        raise FileNotFoundError(f"Coordinates file not found: {INPUT}")

    df = pd.read_csv(INPUT)

    state_col = detect_column(
        df,
        ["state_raw", "state", "matched_state"],
    )
    district_col = detect_column(
        df,
        ["district_raw", "district", "matched_district"],
    )
    latitude_col = detect_column(
        df,
        ["latitude", "lat", "representative_latitude"],
    )
    longitude_col = detect_column(
        df,
        ["longitude", "lon", "lng", "representative_longitude"],
    )

    missing = [
        name
        for name, col in {
            "state": state_col,
            "district": district_col,
            "latitude": latitude_col,
            "longitude": longitude_col,
        }.items()
        if col is None
    ]

    if missing:
        raise ValueError(
            f"Missing coordinate columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    data = df[
        [state_col, district_col, latitude_col, longitude_col]
    ].copy()

    data.columns = [
        "state",
        "district",
        "latitude",
        "longitude",
    ]

    data["latitude"] = pd.to_numeric(
        data["latitude"], errors="coerce"
    )
    data["longitude"] = pd.to_numeric(
        data["longitude"], errors="coerce"
    )

    data["state"] = data["state"].astype("string").str.strip()
    data["district"] = data["district"].astype("string").str.strip()

    valid = (
        data["state"].notna()
        & data["district"].notna()
        & data["latitude"].between(6, 38)
        & data["longitude"].between(68, 98)
    )

    skipped = data.loc[~valid].copy()
    resolved = data.loc[valid].copy().reset_index(drop=True)

    return resolved, skipped


def validate_identity(df):
    """
    Identifies duplicate names without silently merging them.
    """

    temp = df.copy()
    temp["state_key"] = temp["state"].map(normalize)
    temp["district_key"] = temp["district"].map(normalize)

    duplicates = temp.loc[
        temp.duplicated(
            ["state_key", "district_key"],
            keep=False,
        )
    ].copy()

    DUPLICATES.parent.mkdir(parents=True, exist_ok=True)
    duplicates.to_csv(DUPLICATES, index=False)

    if not duplicates.empty:
        print(
            f"WARNING: {len(duplicates)} coordinate rows have "
            "repeated normalized district identities."
        )
        print(f"Review: {DUPLICATES}")

    return duplicates


def cache_path(latitude, longitude):
    """
    Stable cache name based on the actual requested coordinates.
    """

    identity = (
        f"{latitude:.6f}|{longitude:.6f}|"
        f"{START_YEAR}|{END_YEAR}|"
        + ",".join(PARAMETERS)
    )

    digest = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:20]

    return CACHE / f"power_{digest}.json"


def parse_response(payload, state, district, latitude, longitude):
    try:
        parameters = payload["properties"]["parameter"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Invalid NASA POWER response structure"
        ) from exc

    for parameter in PARAMETERS:
        if parameter not in parameters:
            raise ValueError(
                f"Missing NASA parameter: {parameter}"
            )

    # Use one parameter as the authoritative timeline.
    timeline = parameters[PARAMETERS[0]]

    records = []

    for key in timeline:
        key = str(key)

        if len(key) != 6 or not key.isdigit():
            continue

        year = int(key[:4])
        month = int(key[4:])

        if not START_YEAR <= year <= END_YEAR:
            continue

        # NASA POWER can include YYYY13 annual summaries.
        # These must not be treated as monthly observations.
        if not 1 <= month <= 12:
            continue

        row = {
            "state": state,
            "district": district,
            "latitude": latitude,
            "longitude": longitude,
            "year": year,
            "month": month,
        }

        for parameter, column in CLIMATE_COLUMNS.items():
            value = parameters[parameter].get(key)

            try:
                value = float(value)
            except (ValueError, TypeError):
                value = math.nan

            # NASA missing-value sentinel.
            if not math.isfinite(value) or value <= -900:
                value = math.nan

            row[column] = value

        records.append(row)

    result = pd.DataFrame(records, columns=OUTPUT_COLUMNS)

    if len(result) != EXPECTED_MONTHS:
        raise ValueError(
            f"Expected {EXPECTED_MONTHS} monthly records; "
            f"received {len(result)}"
        )

    if result.duplicated(["year", "month"]).any():
        raise ValueError(
            "Duplicate year-month records inside NASA response"
        )

    expected_periods = {
        (year, month)
        for year in range(START_YEAR, END_YEAR + 1)
        for month in range(1, 13)
    }

    actual_periods = set(
        zip(result["year"], result["month"])
    )

    if actual_periods != expected_periods:
        raise ValueError(
            "Incomplete or unexpected year-month coverage"
        )

    if result["rainfall_total"].dropna().lt(0).any():
        raise ValueError("Negative monthly rainfall detected")

    if result["humidity_mean"].dropna().gt(100).any():
        raise ValueError("Humidity above 100% detected")

    if result["humidity_mean"].dropna().lt(0).any():
        raise ValueError("Negative humidity detected")

    return result


def request_nasa(latitude, longitude):
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": str(START_YEAR),
        "end": str(END_YEAR),
        "format": "JSON",
    }

    last_error = None

    for attempt in range(1, 4):
        try:
            response = SESSION.get(
                URL,
                params=params,
                timeout=120,
            )
            response.raise_for_status()

            payload = response.json()

            if "properties" not in payload:
                raise ValueError(
                    f"Unexpected NASA response: {str(payload)[:300]}"
                )

            return payload

        except (
            requests.RequestException,
            ValueError,
        ) as exc:
            last_error = exc

            if attempt < 3:
                time.sleep(3 * attempt)

    raise RuntimeError(
        f"NASA request failed after 3 attempts: {last_error}"
    )


def get_payload(state, district, latitude, longitude):
    path = cache_path(latitude, longitude)

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)

            parse_response(
                payload,
                state,
                district,
                latitude,
                longitude,
            )

            return payload, "cache"

        except (ValueError, KeyError, TypeError, OSError):
            print(
                "Cached response invalid; downloading again."
            )

    payload = request_nasa(latitude, longitude)

    # Validate BEFORE writing cache.
    parse_response(
        payload,
        state,
        district,
        latitude,
        longitude,
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file)

    return payload, "download"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    coordinates, skipped = load_coordinates()
    identity_duplicates = validate_identity(coordinates)

    print("=" * 65)
    print("AGRIADAPT V4.19 — NASA POWER MONTHLY INGESTION")
    print("=" * 65)

    print(f"Resolved coordinate rows: {len(coordinates)}")
    print(f"Skipped coordinate rows: {len(skipped)}")
    print(f"Expected months per row: {EXPECTED_MONTHS}")

    # Cache the parsed climate data in memory for identical
    # coordinate requests, but retain all original input identities.
    climate_by_coordinate = {}

    all_frames = []
    failed = []
    cache_hits = 0
    downloads = 0
    successes = 0

    for index, row in coordinates.iterrows():
        state = str(row["state"])
        district = str(row["district"])
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])

        print(
            f"[{index + 1}/{len(coordinates)}] "
            f"{state} / {district}"
        )

        coordinate_key = (
            round(latitude, 6),
            round(longitude, 6),
        )

        try:
            if coordinate_key in climate_by_coordinate:
                climate = climate_by_coordinate[
                    coordinate_key
                ].copy()

                climate["state"] = state
                climate["district"] = district

                source = "memory"

            else:
                payload, source = get_payload(
                    state,
                    district,
                    latitude,
                    longitude,
                )

                climate = parse_response(
                    payload,
                    state,
                    district,
                    latitude,
                    longitude,
                )

                climate_by_coordinate[
                    coordinate_key
                ] = climate.copy()

                if source == "cache":
                    cache_hits += 1
                else:
                    downloads += 1

            all_frames.append(climate)
            successes += 1

            print(
                f"  Success — {len(climate)} monthly records "
                f"({source})"
            )

        except Exception as exc:
            failed.append(
                {
                    "state": state,
                    "district": district,
                    "latitude": latitude,
                    "longitude": longitude,
                    "error": str(exc),
                }
            )

            print(f"  FAILED — {exc}")

    pd.DataFrame(
        failed,
        columns=[
            "state",
            "district",
            "latitude",
            "longitude",
            "error",
        ],
    ).to_csv(FAILED, index=False)

    if not all_frames:
        raise RuntimeError(
            "No valid monthly climate records were collected."
        )

    result = pd.concat(
        all_frames,
        ignore_index=True,
    )

    # An input identity repeated twice creates duplicate output
    # keys. Report this instead of silently deleting records.
    result["state_key"] = result["state"].map(normalize)
    result["district_key"] = result["district"].map(normalize)

    identity_key = [
        "state_key",
        "district_key",
        "year",
        "month",
    ]

    duplicate_mask = result.duplicated(
        identity_key,
        keep=False,
    )

    duplicate_count = int(duplicate_mask.sum())

    # Check whether repeated identities disagree on coordinates
    # or climate values.
    conflicting_groups = 0

    if duplicate_count:
        value_columns = [
            "latitude",
            "longitude",
            *CLIMATE_COLUMNS.values(),
        ]

        grouped = result.loc[duplicate_mask].groupby(
            identity_key,
            dropna=False,
        )

        for _, group in grouped:
            if (
                group[value_columns]
                .nunique(dropna=False)
                .gt(1)
                .any()
            ):
                conflicting_groups += 1

    # Validate each retained coordinate identity independently.
    expected_total = successes * EXPECTED_MONTHS

    if len(result) != expected_total:
        raise RuntimeError(
            f"Unexpected total rows: {len(result)} "
            f"vs {expected_total}"
        )

    if not result["month"].between(1, 12).all():
        raise RuntimeError(
            "Invalid month detected in final dataset."
        )

    if not result["year"].between(
        START_YEAR, END_YEAR
    ).all():
        raise RuntimeError(
            "Invalid year detected in final dataset."
        )

    if result[["latitude", "longitude"]].isna().any().any():
        raise RuntimeError(
            "Missing coordinates in final dataset."
        )

    # Remove temporary audit columns, not observations.
    result = result[OUTPUT_COLUMNS]

    result = result.sort_values(
        ["state", "district", "year", "month"]
    ).reset_index(drop=True)

    # Never silently overwrite a previously valid output
    # with a dataset containing ambiguous identity keys.
    safe_to_publish = (
        duplicate_count == 0
        and conflicting_groups == 0
    )

    if safe_to_publish:
        temporary = OUTPUT.with_suffix(".tmp.csv")
        result.to_csv(temporary, index=False)
        temporary.replace(OUTPUT)
        publication_status = "PUBLISHED"
    else:
        review_output = OUTPUT.with_name(
            "nasa_power_district_monthly_v419_REVIEW.csv"
        )
        result.to_csv(review_output, index=False)
        publication_status = (
            f"BLOCKED — review {review_output}"
        )

    missing_counts = result[
        list(CLIMATE_COLUMNS.values())
    ].isna().sum()

    report_lines = [
        "AGRIADAPT V4.19 INGESTION AUDIT",
        "=" * 60,
        f"Input coordinate rows: {len(coordinates)}",
        f"Skipped coordinate rows: {len(skipped)}",
        f"Successful input rows: {successes}",
        f"Failed input rows: {len(failed)}",
        f"Expected months per input row: {EXPECTED_MONTHS}",
        f"Actual monthly rows: {len(result)}",
        f"Expected successful rows: {expected_total}",
        f"Cache hits: {cache_hits}",
        f"New downloads: {downloads}",
        f"Repeated identity input rows: {len(identity_duplicates)}",
        f"Duplicate identity-month rows: {duplicate_count}",
        f"Conflicting duplicate groups: {conflicting_groups}",
        f"Year range: {START_YEAR}-{END_YEAR}",
        f"Publication status: {publication_status}",
        "",
        "MISSING CLIMATE VALUES",
        missing_counts.to_string(),
        "",
        f"Failed records: {FAILED}",
        f"Duplicate identity audit: {DUPLICATES}",
    ]

    REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 65)
    print("V4.19 INGESTION FINISHED")
    print("=" * 65)

    for line in report_lines:
        print(line)

    print(f"\nReport: {REPORT}")

    if not safe_to_publish:
        print(
            "\nACTION REQUIRED: Repeated district identities "
            "were found. Review the duplicate-identity audit "
            "before publishing the dataset."
        )


if __name__ == "__main__":
    main()
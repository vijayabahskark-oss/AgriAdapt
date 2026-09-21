"""
AgriAdapt - NASA POWER weather ingestion.

Downloads monthly NASA POWER Agroclimatology data for a documented set of
Indian agricultural reference locations, preserves raw JSON responses, and
creates annual location-level and India-level aggregates.

NASA POWER Monthly API:
https://power.larc.nasa.gov/docs/services/api/temporal/monthly/
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "weather" / "nasa_power"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "weather"

START_YEAR = 1981
END_YEAR = 2024

PARAMETERS = [
    "T2M",
    "PRECTOTCORR",
    "RH2M",
    "ALLSKY_SFC_SW_DWN",
    "WS2M",
]

# Prototype reference points distributed across major agricultural regions.
# These are NOT intended to be a statistically optimal national climate
# average; they provide a transparent, reproducible first prototype.
LOCATIONS = {
    "Punjab_Ludhiana": (30.9010, 75.8573),
    "Haryana_Hisar": (29.1492, 75.7217),
    "Uttar_Pradesh_Lucknow": (26.8467, 80.9462),
    "Bihar_Patna": (25.5941, 85.1376),
    "West_Bengal_Kolkata": (22.5726, 88.3639),
    "Maharashtra_Nagpur": (21.1458, 79.0882),
    "Telangana_Hyderabad": (17.3850, 78.4867),
    "Karnataka_Bengaluru": (12.9716, 77.5946),
    "Tamil_Nadu_Coimbatore": (11.0168, 76.9558),
    "Andhra_Pradesh_Vijayawada": (16.5062, 80.6480),
}

BASE_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"


def fetch_location(name: str, latitude: float, longitude: float) -> dict:
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "format": "JSON",
        "start": START_YEAR,
        "end": END_YEAR,
    }

    response = requests.get(BASE_URL, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / f"{name}_{START_YEAR}_{END_YEAR}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def payload_to_dataframe(name: str, latitude: float, longitude: float, payload: dict) -> pd.DataFrame:
    parameters = payload["properties"]["parameter"]

    # Monthly API keys are YYYYMM. Convert them to rows.
    rows = []
    for key in parameters["T2M"]:
        year = int(key[:4])
        month = int(key[4:])
        row = {
            "location": name,
            "latitude": latitude,
            "longitude": longitude,
            "year": year,
            "month": month,
        }
        for parameter in PARAMETERS:
            row[parameter] = parameters[parameter].get(key)
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    all_monthly = []

    for name, (latitude, longitude) in LOCATIONS.items():
        print(f"Downloading {name} ...")
        payload = fetch_location(name, latitude, longitude)
        all_monthly.append(
            payload_to_dataframe(name, latitude, longitude, payload)
        )
        # Stay conservative with request frequency.
        time.sleep(1)

    monthly = pd.concat(all_monthly, ignore_index=True)
    monthly = monthly.sort_values(["location", "year", "month"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    monthly_path = PROCESSED_DIR / "nasa_power_india_reference_locations_monthly.csv"
    monthly.to_csv(monthly_path, index=False)

    # NASA POWER's monthly temporal product provides averages by month.
    # For the prototype, annual weather features are calculated as:
    # - mean across months for temperature, humidity, solar radiation, wind
    # - sum across months for precipitation
    annual_location = (
        monthly.groupby(["location", "latitude", "longitude", "year"], as_index=False)
        .agg(
            temperature_mean=("T2M", "mean"),
            rainfall_total=("PRECTOTCORR", "sum"),
            humidity_mean=("RH2M", "mean"),
            solar_radiation_mean=("ALLSKY_SFC_SW_DWN", "mean"),
            wind_speed_mean=("WS2M", "mean"),
        )
    )

    location_path = PROCESSED_DIR / "nasa_power_india_reference_locations_annual.csv"
    annual_location.to_csv(location_path, index=False)

    # First prototype national aggregate: equal-weight mean of reference points.
    # This is explicitly a prototype approximation, not an official national
    # climatological average.
    annual_india = (
        annual_location.groupby("year", as_index=False)
        .agg(
            temperature_mean=("temperature_mean", "mean"),
            rainfall_total=("rainfall_total", "mean"),
            humidity_mean=("humidity_mean", "mean"),
            solar_radiation_mean=("solar_radiation_mean", "mean"),
            wind_speed_mean=("wind_speed_mean", "mean"),
        )
    )

    india_path = PROCESSED_DIR / "nasa_power_india_annual.csv"
    annual_india.to_csv(india_path, index=False)

    print("\nCreated:")
    print(monthly_path)
    print(location_path)
    print(india_path)
    print(f"\nMonthly rows: {len(monthly):,}")
    print(f"Annual location rows: {len(annual_location):,}")
    print(f"India annual rows: {len(annual_india):,}")
    print(f"Years: {annual_india['year'].min()}–{annual_india['year'].max()}")


if __name__ == "__main__":
    main()

"""
AgriAdapt - NASA POWER Weather Data Ingestion

Purpose
-------
Download, validate, clean, and aggregate NASA POWER monthly
weather data for representative Indian agricultural locations.

Outputs
-------
1. Monthly location-level weather data
2. Seasonal location-level weather features
3. Annual location-level weather features
4. Prototype India-level annual weather aggregate

Important methodological notes
------------------------------
- NASA POWER monthly responses may contain month=13, representing
  an annual summary. This is NOT a real calendar month and is removed.
- NASA POWER uses fill values such as -999 for unavailable data.
  These are converted to NaN before aggregation.
- Solar radiation availability differs from meteorological variables.
  Missing solar values are reported explicitly and are NOT blindly
  imputed.
- The India-level aggregate is an equal-weight prototype across the
  selected reference locations. It is NOT an authoritative national
  climate average.
- Location-level annual/seasonal data should be preferred for modelling.
- Seasonal definitions here are calendar-year groupings, not crop-specific
  agronomic seasons.
"""

from pathlib import Path
import json
import time
from typing import Dict, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# File location:
# D:\AgriAdapt\src\data\ingest_nasa_power.py
#
# parents[0] -> data
# parents[1] -> src
# parents[2] -> AgriAdapt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "weather"
    / "nasa_power"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather"
)

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. NASA POWER CONFIGURATION
# ============================================================

BASE_URL = (
    "https://power.larc.nasa.gov/api/temporal/monthly/point"
)

START_YEAR = 1981
END_YEAR = 2024

PARAMETERS = [
    "T2M",
    "PRECTOTCORR_SUM",
    "RH2M",
    "ALLSKY_SFC_SW_DWN",
    "WS2M",
]

COMMUNITY = "AG"

# IMPORTANT:
# Use UTC for the NASA POWER monthly API request.
TIME_STANDARD = "UTC"

EXPECTED_MONTHS = set(range(1, 13))


# ============================================================
# 3. REPRESENTATIVE INDIAN LOCATIONS
# ============================================================

LOCATIONS: Dict[str, Dict[str, float]] = {
    "Punjab_Ludhiana": {
        "latitude": 30.9010,
        "longitude": 75.8573,
    },
    "Haryana_Hisar": {
        "latitude": 29.1492,
        "longitude": 75.7217,
    },
    "Uttar_Pradesh_Lucknow": {
        "latitude": 26.8467,
        "longitude": 80.9462,
    },
    "Bihar_Patna": {
        "latitude": 25.5941,
        "longitude": 85.1376,
    },
    "West_Bengal_Kolkata": {
        "latitude": 22.5726,
        "longitude": 88.3639,
    },
    "Maharashtra_Nagpur": {
        "latitude": 21.1458,
        "longitude": 79.0882,
    },
    "Telangana_Hyderabad": {
        "latitude": 17.3850,
        "longitude": 78.4867,
    },
    "Karnataka_Bengaluru": {
        "latitude": 12.9716,
        "longitude": 77.5946,
    },
    "Tamil_Nadu_Coimbatore": {
        "latitude": 11.0168,
        "longitude": 76.9558,
    },
    "Andhra_Pradesh_Vijayawada": {
        "latitude": 16.5062,
        "longitude": 80.6480,
    },
}


# ============================================================
# 4. HTTP SESSION WITH RETRIES
# ============================================================

def create_session() -> requests.Session:
    """
    Create a requests session with retry support.

    Retries are useful for:
    - HTTP 429 rate limiting
    - HTTP 500
    - HTTP 502
    - HTTP 503
    - HTTP 504
    - transient connection/read failures
    """

    session = requests.Session()

    retry_strategy = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=2,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=[
            "GET",
        ],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
    )

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    session.headers.update(
        {
            "User-Agent": (
                "AgriAdapt/1.0 "
                "(research project; NASA POWER API client)"
            )
        }
    )

    return session


# ============================================================
# 5. BUILD NASA POWER API PARAMETERS
# ============================================================

def build_params(
    latitude: float,
    longitude: float,
) -> Dict[str, str]:
    """
    Build NASA POWER Monthly API query parameters.

    IMPORTANT:
    The NASA POWER Monthly/Annual API expects start/end
    as YEAR values, not YYYYMMDD dates.
    """

    return {
        "start": str(START_YEAR),
        "end": str(END_YEAR),
        "latitude": str(latitude),
        "longitude": str(longitude),
        "community": COMMUNITY,
        "parameters": ",".join(PARAMETERS),
        "format": "JSON",
        "time-standard": TIME_STANDARD,
    }


# ============================================================
# 6. DOWNLOAD ONE LOCATION
# ============================================================

def download_location(
    session: requests.Session,
    location_name: str,
    latitude: float,
    longitude: float,
) -> dict:
    """
    Download NASA POWER data for one location.

    The raw JSON response is saved locally before processing.
    """

    print()
    print("=" * 70)
    print(
        f"Downloading NASA POWER data: "
        f"{location_name}"
    )
    print(
        f"Coordinates: "
        f"{latitude:.4f}, {longitude:.4f}"
    )
    print("=" * 70)

    params = build_params(
        latitude=latitude,
        longitude=longitude,
    )

    raw_file = (
        RAW_DIR
        / f"nasa_power_{location_name}.json"
    )

    response = session.get(
        BASE_URL,
        params=params,
        timeout=120,
    )

    print(
        f"HTTP status: "
        f"{response.status_code}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Print NASA's actual error body before raising.
    #
    # This makes future 422 errors diagnosable instead of
    # showing only "422 Client Error".
    # --------------------------------------------------------

    if not response.ok:

        print()
        print("=" * 70)
        print("NASA POWER API ERROR")
        print("=" * 70)

        print(
            f"Status code: "
            f"{response.status_code}"
        )

        print(
            f"Reason: "
            f"{response.reason}"
        )

        print()
        print("Request URL:")
        print(response.url)

        print()
        print("NASA response body:")

        try:
            error_body = response.json()

            print(
                json.dumps(
                    error_body,
                    indent=2,
                )
            )

        except ValueError:
            print(
                response.text
            )

        print("=" * 70)

        response.raise_for_status()

    # --------------------------------------------------------
    # Parse successful response
    # --------------------------------------------------------

    data = response.json()

    # --------------------------------------------------------
    # Save exact raw API response
    # --------------------------------------------------------

    with open(
        raw_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
        )

    print(
        f"Raw response saved: "
        f"{raw_file}"
    )

    return data


# ============================================================
# 7. PARSE NASA POWER MONTHLY RESPONSE
# ============================================================

def parse_monthly_response(
    data: dict,
    location_name: str,
) -> pd.DataFrame:
    """
    Convert NASA POWER JSON into a tidy monthly DataFrame.
    """

    if "properties" not in data:

        raise ValueError(
            f"{location_name}: missing "
            "'properties' in NASA POWER response."
        )

    properties = data["properties"]

    if "parameter" not in properties:

        raise ValueError(
            f"{location_name}: missing "
            "'parameter' section in response."
        )

    parameter_data = properties["parameter"]

    if not parameter_data:

        raise ValueError(
            f"{location_name}: NASA POWER returned "
            "no parameter data."
        )

    rows: List[dict] = []

    all_periods = set()

    # --------------------------------------------------------
    # Collect all YYYYMM periods
    # --------------------------------------------------------

    for parameter in PARAMETERS:

        if parameter not in parameter_data:

            raise ValueError(
                f"{location_name}: missing parameter "
                f"{parameter} in API response."
            )

        all_periods.update(
            parameter_data[
                parameter
            ].keys()
        )

    # --------------------------------------------------------
    # Convert each YYYYMM record into one row
    # --------------------------------------------------------

    for period in sorted(all_periods):

        if len(period) != 6:

            print(
                f"WARNING: unexpected period "
                f"'{period}' for "
                f"{location_name}"
            )

            continue

        year = int(
            period[:4]
        )

        month = int(
            period[4:]
        )

        row = {
            "location": location_name,
            "year": year,
            "month": month,
        }

        for parameter in PARAMETERS:

            value = (
                parameter_data[
                    parameter
                ].get(period)
            )

            row[parameter] = value

        rows.append(row)

    monthly = pd.DataFrame(
        rows
    )

    return monthly


# ============================================================
# 8. CLEAN MONTHLY NASA POWER DATA
# ============================================================

def clean_monthly_data(
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean NASA POWER monthly data.

    Operations:
    - remove month=13
    - convert fill values (-999) to NaN
    - convert weather variables to numeric
    - validate year range
    - validate calendar months
    - detect duplicates
    """

    print()
    print(
        "Cleaning monthly NASA POWER data..."
    )

    # --------------------------------------------------------
    # Remove month=13
    # --------------------------------------------------------
    #
    # NASA POWER may include an annual summary as month 13.
    # It is NOT a real calendar month.
    #

    month_13_count = int(
        (
            monthly["month"] == 13
        ).sum()
    )

    if month_13_count > 0:

        print(
            f"Removing "
            f"{month_13_count} "
            "month=13 annual-summary records."
        )

    monthly = monthly[
        monthly["month"].between(
            1,
            12,
        )
    ].copy()

    # --------------------------------------------------------
    # Validate year range
    # --------------------------------------------------------

    invalid_years = monthly[
        ~monthly["year"].between(
            START_YEAR,
            END_YEAR,
        )
    ]

    if not invalid_years.empty:

        raise ValueError(
            "Found years outside requested range: "
            f"{invalid_years['year'].unique().tolist()}"
        )

    # --------------------------------------------------------
    # Convert parameters to numeric
    # --------------------------------------------------------

    for parameter in PARAMETERS:

        monthly[parameter] = pd.to_numeric(
            monthly[parameter],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Convert NASA POWER fill values to NaN
    # --------------------------------------------------------

    print()
    print(
        "NASA POWER fill-value counts:"
    )

    for parameter in PARAMETERS:

        count = int(
            monthly[
                parameter
            ]
            .isin(
                [
                    -999,
                    -999.0,
                ]
            )
            .sum()
        )

        print(
            f"  {parameter}: "
            f"{count}"
        )

        monthly.loc[
            monthly[
                parameter
            ].isin(
                [
                    -999,
                    -999.0,
                ]
            ),
            parameter,
        ] = pd.NA

    # --------------------------------------------------------
    # Duplicate validation
    # --------------------------------------------------------

    duplicate_mask = monthly.duplicated(
        subset=[
            "location",
            "year",
            "month",
        ],
        keep=False,
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:

        duplicate_rows = (
            monthly.loc[
                duplicate_mask
            ]
            .sort_values(
                [
                    "location",
                    "year",
                    "month",
                ]
            )
        )

        print()
        print(
            "Duplicate location/year/month records:"
        )

        print(
            duplicate_rows.head(
                20
            ).to_string(
                index=False
            )
        )

        raise ValueError(
            "Duplicate location/year/month "
            "records detected."
        )

    # --------------------------------------------------------
    # Month validation
    # --------------------------------------------------------

    invalid_months = monthly[
        ~monthly["month"].isin(
            EXPECTED_MONTHS
        )
    ]

    if not invalid_months.empty:

        raise ValueError(
            "Invalid calendar month detected: "
            f"{invalid_months['month'].unique().tolist()}"
        )

    return monthly


# ============================================================
# 9. VALIDATE 12 MONTHS PER LOCATION/YEAR
# ============================================================

def validate_month_completeness(
    monthly: pd.DataFrame,
) -> None:
    """
    Ensure every location/year contains exactly 12
    real calendar months.
    """

    counts = (
        monthly
        .groupby(
            [
                "location",
                "year",
            ]
        )["month"]
        .nunique()
        .reset_index(
            name="month_count"
        )
    )

    incomplete = counts[
        counts[
            "month_count"
        ] != 12
    ]

    if not incomplete.empty:

        print()
        print(
            "Incomplete location/year groups:"
        )

        print(
            incomplete.to_string(
                index=False
            )
        )

        raise ValueError(
            "One or more location/year groups "
            "do not contain exactly 12 months."
        )

    print(
        "Monthly completeness validation: PASSED"
    )


# ============================================================
# 10. REPORT SOLAR AVAILABILITY
# ============================================================

def report_solar_availability(
    monthly: pd.DataFrame,
) -> None:
    """
    Report years containing missing solar radiation values.

    Missing values are preserved as NaN.
    """

    solar_missing = monthly[
        monthly[
            "ALLSKY_SFC_SW_DWN"
        ].isna()
    ]

    if solar_missing.empty:

        print()
        print(
            "Solar radiation missing years: NONE"
        )

        return

    missing_years = (
        solar_missing[
            "year"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    print()
    print(
        "Solar radiation unavailable/missing "
        f"for years: {missing_years}"
    )

    print(
        "These values will remain NaN."
    )

    print(
        "They will NOT be blindly imputed."
    )

    missing_by_year = (
        solar_missing
        .groupby(
            "year"
        )
        .size()
        .reset_index(
            name="missing_solar_months"
        )
    )

    print()
    print(
        "Missing solar months by year:"
    )

    print(
        missing_by_year.to_string(
            index=False
        )
    )


# ============================================================
# 11. SAVE MONTHLY DATA
# ============================================================

def save_monthly_data(
    monthly: pd.DataFrame,
) -> Path:
    """
    Save cleaned monthly location-level data.
    """

    output_file = (
        PROCESSED_DIR
        / "nasa_power_india_reference_locations_monthly.csv"
    )

    monthly = (
        monthly
        .sort_values(
            [
                "location",
                "year",
                "month",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    monthly.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        f"Monthly dataset saved: "
        f"{output_file}"
    )

    print(
        f"Monthly rows: "
        f"{len(monthly):,}"
    )

    return output_file


# ============================================================
# 12. ASSIGN CALENDAR SEASON
# ============================================================

def assign_season(
    month: int,
) -> str:
    """
    Assign a calendar-year seasonal grouping.

    NOTE:
    These are calendar groupings rather than
    crop-specific agronomic seasons.
    """

    if month in [
        3,
        4,
        5,
    ]:

        return "pre_monsoon"

    if month in [
        6,
        7,
        8,
        9,
    ]:

        return "monsoon"

    if month in [
        10,
        11,
    ]:

        return "post_monsoon"

    if month in [
        12,
        1,
        2,
    ]:

        return "winter"

    raise ValueError(
        f"Invalid month: {month}"
    )


# ============================================================
# 13. CREATE SEASONAL FEATURES
# ============================================================

def create_seasonal_features(
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate monthly data into calendar-year seasonal features.

    Rainfall:
        SUM

    Temperature:
        MEAN / MAX

    Humidity:
        MEAN

    Solar:
        MEAN of available observations

    Wind:
        MEAN
    """

    data = monthly.copy()

    data["season"] = (
        data["month"]
        .apply(
            assign_season
        )
    )

    seasonal = (
        data
        .groupby(
            [
                "location",
                "year",
                "season",
            ]
        )
        .agg(
            temperature_mean=(
                "T2M",
                "mean",
            ),
            temperature_max=(
                "T2M",
                "max",
            ),
            rainfall_total=(
                "PRECTOTCORR_SUM",
                "sum",
            ),
            humidity_mean=(
                "RH2M",
                "mean",
            ),
            solar_mean=(
                "ALLSKY_SFC_SW_DWN",
                "mean",
            ),
            wind_mean=(
                "WS2M",
                "mean",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Convert seasons into columns
    # --------------------------------------------------------

    seasonal_wide = (
        seasonal
        .pivot(
            index=[
                "location",
                "year",
            ],
            columns="season",
            values=[
                "temperature_mean",
                "temperature_max",
                "rainfall_total",
                "humidity_mean",
                "solar_mean",
                "wind_mean",
            ],
        )
    )

    seasonal_wide.columns = [
        f"{metric}_{season}"
        for metric, season
        in seasonal_wide.columns
    ]

    seasonal_wide = (
        seasonal_wide
        .reset_index()
    )

    expected_seasons = [
        "pre_monsoon",
        "monsoon",
        "post_monsoon",
        "winter",
    ]

    expected_metrics = [
        "temperature_mean",
        "temperature_max",
        "rainfall_total",
        "humidity_mean",
        "solar_mean",
        "wind_mean",
    ]

    expected_columns = [
        f"{metric}_{season}"
        for metric in expected_metrics
        for season in expected_seasons
    ]

    for column in expected_columns:

        if column not in seasonal_wide.columns:

            seasonal_wide[column] = pd.NA

    seasonal_wide = seasonal_wide[
        [
            "location",
            "year",
        ]
        + expected_columns
    ]

    seasonal_wide = (
        seasonal_wide
        .sort_values(
            [
                "location",
                "year",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    output_file = (
        PROCESSED_DIR
        / "nasa_power_india_reference_locations_seasonal.csv"
    )

    seasonal_wide.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        f"Seasonal dataset saved: "
        f"{output_file}"
    )

    print(
        f"Seasonal rows: "
        f"{len(seasonal_wide):,}"
    )

    return seasonal_wide


# ============================================================
# 14. CREATE ANNUAL LOCATION FEATURES
# ============================================================

def create_annual_location_features(
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate monthly data into annual location-level
    weather features.

    This is the primary weather dataset for the prototype.
    """

    annual = (
        monthly
        .groupby(
            [
                "location",
                "year",
            ]
        )
        .agg(
            temperature_mean=(
                "T2M",
                "mean",
            ),
            temperature_max=(
                "T2M",
                "max",
            ),
            temperature_min=(
                "T2M",
                "min",
            ),
            temperature_std=(
                "T2M",
                "std",
            ),
            rainfall_total=(
                "PRECTOTCORR_SUM",
                "sum",
            ),
            rainfall_mean_monthly=(
                "PRECTOTCORR_SUM",
                "mean",
            ),
            humidity_mean=(
                "RH2M",
                "mean",
            ),
            humidity_max=(
                "RH2M",
                "max",
            ),
            solar_mean=(
                "ALLSKY_SFC_SW_DWN",
                "mean",
            ),
            solar_min=(
                "ALLSKY_SFC_SW_DWN",
                "min",
            ),
            wind_mean=(
                "WS2M",
                "mean",
            ),
            wind_max=(
                "WS2M",
                "max",
            ),
        )
        .reset_index()
    )

    annual = (
        annual
        .sort_values(
            [
                "location",
                "year",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    output_file = (
        PROCESSED_DIR
        / "nasa_power_india_reference_locations_annual.csv"
    )

    annual.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        "Annual location dataset saved: "
        f"{output_file}"
    )

    print(
        f"Annual location rows: "
        f"{len(annual):,}"
    )

    return annual


# ============================================================
# 15. CREATE PROTOTYPE INDIA AGGREGATE
# ============================================================

def create_india_prototype_aggregate(
    annual_location: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create an equal-weight prototype India-level aggregate.

    IMPORTANT:
    This is NOT a true national climate average.

    Each selected reference location receives equal weight.

    A future production version should use a scientifically
    justified spatial weighting scheme such as:
    - crop-area weighting
    - gridded weather
    - district-level aggregation
    - state-level aggregation
    """

    feature_columns = [
        column
        for column in annual_location.columns
        if column not in [
            "location",
            "year",
        ]
    ]

    india_annual = (
        annual_location
        .groupby(
            "year"
        )[
            feature_columns
        ]
        .mean(
            numeric_only=True
        )
        .reset_index()
    )

    india_annual.insert(
        1,
        "aggregation_method",
        "equal_weight_reference_locations_prototype",
    )

    india_annual.insert(
        2,
        "num_reference_locations",
        annual_location[
            "location"
        ].nunique(),
    )

    india_annual = (
        india_annual
        .sort_values(
            "year"
        )
        .reset_index(
            drop=True
        )
    )

    output_file = (
        PROCESSED_DIR
        / "nasa_power_india_annual.csv"
    )

    india_annual.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        "Prototype India aggregate saved: "
        f"{output_file}"
    )

    print(
        f"India aggregate rows: "
        f"{len(india_annual):,}"
    )

    print()
    print(
        "WARNING:"
    )

    print(
        "The India-level dataset is an "
        "equal-weight prototype aggregate."
    )

    print(
        "It should NOT be described as an "
        "authoritative national climate average."
    )

    return india_annual


# ============================================================
# 16. DATA QUALITY SUMMARY
# ============================================================

def print_quality_summary(
    monthly: pd.DataFrame,
    annual_location: pd.DataFrame,
) -> None:
    """
    Print final data-quality statistics.
    """

    print()
    print("=" * 70)
    print(
        "FINAL DATA QUALITY SUMMARY"
    )
    print("=" * 70)

    print(
        f"Locations: "
        f"{monthly['location'].nunique()}"
    )

    print(
        f"Years: "
        f"{monthly['year'].min()} - "
        f"{monthly['year'].max()}"
    )

    print(
        f"Monthly rows: "
        f"{len(monthly):,}"
    )

    print(
        f"Annual location rows: "
        f"{len(annual_location):,}"
    )

    expected_monthly_rows = (
        len(LOCATIONS)
        * (
            END_YEAR
            - START_YEAR
            + 1
        )
        * 12
    )

    print(
        f"Expected monthly rows: "
        f"{expected_monthly_rows:,}"
    )

    if (
        len(monthly)
        != expected_monthly_rows
    ):

        print(
            "WARNING: Actual monthly row count "
            "does not match expected count."
        )

    else:

        print(
            "Monthly row-count validation: PASSED"
        )

    print()
    print(
        "Missing values by weather parameter:"
    )

    for parameter in PARAMETERS:

        missing_count = int(
            monthly[
                parameter
            ]
            .isna()
            .sum()
        )

        missing_percentage = (
            missing_count
            / len(monthly)
            * 100
        )

        print(
            f"  {parameter}: "
            f"{missing_count:,} "
            f"({missing_percentage:.2f}%)"
        )

    print()
    print(
        "Data quality checks completed."
    )


# ============================================================
# 17. MAIN PIPELINE
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print(
        "AgriAdapt - NASA POWER Data Ingestion"
    )
    print("=" * 70)

    print()
    print(
        f"Project root: "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Raw directory: "
        f"{RAW_DIR}"
    )

    print(
        f"Processed directory: "
        f"{PROCESSED_DIR}"
    )

    print()
    print(
        f"Requested period: "
        f"{START_YEAR} - {END_YEAR}"
    )

    print()
    print(
        "Parameters:"
    )

    for parameter in PARAMETERS:

        print(
            f"  - {parameter}"
        )

    print()
    print(
        f"Reference locations: "
        f"{len(LOCATIONS)}"
    )

    print()
    print(
        f"Time standard: "
        f"{TIME_STANDARD}"
    )

    # --------------------------------------------------------
    # Create HTTP session
    # --------------------------------------------------------

    session = create_session()

    all_monthly_data = []

    # --------------------------------------------------------
    # Download every location
    # --------------------------------------------------------

    for index, (
        location_name,
        coordinates,
    ) in enumerate(
        LOCATIONS.items(),
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(LOCATIONS)}] "
            f"{location_name}"
        )

        try:

            data = download_location(
                session=session,
                location_name=location_name,
                latitude=coordinates[
                    "latitude"
                ],
                longitude=coordinates[
                    "longitude"
                ],
            )

            monthly = parse_monthly_response(
                data=data,
                location_name=location_name,
            )

            all_monthly_data.append(
                monthly
            )

            print(
                f"Parsed rows: "
                f"{len(monthly):,}"
            )

        except Exception as exc:

            print()
            print(
                f"ERROR while processing "
                f"{location_name}:"
            )

            print(
                exc
            )

            raise

        # ----------------------------------------------------
        # Conservative delay between API requests
        # ----------------------------------------------------

        if index < len(LOCATIONS):

            print(
                "Waiting 1 second before "
                "next request..."
            )

            time.sleep(1)

    # --------------------------------------------------------
    # Ensure data exists
    # --------------------------------------------------------

    if not all_monthly_data:

        raise RuntimeError(
            "No NASA POWER data was downloaded."
        )

    # --------------------------------------------------------
    # Combine locations
    # --------------------------------------------------------

    monthly = pd.concat(
        all_monthly_data,
        ignore_index=True,
    )

    print()
    print(
        f"Combined raw monthly rows: "
        f"{len(monthly):,}"
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    monthly = clean_monthly_data(
        monthly
    )

    # --------------------------------------------------------
    # Validate 12 months/year
    # --------------------------------------------------------

    validate_month_completeness(
        monthly
    )

    # --------------------------------------------------------
    # Report solar availability
    # --------------------------------------------------------

    report_solar_availability(
        monthly
    )

    # --------------------------------------------------------
    # Save monthly dataset
    # --------------------------------------------------------

    save_monthly_data(
        monthly
    )

    # --------------------------------------------------------
    # Create seasonal features
    # --------------------------------------------------------

    create_seasonal_features(
        monthly
    )

    # --------------------------------------------------------
    # Create annual location features
    # --------------------------------------------------------

    annual_location = (
        create_annual_location_features(
            monthly
        )
    )

    # --------------------------------------------------------
    # Create prototype India aggregate
    # --------------------------------------------------------

    create_india_prototype_aggregate(
        annual_location
    )

    # --------------------------------------------------------
    # Final quality summary
    # --------------------------------------------------------

    print_quality_summary(
        monthly=monthly,
        annual_location=annual_location,
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "NASA POWER INGESTION COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print()
    print(
        "Generated files:"
    )

    print(
        PROCESSED_DIR
        / "nasa_power_india_reference_locations_monthly.csv"
    )

    print(
        PROCESSED_DIR
        / "nasa_power_india_reference_locations_seasonal.csv"
    )

    print(
        PROCESSED_DIR
        / "nasa_power_india_reference_locations_annual.csv"
    )

    print(
        PROCESSED_DIR
        / "nasa_power_india_annual.csv"
    )

    print()
    print(
        "Raw API responses:"
    )

    print(
        RAW_DIR
    )

    print()
    print(
        "IMPORTANT FOR MODEL TRAINING:"
    )

    print(
        "Use the location-level annual/seasonal "
        "datasets as the primary weather source."
    )

    print(
        "Treat nasa_power_india_annual.csv as "
        "a prototype equal-weight aggregate only."
    )


# ============================================================
# 18. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
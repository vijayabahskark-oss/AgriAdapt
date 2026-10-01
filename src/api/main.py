from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    ROOT
    / "models"
    / "production"
    / "v5"
)

MODEL_MANIFEST = (
    MODEL_DIR
    / "current_model.json"
)

PREDICTION_LOG_DIR = (
    ROOT
    / "data"
    / "production"
    / "v5"
)

PREDICTION_LOG = (
    PREDICTION_LOG_DIR
    / "predictions.csv"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AgriAdapt Production API",
    description=(
        "Production crop-yield prediction API "
        "for the AgriAdapt self-adaptive MLOps system."
    ),
    version="5.9.0",
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class PredictionRequest(BaseModel):

    crop: str = Field(
        ...,
        description="Crop name",
        examples=["Rice"],
    )

    season: str = Field(
        ...,
        description="Agricultural season",
        examples=["Kharif"],
    )

    temperature_mean: float = Field(
        ...,
        description="Mean temperature",
    )

    rainfall_total: float = Field(
        ...,
        description="Total seasonal rainfall",
    )

    humidity_mean: float = Field(
        ...,
        description="Mean relative humidity",
    )

    solar_radiation_mean: float = Field(
        ...,
        description="Mean solar radiation",
    )

    wind_speed_mean: float = Field(
        ...,
        description="Mean wind speed",
    )

    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Latitude",
    )

    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Longitude",
    )


# ============================================================
# MODEL MANAGER
# ============================================================

class ProductionModel:

    def __init__(self):

        self.model = None
        self.model_path = None
        self.version = None
        self.status = None

        self.load_model()

    def load_model(self):

        if not MODEL_MANIFEST.exists():

            raise FileNotFoundError(
                f"Production model manifest not found: "
                f"{MODEL_MANIFEST}"
            )

        with open(
            MODEL_MANIFEST,
            "r",
            encoding="utf-8",
        ) as f:

            manifest = json.load(f)

        active_model = manifest.get(
            "active_model"
        )

        if not active_model:

            raise ValueError(
                "current_model.json does not contain "
                "'active_model'."
            )

        self.model_path = (
            MODEL_DIR
            / active_model
        )

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Active production model not found: "
                f"{self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        self.version = manifest.get(
            "version"
        )

        self.status = manifest.get(
            "status",
            "unknown",
        )

    def reload_if_changed(self):

        if not MODEL_MANIFEST.exists():
            return

        with open(
            MODEL_MANIFEST,
            "r",
            encoding="utf-8",
        ) as f:

            manifest = json.load(f)

        active_model = manifest.get(
            "active_model"
        )

        version = manifest.get(
            "version"
        )

        if (
            active_model != self.model_path.name
            or version != self.version
        ):

            self.load_model()


# ============================================================
# PREDICTION LOGGER
# ============================================================

def log_prediction(
    request,
    prediction,
    model_version,
    model_name,
):
    """
    Append one production prediction event
    to the prediction monitoring dataset.
    """

    PREDICTION_LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "prediction_id":
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y%m%d%H%M%S%f"
            ),

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "crop":
            request.crop,

        "season":
            request.season,

        "temperature_mean":
            request.temperature_mean,

        "rainfall_total":
            request.rainfall_total,

        "humidity_mean":
            request.humidity_mean,

        "solar_radiation_mean":
            request.solar_radiation_mean,

        "wind_speed_mean":
            request.wind_speed_mean,

        "latitude":
            request.latitude,

        "longitude":
            request.longitude,

        "predicted_yield_tonnes_per_ha":
            prediction,

        "model_version":
            model_version,

        "model_name":
            model_name,
    }

    new_row = pd.DataFrame(
        [record]
    )

    if PREDICTION_LOG.exists():

        new_row.to_csv(
            PREDICTION_LOG,
            mode="a",
            header=False,
            index=False,
        )

    else:

        new_row.to_csv(
            PREDICTION_LOG,
            index=False,
        )


# ============================================================
# LOAD PRODUCTION MODEL
# ============================================================

try:

    production_model = ProductionModel()

except Exception as exc:

    production_model = None
    MODEL_LOAD_ERROR = str(exc)

else:

    MODEL_LOAD_ERROR = None


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    if production_model is None:

        return {
            "status": "unhealthy",
            "service": "AgriAdapt Production API",
            "model_loaded": False,
            "error": MODEL_LOAD_ERROR,
        }

    return {
        "status": "healthy",
        "service": "AgriAdapt Production API",
        "model_loaded": True,
        "model_status": production_model.status,
        "model_version": production_model.version,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model")
def model_info():

    if production_model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Production model is not available."
            ),
        )

    production_model.reload_if_changed()

    return {
        "model_name":
            production_model.model_path.name,

        "model_version":
            production_model.version,

        "model_status":
            production_model.status,

        "model_path":
            str(
                production_model.model_path
            ),
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    request: PredictionRequest,
):

    if production_model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Production model is not available."
            ),
        )

    try:

        # Check whether a newer model has
        # been promoted.
        production_model.reload_if_changed()

        input_data = pd.DataFrame(
            [
                {
                    "crop": request.crop,
                    "season": request.season,
                    "temperature_mean":
                        request.temperature_mean,
                    "rainfall_total":
                        request.rainfall_total,
                    "humidity_mean":
                        request.humidity_mean,
                    "solar_radiation_mean":
                        request.solar_radiation_mean,
                    "wind_speed_mean":
                        request.wind_speed_mean,
                    "latitude":
                        request.latitude,
                    "longitude":
                        request.longitude,
                }
            ]
        )

        prediction = (
            production_model.model.predict(
                input_data
            )
        )

        predicted_yield = float(
            prediction[0]
        )

        # ----------------------------------------------------
        # V5.9 PREDICTION MONITORING
        # ----------------------------------------------------

        log_prediction(
            request=request,
            prediction=predicted_yield,
            model_version=(
                production_model.version
            ),
            model_name=(
                production_model.model_path.name
            ),
        )

        return {
            "predicted_yield_tonnes_per_ha":
                predicted_yield,

            "model_version":
                production_model.version,

            "model_status":
                production_model.status,

            "model_name":
                production_model.model_path.name,

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                + str(exc)
            ),
        )
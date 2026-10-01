from pathlib import Path
from datetime import datetime
import json

import mlflow
from mlflow import MlflowClient


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

MANIFEST = (
    MODEL_DIR
    / "current_model.json"
)

MLFLOW_DB = ROOT / "mlflow.db"


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

client = MlflowClient(
    tracking_uri=mlflow.get_tracking_uri()
)


# ============================================================
# REGISTRY CONFIGURATION
# ============================================================

REGISTERED_MODEL_NAME = (
    "AgriAdapt_Crop_Yield_Model"
)


# ============================================================
# HELPERS
# ============================================================

def read_manifest():

    if not MANIFEST.exists():

        raise FileNotFoundError(
            f"Production manifest not found:\n{MANIFEST}"
        )

    with open(
        MANIFEST,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def register_model_if_needed():

    try:

        client.get_registered_model(
            REGISTERED_MODEL_NAME
        )

        print(
            f"Registered model already exists: "
            f"{REGISTERED_MODEL_NAME}"
        )

    except Exception:

        client.create_registered_model(
            REGISTERED_MODEL_NAME,
            description=(
                "AgriAdapt crop yield prediction "
                "model managed through the "
                "self-adaptive MLOps lifecycle."
            ),
        )

        print(
            f"Created registered model: "
            f"{REGISTERED_MODEL_NAME}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AGRIADAPT V5.6")
    print("MLFLOW MODEL REGISTRY")
    print("=" * 70)

    # --------------------------------------------------------
    # Read current production pointer
    # --------------------------------------------------------

    manifest = read_manifest()

    model_name = manifest[
        "active_model"
    ]

    model_version = manifest[
        "version"
    ]

    model_path = (
        MODEL_DIR
        / model_name
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Active production model not found:\n"
            f"{model_path}"
        )

    print()
    print("CURRENT PRODUCTION MODEL")
    print("-" * 70)

    print(
        f"Model   : {model_name}"
    )

    print(
        f"Version : {model_version}"
    )

    print(
        f"Path    : {model_path}"
    )

    # --------------------------------------------------------
    # Create registered model
    # --------------------------------------------------------

    register_model_if_needed()

    # --------------------------------------------------------
    # Create MLflow run
    # --------------------------------------------------------

    experiment = mlflow.set_experiment(
        "AgriAdapt_V5_Model_Registry"
    )

    with mlflow.start_run(
        run_name=(
            "register_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        ),
        experiment_id=experiment.experiment_id,
    ) as run:

        # ----------------------------------------------------
        # Log lineage
        # ----------------------------------------------------

        mlflow.set_tag(
            "project",
            "AgriAdapt",
        )

        mlflow.set_tag(
            "pipeline_version",
            "V5.6",
        )

        mlflow.set_tag(
            "source_model_version",
            model_version,
        )

        mlflow.set_tag(
            "registry_model",
            REGISTERED_MODEL_NAME,
        )

        mlflow.set_tag(
            "production_status",
            "production",
        )

        mlflow.log_param(
            "source_model",
            model_name,
        )

        mlflow.log_param(
            "source_model_path",
            str(model_path),
        )

        # ----------------------------------------------------
        # Log model artifact
        # ----------------------------------------------------

        artifact_path = "model"

        model_uri = (
            f"runs:/{run.info.run_id}/"
            f"{artifact_path}"
        )

        # ----------------------------------------------------
        # Register sklearn model
        # ----------------------------------------------------

        import joblib

        model = joblib.load(
            model_path
        )

        mlflow.sklearn.log_model(
           model,
           name=artifact_path,
           skops_trusted_types=[
              "numpy.dtype",
              "sklearn.tree._tree.Tree",
            ],
   )

        # ----------------------------------------------------
        # Register model version
        # ----------------------------------------------------

        registered_version = (
            mlflow.register_model(
                model_uri=model_uri,
                name=REGISTERED_MODEL_NAME,
            )
        )

        # ----------------------------------------------------
        # Add version metadata
        # ----------------------------------------------------

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            registered_version.version,
            "source_model_version",
            model_version,
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            registered_version.version,
            "status",
            "production",
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            registered_version.version,
            "registration_pipeline",
            "AgriAdapt V5.6",
        )

        print()
        print("=" * 70)
        print("MODEL REGISTERED")
        print("=" * 70)

        print(
            f"Registry name : "
            f"{REGISTERED_MODEL_NAME}"
        )

        print(
            f"Registry version : "
            f"{registered_version.version}"
        )

        print(
            f"Run ID : "
            f"{run.info.run_id}"
        )

        print(
            f"Source model : "
            f"{model_name}"
        )


if __name__ == "__main__":
    main()
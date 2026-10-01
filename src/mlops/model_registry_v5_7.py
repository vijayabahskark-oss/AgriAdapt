from pathlib import Path
import joblib
import mlflow
from mlflow import MlflowClient


ROOT = Path(__file__).resolve().parents[2]

MLFLOW_DB = ROOT / "mlflow.db"

REGISTERED_MODEL_NAME = "AgriAdapt_Crop_Yield_Model"

TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
]


def configure_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(
        "AgriAdapt_V5_Adaptation"
    )

    return MlflowClient(
        tracking_uri=TRACKING_URI
    )


def ensure_registered_model(client):
    try:
        client.get_registered_model(
            REGISTERED_MODEL_NAME
        )
        return
    except Exception:
        pass

    client.create_registered_model(
        REGISTERED_MODEL_NAME,
        description=(
            "Production crop-yield prediction "
            "model for the AgriAdapt "
            "self-adaptive MLOps system."
        ),
    )


def register_promoted_model(
    model_path,
    production_version,
):
    """
    Register ONLY a successfully promoted
    production model.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Promoted model does not exist: "
            f"{model_path}"
        )

    client = configure_mlflow()

    ensure_registered_model(client)

    with mlflow.start_run(
        run_name=(
            f"V5.7_Registry_{production_version}"
        )
    ) as run:

        run_id = run.info.run_id

        mlflow.set_tags(
            {
                "project": "AgriAdapt",
                "pipeline_version": "V5.7",
                "lifecycle": "production",
                "source": "AgriAdapt V5.7",
                "production_model_version":
                    production_version,
                "registry_model_name":
                    REGISTERED_MODEL_NAME,
            }
        )

        mlflow.log_param(
            "production_model_path",
            str(model_path),
        )

        model = joblib.load(model_path)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=TRUSTED_TYPES,
        )

        model_uri = model_info.model_uri

        registered = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME,
        )

        registry_version = registered.version

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            registry_version,
            "lifecycle",
            "production",
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            registry_version,
            "source",
            "AgriAdapt V5.7",
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            registry_version,
            "production_model_version",
            production_version,
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            registry_version,
            "mlflow_run_id",
            run_id,
        )

        alias = None

        if hasattr(
            client,
            "set_registered_model_alias",
        ):
            client.set_registered_model_alias(
                REGISTERED_MODEL_NAME,
                "production",
                registry_version,
            )

            alias = "production"

        return {
            "registered_model_name":
                REGISTERED_MODEL_NAME,
            "registry_version":
                str(registry_version),
            "production_model_version":
                production_version,
            "mlflow_run_id":
                run_id,
            "model_uri":
                model_uri,
            "alias":
                alias,
        }
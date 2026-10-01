from pathlib import Path
import json

import pandas as pd
import requests
import streamlit as st
import mlflow
from mlflow import MlflowClient


# ============================================================
# AGRIADAPT V5.11 DASHBOARD
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------
# Project directories
# ------------------------------------------------------------

MODEL_DIR = ROOT / "models" / "production" / "v5"
DATA_DIR = ROOT / "data" / "production" / "v5"

REPORT_DIR = ROOT / "reports"
LIFECYCLE_DIR = REPORT_DIR / "lifecycle" / "v5"
MONITORING_DIR = REPORT_DIR / "monitoring" / "v5"

# ------------------------------------------------------------
# Production artifacts
# ------------------------------------------------------------

MANIFEST_FILE = (
    MODEL_DIR / "current_model.json"
)

PREDICTIONS_FILE = (
    DATA_DIR / "predictions.csv"
)

EVALUATED_FILE = (
    DATA_DIR / "evaluated_predictions.csv"
)

# ------------------------------------------------------------
# Monitoring reports
# ------------------------------------------------------------

INPUT_DRIFT_FILE = (
    MONITORING_DIR
    / "drifted_drift_report.json"
)

PERFORMANCE_DRIFT_FILE = (
    MONITORING_DIR
    / "performance_drift_report_v5_10.json"
)

# ------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------

API_URL = "http://127.0.0.1:8000"

# ------------------------------------------------------------
# MLflow
# ------------------------------------------------------------

MLFLOW_DB = ROOT / "mlflow.db"

MLFLOW_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

MLFLOW_EXPERIMENT_NAME = (
    "AgriAdapt_V5_Adaptation"
)

MLFLOW_REGISTERED_MODEL = (
    "AgriAdapt_Crop_Yield_Model"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriAdapt",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LIGHT CUSTOM CSS
# ============================================================
# Only minimal CSS is used.
# No HTML-based hero/pipeline rendering is used.
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_json(path):
    """Safely load a JSON file."""

    if not path.exists():
        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return None


def load_csv(path):
    """Safely load a CSV file."""

    if not path.exists():
        return pd.DataFrame()

    try:

        return pd.read_csv(path)

    except Exception:

        return pd.DataFrame()


def latest_json(directory, pattern):
    """
    Return the newest JSON file matching
    the supplied pattern.
    """

    if not directory.exists():
        return None

    files = sorted(
        directory.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not files:
        return None

    return load_json(files[0])


def format_metric(value):
    """Format a metric safely."""

    if value is None:
        return "N/A"

    try:

        return f"{float(value):.4f}"

    except Exception:

        return str(value)


# ============================================================
# MLFLOW HELPERS
# ============================================================

def configure_mlflow():
    """
    Configure MLflow using the existing AgriAdapt
    SQLite tracking database.
    """

    try:

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        client = MlflowClient(
            tracking_uri=MLFLOW_TRACKING_URI
        )

        return client

    except Exception:

        return None


def get_registered_model_info(client):
    """
    Load the registered AgriAdapt production model.
    """

    if client is None:
        return None, []

    try:

        registered_model = (
            client.get_registered_model(
                MLFLOW_REGISTERED_MODEL
            )
        )

    except Exception:

        return None, []

    try:

        model_versions = (
            client.search_model_versions(
                f"name='{MLFLOW_REGISTERED_MODEL}'"
            )
        )

        model_versions = sorted(
            model_versions,
            key=lambda item: int(
                item.version
            ),
            reverse=True,
        )

    except Exception:

        model_versions = []

    return (
        registered_model,
        model_versions,
    )


def get_production_registry_version(
    client
):
    """
    Retrieve the MLflow 'production' alias.
    """

    if client is None:
        return None

    try:

        return (
            client.get_model_version_by_alias(
                MLFLOW_REGISTERED_MODEL,
                "production",
            )
        )

    except Exception:

        return None


def load_mlflow_runs(client):
    """
    Load recent adaptation runs from the
    AgriAdapt_V5_Adaptation experiment.
    """

    if client is None:
        return pd.DataFrame()

    try:

        experiment = (
            client.get_experiment_by_name(
                MLFLOW_EXPERIMENT_NAME
            )
        )

        if experiment is None:
            return pd.DataFrame()

        runs = client.search_runs(
            experiment_ids=[
                experiment.experiment_id
            ],
            order_by=[
                "attribute.start_time DESC"
            ],
            max_results=25,
        )

    except Exception:

        return pd.DataFrame()

    rows = []

    for run in runs:

        params = run.data.params
        metrics = run.data.metrics
        tags = run.data.tags

        state = tags.get(
            "lifecycle_state",
            tags.get(
                "state",
                "UNKNOWN",
            ),
        )

        promotion = tags.get(
            "promotion",
            "UNKNOWN",
        )

        drifted_columns = params.get(
            "drifted_columns",
            "N/A",
        )

        ground_truth = params.get(
            "ground_truth_available",
            "N/A",
        )

        current_mae = metrics.get(
            "current_mae"
        )

        candidate_mae = metrics.get(
            "candidate_mae"
        )

        mae_improvement = metrics.get(
            "mae_improvement_percent"
        )

        # Some versions of the tracking script may
        # store the value as a parameter rather than
        # a metric. Support both.
        if mae_improvement is None:

            mae_improvement = params.get(
                "actual_mae_improvement_percent"
            )

        try:

            start_time = pd.to_datetime(
                run.info.start_time,
                unit="ms",
            )

        except Exception:

            start_time = None

        rows.append(
            {
                "Run ID": run.info.run_id[:12],

                "State": state,

                "Promotion": promotion,

                "Drifted Columns": (
                    drifted_columns
                ),

                "Ground Truth": (
                    ground_truth
                ),

                "Current MAE": (
                    current_mae
                ),

                "Candidate MAE": (
                    candidate_mae
                ),

                "MAE Improvement %": (
                    mae_improvement
                ),

                "Start Time": start_time,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ============================================================
# LOAD PRODUCTION DATA
# ============================================================

manifest = load_json(
    MANIFEST_FILE
)

predictions = load_csv(
    PREDICTIONS_FILE
)

evaluated = load_csv(
    EVALUATED_FILE
)

lifecycle = latest_json(
    LIFECYCLE_DIR,
    "lifecycle_*.json",
)

input_drift = load_json(
    INPUT_DRIFT_FILE
)

performance_drift = load_json(
    PERFORMANCE_DRIFT_FILE
)


# ============================================================
# FASTAPI HEALTH CHECK
# ============================================================

api_online = False

try:

    response = requests.get(
        f"{API_URL}/health",
        timeout=2,
    )

    api_online = (
        response.status_code == 200
    )

except Exception:

    api_online = False


# ============================================================
# MLFLOW CONNECTION
# ============================================================

mlflow_client = configure_mlflow()

(
    registered_model,
    model_versions,
) = get_registered_model_info(
    mlflow_client
)

production_registry_version = (
    get_production_registry_version(
        mlflow_client
    )
)

mlflow_runs = load_mlflow_runs(
    mlflow_client
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🌾 AgriAdapt"
    )

    st.caption(
        "MLOps Control Center"
    )

    st.divider()

    st.subheader(
        "System Status"
    )

    # --------------------------------------------------------
    # FastAPI
    # --------------------------------------------------------

    if api_online:

        st.success(
            "FastAPI • ONLINE"
        )

    else:

        st.error(
            "FastAPI • OFFLINE"
        )

    # --------------------------------------------------------
    # Production model
    # --------------------------------------------------------

    if manifest:

        st.success(
            "Production Model • ACTIVE"
        )

        st.caption(
            "Version: "
            + str(
                manifest.get(
                    "version",
                    "UNKNOWN",
                )
            )
        )

    else:

        st.warning(
            "Production model manifest unavailable"
        )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    if mlflow_client:

        st.success(
            "MLflow • CONNECTED"
        )

    else:

        st.error(
            "MLflow • UNAVAILABLE"
        )

    st.divider()

    # --------------------------------------------------------
    # Production data
    # --------------------------------------------------------

    st.subheader(
        "Production Data"
    )

    st.write(
        f"Predictions: **{len(predictions)}**"
    )

    st.write(
        f"Evaluated: **{len(evaluated)}**"
    )

    st.divider()

    # --------------------------------------------------------
    # Monitoring
    # --------------------------------------------------------

    st.subheader(
        "Monitoring"
    )

    if input_drift:

        drift_count = input_drift.get(
            "DriftedColumnsCount",
            0,
        )

        st.write(
            f"Input drift: "
            f"**{drift_count} columns**"
        )

    if performance_drift:

        st.write(
            "Performance:"
        )

        st.write(
            f"`{performance_drift.get('state', 'UNKNOWN')}`"
        )

    st.divider()

    if st.button(
        "🔄 Refresh Dashboard",
        use_container_width=True,
    ):

        st.rerun()


# ============================================================
# HERO
# ============================================================

st.title(
    "🌾 AgriAdapt"
)

st.subheader(
    "Self-Adaptive Crop Yield Prediction System"
)

st.caption(
    "MLOps and Climate-Aware Continuous Learning"
)

st.divider()


# ============================================================
# MLOPS LIFECYCLE
# ============================================================

st.header(
    "MLOps Lifecycle"
)

pipeline = [
    "1. Data",
    "2. Prediction",
    "3. Drift Detection",
    "4. Ground Truth",
    "5. Performance",
    "6. Retraining",
    "7. Registry",
    "8. Production",
]

pipeline_columns = st.columns(
    len(pipeline)
)

for column, step in zip(
    pipeline_columns,
    pipeline,
):

    with column:

        st.info(
            step
        )


# ============================================================
# PRODUCTION MODEL
# ============================================================

st.header(
    "Production Model"
)

if manifest:

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Status",
            manifest.get(
                "status",
                "UNKNOWN",
            ),
        )

    with c2:

        st.metric(
            "Model Version",
            manifest.get(
                "version",
                "UNKNOWN",
            ),
        )

    with c3:

        st.metric(
            "Predictions",
            len(predictions),
        )

    with c4:

        st.metric(
            "Evaluated",
            len(evaluated),
        )

    st.caption(
        "Active model: "
        + str(
            manifest.get(
                "active_model",
                "UNKNOWN",
            )
        )
    )

else:

    st.error(
        "Production model manifest not found."
    )


# ============================================================
# MONITORING SIGNALS
# ============================================================

st.header(
    "Monitoring Signals"
)

c1, c2, c3, c4 = st.columns(4)


# ------------------------------------------------------------
# INPUT DATA DRIFT
# ------------------------------------------------------------

if input_drift:

    drift_count = input_drift.get(
        "DriftedColumnsCount",
        0,
    )

    if drift_count > 0:

        c1.error(
            f"Input Drift\n\n"
            f"{drift_count} columns"
        )

    else:

        c1.success(
            "Input Drift\n\n"
            "Stable"
        )

else:

    c1.warning(
        "Input Drift\n\n"
        "No report"
    )


# ------------------------------------------------------------
# PERFORMANCE DRIFT
# ------------------------------------------------------------

if performance_drift:

    performance_state = (
        performance_drift.get(
            "state",
            "UNKNOWN",
        )
    )

    if performance_state == (
        "PERFORMANCE_DRIFT_DETECTED"
    ):

        c2.error(
            "Performance Drift\n\n"
            "Detected"
        )

    elif performance_state == (
        "PERFORMANCE_STABLE"
    ):

        c2.success(
            "Performance Drift\n\n"
            "Stable"
        )

    elif performance_state == (
        "INSUFFICIENT_GROUND_TRUTH"
    ):

        c2.warning(
            "Performance\n\n"
            "Waiting for ground truth"
        )

    else:

        c2.info(
            f"Performance\n\n"
            f"{performance_state}"
        )

else:

    c2.warning(
        "Performance\n\n"
        "Not evaluated"
    )


# ------------------------------------------------------------
# GROUND TRUTH
# ------------------------------------------------------------

if not evaluated.empty:

    c3.success(
        f"Ground Truth\n\n"
        f"{len(evaluated)} records"
    )

else:

    c3.warning(
        "Ground Truth\n\n"
        "Waiting"
    )


# ------------------------------------------------------------
# LAST ADAPTATION EVENT
# ------------------------------------------------------------

if lifecycle:

    lifecycle_promotion = (
        lifecycle.get(
            "promotion",
            "UNKNOWN",
        )
    )

    lifecycle_state = (
        lifecycle.get(
            "state",
            "UNKNOWN",
        )
    )

    if lifecycle_promotion == "PROMOTED":

        c4.success(
            "Last Adaptation\n\n"
            "Model Promoted"
        )

    elif lifecycle_promotion == "REJECTED":

        c4.warning(
            "Last Adaptation\n\n"
            "Model Rejected"
        )

    else:

        c4.info(
            "Last Adaptation\n\n"
            f"{lifecycle_state}"
        )

else:

    c4.info(
        "Last Adaptation\n\n"
        "No event"
    )


# ============================================================
# PRODUCTION PERFORMANCE
# ============================================================

st.header(
    "Production Performance"
)

if not evaluated.empty:

    required_columns = {
        "actual_yield_tonnes_per_ha",
        "predicted_yield_tonnes_per_ha",
    }

    if required_columns.issubset(
        evaluated.columns
    ):

        actual = pd.to_numeric(
            evaluated[
                "actual_yield_tonnes_per_ha"
            ],
            errors="coerce",
        )

        predicted = pd.to_numeric(
            evaluated[
                "predicted_yield_tonnes_per_ha"
            ],
            errors="coerce",
        )

        valid = pd.DataFrame(
            {
                "Actual": actual,
                "Predicted": predicted,
            }
        ).dropna()

        if not valid.empty:

            error = (
                valid["Predicted"]
                - valid["Actual"]
            )

            mae = (
                error
                .abs()
                .mean()
            )

            rmse = (
                error
                .pow(2)
                .mean()
            ) ** 0.5

            r2 = None

            if len(valid) >= 2:

                ss_res = (
                    (
                        valid["Actual"]
                        - valid["Predicted"]
                    ) ** 2
                ).sum()

                ss_tot = (
                    (
                        valid["Actual"]
                        - valid["Actual"].mean()
                    ) ** 2
                ).sum()

                if ss_tot > 0:

                    r2 = (
                        1
                        - ss_res / ss_tot
                    )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "MAE",
                    f"{mae:.4f} t/ha",
                )

            with c2:

                st.metric(
                    "RMSE",
                    f"{rmse:.4f} t/ha",
                )

            with c3:

                st.metric(
                    "R²",
                    (
                        "N/A"
                        if r2 is None
                        else f"{r2:.4f}"
                    ),
                )

            if len(valid) >= 2:

                st.subheader(
                    "Actual vs Predicted"
                )

                chart_data = (
                    valid
                    .reset_index(drop=True)
                )

                st.line_chart(
                    chart_data
                )

            else:

                st.info(
                    "Only one ground-truth observation "
                    "is currently available. Additional "
                    "production observations are required "
                    "before displaying a meaningful "
                    "actual-vs-predicted performance trend."
                )

        else:

            st.info(
                "No valid evaluated observations."
            )

    else:

        st.warning(
            "Evaluation file does not contain the "
            "required performance columns."
        )

else:

    st.info(
        "Waiting for delayed ground-truth observations."
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.header(
    "Prediction History"
)

if not predictions.empty:

    display_columns = [
        "timestamp",
        "crop",
        "season",
        "temperature_mean",
        "rainfall_total",
        "predicted_yield_tonnes_per_ha",
        "model_version",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in predictions.columns
    ]

    history = (
        predictions[
            available_columns
        ]
        .sort_values(
            "timestamp",
            ascending=False,
        )
    )

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No production predictions recorded."
    )


# ============================================================
# INPUT DATA DRIFT
# ============================================================

st.header(
    "Input Data Drift"
)

if input_drift:

    drift_count = input_drift.get(
        "DriftedColumnsCount",
        0,
    )

    drift_share = input_drift.get(
        "DriftedColumnsShare",
        0,
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Drifted Columns",
            drift_count,
        )

    with c2:

        if drift_share is not None:

            st.metric(
                "Drift Share",
                f"{float(drift_share) * 100:.1f}%",
            )

        else:

            st.metric(
                "Drift Share",
                "N/A",
            )

    with st.expander(
        "View Evidently Drift Report"
    ):

        st.json(
            input_drift
        )

else:

    st.info(
        "Input drift report unavailable."
    )


# ============================================================
# PERFORMANCE DRIFT
# ============================================================

st.header(
    "Performance Drift"
)

if performance_drift:

    performance_state = (
        performance_drift.get(
            "state",
            "UNKNOWN",
        )
    )

    performance_decision = (
        performance_drift.get(
            "decision",
            "UNKNOWN",
        )
    )

    if performance_state == (
        "PERFORMANCE_DRIFT_DETECTED"
    ):

        st.error(
            "Performance drift detected.\n\n"
            f"Decision: `{performance_decision}`"
        )

    elif performance_state == (
        "PERFORMANCE_STABLE"
    ):

        st.success(
            "Production performance is stable.\n\n"
            f"Decision: `{performance_decision}`"
        )

    elif performance_state == (
        "INSUFFICIENT_GROUND_TRUTH"
    ):

        st.warning(
            "Insufficient ground truth.\n\n"
            "The system is waiting for additional "
            "post-harvest observations before making "
            "a performance-drift decision."
        )

    else:

        st.info(
            f"Performance state: "
            f"`{performance_state}`"
        )

    baseline = performance_drift.get(
        "baseline"
    )

    current = performance_drift.get(
        "current"
    )

    if baseline and current:

        c1, c2 = st.columns(2)

        with c1:

            st.subheader(
                "Baseline"
            )

            st.metric(
                "MAE",
                format_metric(
                    baseline.get(
                        "mae_tonnes_per_ha"
                    )
                ),
            )

            st.metric(
                "RMSE",
                format_metric(
                    baseline.get(
                        "rmse_tonnes_per_ha"
                    )
                ),
            )

            st.metric(
                "R²",
                format_metric(
                    baseline.get(
                        "r2"
                    )
                ),
            )

        with c2:

            st.subheader(
                "Current"
            )

            st.metric(
                "MAE",
                format_metric(
                    current.get(
                        "mae_tonnes_per_ha"
                    )
                ),
            )

            st.metric(
                "RMSE",
                format_metric(
                    current.get(
                        "rmse_tonnes_per_ha"
                    )
                ),
            )

            st.metric(
                "R²",
                format_metric(
                    current.get(
                        "r2"
                    )
                ),
            )

    degradation = (
        performance_drift.get(
            "degradation"
        )
    )

    if degradation:

        mae_degradation = (
            degradation.get(
                "mae_degradation_percent"
            )
        )

        if mae_degradation is not None:

            st.write(
                "MAE degradation: "
                f"**{float(mae_degradation):.2f}%**"
            )

    with st.expander(
        "View Performance Drift Report"
    ):

        st.json(
            performance_drift
        )

else:

    st.info(
        "Performance drift report unavailable."
    )


# ============================================================
# ADAPTATION LIFECYCLE
# ============================================================

st.header(
    "Adaptation Lifecycle"
)


# ------------------------------------------------------------
# CURRENT PRODUCTION STATE
# ------------------------------------------------------------

current_status = (
    manifest.get(
        "status",
        "UNKNOWN",
    )
    if manifest
    else "UNKNOWN"
)

current_version = (
    manifest.get(
        "version",
        "UNKNOWN",
    )
    if manifest
    else "UNKNOWN"
)

c1, c2 = st.columns(2)


with c1:

    st.subheader(
        "Current Production"
    )

    if (
        str(current_status).lower()
        == "production"
    ):

        st.success(
            "🟢 PRODUCTION\n\n"
            f"Version: `{current_version}`"
        )

    else:

        st.warning(
            f"Status: `{current_status}`"
        )


# ------------------------------------------------------------
# LAST ADAPTATION EVENT
# ------------------------------------------------------------

with c2:

    st.subheader(
        "Last Adaptation Event"
    )

    if lifecycle:

        last_state = lifecycle.get(
            "state",
            "UNKNOWN",
        )

        last_action = lifecycle.get(
            "action",
            "UNKNOWN",
        )

        last_promotion = lifecycle.get(
            "promotion",
            "UNKNOWN",
        )

        if last_promotion == "PROMOTED":

            st.success(
                "🟢 MODEL PROMOTED\n\n"
                f"State: `{last_state}`"
            )

        elif last_promotion == "REJECTED":

            st.warning(
                "🟡 MODEL REJECTED\n\n"
                f"State: `{last_state}`"
            )

        else:

            st.info(
                f"State: `{last_state}`\n\n"
                f"Action: `{last_action}`"
            )

    else:

        st.info(
            "No adaptation event recorded."
        )


# ------------------------------------------------------------
# LIFECYCLE DETAILS
# ------------------------------------------------------------

if lifecycle:

    with st.expander(
        "View Last Adaptation Event"
    ):

        lifecycle_table = pd.DataFrame(
            [
                {
                    "State": lifecycle.get(
                        "state",
                        "UNKNOWN",
                    ),
                    "Action": lifecycle.get(
                        "action",
                        "UNKNOWN",
                    ),
                    "Promotion": lifecycle.get(
                        "promotion",
                        "UNKNOWN",
                    ),
                }
            ]
        )

        st.dataframe(
            lifecycle_table,
            use_container_width=True,
            hide_index=True,
        )

        st.json(
            lifecycle
        )

else:

    st.info(
        "No adaptation lifecycle report available."
    )


# ============================================================
# MODEL LINEAGE
# ============================================================

st.header(
    "Model Lineage"
)

if manifest:

    lineage = pd.DataFrame(
        [
            {
                "Model": manifest.get(
                    "active_model",
                    "UNKNOWN",
                ),
                "Version": manifest.get(
                    "version",
                    "UNKNOWN",
                ),
                "Status": manifest.get(
                    "status",
                    "UNKNOWN",
                ),
                "Updated": manifest.get(
                    "updated_at",
                    "UNKNOWN",
                ),
            }
        ]
    )

    st.dataframe(
        lineage,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Model lineage unavailable."
    )


# ============================================================
# V5.11 — MLFLOW MODEL REGISTRY
# ============================================================

st.header(
    "MLflow Model Registry"
)

if registered_model:

    # --------------------------------------------------------
    # Registry summary
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Registered Model",
            registered_model.name,
        )

    with c2:

        st.metric(
            "Registry Versions",
            len(model_versions),
        )

    with c3:

        st.metric(
            "Registry Status",
            "CONNECTED",
        )

    if registered_model.description:

        st.caption(
            registered_model.description
        )

    else:

        st.caption(
            "AgriAdapt production model registered in MLflow."
        )

    # --------------------------------------------------------
    # Production alias
    # --------------------------------------------------------

    st.subheader(
        "Production Registry Version"
    )

    if production_registry_version:

        production_tags = (
            production_registry_version.tags
        )

        st.success(
            "🟢 Production alias is active"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Registry Version",
                production_registry_version.version,
            )

        with c2:

            st.metric(
                "Lifecycle",
                production_tags.get(
                    "lifecycle",
                    "N/A",
                ),
            )

        with c3:

            st.metric(
                "Production Version",
                production_tags.get(
                    "production_model_version",
                    "N/A",
                ),
            )

        with c4:

            registry_run_id = (
                production_tags.get(
                    "mlflow_run_id",
                    "N/A",
                )
            )

            st.metric(
                "Run ID",
                (
                    registry_run_id[:12]
                    if registry_run_id != "N/A"
                    else "N/A"
                ),
            )

        source_model = (
            production_tags.get(
                "source_model",
                "N/A",
            )
        )

        st.caption(
            f"Source model: `{source_model}`"
        )

    else:

        st.warning(
            "No `production` alias is currently assigned "
            "to the registered model."
        )

    # --------------------------------------------------------
    # Registry version table
    # --------------------------------------------------------

    if model_versions:

        st.subheader(
            "Registered Model Versions"
        )

        registry_rows = []

        for version in model_versions:

            tags = version.tags

            registry_rows.append(
                {
                    "Version": version.version,

                    "Lifecycle": tags.get(
                        "lifecycle",
                        "N/A",
                    ),

                    "Production Version": (
                        tags.get(
                            "production_model_version",
                            "N/A",
                        )
                    ),

                    "Source": tags.get(
                        "source_model",
                        "N/A",
                    ),

                    "Created": pd.to_datetime(
                        version.creation_timestamp,
                        unit="ms",
                    ),

                    "Run ID": (
                        tags.get(
                            "mlflow_run_id",
                            "N/A",
                        )[:12]
                    ),
                }
            )

        registry_df = pd.DataFrame(
            registry_rows
        )

        st.dataframe(
            registry_df,
            use_container_width=True,
            hide_index=True,
        )

else:

    st.warning(
        "MLflow Model Registry is unavailable."
    )

    st.caption(
        "Tracking URI: "
        f"`{MLFLOW_TRACKING_URI}`"
    )


# ============================================================
# V5.11 — MLFLOW ADAPTATION RUNS
# ============================================================

st.header(
    "MLflow Adaptation Runs"
)

if not mlflow_runs.empty:

    # --------------------------------------------------------
    # Run summary
    # --------------------------------------------------------

    total_runs = len(
        mlflow_runs
    )

    promoted_runs = len(
        mlflow_runs[
            mlflow_runs["Promotion"]
            == "PROMOTED"
        ]
    )

    rejected_runs = len(
        mlflow_runs[
            mlflow_runs["Promotion"]
            == "REJECTED"
        ]
    )

    stable_runs = len(
        mlflow_runs[
            mlflow_runs["State"]
            == "STABLE"
        ]
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Total Runs",
            total_runs,
        )

    with c2:

        st.metric(
            "Promoted",
            promoted_runs,
        )

    with c3:

        st.metric(
            "Rejected",
            rejected_runs,
        )

    with c4:

        st.metric(
            "Stable",
            stable_runs,
        )

    # --------------------------------------------------------
    # Recent runs
    # --------------------------------------------------------

    st.subheader(
        "Recent Adaptation Experiments"
    )

    display_runs = (
        mlflow_runs.copy()
    )

    if "Current MAE" in display_runs.columns:

        display_runs[
            "Current MAE"
        ] = pd.to_numeric(
            display_runs[
                "Current MAE"
            ],
            errors="coerce",
        ).round(4)

    if "Candidate MAE" in display_runs.columns:

        display_runs[
            "Candidate MAE"
        ] = pd.to_numeric(
            display_runs[
                "Candidate MAE"
            ],
            errors="coerce",
        ).round(4)

    if "MAE Improvement %" in display_runs.columns:

        display_runs[
            "MAE Improvement %"
        ] = pd.to_numeric(
            display_runs[
                "MAE Improvement %"
            ],
            errors="coerce",
        ).round(2)

    st.dataframe(
        display_runs,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Latest run
    # --------------------------------------------------------

    latest_run = (
        mlflow_runs.iloc[0]
    )

    st.subheader(
        "Latest MLflow Run"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.write(
            f"**Run ID:** "
            f"`{latest_run['Run ID']}`"
        )

        st.write(
            f"**State:** "
            f"`{latest_run['State']}`"
        )

        st.write(
            f"**Promotion:** "
            f"`{latest_run['Promotion']}`"
        )

        st.write(
            f"**Drifted Columns:** "
            f"`{latest_run['Drifted Columns']}`"
        )

        st.write(
            f"**Ground Truth:** "
            f"`{latest_run['Ground Truth']}`"
        )

    with c2:

        st.write(
            f"**Current MAE:** "
            f"`{latest_run['Current MAE']}`"
        )

        st.write(
            f"**Candidate MAE:** "
            f"`{latest_run['Candidate MAE']}`"
        )

        st.write(
            f"**MAE Improvement:** "
            f"`{latest_run['MAE Improvement %']}`"
        )

        st.write(
            f"**Start Time:** "
            f"`{latest_run['Start Time']}`"
        )

else:

    st.info(
        "No MLflow adaptation runs found."
    )


# ============================================================
# LIVE CROP YIELD PREDICTION
# ============================================================

st.header(
    "Live Crop Yield Prediction"
)

st.caption(
    "The prediction is served by the active production "
    "model through FastAPI."
)

with st.form(
    "prediction_form"
):

    c1, c2, c3 = st.columns(3)

    # --------------------------------------------------------
    # Crop / Season / Temperature
    # --------------------------------------------------------

    with c1:

        crop = st.selectbox(
            "Crop",
            [
                "Rice",
                "Wheat",
                "Maize",
            ],
        )

        season = st.selectbox(
            "Season",
            [
                "Kharif",
                "Rabi",
                "Summer",
                "Winter",
                "Autumn",
                "Whole year",
            ],
        )

        temperature = st.number_input(
            "Temperature Mean (°C)",
            value=28.4,
            step=0.1,
        )

    # --------------------------------------------------------
    # Rainfall / Humidity / Solar
    # --------------------------------------------------------

    with c2:

        rainfall = st.number_input(
            "Rainfall Total (mm)",
            value=1120.5,
            step=1.0,
        )

        humidity = st.number_input(
            "Humidity Mean (%)",
            value=76.2,
            step=0.1,
        )

        solar = st.number_input(
            "Solar Radiation",
            value=18.4,
            step=0.1,
        )

    # --------------------------------------------------------
    # Wind / Coordinates
    # --------------------------------------------------------

    with c3:

        wind = st.number_input(
            "Wind Speed",
            value=2.7,
            step=0.1,
        )

        latitude = st.number_input(
            "Latitude",
            value=13.08,
            step=0.01,
        )

        longitude = st.number_input(
            "Longitude",
            value=80.27,
            step=0.01,
        )

    submitted = st.form_submit_button(
        "🌱 Predict Yield",
        use_container_width=True,
    )


# ============================================================
# PREDICTION REQUEST
# ============================================================

if submitted:

    payload = {
        "crop": crop,
        "season": season,
        "temperature_mean": temperature,
        "rainfall_total": rainfall,
        "humidity_mean": humidity,
        "solar_radiation_mean": solar,
        "wind_speed_mean": wind,
        "latitude": latitude,
        "longitude": longitude,
    }

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10,
        )

        if response.status_code == 200:

            result = response.json()

            st.success(
                "Prediction generated successfully."
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Predicted Yield",
                    (
                        f"{result['predicted_yield_tonnes_per_ha']:.4f} "
                        "t/ha"
                    ),
                )

            with c2:

                st.metric(
                    "Model Version",
                    result.get(
                        "model_version",
                        "UNKNOWN",
                    ),
                )

            with c3:

                st.metric(
                    "Model Status",
                    result.get(
                        "model_status",
                        "UNKNOWN",
                    ),
                )

            with st.expander(
                "View Prediction Response"
            ):

                st.json(
                    result
                )

        else:

            st.error(
                f"FastAPI returned "
                f"{response.status_code}:\n\n"
                f"{response.text}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "FastAPI is not running.\n\n"
            "Start it with:\n\n"
            "uvicorn src.api.main:app "
            "--host 127.0.0.1 --port 8000"
        )

    except Exception as exc:

        st.error(
            f"Prediction request failed:\n\n"
            f"{exc}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AgriAdapt • MLOps Research Prototype • "
    "Climate-Aware Crop Yield Prediction • V5.11"
)
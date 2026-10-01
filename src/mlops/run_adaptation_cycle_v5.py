from pathlib import Path
from datetime import datetime
import json
import shutil

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REFERENCE = (
    ROOT
    / "models"
    / "production"
    / "v5"
    / "reference_training_data.csv"
)

PRODUCTION_DIR = (
    ROOT
    / "data"
    / "production"
    / "v5"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "production"
    / "v5"
)

MONITORING_DIR = (
    ROOT
    / "reports"
    / "monitoring"
    / "v5"
)

LIFECYCLE_DIR = (
    ROOT
    / "reports"
    / "lifecycle"
    / "v5"
)

CURRENT_MODEL_MANIFEST = (
    MODEL_DIR
    / "current_model.json"
)


# ============================================================
# FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    "crop",
    "season",
]

NUMERIC_FEATURES = [
    "temperature_mean",
    "rainfall_total",
    "humidity_mean",
    "solar_radiation_mean",
    "wind_speed_mean",
    "latitude",
    "longitude",
]

FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)

DRIFT_THRESHOLD = 3


# ============================================================
# MODEL MANIFEST
# ============================================================

def initialize_model_manifest():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if CURRENT_MODEL_MANIFEST.exists():
        return

    manifest = {
        "project": "AgriAdapt",
        "status": "production",
        "active_model": "agriadapt_production_model.joblib",
        "version": "v5.1",
        "updated_at": datetime.now().isoformat(),
    }

    with open(
        CURRENT_MODEL_MANIFEST,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )


def read_model_manifest():

    initialize_model_manifest()

    with open(
        CURRENT_MODEL_MANIFEST,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def update_model_manifest(
    model_name,
    version,
):

    manifest = {
        "project": "AgriAdapt",
        "status": "production",
        "active_model": model_name,
        "version": version,
        "updated_at": datetime.now().isoformat(),
    }

    with open(
        CURRENT_MODEL_MANIFEST,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )


# ============================================================
# DRIFT DETECTION
# ============================================================

def run_drift_detection(
    reference,
    production,
    report_name,
):

    reference_features = (
        reference[FEATURES]
        .copy()
    )

    production_features = (
        production[FEATURES]
        .copy()
    )

    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    snapshot = report.run(
        current_data=production_features,
        reference_data=reference_features,
    )

    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    html_path = (
        MONITORING_DIR
        / f"{report_name}_drift_report.html"
    )

    json_path = (
        MONITORING_DIR
        / f"{report_name}_drift_report.json"
    )

    snapshot.save_html(
        str(html_path)
    )

    try:

        snapshot.save_json(
            str(json_path)
        )

    except Exception:

        json_path = None

    drifted_columns = None

    for metric in snapshot.dict().get(
        "metrics",
        [],
    ):

        metric_name = metric.get(
            "metric_name",
            "",
        )

        if metric_name.startswith(
            "DriftedColumnsCount"
        ):

            value = metric.get(
                "value",
                {},
            )

            if isinstance(
                value,
                dict,
            ):

                drifted_columns = int(
                    value.get(
                        "count",
                        0,
                    )
                )

            break

    if drifted_columns is None:

        raise ValueError(
            "Could not determine DriftedColumnsCount."
        )

    return {
        "drifted_columns": drifted_columns,
        "html_report": str(html_path),
        "json_report": (
            str(json_path)
            if json_path
            else None
        ),
    }


# ============================================================
# LIFECYCLE
# ============================================================

def run_cycle(
    batch_name,
    labels_available=False,
):

    timestamp = datetime.now()

    run_id = timestamp.strftime(
        "%Y%m%d_%H%M%S"
    )

    LIFECYCLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    initialize_model_manifest()

    manifest = read_model_manifest()

    batch_path = (
        PRODUCTION_DIR
        / batch_name
    )

    if not REFERENCE.exists():

        raise FileNotFoundError(
            f"Reference data not found:\n{REFERENCE}"
        )

    if not batch_path.exists():

        raise FileNotFoundError(
            f"Production batch not found:\n{batch_path}"
        )

    reference = pd.read_csv(
        REFERENCE
    )

    production = pd.read_csv(
        batch_path
    )

    required_columns = FEATURES

    missing = [
        column
        for column in required_columns
        if column not in production.columns
    ]

    if missing:

        raise ValueError(
            f"Production batch missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Drift detection
    # --------------------------------------------------------

    drift = run_drift_detection(
        reference,
        production,
        f"lifecycle_{run_id}",
    )

    drifted_columns = drift[
        "drifted_columns"
    ]

    # --------------------------------------------------------
    # Initial lifecycle record
    # --------------------------------------------------------

    lifecycle = {

        "project": "AgriAdapt",

        "version": "V5.4",

        "run_id": run_id,

        "timestamp": timestamp.isoformat(),

        "production_batch": str(
            batch_path
        ),

        "production_rows": int(
            len(production)
        ),

        "reference_rows": int(
            len(reference)
        ),

        "drift": {

            "drifted_columns": int(
                drifted_columns
            ),

            "threshold": DRIFT_THRESHOLD,

            "detected": bool(
                drifted_columns
                >= DRIFT_THRESHOLD
            ),

            "html_report": drift[
                "html_report"
            ],

            "json_report": drift[
                "json_report"
            ],
        },

        "ground_truth": {

            "available": bool(
                labels_available
            ),

            "note": (
                "Prototype mode uses "
                "historical labeled data "
                "as simulated post-harvest "
                "ground truth."
            ),
        },

        "previous_model": manifest,

        "state": None,

        "action": None,

        "promotion": None,

    }

    # --------------------------------------------------------
    # Branch 1 — Stable
    # --------------------------------------------------------

    if drifted_columns < DRIFT_THRESHOLD:

        lifecycle["state"] = "STABLE"

        lifecycle["action"] = "NO_ACTION"

        lifecycle["promotion"] = (
            "KEEP_PRODUCTION_MODEL"
        )

        output = (
            LIFECYCLE_DIR
            / f"lifecycle_{run_id}.json"
        )

        with open(
            output,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                lifecycle,
                f,
                indent=2,
            )

        print()
        print("=" * 70)
        print("AGRIADAPT V5.4")
        print("ADAPTATION CYCLE")
        print("=" * 70)

        print(
            f"Drifted columns : {drifted_columns}"
        )

        print(
            "STATE            : STABLE"
        )

        print(
            "ACTION           : NO_ACTION"
        )

        print(
            f"Lifecycle report : {output}"
        )

        return lifecycle

    # --------------------------------------------------------
    # Branch 2 — Drift but no labels
    # --------------------------------------------------------

    if not labels_available:

        lifecycle["state"] = (
            "DRIFT_DETECTED_WAIT_FOR_LABELS"
        )

        lifecycle["action"] = (
            "WAIT_FOR_GROUND_TRUTH"
        )

        lifecycle["promotion"] = (
            "KEEP_PRODUCTION_MODEL"
        )

        output = (
            LIFECYCLE_DIR
            / f"lifecycle_{run_id}.json"
        )

        with open(
            output,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                lifecycle,
                f,
                indent=2,
            )

        print()
        print("=" * 70)
        print("AGRIADAPT V5.4")
        print("ADAPTATION CYCLE")
        print("=" * 70)

        print(
            f"Drifted columns : {drifted_columns}"
        )

        print(
            "STATE            : "
            "DRIFT_DETECTED_WAIT_FOR_LABELS"
        )

        print(
            "ACTION           : "
            "WAIT_FOR_GROUND_TRUTH"
        )

        print(
            f"Lifecycle report : {output}"
        )

        return lifecycle

    # --------------------------------------------------------
    # Branch 3 — Drift + labels
    # --------------------------------------------------------

    lifecycle["state"] = (
        "RETRAINING_REQUIRED"
    )

    lifecycle["action"] = (
        "CONTROLLED_RETRAINING"
    )

    print()
    print("=" * 70)
    print("AGRIADAPT V5.4")
    print("ADAPTATION CYCLE")
    print("=" * 70)

    print(
        f"Drifted columns : {drifted_columns}"
    )

    print(
        "STATE            : RETRAINING_REQUIRED"
    )

    print(
        "GROUND TRUTH     : AVAILABLE"
    )

    # --------------------------------------------------------
    # Reuse V5.3 retraining implementation
    # --------------------------------------------------------

    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    if str(PROJECT_ROOT) not in sys.path:
       sys.path.insert(0, str(PROJECT_ROOT))

    from src.retraining.controlled_retraining_v5 import (
       build_pipeline,
       metrics,
)

    import joblib

    production_model_path = (
        MODEL_DIR
        / manifest["active_model"]
    )

    dataset_path = (
        ROOT
        / "data"
        / "processed"
        / "v4"
        / "v4_climate_yield_dataset.csv"
    )

    if not production_model_path.exists():

        raise FileNotFoundError(
            f"Active production model not found:\n"
            f"{production_model_path}"
        )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Training dataset not found:\n"
            f"{dataset_path}"
        )

    production_model = joblib.load(
        production_model_path
    )

    df = pd.read_csv(
        dataset_path
    )

    TARGET = "yield_tonnes_per_ha"

    df["year_start"] = (
        df["year"]
        .astype(str)
        .str[:4]
        .astype(int)
    )

    train = df[
        df["year_start"] <= 2010
    ].copy()

    holdout = df[
        df["year_start"] > 2010
    ].copy()

    current_metrics = metrics(
        production_model,
        holdout,
    )

    candidate = build_pipeline()

    candidate.fit(
        train[FEATURES],
        train[TARGET],
    )

    candidate_metrics = metrics(
        candidate,
        holdout,
    )

    mae_improvement = (
        current_metrics["mae"]
        - candidate_metrics["mae"]
    ) / current_metrics["mae"]

    candidate_accepted = (
        mae_improvement >= 0.02
        and candidate_metrics["r2"]
        >= current_metrics["r2"]
    )

    candidate_path = (
        MODEL_DIR
        / "candidate_model_v5_4.joblib"
    )

    joblib.dump(
        candidate,
        candidate_path,
    )

    lifecycle["candidate_metrics"] = (
        candidate_metrics
    )

    lifecycle["current_metrics"] = (
        current_metrics
    )

    lifecycle["mae_improvement_fraction"] = (
        float(mae_improvement)
    )

    lifecycle["promotion_gate"] = {

        "minimum_mae_improvement": 0.02,

        "actual_mae_improvement": float(
            mae_improvement
        ),

        "candidate_r2": float(
            candidate_metrics["r2"]
        ),

        "current_r2": float(
            current_metrics["r2"]
        ),

        "accepted": bool(
            candidate_accepted
        ),
    }

    lifecycle["candidate_model"] = str(
        candidate_path
    )

    # --------------------------------------------------------
    # Promotion
    # --------------------------------------------------------

    if candidate_accepted:

        promoted_name = (
            f"agriadapt_production_model_"
            f"{run_id}.joblib"
        )

        promoted_path = (
            MODEL_DIR
            / promoted_name
        )

        shutil.copy2(
            candidate_path,
            promoted_path,
        )

        update_model_manifest(
            promoted_name,
            f"v5.4-{run_id}",
        )

        lifecycle["state"] = (
            "MODEL_PROMOTED"
        )

        lifecycle["promotion"] = "PROMOTED"

        lifecycle["promoted_model"] = (
            str(promoted_path)
        )

    else:

        lifecycle["state"] = (
            "MODEL_REJECTED"
        )

        lifecycle["promotion"] = (
            "REJECTED"
        )

        lifecycle[
            "promoted_model"
        ] = None

    # --------------------------------------------------------
    # Final audit record
    # --------------------------------------------------------

    lifecycle["final_model"] = (
        read_model_manifest()
    )

    output = (
        LIFECYCLE_DIR
        / f"lifecycle_{run_id}.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            lifecycle,
            f,
            indent=2,
        )

    print()
    print(
        f"Candidate MAE : "
        f"{candidate_metrics['mae']:.4f}"
    )

    print(
        f"Current MAE  : "
        f"{current_metrics['mae']:.4f}"
    )

    print(
        f"MAE improvement : "
        f"{mae_improvement * 100:.2f}%"
    )

    print(
        f"Promotion : "
        f"{lifecycle['promotion']}"
    )

    print(
        f"Lifecycle report : {output}"
    )

    return lifecycle


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "AgriAdapt V5.4 "
            "end-to-end adaptation cycle"
        )
    )

    parser.add_argument(
        "--batch",
        default="stable_production_batch.csv",
        help=(
            "Production batch filename "
            "inside data/production/v5"
        ),
    )

    parser.add_argument(
        "--labels",
        action="store_true",
        help=(
            "Indicate that ground-truth "
            "labels are available."
        ),
    )

    args = parser.parse_args()

    run_cycle(
        batch_name=args.batch,
        labels_available=args.labels,
    )
from pathlib import Path
from datetime import datetime
import json
import sys

import mlflow


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

LIFECYCLE_DIR = (
    ROOT
    / "reports"
    / "lifecycle"
    / "v5"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "production"
    / "v5"
)

CURRENT_MODEL_MANIFEST = (
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

mlflow.set_experiment(
    "AgriAdapt_V5_Adaptation"
)


# ============================================================
# IMPORT V5.4
# ============================================================

SRC_MLOPS = Path(__file__).resolve().parent

if str(SRC_MLOPS) not in sys.path:
    sys.path.insert(0, str(SRC_MLOPS))

from run_adaptation_cycle_v5 import run_cycle
from model_registry_v5_7 import register_promoted_model


# ============================================================
# MODEL LINEAGE
# ============================================================

def read_model_manifest():

    if not CURRENT_MODEL_MANIFEST.exists():

        return {
            "status": "unknown",
            "active_model": None,
            "version": None,
        }

    with open(
        CURRENT_MODEL_MANIFEST,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# LIFECYCLE REPORT
# ============================================================

def find_lifecycle_report(
    run_started_at,
):

    reports = list(
        LIFECYCLE_DIR.glob(
            "lifecycle_*.json"
        )
    )

    if not reports:
        return None

    # Prefer reports created after this MLflow
    # run started.
    newer_reports = [
        path
        for path in reports
        if path.stat().st_mtime >= run_started_at
    ]

    if newer_reports:

        return max(
            newer_reports,
            key=lambda path: path.stat().st_mtime,
        )

    return max(
        reports,
        key=lambda path: path.stat().st_mtime,
    )


# ============================================================
# LOG MODEL LINEAGE
# ============================================================

def log_model_lineage(
    lifecycle,
    initial_manifest,
):

    # --------------------------------------------------------
    # Production model before adaptation
    # --------------------------------------------------------

    mlflow.set_tag(
        "production_model",
        str(
            initial_manifest.get(
                "active_model"
            )
        ),
    )

    mlflow.set_tag(
        "production_model_version",
        str(
            initial_manifest.get(
                "version"
            )
        ),
    )

    # --------------------------------------------------------
    # Candidate
    # --------------------------------------------------------

    candidate_model = lifecycle.get(
        "candidate_model"
    )

    if candidate_model:

        candidate_path = Path(
            candidate_model
        )

        mlflow.set_tag(
            "candidate_model",
            candidate_path.name,
        )

        mlflow.log_param(
            "candidate_model_path",
            str(candidate_path),
        )

    # --------------------------------------------------------
    # Promoted model
    # --------------------------------------------------------

    promoted_model = lifecycle.get(
        "promoted_model"
    )

    if promoted_model:

        promoted_path = Path(
            promoted_model
        )

        mlflow.set_tag(
            "promoted_model",
            promoted_path.name,
        )

    # --------------------------------------------------------
    # Final production model
    # --------------------------------------------------------

    final_manifest = lifecycle.get(
        "final_model"
    )

    if final_manifest:

        mlflow.set_tag(
            "final_production_model",
            str(
                final_manifest.get(
                    "active_model"
                )
            ),
        )

        mlflow.set_tag(
            "final_production_version",
            str(
                final_manifest.get(
                    "version"
                )
            ),
        )


# ============================================================
# MAIN
# ============================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "AgriAdapt V5.5 "
            "MLflow-tracked adaptation cycle "
            "with V5.7 automatic model registry update"
        )
    )

    parser.add_argument(
        "--batch",
        default="stable_production_batch.csv",
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

    timestamp = datetime.now()

    run_name = (
        "adaptation_"
        + timestamp.strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    run_started_at = timestamp.timestamp()

    # --------------------------------------------------------
    # Capture production model BEFORE adaptation
    # --------------------------------------------------------

    initial_manifest = (
        read_model_manifest()
    )

    print("=" * 70)
    print("AGRIADAPT V5.5")
    print("MLFLOW-TRACKED ADAPTATION")
    print("V5.7 AUTOMATIC MODEL REGISTRY")
    print("=" * 70)

    # This variable will only contain Registry
    # information after a successful promotion.
    registry_result = None

    with mlflow.start_run(
        run_name=run_name
    ):

        # ====================================================
        # RUN ID
        # ====================================================

        run_id = (
            mlflow.active_run()
            .info
            .run_id
        )

        mlflow.set_tag(
            "project",
            "AgriAdapt",
        )

        mlflow.set_tag(
            "pipeline",
            "V5.4 adaptation cycle",
        )

        mlflow.set_tag(
            "mlops_version",
            "V5.5+V5.7",
        )

        mlflow.set_tag(
            "batch",
            args.batch,
        )

        mlflow.set_tag(
            "ground_truth_available",
            str(args.labels),
        )

        # ====================================================
        # MODEL LINEAGE — BEFORE
        # ====================================================

        mlflow.set_tag(
            "initial_production_model",
            str(
                initial_manifest.get(
                    "active_model"
                )
            ),
        )

        mlflow.set_tag(
            "initial_production_version",
            str(
                initial_manifest.get(
                    "version"
                )
            ),
        )

        # ====================================================
        # RUN V5.4
        # ====================================================

        lifecycle = run_cycle(
            batch_name=args.batch,
            labels_available=args.labels,
        )

        # ====================================================
        # DATA / DRIFT
        # ====================================================

        mlflow.log_params(
            {
                "drifted_columns":
                    lifecycle["drift"][
                        "drifted_columns"
                    ],

                "drift_threshold":
                    lifecycle["drift"][
                        "threshold"
                    ],

                "production_rows":
                    lifecycle[
                        "production_rows"
                    ],

                "reference_rows":
                    lifecycle[
                        "reference_rows"
                    ],

                "ground_truth_available":
                    int(args.labels),
            }
        )

        mlflow.set_tag(
            "lifecycle_state",
            lifecycle["state"],
        )

        mlflow.set_tag(
            "action",
            lifecycle["action"],
        )

        mlflow.set_tag(
            "promotion",
            lifecycle["promotion"],
        )

        # ====================================================
        # MODEL LINEAGE
        # ====================================================

        log_model_lineage(
            lifecycle,
            initial_manifest,
        )

        # ====================================================
        # CURRENT MODEL METRICS
        # ====================================================

        if "current_metrics" in lifecycle:

            current = lifecycle[
                "current_metrics"
            ]

            mlflow.log_metrics(
                {
                    "current_mae":
                        current["mae"],

                    "current_rmse":
                        current["rmse"],

                    "current_r2":
                        current["r2"],
                }
            )

        # ====================================================
        # CANDIDATE MODEL METRICS
        # ====================================================

        if "candidate_metrics" in lifecycle:

            candidate = lifecycle[
                "candidate_metrics"
            ]

            mlflow.log_metrics(
                {
                    "candidate_mae":
                        candidate["mae"],

                    "candidate_rmse":
                        candidate["rmse"],

                    "candidate_r2":
                        candidate["r2"],
                }
            )

        # ====================================================
        # PROMOTION GATE
        # ====================================================

        if "promotion_gate" in lifecycle:

            gate = lifecycle[
                "promotion_gate"
            ]

            mlflow.log_params(
                {
                    "minimum_mae_improvement":
                        gate[
                            "minimum_mae_improvement"
                        ],

                    "actual_mae_improvement":
                        gate[
                            "actual_mae_improvement"
                        ],
                }
            )

            mlflow.set_tag(
                "promotion_gate_accepted",
                str(
                    gate[
                        "accepted"
                    ]
                ),
            )

        elif (
            "mae_improvement_fraction"
            in lifecycle
        ):

            mlflow.log_metric(
                "mae_improvement",
                lifecycle[
                    "mae_improvement_fraction"
                ],
            )

        # ====================================================
        # LIFECYCLE REPORT
        # ====================================================

        report_path = (
            find_lifecycle_report(
                run_started_at
            )
        )

        if report_path:

            mlflow.log_artifact(
                str(report_path),
                artifact_path="lifecycle",
            )

        # ====================================================
        # DRIFT REPORT
        # ====================================================

        drift_json = lifecycle[
            "drift"
        ].get(
            "json_report"
        )

        if drift_json:

            drift_path = Path(
                drift_json
            )

            if drift_path.exists():

                mlflow.log_artifact(
                    str(drift_path),
                    artifact_path="drift",
                )

        # ====================================================
        # PROMOTED MODEL + V5.7 REGISTRY
        # ====================================================

        promoted_model = lifecycle.get(
            "promoted_model"
        )

        # IMPORTANT:
        # Registry update is allowed ONLY when
        # the promotion gate has actually accepted
        # the candidate.
        if (
            lifecycle.get("promotion")
            == "PROMOTED"
            and promoted_model
        ):

            promoted_path = Path(
                promoted_model
            )

            if not promoted_path.exists():

                raise FileNotFoundError(
                    "Promotion reported success, "
                    "but the promoted model does not exist: "
                    f"{promoted_path}"
                )

            # ------------------------------------------------
            # Keep promoted model as an MLflow artifact
            # ------------------------------------------------

            mlflow.log_artifact(
                str(promoted_path),
                artifact_path="models",
            )

            # ------------------------------------------------
            # Read the NEW production version
            # from V5.4 final manifest.
            # ------------------------------------------------

            final_manifest = lifecycle.get(
                "final_model"
            )

            if not final_manifest:

                raise RuntimeError(
                    "Promotion succeeded, "
                    "but lifecycle['final_model'] "
                    "is missing."
                )

            production_version = (
                final_manifest.get(
                    "version"
                )
            )

            if not production_version:

                raise RuntimeError(
                    "Promotion succeeded, "
                    "but the final production "
                    "version is missing."
                )

            # ------------------------------------------------
            # V5.7 AUTOMATIC MODEL REGISTRY
            # ------------------------------------------------

            registry_result = (
                register_promoted_model(
                    model_path=promoted_path,
                    production_version=(
                        production_version
                    ),
                )
            )

            # ------------------------------------------------
            # Record Registry lineage in the
            # V5.5 adaptation MLflow run.
            # ------------------------------------------------

            mlflow.set_tag(
                "registry_model_name",
                registry_result[
                    "registered_model_name"
                ],
            )

            mlflow.set_tag(
                "registry_version",
                registry_result[
                    "registry_version"
                ],
            )

            mlflow.set_tag(
                "registry_alias",
                str(
                    registry_result[
                        "alias"
                    ]
                ),
            )

            mlflow.set_tag(
                "registry_run_id",
                registry_result[
                    "mlflow_run_id"
                ],
            )

            mlflow.log_param(
                "registry_model_uri",
                registry_result[
                    "model_uri"
                ],
            )

            print()
            print("=" * 70)
            print(
                "MLFLOW MODEL REGISTRY UPDATED"
            )
            print("=" * 70)

            print(
                "Registered Model : "
                + registry_result[
                    "registered_model_name"
                ]
            )

            print(
                "Registry Version : "
                + registry_result[
                    "registry_version"
                ]
            )

            print(
                "Production Version : "
                + registry_result[
                    "production_model_version"
                ]
            )

            print(
                "Production Alias : "
                + str(
                    registry_result[
                        "alias"
                    ]
                )
            )

        # ====================================================
        # FINAL MODEL MANIFEST
        # ====================================================

        if CURRENT_MODEL_MANIFEST.exists():

            mlflow.log_artifact(
                str(
                    CURRENT_MODEL_MANIFEST
                ),
                artifact_path="model_registry",
            )

        # ====================================================
        # SUMMARY
        # ====================================================

        print()
        print("=" * 70)
        print("MLFLOW RUN COMPLETE")
        print("=" * 70)

        print(
            f"Run ID : {run_id}"
        )

        print(
            f"State  : "
            f"{lifecycle['state']}"
        )

        print(
            f"Action : "
            f"{lifecycle['action']}"
        )

        print(
            f"Promotion : "
            f"{lifecycle['promotion']}"
        )

        print(
            f"Production model : "
            f"{initial_manifest.get('active_model')}"
        )

        print(
            f"Production version : "
            f"{initial_manifest.get('version')}"
        )

        # ----------------------------------------------------
        # Registry summary
        # ----------------------------------------------------

        if registry_result:

            print(
                f"Registry model : "
                f"{registry_result['registered_model_name']}"
            )

            print(
                f"Registry version : "
                f"{registry_result['registry_version']}"
            )

            print(
                f"Registry alias : "
                f"{registry_result['alias']}"
            )


if __name__ == "__main__":
    main()
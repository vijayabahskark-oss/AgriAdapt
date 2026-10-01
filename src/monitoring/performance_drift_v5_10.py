from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_DIR = ROOT / "data" / "production" / "v5"
REPORT_DIR = ROOT / "reports" / "monitoring" / "v5"

EVALUATED_FILE = PRODUCTION_DIR / "evaluated_predictions.csv"
PERFORMANCE_REPORT = REPORT_DIR / "performance_report_v5_9.json"

DRIFT_REPORT = REPORT_DIR / "performance_drift_report_v5_10.json"


# ============================================================
# CONFIGURATION
# ============================================================

# Minimum number of ground-truth observations required before
# making a performance-drift decision.
MIN_EVALUATED_RECORDS = 10

# Prototype performance-drift thresholds.
#
# MAE degradation:
#     10% or more = performance drift
#
# R² degradation:
#     0.05 or more = performance drift
#
# These are engineering thresholds for the prototype and
# should later be calibrated using real production history.
MAE_DEGRADATION_THRESHOLD = 0.10
R2_DROP_THRESHOLD = 0.05


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    raise RuntimeError(message)


def calculate_metrics(df):
    y_true = df["actual_yield_tonnes_per_ha"].to_numpy()
    y_pred = df["predicted_yield_tonnes_per_ha"].to_numpy()

    mae = mean_absolute_error(y_true, y_pred)

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )

    if len(y_true) >= 2 and np.std(y_true) > 0:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = None

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": r2,
    }


# ============================================================
# LOAD EVALUATED PREDICTIONS
# ============================================================

def load_evaluated_predictions():

    if not EVALUATED_FILE.exists():
        fail(
            f"Evaluated prediction file not found:\n"
            f"{EVALUATED_FILE}\n\n"
            f"Run V5.9 evaluation first."
        )

    df = pd.read_csv(EVALUATED_FILE)

    required_columns = {
        "prediction_id",
        "predicted_yield_tonnes_per_ha",
        "actual_yield_tonnes_per_ha",
        "model_version",
        "model_name",
    }

    missing = required_columns - set(df.columns)

    if missing:
        fail(
            "Evaluated prediction file is missing columns:\n"
            + "\n".join(sorted(missing))
        )

    df["predicted_yield_tonnes_per_ha"] = pd.to_numeric(
        df["predicted_yield_tonnes_per_ha"],
        errors="coerce",
    )

    df["actual_yield_tonnes_per_ha"] = pd.to_numeric(
        df["actual_yield_tonnes_per_ha"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "predicted_yield_tonnes_per_ha",
            "actual_yield_tonnes_per_ha",
        ]
    )

    return df


# ============================================================
# BASELINE
# ============================================================

def load_baseline_metrics():

    if not PERFORMANCE_REPORT.exists():
        fail(
            f"Performance baseline report not found:\n"
            f"{PERFORMANCE_REPORT}"
        )

    with open(
        PERFORMANCE_REPORT,
        "r",
        encoding="utf-8",
    ) as f:
        report = json.load(f)

    performance = report.get("performance", {})

    mae = performance.get("mae_tonnes_per_ha")
    rmse = performance.get("rmse_tonnes_per_ha")
    r2 = performance.get("r2")

    if mae is None or rmse is None:
        fail(
            "Baseline performance report does not contain "
            "MAE and RMSE."
        )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": None if r2 is None else float(r2),
    }


# ============================================================
# PERFORMANCE DRIFT
# ============================================================

def evaluate_performance_drift():

    print("=" * 70)
    print("AGRIADAPT V5.10")
    print("PERFORMANCE DRIFT DETECTION")
    print("=" * 70)

    evaluated = load_evaluated_predictions()

    print(
        f"Evaluated records : {len(evaluated)}"
    )

    # --------------------------------------------------------
    # Minimum sample guard
    # --------------------------------------------------------

    if len(evaluated) < MIN_EVALUATED_RECORDS:

        print()
        print(
            f"INSUFFICIENT_GROUND_TRUTH"
        )

        print(
            f"Required records : "
            f"{MIN_EVALUATED_RECORDS}"
        )

        print(
            f"Available records: "
            f"{len(evaluated)}"
        )

        report = {
            "project": "AgriAdapt",
            "pipeline": "V5.10 Performance Drift Detection",
            "evaluated_at": datetime.now(
                timezone.utc
            ).isoformat(),

            "state": "INSUFFICIENT_GROUND_TRUTH",

            "decision": "WAIT_FOR_MORE_GROUND_TRUTH",

            "records": {
                "available": int(len(evaluated)),
                "minimum_required": int(
                    MIN_EVALUATED_RECORDS
                ),
            },

            "thresholds": {
                "mae_degradation": MAE_DEGRADATION_THRESHOLD,
                "r2_drop": R2_DROP_THRESHOLD,
            },

            "reason": (
                "The evaluated prediction batch is too small "
                "to make a reliable performance-drift decision."
            ),
        }

        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            DRIFT_REPORT,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
            )

        print()
        print(
            f"Report: {DRIFT_REPORT}"
        )
        print("=" * 70)

        return report

    # --------------------------------------------------------
    # Current performance
    # --------------------------------------------------------

    current = calculate_metrics(evaluated)

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = load_baseline_metrics()

    print()
    print("BASELINE PERFORMANCE")
    print(
        f"MAE  : {baseline['mae']:.4f}"
    )
    print(
        f"RMSE : {baseline['rmse']:.4f}"
    )
    print(
        f"R²   : "
        f"{baseline['r2']:.4f}"
        if baseline["r2"] is not None
        else "R²   : N/A"
    )

    print()
    print("CURRENT PERFORMANCE")
    print(
        f"MAE  : {current['mae']:.4f}"
    )
    print(
        f"RMSE : {current['rmse']:.4f}"
    )
    print(
        f"R²   : "
        f"{current['r2']:.4f}"
        if current["r2"] is not None
        else "R²   : N/A"
    )

    # --------------------------------------------------------
    # MAE degradation
    # --------------------------------------------------------

    if baseline["mae"] > 0:

        mae_degradation = (
            current["mae"] - baseline["mae"]
        ) / baseline["mae"]

    else:
        mae_degradation = 0.0

    # --------------------------------------------------------
    # R² degradation
    # --------------------------------------------------------

    if (
        baseline["r2"] is not None
        and current["r2"] is not None
    ):

        r2_drop = (
            baseline["r2"] - current["r2"]
        )

    else:

        r2_drop = None

    mae_drift = (
        mae_degradation
        >= MAE_DEGRADATION_THRESHOLD
    )

    r2_drift = (
        r2_drop is not None
        and r2_drop >= R2_DROP_THRESHOLD
    )

    performance_drift = (
        mae_drift
        or r2_drift
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if performance_drift:

        state = "PERFORMANCE_DRIFT_DETECTED"
        decision = "ADAPTATION_REVIEW_REQUIRED"

    else:

        state = "PERFORMANCE_STABLE"
        decision = "KEEP_PRODUCTION_MODEL"

    # --------------------------------------------------------
    # Model lineage
    # --------------------------------------------------------

    model_versions = (
        evaluated["model_version"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    model_names = (
        evaluated["model_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "project": "AgriAdapt",
        "pipeline": "V5.10 Performance Drift Detection",

        "evaluated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "state": state,

        "decision": decision,

        "records": {
            "evaluated": int(len(evaluated)),
            "minimum_required": int(
                MIN_EVALUATED_RECORDS
            ),
        },

        "baseline": {
            "mae_tonnes_per_ha": baseline["mae"],
            "rmse_tonnes_per_ha": baseline["rmse"],
            "r2": baseline["r2"],
        },

        "current": {
            "mae_tonnes_per_ha": current["mae"],
            "rmse_tonnes_per_ha": current["rmse"],
            "r2": current["r2"],
        },

        "degradation": {
            "mae_degradation_fraction": float(
                mae_degradation
            ),
            "mae_degradation_percent": float(
                mae_degradation * 100
            ),
            "r2_drop": (
                None
                if r2_drop is None
                else float(r2_drop)
            ),
        },

        "thresholds": {
            "minimum_evaluated_records": (
                MIN_EVALUATED_RECORDS
            ),
            "mae_degradation_threshold": (
                MAE_DEGRADATION_THRESHOLD
            ),
            "r2_drop_threshold": (
                R2_DROP_THRESHOLD
            ),
        },

        "signals": {
            "mae_drift": bool(mae_drift),
            "r2_drift": bool(r2_drift),
            "performance_drift": bool(
                performance_drift
            ),
        },

        "model_lineage": {
            "model_versions": model_versions,
            "model_names": model_names,
        },

        "methodology": {
            "mae_rule": (
                "Performance drift is signaled when "
                "current MAE increases by at least "
                "10 percent relative to the baseline."
            ),
            "r2_rule": (
                "Performance drift is signaled when "
                "R² decreases by at least 0.05."
            ),
            "minimum_sample_rule": (
                "No performance-drift decision is made "
                "until at least 10 ground-truth observations "
                "are available."
            ),
        },
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        DRIFT_REPORT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print("-" * 70)

    print(
        f"MAE degradation : "
        f"{mae_degradation * 100:.2f}%"
    )

    if r2_drop is not None:

        print(
            f"R² drop         : "
            f"{r2_drop:.4f}"
        )

    else:

        print(
            "R² drop         : N/A"
        )

    print()

    print(
        f"MAE drift       : "
        f"{'YES' if mae_drift else 'NO'}"
    )

    print(
        f"R² drift        : "
        f"{'YES' if r2_drift else 'NO'}"
    )

    print()

    print(
        f"STATE           : {state}"
    )

    print(
        f"DECISION        : {decision}"
    )

    print()

    print(
        f"Report          : {DRIFT_REPORT}"
    )

    print("=" * 70)

    return report


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    evaluate_performance_drift()
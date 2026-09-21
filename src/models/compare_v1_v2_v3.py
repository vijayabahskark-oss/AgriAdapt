from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(r"D:\AgriAdapt")

REPORT_DIR = PROJECT_ROOT / "reports" / "comparison_v1_v2_v3"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

V1_FILE = PROJECT_ROOT / "reports" / "benchmark" / "benchmark_results.csv"
V2_FILE = PROJECT_ROOT / "reports" / "benchmark_v2" / "benchmark_results_v2.csv"
V3_FILE = PROJECT_ROOT / "reports" / "benchmark_v3" / "benchmark_results_v3.csv"


def load_benchmark(path, version):
    df = pd.read_csv(path)

    # Normalize column names for easier matching
    df.columns = [str(c).strip() for c in df.columns]

    # Create a case-insensitive lookup
    column_lookup = {
        c.lower(): c
        for c in df.columns
    }

    # =========================================================
    # CASE 1: LONG FORMAT
    #
    # V1:
    # model | split | MAE | RMSE | R2
    #
    # Also supports:
    # Model | Split | MAE | RMSE | R2
    # =========================================================

    long_required = {
        "model",
        "split",
        "mae",
        "rmse",
        "r2",
    }

    if long_required.issubset(column_lookup.keys()):

        result = df[
            [
                column_lookup["model"],
                column_lookup["split"],
                column_lookup["mae"],
                column_lookup["rmse"],
                column_lookup["r2"],
            ]
        ].copy()

        result.columns = [
            "Model",
            "Split",
            "MAE",
            "RMSE",
            "R2",
        ]

    # =========================================================
    # CASE 2: WIDE FORMAT
    #
    # V2 / V3:
    #
    # Model
    # Validation_MAE
    # Validation_RMSE
    # Validation_R2
    # Test_MAE
    # Test_RMSE
    # Test_R2
    # =========================================================

    else:

        wide_required = {
            "model",
            "validation_mae",
            "validation_rmse",
            "validation_r2",
            "test_mae",
            "test_rmse",
            "test_r2",
        }

        if not wide_required.issubset(column_lookup.keys()):
            raise ValueError(
                f"{version}: Unsupported benchmark schema.\n"
                f"Available columns: {list(df.columns)}"
            )

        model_col = column_lookup["model"]

        # -------------------------
        # Validation
        # -------------------------

        validation = df[
            [
                model_col,
                column_lookup["validation_mae"],
                column_lookup["validation_rmse"],
                column_lookup["validation_r2"],
            ]
        ].copy()

        validation["Split"] = "Validation"

        validation = validation.rename(
            columns={
                model_col: "Model",
                column_lookup["validation_mae"]: "MAE",
                column_lookup["validation_rmse"]: "RMSE",
                column_lookup["validation_r2"]: "R2",
            }
        )

        validation = validation[
            [
                "Model",
                "Split",
                "MAE",
                "RMSE",
                "R2",
            ]
        ]

        # -------------------------
        # Test
        # -------------------------

        test = df[
            [
                model_col,
                column_lookup["test_mae"],
                column_lookup["test_rmse"],
                column_lookup["test_r2"],
            ]
        ].copy()

        test["Split"] = "Test"

        test = test.rename(
            columns={
                model_col: "Model",
                column_lookup["test_mae"]: "MAE",
                column_lookup["test_rmse"]: "RMSE",
                column_lookup["test_r2"]: "R2",
            }
        )

        test = test[
            [
                "Model",
                "Split",
                "MAE",
                "RMSE",
                "R2",
            ]
        ]

        # Combine validation + test
        result = pd.concat(
            [validation, test],
            ignore_index=True,
        )

    # Add experiment version
    result["Version"] = version

    # Make sure metrics are numeric
    result["MAE"] = pd.to_numeric(
        result["MAE"],
        errors="coerce",
    )

    result["RMSE"] = pd.to_numeric(
        result["RMSE"],
        errors="coerce",
    )

    result["R2"] = pd.to_numeric(
        result["R2"],
        errors="coerce",
    )

 # Normalize split names across benchmark versions
    result["Split"] = (
        result["Split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "validation": "Validation",
            "test": "Test",
        })
    )

    # Normalize model names
    result["Model"] = (
        result["Model"]
        .astype(str)
        .str.strip()
    )

    return result   


def main():
    print("=" * 70)
    print("AgriAdapt V1 vs V2 vs V3 Benchmark Comparison")
    print("=" * 70)

    v1 = load_benchmark(V1_FILE, "V1")
    v2 = load_benchmark(V2_FILE, "V2")
    v3 = load_benchmark(V3_FILE, "V3")

    comparison = pd.concat([v1, v2, v3], ignore_index=True)

    # Consistent ordering
    version_order = ["V1", "V2", "V3"]
    model_order = [
        "Linear Regression",
        "Random Forest",
        "XGBoost",
    ]
    split_order = ["Validation", "Test"]

    comparison["Version"] = pd.Categorical(
    comparison["Version"],
    categories=version_order,
    ordered=True,
)

    comparison["Model"] = pd.Categorical(
    comparison["Model"],
    categories=model_order,
    ordered=True,
)

    comparison["Split"] = pd.Categorical(
    comparison["Split"],
    categories=split_order,
    ordered=True,
)

    comparison = comparison.sort_values(
        ["Model", "Version", "Split"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Full comparison
    # ---------------------------------------------------------

    full_file = REPORT_DIR / "v1_v2_v3_full_comparison.csv"
    comparison.to_csv(full_file, index=False)

    # ---------------------------------------------------------
    # Test-only comparison
    # ---------------------------------------------------------

    test = comparison[
        comparison["Split"] == "Test"
    ].copy()

    test_file = REPORT_DIR / "v1_v2_v3_test_comparison.csv"
    test.to_csv(test_file, index=False)

    # ---------------------------------------------------------
    # Validation-only comparison
    # ---------------------------------------------------------

    validation = comparison[
        comparison["Split"] == "Validation"
    ].copy()

    validation_file = REPORT_DIR / "v1_v2_v3_validation_comparison.csv"
    validation.to_csv(validation_file, index=False)

    # ---------------------------------------------------------
    # Pivot tables
    # ---------------------------------------------------------

    test_mae = test.pivot(
        index="Model",
        columns="Version",
        values="MAE",
    )

    test_rmse = test.pivot(
        index="Model",
        columns="Version",
        values="RMSE",
    )

    test_r2 = test.pivot(
        index="Model",
        columns="Version",
        values="R2",
    )

    test_mae.to_csv(REPORT_DIR / "test_mae_v1_v2_v3.csv")
    test_rmse.to_csv(REPORT_DIR / "test_rmse_v1_v2_v3.csv")
    test_r2.to_csv(REPORT_DIR / "test_r2_v1_v2_v3.csv")

    # ---------------------------------------------------------
    # Human-readable summary
    # ---------------------------------------------------------

    summary_file = REPORT_DIR / "comparison_summary.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("AgriAdapt V1 vs V2 vs V3 Benchmark Comparison\n")
        f.write("=" * 60 + "\n\n")

        f.write("Dataset / evaluation design\n")
        f.write("-" * 60 + "\n")
        f.write("Training:   1984-2015\n")
        f.write("Validation: 2016-2019\n")
        f.write("Test:       2020-2024\n")
        f.write("Random shuffle: No\n")
        f.write("Target: yield_tonnes_per_ha\n\n")

        f.write("TEST RESULTS\n")
        f.write("-" * 60 + "\n")

        for model in model_order:
            rows = test[test["Model"] == model]

            f.write(f"\n{model}\n")

            for _, row in rows.iterrows():
                f.write(
                    f"  {row['Version']}: "
                    f"MAE={row['MAE']:.4f}, "
                    f"RMSE={row['RMSE']:.4f}, "
                    f"R2={row['R2']:.4f}\n"
                )

        f.write("\n\nVALIDATION RESULTS\n")
        f.write("-" * 60 + "\n")

        for model in model_order:
            rows = validation[validation["Model"] == model]

            f.write(f"\n{model}\n")

            for _, row in rows.iterrows():
                f.write(
                    f"  {row['Version']}: "
                    f"MAE={row['MAE']:.4f}, "
                    f"RMSE={row['RMSE']:.4f}, "
                    f"R2={row['R2']:.4f}\n"
                )

        f.write("\n\nINTERPRETATION\n")
        f.write("-" * 60 + "\n")
        f.write(
            "The comparison evaluates whether additional feature engineering "
            "improves temporal generalization to the 2020-2024 test period.\n\n"
        )

        f.write(
            "V1 uses annual climate features.\n"
            "V2 adds seasonal climate features.\n"
            "V3 adds spatial statistics, crop-aware climate features, "
            "and training-period anomaly features.\n\n"
        )

        f.write(
            "Important methodological limitation: the current dataset contains "
            "three crop rows for each of 41 independent climate years. "
            "Therefore, increasing the number of engineered features does not "
            "increase the number of independent climate-year observations.\n"
        )

    print("\nComparison completed successfully.")
    print(f"\nReports saved to:\n{REPORT_DIR}")

    print("\nTEST COMPARISON:")
    print(
        test[
            ["Model", "Version", "MAE", "RMSE", "R2"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
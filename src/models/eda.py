import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
 
from src.database.get_model_data import load_dataset_from_postgres
 
TARGET = "site_eui_gj_m2"
 
# Variables to treat with caution as predictors — they may be strongly
# related to (or derived from) the target, so don't include them in X
# without investigating first.
WATCHLIST = [
    "source_eui_gj_m2",
    "weather_normalized_site_eui_gj_m2",
    "weather_normalized_source_eui_gj_m2",
    "electricity_intensity_gj_m2",
    "gas_intensity_gj_m2",
]
 
output_dir = Path("reports/eda")
output_dir.mkdir(parents=True, exist_ok=True)
 
 
def inspect_shape(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("1. SHAPE")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
 
 
def inspect_dtypes(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("2. DTYPES")
    print(df.dtypes)
 
 
def inspect_missing_values(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("3. MISSING VALUES")
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(1)
    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct
    })
    summary = summary[summary["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )
    if summary.empty:
        print("No missing values found.")
    else:
        print(summary)
 
    target_missing = df[TARGET].isna().sum()
    print(f"\nTarget '{TARGET}' missing values: {target_missing} "
          f"({target_missing / len(df) * 100:.1f}% of rows)")
 
 
def describe_target(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(f"4. DESCRIPTIVE STATISTICS: {TARGET}")
    stats = df[TARGET].describe()
    print(stats)
    print(f"\nMedian: {df[TARGET].median():.3f}")
    print(f"Standard deviation: {df[TARGET].std():.3f}")
 
 
def inspect_property_types(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("5. PROPERTY TYPE COUNTS")
    print(df["primary_property_type"].value_counts())
 
 
def check_outliers(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(f"6. OUTLIER CHECK ({TARGET}, IQR method)")
 
    # Base the percentage on non-missing target observations only, so
    # missing rows in the denominator don't understate the outlier rate.
    non_missing = df[TARGET].dropna()
    n_non_missing = len(non_missing)
 
    q1 = non_missing.quantile(0.25)
    q3 = non_missing.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
 
    outlier_mask = (df[TARGET] < lower_bound) | (df[TARGET] > upper_bound)
    outliers = df[outlier_mask]
 
    print(f"Non-missing target observations: {n_non_missing}")
    print(f"IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"Outliers found: {len(outliers)} "
          f"({len(outliers) / n_non_missing * 100:.1f}% of non-missing observations)")
 
    if not outliers.empty:
        print(outliers[["ewrb_id", "primary_property_type", TARGET]]
              .sort_values(TARGET, ascending=False)
              .head(10))
 
    print(
        "\nNote: these are candidates to investigate, not automatically "
        "remove. A building can legitimately have very high energy "
        "intensity — a statistical outlier isn't necessarily an error."
    )
 
 
def plot_target_distribution(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("7. DISTRIBUTION PLOTS")
 
    plt.figure(figsize=(8, 5))
    df[TARGET].dropna().hist(bins=40, edgecolor="black")
    plt.title(f"Distribution of {TARGET}")
    plt.xlabel(TARGET)
    plt.ylabel("Count")
    plt.tight_layout()
    hist_path = output_dir / "site_eui_histogram.png"
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved histogram to {hist_path}")
    print(
        "Check whether the distribution is right-skewed with a long tail — "
        "that will matter for model/transformation choices later. Don't "
        "log-transform anything yet; just note what you observe."
    )
 
    # Boxplot by property type — limit to top N types so it stays readable.
    # Types with very few buildings can produce misleading boxplots, so
    # this is worth revisiting with a minimum-count threshold later.
    top_types = df["primary_property_type"].value_counts().nlargest(8).index
    subset = df[df["primary_property_type"].isin(top_types)]
 
    plt.figure(figsize=(10, 6))
    subset.boxplot(column=TARGET, by="primary_property_type", rot=45)
    plt.title(f"{TARGET} by Property Type (top 8 types)")
    plt.suptitle("")
    plt.xlabel("Property Type")
    plt.ylabel(TARGET)
    plt.tight_layout()
    box_path = output_dir / "site_eui_by_property_type_boxplot.png"
    plt.savefig(box_path)
    plt.close()
    print(f"Saved boxplot to {box_path}")
 
 
def target_vs_numerical(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(f"8. {TARGET} vs. NUMERICAL VARIABLES (correlation)")
 
    numeric_df = df.select_dtypes(include="number")
    correlations = numeric_df.corr()[TARGET].drop(TARGET).sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    print(correlations)
 
    print("\n--- Variables on the watchlist (possible leakage risk) ---")
    for col in WATCHLIST:
        if col in correlations.index:
            print(f"{col}: correlation = {correlations[col]:.3f}")
    print(
        "\nA high correlation here is a prompt to investigate WHY, not "
        "a signal to automatically include the variable. Correlation "
        "isn't a predictor-selection mechanism on its own — especially "
        "since several of these are themselves other energy-use measures "
        "derived from, or closely tied to, the target."
    )
 
 
def verify_one_row_per_building(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("9. ROW COUNT / REPORTING YEAR CHECK")
    n_rows = len(df)
    n_unique_buildings = df["ewrb_id"].nunique()
    print(f"Total rows: {n_rows}")
    print(f"Unique ewrb_id values: {n_unique_buildings}")
 
    if n_rows == n_unique_buildings:
        print("Confirmed: one row per building (this is a deliberate "
              "modeling choice from get_model_data.py, not a universal fact "
              "about the data — worth re-checking if the query changes).")
    else:
        print(
            "WARNING: row count does not match unique building count — "
            "there may be duplicate ewrb_id rows, which would break the "
            "'one row per building' assumption from get_model_data.py."
        )
 
    print("\nReporting year distribution (which year each building's row came from):")
    print(df["reporting_year"].value_counts().sort_index())
 
 
def main():
    print("Loading modeling dataset from PostgreSQL...")
    df = load_dataset_from_postgres()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
 
    inspect_shape(df)
    inspect_dtypes(df)
    inspect_missing_values(df)
    describe_target(df)
    inspect_property_types(df)
    check_outliers(df)
    plot_target_distribution(df)
    target_vs_numerical(df)
    verify_one_row_per_building(df)
 
    print("\n" + "=" * 60)
    print("EDA complete. Review the printed output and plots in "
          f"'{output_dir}/' before deciding on features in prepare_data.py.")
 
 
if __name__ == "__main__":
    main()
"""
src/models/eda.py

Pure exploratory data analysis on the modeling dataset. This script does
NOT prepare features or train a model.

Target variable: site_eui_gj_m2

This script answers:
    1. How many observations are there?
    2. How many target values are missing?
    3. What are the mean, median, min, max, and standard deviation?
    4. What does the distribution look like?
    5. Are there extreme outliers?
    6. How does Site EUI vary by property type?
    7. How does Site EUI relate to the numerical variables?

It also verifies the row count / reporting-year distribution, since
get_model_data.py keeps only the latest reporting_year per building.
This dataset should be one row per building.

Run from the project root as a module (not as a standalone script), so
the package imports below resolve correctly:

    python -m src.models.eda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.database.get_model_data import load_dataset_from_postgres
from src.models.prepare_data import check_property_type_redundancy

TARGET = "site_eui_gj_m2"

# Variables to treat with caution as predictors as they may be strongly
# related to (or derived from) the target.
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
        "intensity."
    )

def plot_target_distribution(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("7. DISTRIBUTION PLOTS")

    plt.figure(figsize=(8, 5))
    df[TARGET].dropna().hist(bins=40, edgecolor="black")
    plt.title(f"Distribution of {TARGET} (raw scale)")
    plt.xlabel(TARGET)
    plt.ylabel("Count")
    plt.tight_layout()
    hist_path = output_dir / "site_eui_histogram.png"
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved histogram to {hist_path}")
    print(
        "Note: the raw-scale histogram is dominated by a handful of "
        "extreme values and isn't actually readable on its own — see the "
        "log-scale version below, which is what makes the skew visible."
    )

    # The raw-scale histogram above is visually useless because a handful
    # of extreme outliers (up to ~16117) compress every other building
    # into a single bar. Setting a log x-axis alone doesn't fix this —
    # the bin edges themselves must be log-spaced, or nearly all data
    # still falls in one bin. This is a visualization choice only, not
    # the log1p target transform used in train.py.
    values = df[TARGET].dropna()
    values_positive = values[values > 0]  # log-spaced bins require > 0
    log_bins = np.logspace(
        np.log10(values_positive.min()), np.log10(values_positive.max()), 40
    )

    plt.figure(figsize=(8, 5))
    plt.hist(values_positive, bins=log_bins, edgecolor="black")
    plt.xscale("log")
    plt.title(f"Distribution of {TARGET} (log-scale x-axis, log-spaced bins)")
    plt.xlabel(f"{TARGET} (log scale)")
    plt.ylabel("Count")
    plt.tight_layout()
    hist_log_path = output_dir / "site_eui_histogram_log_scale.png"
    plt.savefig(hist_log_path)
    plt.close()
    print(f"Saved log-scale histogram to {hist_log_path}")

    # Boxplot by property type. Limit to the top 8 types so the chart stays
    # readable. A horizontal layout is used so long property-type names do
    # not overlap; the log scale is applied to the x-axis instead of the
    # y-axis. Zero-valued EUI observations are excluded from this plot
    # because a logarithmic axis cannot display zero.
    top_types = df["primary_property_type"].value_counts().nlargest(8).index.tolist()
    subset = df[df["primary_property_type"].isin(top_types)][
        ["primary_property_type", TARGET]
    ].dropna(subset=[TARGET])
    subset = subset[subset[TARGET] > 0]

    grouped = [
        subset.loc[subset["primary_property_type"] == prop_type, TARGET].values
        for prop_type in top_types
    ]
    labels = [
        prop_type.replace("/", "/\n") if len(prop_type) > 24 else prop_type
        for prop_type in top_types
    ]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.boxplot(
        grouped,
        vert=False,
        patch_artist=True,
        tick_labels=labels,
        showfliers=True,
    )
    ax.set_xscale("log")
    fig.suptitle(
        "Site Energy Use Intensity by Primary Property Type",
        fontsize=15,
        y=0.98,
    )
    fig.text(
        0.5,
        0.945,
        "Top 8 property types by building count | positive EUI values | log scale",
        ha="center",
        va="top",
        fontsize=10,
    )
    ax.set_xlabel("Site EUI (GJ/m²) — logarithmic scale")
    ax.set_ylabel("Primary Property Type")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.tick_params(axis="y", labelsize=9)
    ax.margins(y=0.05)
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    box_path = output_dir / "site_eui_by_property_type_boxplot.png"
    fig.savefig(box_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved boxplot to {box_path}")
    print(
        "Note: this plot uses the top 8 property types by building count, "
        "a horizontal layout to keep long labels readable, and a log-scale "
        "x-axis so extreme EUI values do not compress the boxes. Zero-valued "
        "EUI observations are excluded because they cannot be shown on a log scale."
    )

def target_vs_numerical(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(f"8. {TARGET} vs. NUMERICAL VARIABLES (correlation)")

    # Exclude ewrb_id (identifier, not a measurement) and reporting_year
    # (constant at 2024 in the current dataset).
    exclude_cols = ["ewrb_id", "reporting_year"]
    numeric_df = df.select_dtypes(include="number").drop(
        columns=[c for c in exclude_cols if c in df.columns], errors="ignore"
    )
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

def plot_missing_values(df: pd.DataFrame) -> None:
    """Bar chart of missing-value counts per column, so it's visible at a
    glance where missing data is concentrated (e.g. third_party_certification
    at ~98% missing vs. the target's much smaller gap)."""
    missing_count = df.isna().sum()
    missing_count = missing_count[missing_count > 0].sort_values(ascending=False)
    if missing_count.empty:
        print("\nNo missing values to plot.")
        return

    plt.figure(figsize=(9, 5))
    missing_count.plot(kind="bar", edgecolor="black")
    plt.title("Missing Values by Column")
    plt.ylabel("Missing count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = output_dir / "missing_values.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved missing values chart to {path}")

def plot_site_eui_by_city(df: pd.DataFrame, top_n: int = 10) -> None:
    """Boxplot of site_eui_gj_m2 for the top N cities by building count: 
    limited for readability, same reasoning as the property-type boxplot."""
    top_cities = df["city"].value_counts().nlargest(top_n).index
    subset = df[df["city"].isin(top_cities)]

    plt.figure(figsize=(10, 6))
    subset.boxplot(column=TARGET, by="city", rot=45)
    plt.yscale("log")
    plt.title(f"{TARGET} by City (top {top_n} by building count, log-scale y-axis)")
    plt.suptitle("")
    plt.xlabel("City")
    plt.ylabel(f"{TARGET} (log scale)")
    plt.tight_layout()
    path = output_dir / "site_eui_by_city.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved city boxplot to {path}")

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Heatmap of correlations among all numeric columns: a compact view
    of what target_vs_numerical() reports as text, plus relationships
    between the non-target numeric variables themselves.

    Excludes ewrb_id (an identifier, not a measurement) and
    reporting_year (constant at 2024 in the current dataset, so its
    correlation with everything is undefined); neither is meaningful
    here."""
    exclude_cols = ["ewrb_id", "reporting_year"]
    numeric_df = df.select_dtypes(include="number").drop(
        columns=[c for c in exclude_cols if c in df.columns], errors="ignore"
    )
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix (numeric columns)")
    plt.tight_layout()
    path = output_dir / "correlation_matrix.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved correlation matrix to {path}")

def main():
    print("Loading modeling dataset from PostgreSQL...")
    df = load_dataset_from_postgres()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    inspect_shape(df)
    inspect_dtypes(df)
    inspect_missing_values(df)
    describe_target(df)
    inspect_property_types(df)
    check_property_type_redundancy(df)
    check_outliers(df)
    plot_target_distribution(df)
    plot_missing_values(df)
    plot_site_eui_by_city(df)
    plot_correlation_matrix(df)
    target_vs_numerical(df)
    verify_one_row_per_building(df)

    print("\n" + "=" * 60)
    print("EDA complete. Review the printed output and plots in "
          f"'{output_dir}/'.")

if __name__ == "__main__":
    main()
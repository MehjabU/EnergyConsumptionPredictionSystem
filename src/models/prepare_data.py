"""
src/models/prepare_data.py

Provides reusable data preparation functions: load, sort, clean, engineer
features, and split. This file does NOT train any model and does NOT run
dataset investigation/diagnostics — that belongs in eda.py. Experiments
live in train.py.

Run from the project root as a module for a quick sanity check during
development:

    python -m src.models.prepare_data
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.database.get_model_data import load_dataset_from_postgres

TARGET = "site_eui_gj_m2"

# First-pass, deliberately conservative feature set (v1). Everything else
# is excluded either because it's a different energy-use measure that's
# highly correlated with the target (source_eui, weather-normalized
# variants, electricity/gas intensity, ghg intensity, energy_star_score),
# because it's mostly missing (third_party_certification, used instead as
# a derived flag in v2), or because it's constant in the current dataset
# and carries no predictive information (reporting_year, currently always
# 2024).
CATEGORICAL_FEATURES = [
    "primary_property_type",
    "self_property_type",
    "largest_property_type",
    "city",
    "postal_code",  # Used as a geographic categorical variable (Canadian
                    # FSA, e.g. "M5V" - the first 3 characters of a postal
                    # code, representing an area), NOT a numeric field.
]

# v2 adds three engineered features derived from columns that were already
# present but unused: all_property_types and third_party_certification.
NEW_NUMERIC_FEATURES = [
    "n_property_types",
    "is_certified",
    "has_data_center",
]

# Each feature set stores its own categorical/numeric split rather than a
# flat list, so adding a v3 or v4 later doesn't require new
# `if feature_set == "..."` branches anywhere downstream then the categorical
# and numeric subsets are always read directly off the selected entry.
FEATURE_SETS = {
    "v1": {
        "categorical": CATEGORICAL_FEATURES,
        "numeric": [],
    },
    "v2": {
        "categorical": CATEGORICAL_FEATURES,
        "numeric": NEW_NUMERIC_FEATURES,
    },
}

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_modeling_dataframe() -> pd.DataFrame:
    """Step 1: load the modeling DataFrame from PostgreSQL."""
    return load_dataset_from_postgres()


def sort_for_reproducibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: sort deterministically by ewrb_id before any split.

    The SQL query in get_model_data.py has no ORDER BY, so PostgreSQL
    does not guarantee row order between runs. train_test_split() splits
    based on row position, so an unordered result set means the same
    random_state=42 could still produce a different train/test split
    from one run to the next. Sorting here makes the split deterministic
    regardless of what order the database happens to return rows in.
    """
    return df.sort_values("ewrb_id").reset_index(drop=True)


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: remove rows with a missing target. Never impute the target."""
    n_before = len(df)
    df_clean = df.dropna(subset=[TARGET]).copy()
    n_after = len(df_clean)
    n_dropped = n_before - n_after
    print(f"Dropped {n_dropped} rows with missing '{TARGET}' "
          f"({n_before} -> {n_after} rows).")
    return df_clean


def engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Step 4: add legitimate, interpretable features derived from columns
    that were already loaded but unused. Each one describes something
    about the building itself, not its energy performance.

    n_property_types  - count of use types listed in all_property_types
                         (assumes a consistently comma-separated string —
                         worth spot-checking a few raw values if this
                         ever looks off).
    is_certified      - 1 if third_party_certification is populated
                         (LEED, BOMA, etc.), else 0.
    has_data_center   - 1 if "Data Center" appears in all_property_types.

    These are two very sparse binary flags (third_party_certification alone is
    missing for ~98% of rows).
    """
    df = df.copy()
    df["n_property_types"] = df["all_property_types"].apply(
        lambda x: len(str(x).split(","))
    )
    df["is_certified"] = df["third_party_certification"].notna().astype(int)
    df["has_data_center"] = df["all_property_types"].str.contains(
        "Data Center", case=False, na=False
    ).astype(int)

    if verbose:
        print("\nEngineered feature distributions:")
        print(f"  n_property_types: min={df['n_property_types'].min()}, "
              f"max={df['n_property_types'].max()}, "
              f"mean={df['n_property_types'].mean():.2f}")
        print(f"  is_certified: {df['is_certified'].value_counts().to_dict()}")
        print(f"  has_data_center: {df['has_data_center'].value_counts().to_dict()}")

    return df


def check_property_type_redundancy(df: pd.DataFrame) -> None:
    """
    Diagnostic/EDA function: how often do primary_property_type,
    self_property_type, and largest_property_type match each other? This
    is dataset investigation, not data preparation.
    """
    cols = ["primary_property_type", "self_property_type", "largest_property_type"]
    if not all(c in df.columns for c in cols):
        return

    n = len(df)
    print("\nProperty type redundancy check:")
    pairs = [
        ("primary_property_type", "self_property_type"),
        ("primary_property_type", "largest_property_type"),
        ("self_property_type", "largest_property_type"),
    ]
    for col_a, col_b in pairs:
        match_pct = (df[col_a] == df[col_b]).mean() * 100
        print(f"  {col_a} == {col_b}: {match_pct:.1f}% of rows match")

    all_match_pct = (
        (df["primary_property_type"] == df["self_property_type"])
        & (df["primary_property_type"] == df["largest_property_type"])
    ).mean() * 100
    print(f"  All three match: {all_match_pct:.1f}% of rows ({n} rows total)")
    print(
        "  High match rates mean these columns contribute overlapping "
        "information rather than three independent signals."
    )


def define_features_and_target(df: pd.DataFrame, feature_set: str = "v1"):
    """Step 5: define X and y for the requested feature set ('v1' or 'v2')."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set '{feature_set}'. "
                          f"Choose from {list(FEATURE_SETS.keys())}.")

    categorical = FEATURE_SETS[feature_set]["categorical"]
    numeric = FEATURE_SETS[feature_set]["numeric"]
    feature_cols = categorical + numeric

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Expected feature columns not found in DataFrame: {missing_cols}")

    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y


def build_preprocessor(
    categorical_features: list[str], numeric_features: list[str] | None = None
) -> ColumnTransformer:
    """
    Step 6: build a ColumnTransformer that one-hot encodes categorical
    columns and passes numeric columns through unchanged. This transformer
    is NOT fit here.
    """
    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
    if numeric_features:
        transformers.append(("num", "passthrough", numeric_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def split_data(X: pd.DataFrame, y: pd.Series):
    """Step 7: create the train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


def prepare_data(feature_set: str = "v1"):
    """
    Runs the full preparation sequence and returns everything train.py
    needs: the train/test split, the (unfit) preprocessor, and the full
    cleaned and engineered DataFrame.

    feature_set: "v1" (5 categorical features) or "v2" (v1 + 3 engineered
    features: n_property_types, is_certified, has_data_center).

    The DataFrame is returned so error-analysis code in train.py can look
    up a test row's identifying/contextual details by index.

    """
    df = load_modeling_dataframe()
    df = sort_for_reproducibility(df)
    df = drop_missing_target(df)
    df = engineer_features(df)

    X, y = define_features_and_target(df, feature_set=feature_set)

    categorical = FEATURE_SETS[feature_set]["categorical"]
    numeric = FEATURE_SETS[feature_set]["numeric"]
    # NOTE: `preprocessor` is intentionally UNFIT. It's built here but not
    # fitted, so that train.py can fit it inside a sklearn Pipeline on
    # X_train only.
    preprocessor = build_preprocessor(categorical, numeric)
    X_train, X_test, y_train, y_test = split_data(X, y)

    return X_train, X_test, y_train, y_test, preprocessor, df


def main():
    """
    Development-only sanity check. Not part of the final pipeline.
    """
    for feature_set in FEATURE_SETS:
        print("\n" + "#" * 60)
        print(f"FEATURE SET: {feature_set}")
        X_train, X_test, y_train, y_test, preprocessor, df = prepare_data(feature_set)

        print(f"\nFeatures used: {FEATURE_SETS[feature_set]}")
        print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

        y_train_missing = y_train.isna().sum()
        y_test_missing = y_test.isna().sum()
        print(f"y_train missing values: {y_train_missing} "
              f"({'OK' if y_train_missing == 0 else 'PROBLEM'})")
        print(f"y_test missing values: {y_test_missing} "
              f"({'OK' if y_test_missing == 0 else 'PROBLEM'})")


if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
 
from src.database.get_model_data import load_dataset_from_postgres
 
TARGET = "site_eui_gj_m2"

CATEGORICAL_FEATURES = [
    "primary_property_type",
    "self_property_type",
    "largest_property_type",
    "city",
    "postal_code",  # Used as a geographic categorical variable (Canadian
                    # FSA, e.g. "M5V" - the first 3 characters of a postal
                    # code, representing an area), NOT a numeric field.
]
 
TEST_SIZE = 0.2
RANDOM_STATE = 42
 
 
def load_modeling_dataframe() -> pd.DataFrame:
    """Step 1: load the modeling DataFrame from PostgreSQL."""
    return load_dataset_from_postgres()
 
 
def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: remove rows with a missing target. Never impute the target."""
    n_before = len(df)
    df_clean = df.dropna(subset=[TARGET]).copy()
    n_after = len(df_clean)
    n_dropped = n_before - n_after
    print(f"Dropped {n_dropped} rows with missing '{TARGET}' "
          f"({n_before} -> {n_after} rows).")
    return df_clean
 
 
def define_features_and_target(df: pd.DataFrame):
    """Step 3: define X and y."""
    missing_cols = [c for c in CATEGORICAL_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Expected feature columns not found in DataFrame: {missing_cols}")
 
    X = df[CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y
 
 
def check_property_type_redundancy(df: pd.DataFrame) -> None:
    """
    primary_property_type, self_property_type, and largest_property_type
    may carry overlapping information (a building could have the same
    value across all three). This doesn't mean any of them should be
    dropped, but it's worth knowing how redundant they are before relying
    on all three as separate features.
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
        "information rather than three independent signals — worth "
        "keeping in mind when interpreting a model that uses all three, "
        "though nothing needs to be dropped for the baseline."
    )
 
 
def build_preprocessor(categorical_features: list[str]) -> ColumnTransformer:
    """
    Steps 4-5: separate categorical columns and prepare a ColumnTransformer
    that one-hot encodes them. This transformer is NOT fit here — train.py
    will fit it as part of a full sklearn Pipeline alongside the model, so
    the encoding is learned only on the training split (avoids leaking
    test-set categories into training).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            )
        ],
        remainder="drop",
    )
    return preprocessor
 
 
def split_data(X: pd.DataFrame, y: pd.Series):
    """Step 6: create the train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test
 
 
def prepare_data():
    """
    Runs the full preparation sequence and returns everything train.py
    needs: the train/test split, the (unfit) preprocessor, and the full
    cleaned DataFrame (df_clean).
 
    df_clean is returned so error-analysis code in train.py can look up a
    test row's identifying/contextual details (ewrb_id, city, property
    type, etc.) by index — X_train/X_test only contain the encoded model
    features, not an identifier, and train_test_split preserves the
    original DataFrame index, so df_clean.loc[X_test.index] recovers the
    full row for any test observation.
    """
    df = load_modeling_dataframe()
    df = drop_missing_target(df)
 
    check_property_type_redundancy(df)
 
    X, y = define_features_and_target(df)
    # NOTE: `preprocessor` is intentionally UNFIT. It's built here but not
    # fitted, so that whichever file trains a model can fit it inside a sklearn Pipeline on X_train only
    preprocessor = build_preprocessor(CATEGORICAL_FEATURES)
    X_train, X_test, y_train, y_test = split_data(X, y)
 
    return X_train, X_test, y_train, y_test, preprocessor, df
 
 
def main():
    X_train, X_test, y_train, y_test, preprocessor, df = prepare_data()
 
    print("\nX_train preview:")
    print(X_train.head())
    print("\ny_train preview:")
    print(y_train.head())
    print(f"\nCategorical features to encode: {CATEGORICAL_FEATURES}")
 
    # --- Verification ---
    print("\n" + "=" * 60)
    print("VERIFICATION (run before writing train.py)")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Expected split (~80/20 of dropped-target rows): "
          f"~{round((len(X_train) + len(X_test)) * 0.8)} / "
          f"~{round((len(X_train) + len(X_test)) * 0.2)}")
 
    y_train_missing = y_train.isna().sum()
    y_test_missing = y_test.isna().sum()
    print(f"y_train missing values: {y_train_missing} "
          f"({'OK' if y_train_missing == 0 else 'PROBLEM'})")
    print(f"y_test missing values: {y_test_missing} "
          f"({'OK' if y_test_missing == 0 else 'PROBLEM'})")
    print(f"y_train dtype: {y_train.dtype}, y_test dtype: {y_test.dtype}")
 
if __name__ == "__main__":
    main()
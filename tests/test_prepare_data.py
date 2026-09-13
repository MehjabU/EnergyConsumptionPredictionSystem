"""
tests/test_prepare_data.py

Unit tests for the pure functions in src/models/prepare_data.py. These
don't touch the database or the Excel files — they construct small
synthetic DataFrames so the tests run anywhere, fast, with no I/O.
"""

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.models.prepare_data import (
    TARGET,
    FEATURE_SETS,
    drop_missing_target,
    engineer_features,
    define_features_and_target,
    build_preprocessor,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ewrb_id": ["1", "2", "3", "4"],
        "city": ["Toronto", "Toronto", "Ottawa", "Ottawa"],
        "postal_code": ["M5V", "M5V", "K1A", "K1A"],
        "primary_property_type": ["Office", "Office", "Retail", "Retail"],
        "self_property_type": ["Office", "Office", "Retail", "Retail"],
        "largest_property_type": ["Office", "Office", "Retail", "Retail"],
        "all_property_types": [
            "Office", "Office,Parking", "Retail,Data Center", "Retail"
        ],
        "third_party_certification": [None, "LEED", None, None],
        TARGET: [1.0, 2.0, None, 4.0],
    })


def test_drop_missing_target_removes_only_null_rows(sample_df):
    result = drop_missing_target(sample_df)
    assert len(result) == 3
    assert result[TARGET].isna().sum() == 0


def test_engineer_features_creates_expected_columns(sample_df):
    result = engineer_features(sample_df, verbose=False)
    assert "n_property_types" in result.columns
    assert "is_certified" in result.columns
    assert "has_data_center" in result.columns


def test_n_property_types_counts_comma_separated_values(sample_df):
    result = engineer_features(sample_df, verbose=False)
    # "Office" -> 1, "Office,Parking" -> 2
    assert result.loc[0, "n_property_types"] == 1
    assert result.loc[1, "n_property_types"] == 2


def test_is_certified_flags_only_populated_certifications(sample_df):
    result = engineer_features(sample_df, verbose=False)
    assert result.loc[0, "is_certified"] == 0
    assert result.loc[1, "is_certified"] == 1


def test_has_data_center_is_case_insensitive(sample_df):
    result = engineer_features(sample_df, verbose=False)
    # row 2 has "Retail,Data Center" -> should be flagged
    assert result.loc[2, "has_data_center"] == 1
    assert result.loc[0, "has_data_center"] == 0


def test_define_features_and_target_raises_on_unknown_feature_set(sample_df):
    with pytest.raises(ValueError):
        define_features_and_target(sample_df, feature_set="v99")


def test_define_features_and_target_v1_excludes_engineered_columns(sample_df):
    df = engineer_features(sample_df, verbose=False)
    X, y = define_features_and_target(df, feature_set="v1")
    assert "n_property_types" not in X.columns
    assert list(X.columns) == FEATURE_SETS["v1"]["categorical"]
    assert (y == df[TARGET]).all()


def test_define_features_and_target_v2_includes_engineered_columns(sample_df):
    df = engineer_features(sample_df, verbose=False)
    X, y = define_features_and_target(df, feature_set="v2")
    for col in FEATURE_SETS["v2"]["numeric"]:
        assert col in X.columns


def test_build_preprocessor_returns_column_transformer():
    preprocessor = build_preprocessor(["city"], ["n_property_types"])
    assert isinstance(preprocessor, ColumnTransformer)
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "cat" in transformer_names
    assert "num" in transformer_names


def test_build_preprocessor_without_numeric_features_omits_num_step():
    preprocessor = build_preprocessor(["city"], None)
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" not in transformer_names
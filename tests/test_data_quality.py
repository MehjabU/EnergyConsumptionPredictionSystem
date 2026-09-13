"""
tests/test_data_quality.py

Validates assumptions the rest of the pipeline depends on: primary key
integrity, referential integrity between buildings and energy_performance,
and expected schema. These are cheap checks that catch a broken upstream
export before it silently corrupts downstream modeling.
"""

import pandas as pd

EXPECTED_BUILDINGS_COLUMNS = {
    "ewrb_id", "city", "postal_code", "primary_property_type",
    "self_property_type", "largest_property_type", "all_property_types",
    "third_party_certification",
}

EXPECTED_ENERGY_COLUMNS = {
    "ewrb_id", "reporting_year", "electricity_intensity_gj_m2",
    "gas_intensity_gj_m2", "water_intensity_m3_m2",
    "indoor_water_intensity_m3_m2", "site_eui_gj_m2",
    "weather_normalized_site_eui_gj_m2", "source_eui_gj_m2",
    "weather_normalized_source_eui_gj_m2", "ghg_intensity_kgco2e_m2",
    "energy_star_score",
}

def test_buildings_has_expected_columns(buildings_df: pd.DataFrame):
    assert EXPECTED_BUILDINGS_COLUMNS.issubset(set(buildings_df.columns))

def test_energy_has_expected_columns(energy_df: pd.DataFrame):
    assert EXPECTED_ENERGY_COLUMNS.issubset(set(energy_df.columns))

def test_ewrb_id_is_unique_in_buildings(buildings_df: pd.DataFrame):
    """ewrb_id is the primary key of buildings — must have no duplicates."""
    duplicate_count = buildings_df["ewrb_id"].duplicated().sum()
    assert duplicate_count == 0, (
        f"Found {duplicate_count} duplicate ewrb_id values in buildings — "
        f"this would violate the PRIMARY KEY constraint on load."
    )

def test_ewrb_id_reporting_year_is_unique_in_energy(energy_df: pd.DataFrame):
    """Matches the UNIQUE (ewrb_id, reporting_year) constraint added to
    energy_performance — each building should report at most once per year."""
    duplicate_count = energy_df.duplicated(subset=["ewrb_id", "reporting_year"]).sum()
    assert duplicate_count == 0, (
        f"Found {duplicate_count} duplicate (ewrb_id, reporting_year) pairs "
        f"in energy_performance."
    )

def test_every_energy_row_has_a_matching_building(buildings_df: pd.DataFrame, energy_df: pd.DataFrame):
    """Referential integrity: every ewrb_id in energy_performance should
    exist in buildings (matches the REFERENCES buildings(ewrb_id) FK)."""
    orphaned = ~energy_df["ewrb_id"].isin(buildings_df["ewrb_id"])
    orphan_count = orphaned.sum()
    assert orphan_count == 0, (
        f"Found {orphan_count} energy_performance rows referencing an "
        f"ewrb_id not present in buildings."
    )

def test_site_eui_is_non_negative(energy_df: pd.DataFrame):
    """site_eui_gj_m2 is a physical intensity measure — negative values
    would indicate a data error, not a valid observation."""
    negative_count = (energy_df["site_eui_gj_m2"] < 0).sum()
    assert negative_count == 0, (
        f"Found {negative_count} negative site_eui_gj_m2 values."
    )

def test_reporting_year_is_reasonable(energy_df: pd.DataFrame):
    """Sanity bound — catches an obviously corrupted year value (e.g. a
    unit or parsing error) without hard-coding a single expected year."""
    years = energy_df["reporting_year"].dropna()
    assert years.between(2000, 2100).all(), (
        "Found reporting_year values outside a plausible range (2000-2100)."
    )
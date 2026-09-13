"""
tests/conftest.py

Shared fixtures. Data-quality tests read the actual cleaned Excel files
from data/processed/ (the same files load_data.py loads into Postgres) —
they're skipped automatically if those files aren't present, so the test
suite still runs in an environment without the data (e.g. CI) without
failing.
"""

from pathlib import Path
import pandas as pd
import pytest

BUILDINGS_PATH = Path("data/processed/odc_building_dataset_2024_cleaned.xlsx")
ENERGY_PATH = Path("data/processed/odc_energy_performance_dataset_2024_cleaned.xlsx")


def _skip_if_missing(path: Path):
    if not path.exists():
        pytest.skip(f"Data file not found: {path} (skipping data-dependent test)")


@pytest.fixture(scope="session")
def buildings_df() -> pd.DataFrame:
    _skip_if_missing(BUILDINGS_PATH)
    return pd.read_excel(BUILDINGS_PATH)


@pytest.fixture(scope="session")
def energy_df() -> pd.DataFrame:
    _skip_if_missing(ENERGY_PATH)
    return pd.read_excel(ENERGY_PATH)


@pytest.fixture(scope="session")
def joined_df(buildings_df, energy_df) -> pd.DataFrame:
    return buildings_df.merge(energy_df, on="ewrb_id", how="inner")
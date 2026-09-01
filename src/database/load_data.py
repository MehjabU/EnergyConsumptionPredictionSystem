import pandas as pd
from pathlib import Path
from connection import get_connection

def convert_missing(value):
    if pd.isna(value):
        return None
    return value

file_path = Path("data/processed/odc_building_dataset_2024_cleaned.xlsx")
buildings_df = pd.read_excel(file_path)

energy_file_path = Path("data/processed/odc_energy_performance_dataset_2024_cleaned.xlsx")
energy_df = pd.read_excel(energy_file_path)

insert_query = """
INSERT INTO buildings (
    ewrb_id,
    city,
    postal_code,
    primary_property_type,
    self_property_type,
    largest_property_type,
    all_property_types,
    third_party_certification
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (ewrb_id) DO NOTHING;
"""

energy_insert_query = """
INSERT INTO energy_performance (
    ewrb_id,
    reporting_year,
    electricity_intensity_gj_m2,
    gas_intensity_gj_m2,
    water_intensity_m3_m2,
    indoor_water_intensity_m3_m2,
    site_eui_gj_m2,
    weather_normalized_site_eui_gj_m2,
    source_eui_gj_m2,
    weather_normalized_source_eui_gj_m2,
    ghg_intensity_kgco2e_m2,
    energy_star_score
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (ewrb_id, reporting_year) DO NOTHING;
"""

connection = get_connection()
cursor = connection.cursor()

try:
    for row in buildings_df.itertuples():
        values = (
            row.ewrb_id,
            row.city,
            row.postal_code,
            row.primary_property_type,
            row.self_property_type,
            row.largest_property_type,
            row.all_property_types,
            convert_missing(row.third_party_certification)
        )
        cursor.execute(insert_query, values)

    connection.commit()

    for row in energy_df.itertuples():
        energy_values = (
            row.ewrb_id,
            row.reporting_year,
            convert_missing(row.electricity_intensity_gj_m2),
            convert_missing(row.gas_intensity_gj_m2),
            convert_missing(row.water_intensity_m3_m2),
            convert_missing(row.indoor_water_intensity_m3_m2),
            convert_missing(row.site_eui_gj_m2),
            convert_missing(row.weather_normalized_site_eui_gj_m2),
            convert_missing(row.source_eui_gj_m2),
            convert_missing(row.weather_normalized_source_eui_gj_m2),
            convert_missing(row.ghg_intensity_kgco2e_m2),
            convert_missing(row.energy_star_score)
        )
        cursor.execute(energy_insert_query, energy_values)

    connection.commit()

except Exception:
    connection.rollback()
    raise

finally:
    cursor.close()
    connection.close()
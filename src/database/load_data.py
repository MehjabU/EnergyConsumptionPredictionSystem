import pandas as pd
from pathlib import Path
from connection import get_connection

file_path = Path("data/processed/odc_building_dataset_2024_cleaned.xlsx")
buildings_df = pd.read_excel(file_path)

connection = get_connection()
cursor = connection.cursor()

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

for row in buildings_df.itertuples():
    if pd.isna(row.third_party_certification):
        certification = None
    else:
        certification = row.third_party_certification

    values = (
        row.ewrb_id,
        row.city,
        row.postal_code,
        row.primary_property_type,
        row.self_property_type,
        row.largest_property_type,
        row.all_property_types,
        certification
    )
    cursor.execute(insert_query, values)

connection.commit()

energy_file_path = Path("data/processed/odc_energy_performance_dataset_2024_cleaned.xlsx")
energy_df = pd.read_excel(energy_file_path)

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

for row in energy_df.itertuples():
    if pd.isna(row.electricity_intensity_gj_m2):
        electricity_intensity_gj_m2 = None
    else:
        electricity_intensity_gj_m2 = row.electricity_intensity_gj_m2

    if pd.isna(row.gas_intensity_gj_m2):
        gas_intensity_gj_m2 = None
    else:
        gas_intensity_gj_m2 = row.gas_intensity_gj_m2

    if pd.isna(row.water_intensity_m3_m2):
        water_intensity_m3_m2 = None
    else:
        water_intensity_m3_m2 = row.water_intensity_m3_m2

    if pd.isna(row.indoor_water_intensity_m3_m2):
        indoor_water_intensity_m3_m2 = None
    else:
        indoor_water_intensity_m3_m2 = row.indoor_water_intensity_m3_m2

    if pd.isna(row.site_eui_gj_m2):
        site_eui_gj_m2 = None
    else:
        site_eui_gj_m2 = row.site_eui_gj_m2

    if pd.isna(row.weather_normalized_site_eui_gj_m2):
        weather_normalized_site_eui_gj_m2 = None
    else:
        weather_normalized_site_eui_gj_m2 = row.weather_normalized_site_eui_gj_m2

    if pd.isna(row.source_eui_gj_m2):
        source_eui_gj_m2 = None
    else:
        source_eui_gj_m2 = row.source_eui_gj_m2

    if pd.isna(row.weather_normalized_source_eui_gj_m2):
        weather_normalized_source_eui_gj_m2 = None
    else:
        weather_normalized_source_eui_gj_m2 = row.weather_normalized_source_eui_gj_m2

    if pd.isna(row.ghg_intensity_kgco2e_m2):
        ghg_intensity_kgco2e_m2 = None
    else:
        ghg_intensity_kgco2e_m2 = row.ghg_intensity_kgco2e_m2

    if pd.isna(row.energy_star_score):
        energy_star_score = None
    else:
        energy_star_score = row.energy_star_score

    energy_values = (
        row.ewrb_id,
        row.reporting_year,
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
    cursor.execute(energy_insert_query, energy_values)

connection.commit()
cursor.close()
connection.close()
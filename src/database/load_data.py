import pandas as pd
# import openpyxl as op
# import numpy as np
# import psycopg2
from pathlib import Path
from connection import get_connection

buildings_file_path = Path("data/processed/odc_building_dataset_2024_cleaned.xlsx")
energy_file_path = Path("data/processed/odc_energy_performance_dataset_2024_cleaned.xlsx")

buildings_df = pd.read_excel(buildings_file_path)
energy_df = pd.read_excel(energy_file_path)

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
cursor.close()
connection.close()
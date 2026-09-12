import pandas as pd
from pathlib import Path
from connection import get_connection
 
query = """
WITH latest_year AS (
    SELECT ewrb_id, MAX(reporting_year) AS max_year
    FROM energy_performance
    GROUP BY ewrb_id
)
SELECT
    b.ewrb_id,
    b.city,
    b.postal_code,
    b.primary_property_type,
    b.self_property_type,
    b.largest_property_type,
    b.all_property_types,
    b.third_party_certification,
    e.reporting_year,
    e.electricity_intensity_gj_m2,
    e.gas_intensity_gj_m2,
    e.water_intensity_m3_m2,
    e.indoor_water_intensity_m3_m2,
    e.site_eui_gj_m2,
    e.weather_normalized_site_eui_gj_m2,
    e.source_eui_gj_m2,
    e.weather_normalized_source_eui_gj_m2,
    e.ghg_intensity_kgco2e_m2,
    e.energy_star_score
FROM buildings b
JOIN energy_performance e
    ON b.ewrb_id = e.ewrb_id
JOIN latest_year ly
    ON e.ewrb_id = ly.ewrb_id
   AND e.reporting_year = ly.max_year;
"""
 
output_path = Path("data/processed/model_dataset.csv")
 
 
def load_dataset_from_postgres() -> pd.DataFrame:
    """Run the JOIN query and return the result as a pandas DataFrame."""
    connection = get_connection()
    try:
        df = pd.read_sql_query(query, connection)
    finally:
        connection.close()
    return df 

def main():
    print("Connecting to PostgreSQL and running JOIN query...")

    df = load_dataset_from_postgres()

    print(f"Retrieved {len(df)} rows, {len(df.columns)} columns.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved modeling dataset to {output_path}")
 
 
if __name__ == "__main__":
    main()
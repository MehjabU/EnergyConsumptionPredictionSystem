import pandas as pd
from pathlib import Path
import openpyxl as op
import numpy as np

file_path = Path("data/raw/odc_final_dataset_2024.xlsx")

df = pd.read_excel(file_path)
columns = ['EWRB_ID','WN_Sit_Elc_Int1', 'WN_Sit_Gas_Int1', 'All_Water_Int1', 'Ind_Water_Int1', 
            'Site_EUI1', 'WN_Site_EUI1', 'Source_EUI1', 'WN_Source_EUI1', 'GHG_Emiss_Int1', 'Ener_Star_Score']

performance_df = df[columns].copy()

print(performance_df.head())
print(performance_df.shape) #for this file, should return (6739, 11) -> (rows, cols)
print(performance_df.dtypes) #check data types of each column

for column_name in columns:
    if column_name != 'Ener_Star_Score' and column_name != 'EWRB_ID':
        performance_df[column_name] = performance_df[column_name].replace('Not Available', np.nan)
        performance_df[column_name] = pd.to_numeric(performance_df[column_name], errors='coerce')

performance_df['Ener_Star_Score'] = performance_df['Ener_Star_Score'].replace('Not Available', np.nan).astype('Int64')

print(f"\nData types after conversion:\n{performance_df.dtypes}") #check data types of each column after conversion

#this removes \uffd from some of the rows in the only column that has it - EWRB_ID
performance_df['EWRB_ID'] = performance_df['EWRB_ID'].str.replace('\ufffd', '', regex=False)

# Here I rename columns to match what I will put in postgreSQL

df2 = performance_df.copy()

df2 = df2.rename(columns={
    'EWRB_ID': 'ewrb_id',
    'WN_Sit_Elc_Int1': 'electricity_intensity_gj_m2',
    'WN_Sit_Gas_Int1': 'gas_intensity_gj_m2',
    'All_Water_Int1': 'water_intensity_m3_m2',
    'Ind_Water_Int1': 'indoor_water_intensity_m3_m2',
    'Site_EUI1': 'site_eui_gj_m2',
    'WN_Site_EUI1': 'weather_normalized_site_eui_gj_m2',
    'Source_EUI1': 'source_eui_gj_m2',
    'WN_Source_EUI1': 'weather_normalized_source_eui_gj_m2',
    'GHG_Emiss_Int1': 'ghg_intensity_kgco2_m2',
    'Ener_Star_Score': 'energy_star_score'})

# added a new column for reporting year, which will be 2024 for this dataset
df2.insert(1, 'reporting_year', 2024)
df2.to_excel("data/processed/odc_final_dataset_2024_cleaned.xlsx", index=False)

print(df2.dtypes) #check data types of each column after conversion
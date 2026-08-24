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

import pandas as pd
from pathlib import Path
import openpyxl as op

file_path = Path("data/raw/odc_final_dataset_2024.xlsx")

df = pd.read_excel(file_path)
performance_df = df[['EWRB_ID','WN_Sit_Elc_Int1', 'WN_Sit_Gas_Int1', 'All_Water_Int1', 'Ind_Water_Int1', 
                     'Site_EUI1', 'WN_Site_EUI1', 'Source_EUI1', 'WN_Source_EUI1', 'GHG_Emiss_Int1', 'Ener_Star_Score']].copy()

print(performance_df.head())
print(performance_df.shape) #for this file, should return (6739, 11) -> (rows, cols)
print(performance_df.dtypes) #check data types of each column
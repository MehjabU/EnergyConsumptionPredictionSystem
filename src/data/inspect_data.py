from pathlib import Path
import pandas as pd

#purpose of this file: inspect xlsx file to check for missing data, max char len, and index of max for each column
#will refactor later

file_path = Path("data/raw/odc_final_dataset_2024.xlsx")

df = pd.read_excel(file_path)
#print(df.head())
print(df.shape) #for this file, should return (6739, 31) -> (rows, cols)

max_city_idx = df['City'].str.len().idxmax()

print(f"\nLongest city name has: {df['City'].str.len().max()} chars")
print(f"Index of maximum length: {max_city_idx}")
print(f"Longest city name: {df.loc[max_city_idx, 'City']}")
print(f"Are there any missing data in the dataset? {df['City'].isnull().sum()}")

print(f"\nLongest city name has postal code: {df['Postal_Code'].str.len().max()} chars")
print(f"Are there any missing data in the dataset? {df['Postal_Code'].isnull().sum()}")

max_primary_idx = df['PrimPropTypCalc'].str.len().idxmax()
print(f"Max char length for primary property type: {df['PrimPropTypCalc'].str.len().max()} chars")
print(f"Index of maximum length for primary property type: {max_primary_idx}")
print(f"name of max index property: {df.loc[max_primary_idx, 'PrimPropTypCalc']}")
print(f"Are there any missing data in the dataset? {df['PrimPropTypCalc'].isnull().sum()}")

max_self_idx = df['PrimPropTypSelf'].str.len().idxmax()
print(f"\nMax char length for primary property type self: {df['PrimPropTypSelf'].str.len().max()} chars")
print(f"Index of maximum length for primary property type self: {max_self_idx}")
print(f"name of max index property: {df.loc[max_self_idx, 'PrimPropTypSelf']}")
print(f"Are there any missing data in the dataset? {df['PrimPropTypSelf'].isnull().sum()}")

max_largest_property_idx = df['Largest_PropTyp'].str.len().idxmax()
print(f"\nMax char length for largest property type: {df['Largest_PropTyp'].str.len().max()} chars")
print(f"Index of maximum length for largest property type: {max_largest_property_idx}")
print(f"name of max index property: {df.loc[max_largest_property_idx, 'Largest_PropTyp']}")
print(f"Are there any missing data in the dataset? {df['Largest_PropTyp'].isnull().sum()}")

maxCert_idx = df['Thrd_Party_Cert'].str.len().idxmax()
print(f"\nMax char length for third party certification: {df['Thrd_Party_Cert'].str.len().max()} chars")
print(f"Index of maximum length for third party certification: {maxCert_idx}")
print(f"name of max index property: {df.loc[maxCert_idx, 'Thrd_Party_Cert']}")
print(f"Are there any missing data in the dataset? {df['Thrd_Party_Cert'].isnull().sum()}")

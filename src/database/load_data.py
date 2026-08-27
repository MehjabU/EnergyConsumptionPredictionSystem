import pandas as pd
import openpyxl as op
import numpy as np
import psycopg2
from pathlib import Path
from connection import get_connection

file_path = Path("data/processed/odc_building_dataset_2024_cleaned.xlsx")

buildings_df = pd.read_excel(file_path)

head = buildings_df.head()

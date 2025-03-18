import pandas as pd
import numpy as np

# Load the Excel file
file_path = "Random_Forest_from_scratch/QoL_Tim_TotalWaste.xlsx"  # Change to your actual file path
df = pd.read_excel(file_path, sheet_name="Features", header=[0, 1])  # Use first two rows as headers

df_2013 = df.loc[:, df.columns.get_level_values(1) == "2013"]

df_2013 = df_2013.dropna(how="all")

df_locations = df.iloc[:, 0].dropna(how="all")  # Selects the first column

df_final = pd.concat([df_locations, df_2013], axis=1)
df_final.columns = df_final.columns.droplevel(1)

for col in df_final.columns[1:]:
    df_final[col] = df_final[col].apply(lambda x: np.nan if isinstance(x, str) else x)

# df_final.to_csv('desired_filename.csv', index=False)





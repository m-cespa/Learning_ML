import pandas as pd

file_path = 'Random_Forest_from_scratch/denmark_waste/cleaned_data.xlsx'
df = pd.read_excel(file_path)

# Assuming your DataFrame is called df and countries are in the 'country' column
# First, identify all numeric columns that you want to demean
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

# Remove the 'year' column from the list if you don't want to demean years
if 'year' in numeric_columns:
    numeric_columns.remove('year')
    
# Create a new DataFrame to store the demeaned data
df_demeaned = df.copy()

# Apply demeaning for each numeric column
for column in numeric_columns:
    # Calculate country-wise means
    country_means = df.groupby('country')[column].transform('mean')
    
    # Create demeaned columns in the new DataFrame
    df_demeaned[column] = df[column] - country_means

# Now df_demeaned contains all your demeaned values while df remains unchanged
print("Original data shape:", df_demeaned.shape)

# Remove rows where country is Copenhagen
if 'country' in df_demeaned.columns:
    df_demeaned = df_demeaned[~df_demeaned['country'].str.contains('Copenhagen', case=False)]
    print("Data shape after removing Copenhagen:", df_demeaned.shape)
else:
    print("Warning: 'country' column not found in dataframe. Could not filter Copenhagen.")

df_demeaned.to_csv('some_csv.csv')

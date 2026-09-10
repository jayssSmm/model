import pandas as pd

# 1. Read the Parquet file
df = pd.read_parquet('combined_df.parquet')

# 2. Convert and save it to a CSV file (index=False prevents an extra column)
df.to_csv('combined_df.csv', index=False)

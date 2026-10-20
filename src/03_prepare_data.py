
import pandas as pd

# Load raw data
df = pd.read_csv("data/raw/online_shoppers_intention.csv")

print("Original dataset shape:", df.shape)

# Inspect duplicate rows before removing anything
print("Duplicate rows:", df.duplicated().sum())

# Check how many duplicate rows have each target value
duplicates = df[df.duplicated(keep=False)]

print("\nTarget distribution among repeated rows:")
print(duplicates["Revenue"].value_counts())

# Create a copy for analysis
df_prepared = df.copy()

# Convert Boolean columns to integers (False=0, True=1)
df_prepared["Weekend"] = df_prepared["Weekend"].astype(int)
df_prepared["Revenue"] = df_prepared["Revenue"].astype(int)

# Save a prepared copy without deleting duplicates yet
df_prepared.to_csv(
    "data/processed/online_shoppers_prepared.csv",
    index=False
)

print("\nPrepared dataset shape:", df_prepared.shape)
print("Saved to data/processed/online_shoppers_prepared.csv")
print("\nData types:")
print(df_prepared.dtypes)

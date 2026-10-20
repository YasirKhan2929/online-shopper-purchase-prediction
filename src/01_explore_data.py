
import pandas as pd

# Load the dataset
file_path = "data/raw/online_shoppers_intention.csv"
df = pd.read_csv(file_path)

# Display basic information
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nPurchase distribution:")
print(df["Revenue"].value_counts())

print("\nPurchase percentages:")
print(df["Revenue"].value_counts(normalize=True) * 100)

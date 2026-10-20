
import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/online_shoppers_intention.csv")

# 1. Statistical summary
print("=== Statistical Summary ===")
print(df.describe().round(2))

# 2. Visitor types
print("\n=== Visitor Types ===")
print(df["VisitorType"].value_counts())

# 3. Purchases by month
print("\n=== Purchases by Month ===")
print(pd.crosstab(df["Month"], df["Revenue"]))

# 4. Average page values for each outcome
print("\n=== Average Page Values ===")
print(df.groupby("Revenue")["PageValues"].mean().round(2))

# 5. Average product-related browsing duration
print("\n=== Product Browsing Duration ===")
print(
    df.groupby("Revenue")["ProductRelated_Duration"]
    .mean()
    .round(2)
)

# 6. Check duplicate rows
print("\n=== Duplicate Rows ===")
print(df.duplicated().sum())

# 7. Check target distribution
print("\n=== Target Distribution ===")
print(df["Revenue"].value_counts(normalize=True).mul(100).round(2))

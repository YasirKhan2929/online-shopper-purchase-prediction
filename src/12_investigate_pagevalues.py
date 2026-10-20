
import pandas as pd

# Load prepared dataset
df = pd.read_csv("data/processed/online_shoppers_prepared.csv")

print("PAGEVALUES LEAKAGE INVESTIGATION")
print("=" * 60)

# 1. Check PageValues by purchase outcome
print("\n1. PageValues statistics by Revenue")
print(
    df.groupby("Revenue")["PageValues"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(3)
)

# 2. Count zero and non-zero values
print("\n2. PageValues zero/non-zero counts by Revenue")
df["PageValues_Status"] = df["PageValues"].eq(0).map({
    True: "Zero",
    False: "Non-zero",
})

print(pd.crosstab(
    df["Revenue"],
    df["PageValues_Status"],
    margins=True,
))

# 3. Percentage of zero values in each class
print("\n3. Percentage of zero PageValues by Revenue")
zero_percent = (
    df.assign(IsZero=df["PageValues"].eq(0))
    .groupby("Revenue")["IsZero"]
    .mean()
    .mul(100)
)

print(zero_percent.round(2).astype(str) + "%")

# 4. Check correlation with Revenue
print("\n4. Correlation between PageValues and Revenue")
print(round(df["PageValues"].corr(df["Revenue"]), 3))

# 5. Check the highest PageValues sessions
print("\n5. Five sessions with highest PageValues")
print(
    df.nlargest(5, "PageValues")[
        ["PageValues", "Revenue", "Month", "VisitorType"]
    ].to_string(index=False)
)

print("\nInvestigation complete.")
print("These statistics identify patterns, not leakage by themselves.")

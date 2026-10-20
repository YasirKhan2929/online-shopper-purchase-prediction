
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Load saved model
saved_data = joblib.load("models/purchase_prediction_model.joblib")
model = saved_data["model"]
threshold = saved_data["threshold"]

# Load data
df = pd.read_csv("data/processed/online_shoppers_prepared.csv")
X = df.drop(columns=["Revenue", "PageValues"])
y = df["Revenue"]

# Recreate the original held-out test set
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Predict only on unseen test records
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Probability": probabilities,
})

print("HELD-OUT TEST SET ANALYSIS")
print("=" * 45)

print("\nPurchase probability by actual outcome:")
print(
    results.groupby("Actual")["Probability"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(3)
)

print("\nMetrics at threshold:", threshold)
print("Precision:", round(precision_score(y_test, predictions), 3))
print("Recall:", round(recall_score(y_test, predictions), 3))
print("F1 score:", round(f1_score(y_test, predictions), 3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

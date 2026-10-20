
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# 1. Load prepared data
df = pd.read_csv("data/processed/online_shoppers_prepared.csv")

# Exclude Revenue (target) and PageValues (potential leakage)
X = df.drop(columns=["Revenue", "PageValues"])
y = df["Revenue"]

# 2. Split data: 60% train, 20% validation, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val,
)

print("DATA SPLIT")
print("-" * 50)
print(f"Training records:   {len(X_train)}")
print(f"Validation records: {len(X_val)}")
print(f"Test records:       {len(X_test)}")

# 3. Identify column types using training data
categorical_cols = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

# 4. Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numerical_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

# 5. Define both models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

# 6. Train and compare on validation data
results = []
threshold_results = []

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_val)[:, 1]
    default_predictions = (probabilities >= 0.50).astype(int)

    results.append({
        "Model": name,
        "Precision": precision_score(
            y_val, default_predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, default_predictions, zero_division=0
        ),
        "F1": f1_score(
            y_val, default_predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_val, probabilities),
    })

    # Check different decision thresholds on validation data
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        predictions = (probabilities >= threshold).astype(int)

        threshold_results.append({
            "Model": name,
            "Threshold": threshold,
            "Precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_val, predictions, zero_division=0
            ),
        })

# 7. Display model comparison at threshold 0.50
comparison = pd.DataFrame(results)

print("\nMODEL COMPARISON AT 50% THRESHOLD")
print("-" * 75)
print(comparison.round(3).to_string(index=False))

# 8. Display best threshold for each model by validation F1
threshold_df = pd.DataFrame(threshold_results)

best_indices = threshold_df.groupby("Model")["F1"].idxmax()
best_thresholds = threshold_df.loc[best_indices].sort_values("Model")

print("\nBEST THRESHOLD PER MODEL (VALIDATION F1)")
print("-" * 75)
print(best_thresholds.round(3).to_string(index=False))

print("\nNote: The test set was not used for training or selection.")

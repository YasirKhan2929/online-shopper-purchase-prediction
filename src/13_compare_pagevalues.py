
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

# Load data
df = pd.read_csv("data/processed/online_shoppers_prepared.csv")

y = df["Revenue"]

# Keep the same split for a fair comparison
X_all = df.drop(columns=["Revenue"])

X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all,
    y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train,
)

# Compare two feature sets on validation data only
feature_sets = {
    "Without PageValues": [
        col for col in X_train.columns if col != "PageValues"
    ],
    "With PageValues": list(X_train.columns),
}

print("PAGEVALUES VALIDATION COMPARISON")
print("=" * 65)

for name, columns in feature_sets.items():

    X_train_model = X_train[columns]
    X_val_model = X_val[columns]

    categorical_cols = X_train_model.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_cols = X_train_model.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

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

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )),
    ])

    model.fit(X_train_model, y_train)

    probabilities = model.predict_proba(X_val_model)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    print(f"\n{name}")
    print("-" * 40)
    print(f"ROC-AUC:   {roc_auc_score(y_val, probabilities):.3f}")
    print(f"Precision: {precision_score(y_val, predictions, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_val, predictions, zero_division=0):.3f}")
    print(f"F1 score:  {f1_score(y_val, predictions, zero_division=0):.3f}")

print("\nValidation comparison complete.")
print("The final test set was not used in this comparison.")

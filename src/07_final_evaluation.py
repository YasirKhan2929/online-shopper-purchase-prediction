
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# Load data
df = pd.read_csv("data/processed/online_shoppers_prepared.csv")

X = df.drop(columns=["Revenue", "PageValues"])
y = df["Revenue"]

categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X.select_dtypes(
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

# Recreate the same held-out test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train using training + validation data
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )),
])

model.fit(X_train_val, y_train_val)

# Evaluate on the untouched test set
probabilities = model.predict_proba(X_test)[:, 1]

threshold = 0.60
predictions = (probabilities >= threshold).astype(int)

print("Selected threshold:", threshold)
print("Test ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

print("\nFinal Classification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["No Purchase", "Purchase"],
    zero_division=0,
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


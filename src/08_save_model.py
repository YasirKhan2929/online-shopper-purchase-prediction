
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load prepared dataset
df = pd.read_csv("data/processed/online_shoppers_prepared.csv")

X = df.drop(columns=["Revenue", "PageValues"])
y = df["Revenue"]

categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

# Preprocessing pipeline
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

# Reserve the same test set used for evaluation
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train final pipeline on training + validation data
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )),
])

model.fit(X_train_val, y_train_val)

# Save model
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "purchase_prediction_model.joblib"

joblib.dump(
    {
        "model": model,
        "threshold": 0.60,
        "feature_columns": X.columns.tolist(),
    },
    model_path,
)

print("Model saved successfully!")
print("Path:", model_path)
print("Threshold:", 0.60)
print("Training rows:", len(X_train_val))
print("Test rows reserved:", len(X_test))

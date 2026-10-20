
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
    confusion_matrix,
    classification_report,
)

# 1. Load prepared dataset
df = pd.read_csv("data/processed/online_shoppers_prepared.csv")

# Exclude target and potential leakage feature
X = df.drop(columns=["Revenue", "PageValues"])
y = df["Revenue"]

# 2. Recreate the same 60/20/20 split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val,
)

print("DATA SPLIT")
print("-" * 50)
print(f"Training:   {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test:       {len(X_test)}")

# 3. Identify feature types using training data only
categorical_cols = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()


def create_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numerical_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])


# 4. Define candidates and their validation-selected thresholds
candidates = [
    (
        "Logistic Regression",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        0.60,
    ),
    (
        "Random Forest",
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        0.40,
    ),
]

# 5. Fit using training data; evaluate once on the test set
results = []

for name, classifier, threshold in candidates:
    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", classifier),
    ])

    model.fit(X_train, y_train)

    # Test predictions
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)
    matrix = confusion_matrix(y_test, predictions)

    results.append({
        "Model": name,
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": auc,
    })

    print("\n" + "=" * 60)
    print(f"MODEL: {name}")
    print(f"Decision threshold: {threshold:.2f}")
    print("-" * 60)

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")

    print("\nConfusion matrix:")
    print(matrix)
    print("\nRows = actual class; columns = predicted class")
    print("Order: [[true negatives, false positives],")
    print("        [false negatives, true positives]]")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["No Purchase", "Purchase"],
        zero_division=0,
    ))

# 6. Summarize test results
summary = pd.DataFrame(results)

print("\n\nFINAL TEST COMPARISON")
print("=" * 75)
print(summary.round(3).to_string(index=False))

print("\nEvaluation complete. Test data was not used for fitting.")

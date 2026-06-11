"""
train_model.py — Heart Disease Classification Model Trainer
Run this once to train and save the model before starting the Flask app.
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os

# ─── scikit-learn imports ─────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

# ─── Load Dataset ─────────────────────────────────────────────────────────────
dataset_path = os.path.join("dataset", "heart.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        f"Dataset not found at '{dataset_path}'.\n"
        "Please download heart.csv from Kaggle and place it in the /dataset folder."
    )

print("Loading dataset...")
df = pd.read_csv(dataset_path)
print(f"Dataset shape: {df.shape}")
print(df.head())

# ─── Basic Data Inspection ────────────────────────────────────────────────────
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nClass distribution (target):")
print(df["target"].value_counts())

# ─── Data Cleaning ────────────────────────────────────────────────────────────
# Fill any missing numeric values with column median (robust to outliers)
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col].fillna(df[col].median(), inplace=True)

# Ensure correct data types
df = df.astype({
    "age":      int,
    "sex":      int,
    "cp":       int,
    "trestbps": int,
    "chol":     int,
    "fbs":      int,
    "restecg":  int,
    "thalach":  int,
    "exang":    int,
    "oldpeak":  float,
    "slope":    int,
    "ca":       int,
    "thal":     int,
    "target":   int
})

# ─── Feature / Target Split ───────────────────────────────────────────────────
feature_cols = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

X = df[feature_cols]
y = 1 - df["target"]

# ─── Train / Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)} | Test samples: {len(X_test)}")

# ─── Define Candidate Classifiers ────────────────────────────────────────────
"""
We compare three classifiers:
1. Random Forest  — robust ensemble, handles non-linearity well
2. Gradient Boosting — strong for tabular data, sequential ensemble
3. Logistic Regression — interpretable linear baseline with scaling
"""
candidates = {
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            class_weight="balanced"
        ))
    ]),
    "Gradient Boosting": Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        ))
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
}

# ─── Evaluate All Candidates via Cross-Validation ────────────────────────────
print("\n--- Cross-Validation Results (5-fold) ---")
best_model_name = None
best_cv_score   = 0

for name, pipeline in candidates.items():
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
    mean_score = cv_scores.mean()
    print(f"{name}: AUC = {mean_score:.4f} ± {cv_scores.std():.4f}")

    if mean_score > best_cv_score:
        best_cv_score   = mean_score
        best_model_name = name

print(f"\nBest model selected: {best_model_name} (AUC={best_cv_score:.4f})")

# ─── Train Best Model on Full Training Set ───────────────────────────────────
best_pipeline = candidates[best_model_name]
best_pipeline.fit(X_train, y_train)

# ─── Evaluate on Test Set ─────────────────────────────────────────────────────
y_pred      = best_pipeline.predict(X_test)
y_pred_prob = best_pipeline.predict_proba(X_test)[:, 1]

print("\n--- Test Set Evaluation ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_pred_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─── Feature Importances (if available) ──────────────────────────────────────
if hasattr(best_pipeline.named_steps.get("clf", None), "feature_importances_"):
    importances = best_pipeline.named_steps["clf"].feature_importances_
    print("\n--- Feature Importances ---")
    for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        print(f"  {feat:10s}: {imp:.4f}")

# ─── Save Model ───────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "heart_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(best_pipeline, f)

print(f"\nModel saved to: {model_path}")
print("Training complete! You can now run app.py.")

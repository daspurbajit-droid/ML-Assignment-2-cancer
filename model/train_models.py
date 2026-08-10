"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset, evaluates each on a held-out test split, and saves:
  - trained model objects (model/*.pkl)
  - the fitted StandardScaler (model/scaler.pkl)
  - the held-out test split as CSV (test_data.csv) for the Streamlit app
  - a metrics comparison table (model/metrics.csv)

Dataset source: UCI Machine Learning Repository / scikit-learn built-in
                 (Wisconsin Diagnostic Breast Cancer data)
                 https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
"""

import pandas as pd
import numpy as np
import pickle
import json

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

# ---------------------------------------------------------------------
# 2. Train / test split (stratified, 80/20)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the test split (features + true label) — this is the CSV that
# gets committed to the repo and uploaded through the Streamlit app.
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("../test_data.csv", index=False)

# ---------------------------------------------------------------------
# 3. Scale features (fit on train only, then reuse for test / app)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_names.json", "w") as f:
    json.dump(list(X.columns), f)

# ---------------------------------------------------------------------
# 4. Define models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)

    # Save the fitted model
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
    with open(fname, "wb") as f:
        pickle.dump(model, f)

    print(name, metrics)

# ---------------------------------------------------------------------
# 5. Save comparison table
# ---------------------------------------------------------------------
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("metrics.csv", index=False)
print("\n=== Comparison Table ===")
print(metrics_df.to_string(index=False))

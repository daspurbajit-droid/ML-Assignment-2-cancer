"""
Streamlit app for Assignment 2 — Breast Cancer classification demo.

Features implemented:
  a. Dataset upload option (CSV of test data)
  b. Model selection dropdown
  c. Display of evaluation metrics
  d. Confusion matrix / classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer Classifier Demo", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl",
}


@st.cache_resource
def load_scaler():
    with open("model/scaler.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_feature_names():
    with open("model/feature_names.json", "r") as f:
        return json.load(f)


st.title("🔬 Breast Cancer Classification — Model Demo")
st.caption(
    "Dataset: Wisconsin Diagnostic Breast Cancer (UCI/sklearn) · "
    "30 features · binary classification (malignant / benign)"
)

# -----------------------------------------------------------------
# a. Dataset upload
# -----------------------------------------------------------------
st.header("1. Upload test data")
uploaded_file = st.file_uploader(
    "Upload a CSV file (must match the schema of test_data.csv in the repo, "
    "including the 'target' column)",
    type=["csv"],
)

feature_names = load_feature_names()

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded — using the bundled test_data.csv as a default.")
    test_df = pd.read_csv("test_data.csv")

st.write("Preview of test data:", test_df.head())

missing_cols = [c for c in feature_names if c not in test_df.columns]
if missing_cols:
    st.error(f"Uploaded file is missing required feature columns: {missing_cols}")
    st.stop()

has_labels = "target" in test_df.columns

X_test = test_df[feature_names]
scaler = load_scaler()
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------
# b. Model selection dropdown
# -----------------------------------------------------------------
st.header("2. Select a model")
model_choice = st.selectbox("Choose a classification model", list(MODEL_FILES.keys()))
model = load_model(MODEL_FILES[model_choice])

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# -----------------------------------------------------------------
# c. Evaluation metrics
# -----------------------------------------------------------------
st.header("3. Evaluation metrics")

if has_labels:
    y_true = test_df["target"]
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
    col2.metric("AUC", f"{roc_auc_score(y_true, y_proba):.4f}")
    col3.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
    col4.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
    col5.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
    col6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")
else:
    st.warning("Uploaded file has no 'target' column — showing predictions only, no metrics.")

# -----------------------------------------------------------------
# Comparison table across all models (precomputed, from model/metrics.csv)
# -----------------------------------------------------------------
st.subheader("Comparison across all models (on the original held-out test split)")
metrics_df = pd.read_csv("model/metrics.csv")
st.dataframe(metrics_df, use_container_width=True)

# -----------------------------------------------------------------
# d. Confusion matrix / classification report
# -----------------------------------------------------------------
st.header("4. Confusion matrix & classification report")

if has_labels:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.text("Classification report:")
    st.text(classification_report(y_true, y_pred, target_names=["Malignant", "Benign"]))
else:
    st.write("Predictions:", pd.Series(y_pred, name="predicted_target"))

st.divider()
st.caption("Assignment 2 — M.Tech (AIML/DSE), Machine Learning · BITS Pilani WILP")

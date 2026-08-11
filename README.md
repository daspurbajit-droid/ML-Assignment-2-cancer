# ML Assignment 2 — Breast Cancer Classification with Streamlit

## a. Problem Statement
Build, evaluate, and deploy multiple supervised classification models that
predict whether a breast tumor is **malignant** or **benign** based on
measurements taken from a digitized image of a fine needle aspirate (FNA) of
a breast mass. The goal is to compare several classical ML algorithms on the
same dataset and expose the trained models through an interactive Streamlit
web application.

## b. Dataset Description
- **Name:** Wisconsin Diagnostic Breast Cancer (WDBC) dataset
- **Source:** UCI Machine Learning Repository
  (https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic),
  also available via `sklearn.datasets.load_breast_cancer()`
- **Instances:** 569 (meets the ≥500 minimum)
- **Features:** 30 numeric features (meets the ≥12 minimum) — mean, standard
  error, and "worst" values of 10 real-valued measurements computed from
  each cell nucleus (radius, texture, perimeter, area, smoothness,
  compactness, concavity, concave points, symmetry, fractal dimension)
- **Target:** Binary — `0 = malignant` (212 cases), `1 = benign` (357 cases)
- **Split used:** 80% train / 20% test, stratified by class, `random_state=42`

## c. GitHub Repository Link
`<https://github.com/daspurbajit-droid/ML-Assignment-2-cancer>`

## d. Models Used

All 5 models were trained on the same 80/20 stratified split of the same
standardized feature set (`StandardScaler` fit on the training set only).

### Comparison Table (on the held-out test set, n=114)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9932 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

*(Exact numbers reproducible by running `model/train_models.py` — `random_state=42` throughout.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer here — the classes are close to linearly separable in this feature space once standardized, so a simple linear decision boundary generalizes very well and gives the highest accuracy, F1, and MCC. |
| Decision Tree | Weakest of the five. A single unpruned tree overfits the training data's specific splits, which hurts generalization to the test set — visible in the lower AUC and MCC compared to the ensemble version of the same idea (Random Forest). |
| kNN | Solid performance; with standardized features, distance-based neighbors work well because no single feature dominates the distance metric. Slightly behind Logistic Regression, likely because it's more sensitive to local noise near the decision boundary. |
| Naive Bayes | Decent accuracy but the independence assumption between features is clearly violated here (many of the 30 features are highly correlated, e.g., radius/perimeter/area), which caps its precision/recall relative to models that can model feature interactions. |
| Random Forest (Ensemble) | Averaging many trees fixes the Decision Tree's overfitting problem — matches kNN on most metrics and posts the second-highest AUC, confirming the ensemble is more robust than its single-tree base learner. |
| **Overall Winner for your dataset?** | **Logistic Regression** — highest Accuracy, AUC, Precision, Recall, F1, and MCC among all five models on this dataset. |

> Replace the wording above with your own phrasing once you've run the
> notebook/script yourself and inspected the actual confusion matrices —
> these are meant as a starting analysis, not a script to copy verbatim.

## Project Structure
```
project-folder/
│-- app.py                  # Streamlit application
│-- requirements.txt
│-- README.md
│-- test_data.csv           # held-out test split (features + target)
│-- model/
│   │-- train_models.py     # trains all 5 models, saves .pkl + metrics.csv
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- scaler.pkl
│   │-- feature_names.json
│   │-- metrics.csv
```

## How to Run Locally
```bash
pip install -r requirements.txt
python model/train_models.py   # regenerates models + test_data.csv + metrics.csv
streamlit run app.py
```

## Deployment
Deployed on Streamlit Community Cloud: `<(https://ml-assignment-2-cancer-k2vyjzo2bexrabznovzpem.streamlit.app/)>`

# 🏭 Semiconductor Wafer Fault Detection & Sensor Attribution (SECOM)

An end-to-end industrial Machine Learning pipeline for diagnosing semiconductor manufacturing defects across 590 continuous sensor telemetry channels.

---

## 📌 Project Overview
- **Dataset**: UCI SECOM Dataset (1,567 wafers, 591 numerical continuous signals)
- **Problem**: Severe Class Imbalance (~14:1 Pass/Fail ratio), High Missingness (>50% missing in several sensors), Collinear Sensor Telemetry.
- **Objective**: Early detection of wafer defects and root-cause sensor attribution for yield optimization.

---

## 🛠️ Architecture & Pipeline Stages

1. **Preprocessing (`src/preprocessing.py`)**:
   - Filter high-null features (>50% missing) & zero-variance (constant) channels.
   - Stratified Train-Test split (80/20) to maintain failure prevalence.
   - Leakage-free Median Imputation and `RobustScaler` transformation.
2. **Feature Selection (`src/feature_selection.py`)**:
   - Collinearity reduction ($|r| > 0.85$).
   - Mutual Information (`SelectKBest`) and Tree-based importance ranking.
3. **Model Benchmarking (`src/models.py`)**:
   - Stratified 5-Fold Cross-Validation on balanced models:
     - Logistic Regression (Balanced)
     - Support Vector Machine (RBF Kernel)
     - Random Forest Classifier (Balanced)
     - Extra Trees Classifier (Balanced)
     - HistGradientBoosting Classifier (Balanced)
     - K-Nearest Neighbors (Distance-Weighted)
   - Evaluated on **PR-AUC (Average Precision)**, **ROC-AUC**, and **Minority F1-Score**.
4. **Hyperparameter Optimization (`src/models.py`)**:
   - GridSearchCV for tree depth, estimator counts, and splitting criteria.
5. **Evaluation & Explainability (`src/evaluate.py`)**:
   - Confusion Matrix heatmap, ROC & Precision-Recall curves.
   - Root-cause sensor attribution bar chart (Top 20 critical failure sensors).

---

## 🚀 Quickstart

Run the complete pipeline using the local Python environment:

```bash
python main.py
```

All figures and evaluation metrics will be generated and saved to `plots/`:
- `plots/model_benchmark.png`
- `plots/confusion_matrix.png`
- `plots/roc_pr_curves.png`
- `plots/top_sensors.png`

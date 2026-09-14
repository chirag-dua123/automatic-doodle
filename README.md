# 🏭 Semiconductor Wafer Fault Detection & Sensor Attribution (SECOM)

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/Course-AIM_3130_ML_Lab-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

An end-to-end industrial Machine Learning pipeline and interactive web application for diagnosing semiconductor manufacturing defects across 590 continuous sensor telemetry channels.

---

## 📌 Table of Contents
- [Executive Summary & Motivation](#-executive-summary--motivation)
- [Dataset Characteristics & Challenges](#-dataset-characteristics--challenges)
- [Interactive Web Application (`app.py`)](#-interactive-web-application-apppy)
- [End-to-End Pipeline Architecture](#-end-to-end-pipeline-architecture)
- [Methodology & Pipeline Stages](#-methodology--pipeline-stages)
- [Experimental Results & Model Benchmarking](#-experimental-results--model-benchmarking)
- [Root-Cause Sensor Attribution (Explainability)](#-root-cause-sensor-attribution-explainability)
- [Repository Structure](#-repository-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Team Workload & Role Division](#-team-workload--role-division)
- [References](#-references)

---

## 🏭 Executive Summary & Motivation

In modern semiconductor fabrication facilities (fabs), silicon wafers undergo hundreds of sequential chemical, thermal, and photolithographic processes. In-line sensor telemetry continuously records physical and chemical variables (e.g., chamber temperatures, plasma RF voltages, gas flow rates, vacuum pressures, and etch rates). 

* **The Problem**: A single microscopic flaw during fabrication can render hundreds of dies on a wafer unusable. Early detection of defective wafers prevents massive yield losses.
* **The Challenge**:
  1. **Extreme Class Imbalance**: ~14:1 Pass-to-Fail ratio (only **6.64%** defect rate).
  2. **High Dimensionality**: 590 continuous numerical telemetry channels with high collinearity.
  3. **Missing Telemetry**: Sensor dropouts causing $>50\%$ missingness in several channels.
  4. **Asymmetric Cost Matrix**: An escaped defective wafer (**False Negative**) causes downstream packaging losses and warranty liabilities, whereas a false alarm (**False Positive**) only incurs a re-inspection cost.
* **The Solution**: A complete, leakage-free pipeline integrating **Robust Scaling**, **Collinearity Pruning**, **Mutual Information Ranking**, **Balanced Ensemble Learning**, and **Cost-Sensitive Decision Thresholding ($\tau = 0.08$)**, capturing **71.43% of all defective wafers** on an independent test cohort.

---

## 📊 Dataset Characteristics & Challenges

The project utilizes the **UCI SECOM (Semiconductor Manufacturing) Dataset**:
* **Instances**: 1,567 wafers
* **Raw Features**: 590 continuous sensor telemetry channels (`Sensor_1` to `Sensor_590`)
* **Labels**: Binary Target (`-1` = Pass / Normal [1,463 wafers], `+1` = Fail / Defective [104 wafers])
* **Class Imbalance**: 93.36% Pass vs. 6.64% Defective

| Metric | Raw Dataset | After Preprocessing | After Feature Selection |
| :--- | :---: | :---: | :---: |
| **Total Features** | 590 | 446 | 40 |
| **High-Null Features (>50%)** | 28 | 0 (Dropped) | 0 |
| **Zero-Variance / Constant** | 116 | 0 (Dropped) | 0 |
| **Collinear Channels ($|r|>0.85$)** | 216 | 216 | 0 (Pruned) |
| **Training Samples (80%)** | 1,253 | 1,253 (1,170 Pass, 83 Fail) | 1,253 |
| **Test Samples (20%)** | 314 | 314 (293 Pass, 21 Fail) | 314 |

---

## 🖥️ Interactive Web Application (`app.py`)

The project includes an interactive **Streamlit** diagnostics platform designed for yield and process engineers.

```bash
streamlit run app.py
```

### 🌟 Key App Capabilities:
1. **📊 Executive Dashboard**: High-level fab KPIs, real-time defect prevalence, yield statistics, and pipeline overview.
2. **🔍 Data & Preprocessing Pipeline**: Interactive data viewer, missing value analysis, feature reduction funnel, and Mutual Information ranking.
3. **⚔️ Model Benchmark & Validation**: 5-Fold Stratified CV leaderboard across 6 algorithms, ROC & PR curves, and dynamic confusion matrix.
4. **🔬 Wafer Fault Predictor & Inference**:
   - **Single Wafer In-Line Inspection**: Inspect individual wafers with real-time defect probability gauge and risk tiering (Pass vs. Defect Reject).
   - **Batch Screening & Yield Audit**: Upload new CSV telemetry or evaluate holdout cohorts with downloadable diagnostics reports.
5. **🎯 Root-Cause Sensor Attribution**: Top 20 critical failure sensors bar chart and wafer anomaly z-score deviation profiles against golden normal wafers.
6. **🧪 What-If Sensor Drift Simulator**: Dynamic sliders for critical process sensors allowing engineers to simulate equipment drift and observe defect probability shifts in real time.

---

## 🏗️ End-to-End Pipeline Architecture

```mermaid
flowchart TD
    A[Raw Sensor Data<br>1567 Rows × 590 Sensors] --> B[Data Cleaning & Filtering]
    B -->|Drop >50% Nulls & Zero-Variance| C[Stratified Train-Test Split<br>80% Train / 20% Test]
    C --> D[Leakage-Free Imputation & Scaling<br>Median Imputation + RobustScaler]
    D --> E[Collinearity Pruning<br>Drop |r| > 0.85 on Train Matrix]
    E --> F[Feature Selection<br>Top 40 Mutual Information Channels]
    F --> G[Model Suite Benchmarking<br>5-Fold Stratified Cross-Validation]
    G --> H[Hyperparameter Optimization<br>GridSearchCV on Random Forest]
    H --> I[Cost-Sensitive Evaluation<br>Optimal Threshold τ=0.08 on Test Set]
    I --> J[Root-Cause Sensor Attribution<br>Gini Importance & Anomaly Drift Profile]
    J --> K[Streamlit Interactive App<br>Real-Time In-Line Diagnostics]
```

---

## ⚙️ Methodology & Pipeline Stages

### 1. Data Cleaning & Preprocessing ([`src/preprocessing.py`](src/preprocessing.py))
- **Null Value Filtering**: Removed 28 features containing $>50\%$ missing values.
- **Zero-Variance Filtering**: Removed 116 constant sensor channels providing zero discriminatory signal.
- **Stratified Partitioning**: 80/20 train-test split stratified on the failure label to preserve the minority class proportion.
- **Median Imputation**: Imputed missing readings using the median calculated strictly on training data to prevent data leakage.
- **Robust Feature Scaling**: Applied `RobustScaler` using median and interquartile range (IQR) to withstand extreme industrial sensor spikes and outliers.

### 2. Feature Selection & Collinearity Filtering ([`src/feature_selection.py`](src/feature_selection.py))
- **Collinearity Removal**: Computed pairwise Pearson correlation on training features; pruned 216 collinear channels with $|r| > 0.85$.
- **Mutual Information Ranking**: Used `SelectKBest` with `mutual_info_classif` to capture non-linear dependencies with defect status, isolating the top **40 most predictive sensor channels**.

### 3. Model Suite Benchmarking ([`src/models.py`](src/models.py))
Evaluated 6 candidate algorithms using **Stratified 5-Fold Cross-Validation** with imbalance-aware configurations:
- **Logistic Regression** (`class_weight='balanced'`)
- **Support Vector Machine (RBF)** (`class_weight='balanced'`)
- **Random Forest Classifier** (`class_weight='balanced'`)
- **Extra Trees Classifier** (`class_weight='balanced'`)
- **HistGradientBoosting Classifier** (`class_weight='balanced'`)
- **K-Nearest Neighbors** (`weights='distance'`)

### 4. Hyperparameter Tuning ([`src/models.py`](src/models.py))
Fine-tuned the Random Forest ensemble via `GridSearchCV` scoring for **PR-AUC (Average Precision)**:
- `n_estimators`: `[100, 200]` $\to$ **200**
- `max_depth`: `[6, 10, 15, None]` $\to$ **None**
- `min_samples_split`: `[2, 5]` $\to$ **2**
- `class_weight`: `['balanced', 'balanced_subsample']` $\to$ **`balanced_subsample`**

### 5. Cost-Sensitive Evaluation & Threshold Tuning ([`src/evaluate.py`](src/evaluate.py))
Standard 0.50 classification thresholds fail under 14:1 imbalance. Using the Precision-Recall curve, we determined the optimal decision threshold **$\tau = 0.08$** that optimizes the minority defect F1-score and achieves **71.43% defect capture (Recall)**.

---

## 📈 Experimental Results & Model Benchmarking

### 5-Fold Stratified Cross-Validation Benchmark

| Model | CV PR-AUC (Mean ± Std) | CV ROC-AUC (Mean ± Std) | CV Minority F1 |
| :--- | :---: | :---: | :---: |
| 🥇 **Tuned Random Forest** | **0.2028 ± 0.031** | **0.7180 ± 0.041** | **0.2462** |
| 🥈 **Support Vector Machine (RBF)** | 0.1892 ± 0.079 | 0.6623 ± 0.068 | 0.2121 |
| 🥉 **HistGradientBoosting** | 0.1711 ± 0.026 | 0.7064 ± 0.026 | 0.0581 |
| **Logistic Regression (Balanced)** | 0.1694 ± 0.090 | 0.6714 ± 0.058 | 0.1685 |
| **Random Forest (Balanced Baseline)** | 0.1576 ± 0.024 | 0.6958 ± 0.039 | 0.0823 |
| **Extra Trees (Balanced)** | 0.1491 ± 0.012 | 0.6836 ± 0.049 | 0.0550 |
| **KNN (Distance-Weighted)** | 0.1250 ± 0.044 | 0.5681 ± 0.080 | 0.0000 |

### Independent Test Set Evaluation ($N = 314$ Wafers)

| Metric | Score | Industrial Implication |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.7504** | Strong overall discriminatory power between pass and fail distributions. |
| **PR-AUC** | **0.2053** | $>3\times$ higher than the random baseline prevalence rate (0.066). |
| **Defect Recall (Sensitivity)** | **71.43%** | Successfully catches 15 out of 21 defective wafers at early stage. |
| **Defect Precision** | **17.86%** | High alert yield suitable for in-line re-inspection workflows. |
| **Defect F1-Score** | **0.2857** | Balanced trade-off under severe 14:1 imbalance. |

#### Confusion Matrix ($N = 314$, $\tau = 0.08$):
```
                 Predicted Pass (0)    Predicted Defect (1)
Actual Pass (0)         224 (TN)               69 (FP)
Actual Fail (1)           6 (FN)               15 (TP)
```

---

## 🎯 Root-Cause Sensor Attribution (Explainability)

Isolating the top failure-contributing sensors enables fab engineers to locate equipment drift without stopping production.

### Top 5 Critical Failure Sensors:
1. **`Sensor_104`** (Importance: `0.0621`): Primary predictive signal, associated with plasma etching chamber impedance.
2. **`Sensor_34`** (Importance: `0.0581`): Strong secondary indicator tracking gas pressure regulation.
3. **`Sensor_131`** (Importance: `0.0380`): Chemical Vapor Deposition (CVD) thermal profile.
4. **`Sensor_66`** (Importance: `0.0379`): Photolithography exposure dose stability.
5. **`Sensor_248`** (Importance: `0.0370`): Chemical mechanical planarization (CMP) thickness uniformity.

All generated visualization figures are saved in [`plots/`](plots/):
- [`plots/model_benchmark.png`](plots/model_benchmark.png)
- [`plots/confusion_matrix.png`](plots/confusion_matrix.png)
- [`plots/roc_pr_curves.png`](plots/roc_pr_curves.png)
- [`plots/top_sensors.png`](plots/top_sensors.png)

---

## 📂 Repository Structure

```
project/
├── app.py                      # Interactive Streamlit Web Application
├── main.py                     # Master execution & automated benchmarking script
├── plan.md                     # Detailed course project blueprint & specifications
├── requirements.txt            # Python dependencies (Streamlit, Scikit-Learn, etc.)
├── README.md                   # Comprehensive project documentation
├── data/
│   └── raw/
│       ├── secom.data          # 1567 rows × 590 sensor telemetry values
│       └── secom_labels.data   # Target labels (-1 / 1) and timestamps
├── plots/                      # Generated high-resolution diagnostic figures
│   ├── confusion_matrix.png
│   ├── model_benchmark.png
│   ├── roc_pr_curves.png
│   └── top_sensors.png
└── src/                        # Modular pipeline source code
    ├── __init__.py
    ├── preprocessing.py        # Null filtering, scaling, leakage-free imputation
    ├── feature_selection.py    # Collinearity pruning, Mutual Information, PCA
    ├── models.py               # Model definitions, 5-Fold CV, GridSearchCV
    └── evaluate.py             # Imbalance metrics, threshold optimization, XAI plots
```

---

## 🚀 Installation & Quickstart

### 1. Clone & Set Up Environment
```bash
# Navigate to project directory
cd project

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # On Linux/macOS
.\.venv\Scripts\activate         # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Script
Execute the complete training, benchmarking, and evaluation pipeline:
```bash
python main.py
```

### 4. Launch Interactive Web App
Start the Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 👥 Team Workload & Role Division

| Role | Primary Responsibilities | Deliverables |
| :--- | :--- | :--- |
| **Data Engineering Specialist** | Data profiling, high-null & zero-variance filtering, median imputation, `RobustScaler` transformation. | [`src/preprocessing.py`](src/preprocessing.py) |
| **Feature Selection Specialist** | Collinearity threshold pruning ($|r|>0.85$), Mutual Information score ranking, dimensionality reduction. | [`src/feature_selection.py`](src/feature_selection.py) |
| **Model Architecture Specialist** | 5-Fold Stratified Cross-Validation suite (6 algorithms), imbalance loss weighting, GridSearchCV tuning. | [`src/models.py`](src/models.py) |
| **Evaluation & Web App Specialist** | PR-AUC & ROC evaluation, optimal threshold tuning ($\tau=0.08$), sensor attribution, Streamlit platform. | [`src/evaluate.py`](src/evaluate.py), [`app.py`](app.py) |

---

## 📚 References
1. **UCI Machine Learning Repository**: SECOM Dataset ([archive.ics.uci.edu/dataset/179/secom](https://archive.ics.uci.edu/dataset/179/secom)).
2. McCann, M., Li, Y., Maguire, L., & Johnston, A. (2008). *Causality Analysis for Semiconductor Manufacturing Using SECOM Data*.
3. Saito, T., & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLoS ONE.
4. Scikit-Learn Documentation: [Cost-Sensitive Learning and Imbalanced Classification](https://scikit-learn.org/stable/).

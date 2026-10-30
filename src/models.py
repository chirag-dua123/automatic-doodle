"""
Machine Learning Models & Cross-Validation Benchmarking Module
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score, f1_score

def get_model_suite(random_state=42):
    """
    Returns a dictionary of candidate classification models configured
    to address class imbalance.
    """
    models = {
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=random_state
        ),
        "Support Vector Machine (RBF)": CalibratedClassifierCV(
            estimator=SVC(
                kernel='rbf', class_weight='balanced', random_state=random_state
            ),
            ensemble=False
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=150, class_weight='balanced', max_depth=10, random_state=random_state
        ),
        "Extra Trees (Balanced)": ExtraTreesClassifier(
            n_estimators=150, class_weight='balanced', max_depth=10, random_state=random_state
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight='balanced', max_iter=150, random_state=random_state
        ),
        "KNN (Distance-Weighted)": KNeighborsClassifier(
            n_neighbors=7, weights='distance'
        )
    }
    return models

def benchmark_models_cv(models, X_train, y_train, cv_splits=5, random_state=42):
    """
    Evaluates candidate models using Stratified K-Fold Cross-Validation across multiple
    imbalance-aware metrics: PR-AUC (average precision), ROC-AUC, and F1 (minority).
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    scoring = {
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision',
        'f1_minority': make_scorer(f1_score, pos_label=1, zero_division=0)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        cv_res = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
        
        # Fit on full training set
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        results.append({
            "Model": name,
            "CV ROC-AUC (Mean)": cv_res['test_roc_auc'].mean(),
            "CV ROC-AUC (Std)": cv_res['test_roc_auc'].std(),
            "CV PR-AUC (Mean)": cv_res['test_pr_auc'].mean(),
            "CV PR-AUC (Std)": cv_res['test_pr_auc'].std(),
            "CV F1 (Minority)": cv_res['test_f1_minority'].mean(),
        })
        
    df_results = pd.DataFrame(results).sort_values(by="CV PR-AUC (Mean)", ascending=False).reset_index(drop=True)
    return df_results, trained_models

def tune_random_forest(X_train, y_train, random_state=42):
    """
    Performs hyperparameter grid search for Random Forest Classifier.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10, 15, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf = RandomForestClassifier(random_state=random_state)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    grid = GridSearchCV(
        rf, param_grid, cv=skf, scoring='average_precision', n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    
    return grid.best_estimator_, grid.best_params_, grid.best_score_

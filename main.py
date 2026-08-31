"""
Main Execution Script for Semiconductor Wafer Fault Detection (SECOM Project)
"""
import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from src.preprocessing import preprocess_data
from src.feature_selection import (
    remove_collinear_features, select_top_k_mutual_info,
    select_top_k_trees, apply_pca_reduction
)
from src.models import get_model_suite, benchmark_models_cv, tune_random_forest
from src.evaluate import (
    evaluate_model_performance, plot_confusion_matrix,
    plot_roc_and_pr_curves, plot_top_sensors, plot_benchmark_comparison
)

def run_pipeline():
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    data_path = os.path.join(script_dir, "data", "raw", "secom.data")
    labels_path = os.path.join(script_dir, "data", "raw", "secom_labels.data")
    
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        print("Error: Dataset files not found. Please ensure secom.data and secom_labels.data exist in data/raw/")
        return
        
    print("=" * 70)
    print("SECOM SEMICONDUCTOR WAFER FAULT DETECTION PIPELINE")
    print("=" * 70)
    
    # -------------------------------------------------------------
    # 1. PREPROCESSING & CLEANING
    # -------------------------------------------------------------
    print("\n[STAGE 1/6] Running Data Cleaning & Preprocessing...")
    X_train, X_test, y_train, y_test, retained_features, prep_summary = preprocess_data(
        data_path, labels_path, test_size=0.2, random_state=42, scaler_type='robust'
    )
    
    print(f"  * Initial Features: {prep_summary['initial_features']}")
    print(f"  * Dropped High-Null Features (>50%): {prep_summary['high_null_dropped']}")
    print(f"  * Dropped Zero-Variance Features: {prep_summary['zero_var_dropped']}")
    print(f"  * Cleaned Features Retained: {prep_summary['retained_features']}")
    print(f"  * Training Samples: {prep_summary['train_samples']} (Pass: {prep_summary['train_neg_class']}, Fail: {prep_summary['train_pos_class']})")
    print(f"  * Test Samples: {prep_summary['test_samples']} (Pass: {prep_summary['test_neg_class']}, Fail: {prep_summary['test_pos_class']})")
    
    # -------------------------------------------------------------
    # 2. FEATURE SELECTION & REDUCTION
    # -------------------------------------------------------------
    print("\n[STAGE 2/6] Feature Selection & Collinearity Filtering...")
    X_train_nc, X_test_nc, dropped_collinear = remove_collinear_features(X_train, X_test, threshold=0.85)
    print(f"  * Collinear Features Dropped (|r| > 0.85): {len(dropped_collinear)}")
    print(f"  * Remaining Features: {X_train_nc.shape[1]}")
    
    print("  * Selecting Top 40 Features via Mutual Information...")
    X_train_mi, X_test_mi, mi_scores = select_top_k_mutual_info(X_train_nc, y_train, X_test_nc, k=40)
    
    # -------------------------------------------------------------
    # 3. MODEL BENCHMARKING (5-FOLD STRATIFIED CV)
    # -------------------------------------------------------------
    print("\n[STAGE 3/6] Benchmarking Candidate Models via 5-Fold Stratified CV...")
    models = get_model_suite(random_state=42)
    df_cv_results, trained_models = benchmark_models_cv(models, X_train_mi, y_train, cv_splits=5, random_state=42)
    
    print("\n--- Model Benchmark Results (Sorted by PR-AUC) ---")
    print(df_cv_results.to_string(index=False))
    
    benchmark_plot_path = os.path.join(plots_dir, "model_benchmark.png")
    plot_benchmark_comparison(df_cv_results, benchmark_plot_path)
    print(f"  * Benchmark comparison plot saved: {benchmark_plot_path}")
    
    # -------------------------------------------------------------
    # 4. HYPERPARAMETER TUNING ON BEST ENSEMBLE
    # -------------------------------------------------------------
    print("\n[STAGE 4/6] Fine-Tuning Random Forest Classifier (GridSearchCV)...")
    best_rf, best_params, best_score = tune_random_forest(X_train_mi, y_train, random_state=42)
    print(f"  * Optimal Hyperparameters: {best_params}")
    print(f"  * Tuned CV PR-AUC Score: {best_score:.4f}")
    
    # -------------------------------------------------------------
    # 5. INDEPENDENT TEST SET EVALUATION
    # -------------------------------------------------------------
    print("\n[STAGE 5/6] Evaluating Best Model on Independent Test Set (N=314)...")
    test_metrics = evaluate_model_performance(best_rf, X_test_mi, y_test)
    
    print(f"  * Optimized Decision Threshold: {test_metrics['Optimal Threshold']:.4f}")
    print(f"  * Confusion Matrix (Optimized Threshold):\n{test_metrics['Confusion Matrix']}")
    print(f"  * Test ROC-AUC: {test_metrics['ROC AUC']:.4f}")
    print(f"  * Test PR-AUC:  {test_metrics['PR AUC']:.4f}")
    print(f"  * Defect Precision: {test_metrics['Precision (Defect)']:.4f}")
    print(f"  * Defect Recall:    {test_metrics['Recall (Defect)']:.4f}")
    print(f"  * Defect F1-Score:  {test_metrics['F1 (Defect)']:.4f}")
    
    # Save Confusion Matrix & Curves
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plot_confusion_matrix(test_metrics['Confusion Matrix'], cm_path, threshold=test_metrics['Optimal Threshold'])
    
    baseline_prev = y_test.mean()
    curves_path = os.path.join(plots_dir, "roc_pr_curves.png")
    plot_roc_and_pr_curves(
        test_metrics['fpr'], test_metrics['tpr'], test_metrics['ROC AUC'],
        test_metrics['precision_curve'], test_metrics['recall_curve'], test_metrics['PR AUC'],
        baseline_prev, curves_path
    )
    print(f"  * Evaluation plots saved to {plots_dir}")
    
    # -------------------------------------------------------------
    # 6. SENSOR ATTRIBUTION & EXPLAINABILITY (ROOT CAUSE)
    # -------------------------------------------------------------
    print("\n[STAGE 6/6] Root-Cause Sensor Attribution (Top 20 Critical Sensors)...")
    importances = best_rf.feature_importances_
    sensors_plot_path = os.path.join(plots_dir, "top_sensors.png")
    plot_top_sensors(X_train_mi.columns, importances, top_n=20, output_path=sensors_plot_path)
    
    top_5 = pd.Series(importances, index=X_train_mi.columns).sort_values(ascending=False).head(5)
    print("  * Top 5 Critical Failure Sensors:")
    for sensor, score in top_5.items():
        print(f"     - {sensor}: Importance = {score:.4f}")
        
    print("\n" + "=" * 70)
    print("[SUCCESS] PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Artifacts & Visualizations saved to: {plots_dir}")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()

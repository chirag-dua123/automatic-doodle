'''metrics_summary.py
Generate a concise textual summary of key evaluation metrics for the SECOM project.
'''\

import os
from pathlib import Path
import pandas as pd

# Import project modules (assumes this script is in project root)
from src.preprocessing import preprocess_data
from src.feature_selection import remove_collinear_features, select_top_k_mutual_info
from src.models import get_model_suite, benchmark_models_cv
from src.evaluate import evaluate_model_performance
from sklearn.ensemble import RandomForestClassifier

def main():
    script_dir = Path(__file__).parent
    data_path = script_dir / "data" / "raw" / "secom.data"
    labels_path = script_dir / "data" / "raw" / "secom_labels.data"

    if not data_path.is_file() or not labels_path.is_file():
        print("Data files not found.")
        return

    # Preprocess
    X_train, X_test, y_train, y_test, retained_features, prep_summary = preprocess_data(
        str(data_path), str(labels_path), test_size=0.2, random_state=42, scaler_type='robust'
    )

    # Feature selection
    X_train_nc, X_test_nc, _ = remove_collinear_features(X_train, X_test, threshold=0.85)
    X_train_mi, X_test_mi, _ = select_top_k_mutual_info(X_train_nc, y_train, X_test_nc, k=40, random_state=42)

    # Benchmark models (just to get CV results)
    models = get_model_suite(random_state=42)
    cv_results, _ = benchmark_models_cv(models, X_train_mi, y_train, cv_splits=5, random_state=42)

    # Train best Random Forest (same as app)
    best_rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        class_weight='balanced_subsample', random_state=42
    )
    best_rf.fit(X_train_mi, y_train)

    # Evaluate on test set with the threshold used in the app (0.08)
    test_metrics = evaluate_model_performance(best_rf, X_test_mi, y_test, threshold=0.08)

    # Print concise summary
    print("--- SECOM Project Evaluation Summary ---")
    print(f"Optimal decision threshold (tau): {test_metrics['Optimal Threshold']:.3f}")
    print(f"ROC AUC: {test_metrics['ROC AUC']:.3f}")
    print(f"PR AUC: {test_metrics['PR AUC']:.3f}")
    print(f"F1 (Defect class): {test_metrics['F1 (Defect)']:.3f}")
    print(f"Precision (Defect): {test_metrics['Precision (Defect)']:.3f}")
    print(f"Recall (Defect): {test_metrics['Recall (Defect)']:.3f}")
    print(f"Confusion Matrix: \n{test_metrics['Confusion Matrix']}")
    print("\nCross‑validation benchmark (mean scores):")
    print(cv_results[['Model','CV ROC-AUC (Mean)','CV PR-AUC (Mean)','CV F1 (Minority)']].to_string(index=False))

if __name__ == "__main__":
    main()

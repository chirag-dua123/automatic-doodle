import os
import pandas as pd
import streamlit as st

from src.preprocessing import preprocess_data
from src.feature_selection import remove_collinear_features, select_top_k_mutual_info
from src.models import get_model_suite, benchmark_models_cv, tune_random_forest
from src.evaluate import (
    evaluate_model_performance,
    plot_confusion_matrix,
    plot_roc_and_pr_curves,
    plot_top_sensors,
    plot_benchmark_comparison,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "secom.data")
LABELS_PATH = os.path.join(BASE_DIR, "data", "raw", "secom_labels.data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


@st.cache_data(show_spinner=False)
def run_pipeline(k_features: int, collinear_threshold: float, random_state: int):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, _, prep_summary = preprocess_data(
        DATA_PATH,
        LABELS_PATH,
        test_size=0.2,
        random_state=random_state,
        scaler_type="robust",
    )

    X_train_nc, X_test_nc, dropped_collinear = remove_collinear_features(
        X_train,
        X_test,
        threshold=collinear_threshold,
    )
    X_train_mi, X_test_mi, _ = select_top_k_mutual_info(
        X_train_nc,
        y_train,
        X_test_nc,
        k=k_features,
        random_state=random_state,
    )

    models = get_model_suite(random_state=random_state)
    df_cv_results, _ = benchmark_models_cv(
        models,
        X_train_mi,
        y_train,
        cv_splits=5,
        random_state=random_state,
    )

    best_rf, best_params, best_score = tune_random_forest(
        X_train_mi,
        y_train,
        random_state=random_state,
    )

    test_metrics = evaluate_model_performance(best_rf, X_test_mi, y_test)

    benchmark_plot_path = os.path.join(PLOTS_DIR, "model_benchmark.png")
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    curves_path = os.path.join(PLOTS_DIR, "roc_pr_curves.png")
    sensors_plot_path = os.path.join(PLOTS_DIR, "top_sensors.png")

    plot_benchmark_comparison(df_cv_results, benchmark_plot_path)
    plot_confusion_matrix(
        test_metrics["Confusion Matrix"],
        cm_path,
        threshold=test_metrics["Optimal Threshold"],
    )
    plot_roc_and_pr_curves(
        test_metrics["fpr"],
        test_metrics["tpr"],
        test_metrics["ROC AUC"],
        test_metrics["precision_curve"],
        test_metrics["recall_curve"],
        test_metrics["PR AUC"],
        y_test.mean(),
        curves_path,
    )

    importances = best_rf.feature_importances_
    plot_top_sensors(X_train_mi.columns, importances, top_n=20, output_path=sensors_plot_path)

    top_sensors = (
        pd.Series(importances, index=X_train_mi.columns)
        .sort_values(ascending=False)
        .head(10)
        .rename("Importance")
        .reset_index()
        .rename(columns={"index": "Sensor"})
    )

    return {
        "prep_summary": prep_summary,
        "dropped_collinear": len(dropped_collinear),
        "remaining_features": X_train_nc.shape[1],
        "selected_features": X_train_mi.shape[1],
        "cv_results": df_cv_results,
        "best_params": best_params,
        "best_cv_pr_auc": best_score,
        "test_metrics": test_metrics,
        "top_sensors": top_sensors,
        "plots": {
            "benchmark": benchmark_plot_path,
            "confusion": cm_path,
            "curves": curves_path,
            "sensors": sensors_plot_path,
        },
    }


def main():
    st.set_page_config(page_title="SECOM Fault Detection", layout="wide")
    st.title("🏭 SECOM Wafer Fault Detection")
    st.write(
        "Train and evaluate the wafer fault detection pipeline, then inspect model metrics and sensor attribution."
    )

    if not os.path.exists(DATA_PATH) or not os.path.exists(LABELS_PATH):
        st.error("Dataset files were not found in data/raw/. Add secom.data and secom_labels.data.")
        return

    with st.sidebar:
        st.header("Run Settings")
        k_features = st.slider("Top features (Mutual Information)", min_value=10, max_value=120, value=40, step=5)
        collinear_threshold = st.slider("Collinearity threshold", min_value=0.70, max_value=0.99, value=0.85, step=0.01)
        random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
        run_clicked = st.button("Run pipeline", type="primary")

    if not run_clicked:
        st.info("Configure settings in the sidebar and click **Run pipeline**.")
        return

    with st.spinner("Running full training and evaluation pipeline..."):
        results = run_pipeline(k_features, collinear_threshold, random_state)

    prep = results["prep_summary"]
    metrics = results["test_metrics"]

    st.subheader("Preprocessing Summary")
    cols = st.columns(4)
    cols[0].metric("Initial features", prep["initial_features"])
    cols[1].metric("Retained features", prep["retained_features"])
    cols[2].metric("High-null dropped", prep["high_null_dropped"])
    cols[3].metric("Zero-variance dropped", prep["zero_var_dropped"])

    cols = st.columns(4)
    cols[0].metric("Train samples", prep["train_samples"])
    cols[1].metric("Test samples", prep["test_samples"])
    cols[2].metric("Collinear dropped", results["dropped_collinear"])
    cols[3].metric("Final selected features", results["selected_features"])

    st.subheader("Cross-validation Benchmark")
    st.dataframe(results["cv_results"], use_container_width=True)

    st.subheader("Best Random Forest Tuning")
    st.write(f"**Best CV PR-AUC:** {results['best_cv_pr_auc']:.4f}")
    st.json(results["best_params"])

    st.subheader("Test Set Metrics")
    cols = st.columns(4)
    cols[0].metric("ROC-AUC", f"{metrics['ROC AUC']:.4f}")
    cols[1].metric("PR-AUC", f"{metrics['PR AUC']:.4f}")
    cols[2].metric("Precision (Defect)", f"{metrics['Precision (Defect)']:.4f}")
    cols[3].metric("Recall (Defect)", f"{metrics['Recall (Defect)']:.4f}")

    cols = st.columns(2)
    with cols[0]:
        st.image(results["plots"]["confusion"], caption="Confusion Matrix", use_container_width=True)
        st.image(results["plots"]["benchmark"], caption="Model Benchmark", use_container_width=True)
    with cols[1]:
        st.image(results["plots"]["curves"], caption="ROC & PR Curves", use_container_width=True)
        st.image(results["plots"]["sensors"], caption="Top Sensor Attribution", use_container_width=True)

    st.subheader("Top 10 Critical Sensors")
    st.dataframe(results["top_sensors"], use_container_width=True)


if __name__ == "__main__":
    main()

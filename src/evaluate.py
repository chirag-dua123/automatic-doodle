"""
Evaluation, Visualization & Root-Cause Attribution Module for SECOM
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score
)

def find_optimal_threshold(y_true, y_proba):
    """
    Finds the probability threshold that maximizes F1-score for the minority (defect) class.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Avoid zero division
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    return best_threshold, best_f1

def evaluate_model_performance(model, X_test, y_test, threshold=None):
    """
    Computes comprehensive test metrics tailored for severe class imbalance.
    Supports optimal thresholding for cost-sensitive defect detection.
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        y_proba = (decision - decision.min()) / (decision.max() - decision.min() + 1e-8)
    else:
        y_proba = model.predict(X_test)
        
    if threshold is None:
        threshold, _ = find_optimal_threshold(y_test, y_proba)
        
    y_pred = (y_proba >= threshold).astype(int)
        
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        "Confusion Matrix": cm,
        "Optimal Threshold": threshold,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc,
        "Precision (Defect)": report_dict['1']['precision'],
        "Recall (Defect)": report_dict['1']['recall'],
        "F1 (Defect)": report_dict['1']['f1-score'],
        "Accuracy": report_dict['accuracy'],
        "y_pred": y_pred,
        "y_proba": y_proba,
        "fpr": fpr, "tpr": tpr,
        "precision_curve": precision, "recall_curve": recall
    }
    
    return metrics

def plot_confusion_matrix(cm, output_path, threshold=0.5):
    """
    Generates and saves a clean confusion matrix heatmap.
    """
    plt.figure(figsize=(6, 5))
    labels = ['Pass (0)', 'Fail / Defect (1)']
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels, cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f'Confusion Matrix (Decision Threshold = {threshold:.3f})', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=11)
    plt.ylabel('Actual Class', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_roc_and_pr_curves(fpr, tpr, roc_auc, precision, recall, pr_auc, baseline_prevalence, output_path):
    """
    Generates dual-panel plot: ROC Curve (left) and Precision-Recall Curve (right).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. ROC Curve
    axes[0].plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Chance')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate (FPR)', fontsize=11)
    axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=11)
    axes[0].set_title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # 2. Precision-Recall Curve
    axes[1].plot(recall, precision, color='#d62728', lw=2.5, label=f'PR Curve (PR-AUC = {pr_auc:.3f})')
    axes[1].axhline(y=baseline_prevalence, color='gray', linestyle='--', lw=1.5, label=f'No-Skill Baseline ({baseline_prevalence:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (Sensitivity)', fontsize=11)
    axes[1].set_ylabel('Precision', fontsize=11)
    axes[1].set_title('Precision-Recall Curve (Imbalanced Metric)', fontsize=12, fontweight='bold')
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_top_sensors(feature_names, importances, top_n=20, output_path=None):
    """
    Plots top N sensor signals attributed to wafer failure.
    """
    df_imp = pd.DataFrame({
        'Sensor': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(9, 7))
    sns.barplot(
        data=df_imp, y='Sensor', x='Importance', hue='Sensor',
        palette='viridis', orient='h', legend=False
    )
    plt.title(f'Top {top_n} Critical Sensors Attributed to Wafer Faults', fontsize=12, fontweight='bold')
    plt.xlabel('Attribution / Importance Score', fontsize=11)
    plt.ylabel('Sensor Telemetry Channel', fontsize=11)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.close()

def plot_benchmark_comparison(df_results, output_path):
    """
    Plots comparative benchmark bar chart across candidate algorithms.
    """
    plt.figure(figsize=(10, 5))
    df_plot = df_results.melt(
        id_vars=['Model'],
        value_vars=['CV ROC-AUC (Mean)', 'CV PR-AUC (Mean)'],
        var_name='Metric', value_name='Score'
    )
    
    sns.barplot(data=df_plot, x='Model', y='Score', hue='Metric', palette='Set2')
    plt.title('Algorithm Benchmark: ROC-AUC vs. PR-AUC across Models', fontsize=12, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=11)
    plt.ylabel('Score (5-Fold Stratified CV)', fontsize=11)
    plt.xticks(rotation=25, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

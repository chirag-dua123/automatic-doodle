"""
Data Preprocessing Module for SECOM Semiconductor Dataset
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold

def load_raw_data(data_path, labels_path):
    """
    Loads raw SECOM feature matrix and label vectors.
    """
    # Load 590 sensor features
    X_df = pd.read_csv(data_path, sep=r'\s+', header=None)
    X_df.columns = [f"Sensor_{i+1}" for i in range(X_df.shape[1])]
    
    # Load labels: First col is class (-1: Pass, 1: Fail), second col is timestamp
    labels_df = pd.read_csv(labels_path, sep=r'\s+', header=None)
    y_raw = labels_df[0]
    
    # Convert labels: -1 -> 0 (Pass / Non-defective), 1 -> 1 (Fail / Defective)
    y = y_raw.map({-1: 0, 1: 1})
    
    timestamps = labels_df[1] if labels_df.shape[1] > 1 else None
    
    return X_df, y, timestamps

def drop_high_null_features(df, threshold=0.5):
    """
    Removes columns with missing value percentage exceeding threshold.
    """
    null_pct = df.isnull().mean()
    cols_to_keep = null_pct[null_pct <= threshold].index
    dropped_cols = null_pct[null_pct > threshold].index
    return df[cols_to_keep], list(dropped_cols)

def drop_constant_features(df):
    """
    Removes zero-variance (constant) features.
    """
    # We measure variance on existing non-null values
    variances = df.var(skipna=True)
    cols_to_keep = variances[variances > 1e-8].index
    dropped_cols = variances[variances <= 1e-8].index
    return df[cols_to_keep], list(dropped_cols)

def preprocess_data(data_path, labels_path, test_size=0.2, random_state=42, scaler_type='robust'):
    """
    Complete preprocessing pipeline avoiding data leakage:
    1. Load data & binary label mapping.
    2. Filter >50% null features and zero-variance features across dataset.
    3. Perform Stratified Train-Test split.
    4. Impute missing values using Median (fitted on train).
    5. Scale numerical features using RobustScaler (fitted on train).
    """
    X_df, y, _ = load_raw_data(data_path, labels_path)
    initial_features_count = X_df.shape[1]
    
    # 1. Drop features with > 50% missing values
    X_filtered, high_null_dropped = drop_high_null_features(X_df, threshold=0.5)
    
    # 2. Drop constant (zero-variance) features
    X_filtered, zero_var_dropped = drop_constant_features(X_filtered)
    
    retained_feature_names = list(X_filtered.columns)
    
    # 3. Stratified Train-Test Split (essential for imbalanced classes)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_filtered, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 4. Impute missing values using Median Imputer
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train_df)
    X_test_imp = imputer.transform(X_test_df)
    
    # 5. Feature Scaling
    scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    # Convert to DataFrames
    X_train = pd.DataFrame(X_train_scaled, columns=retained_feature_names, index=X_train_df.index)
    X_test = pd.DataFrame(X_test_scaled, columns=retained_feature_names, index=X_test_df.index)
    
    summary = {
        "initial_features": initial_features_count,
        "high_null_dropped": len(high_null_dropped),
        "zero_var_dropped": len(zero_var_dropped),
        "retained_features": len(retained_feature_names),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "train_pos_class": int(y_train.sum()),
        "train_neg_class": int(len(y_train) - y_train.sum()),
        "test_pos_class": int(y_test.sum()),
        "test_neg_class": int(len(y_test) - y_test.sum()),
    }
    
    return X_train, X_test, y_train, y_test, retained_feature_names, summary

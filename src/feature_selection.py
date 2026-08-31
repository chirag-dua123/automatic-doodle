"""
Feature Selection & Dimensionality Reduction Module for SECOM Dataset
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

def remove_collinear_features(X_train, X_test, threshold=0.85):
    """
    Identifies and removes highly collinear features (|correlation| > threshold)
    using the correlation matrix calculated strictly on the training set.
    """
    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    X_train_filtered = X_train.drop(columns=to_drop)
    X_test_filtered = X_test.drop(columns=to_drop)
    
    return X_train_filtered, X_test_filtered, to_drop

def select_top_k_mutual_info(X_train, y_train, X_test, k=40, random_state=42):
    """
    Selects top K features according to Mutual Information classification score.
    """
    selector = SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=random_state), k=min(k, X_train.shape[1]))
    selector.fit(X_train, y_train)
    
    selected_mask = selector.get_support()
    selected_features = X_train.columns[selected_mask].tolist()
    scores = pd.Series(selector.scores_, index=X_train.columns).loc[selected_features].sort_values(ascending=False)
    
    X_train_k = pd.DataFrame(selector.transform(X_train), columns=selected_features, index=X_train.index)
    X_test_k = pd.DataFrame(selector.transform(X_test), columns=selected_features, index=X_test.index)
    
    return X_train_k, X_test_k, scores

def select_top_k_trees(X_train, y_train, X_test, k=40, random_state=42):
    """
    Selects top K features using feature importances from ExtraTreesClassifier.
    """
    et = ExtraTreesClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    et.fit(X_train, y_train)
    
    importances = pd.Series(et.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    top_features = importances.head(k).index.tolist()
    
    X_train_k = X_train[top_features]
    X_test_k = X_test[top_features]
    
    return X_train_k, X_test_k, importances.head(k)

def apply_pca_reduction(X_train, X_test, variance_ratio=0.90, random_state=42):
    """
    Applies Principal Component Analysis (PCA) to retain specified cumulative variance ratio.
    """
    pca = PCA(n_components=variance_ratio, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    pca_cols = [f"PC_{i+1}" for i in range(X_train_pca.shape[1])]
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_cols, index=X_train.index)
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_cols, index=X_test.index)
    
    return X_train_pca_df, X_test_pca_df, pca

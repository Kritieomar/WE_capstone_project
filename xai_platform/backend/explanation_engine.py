"""
explanation_engine.py

Purpose:
Generate explainability insights using SHAP.
Provides functions for computing SHAP values, extracting feature importance,
and explaining individual predictions.
"""
import shap
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_shap_values(model, X):
    """
    Detects the model type and computes SHAP values.

    - TreeExplainer is used for tree-based models (RandomForest, XGBoost, etc.).
    - KernelExplainer is a model-agnostic fallback.

    Args:
        model: Trained machine learning model.
        X (pd.DataFrame): Feature dataset.

    Returns:
        tuple: (shap_values, expected_value)
    """
    model_name = type(model).__name__
    num_samples = len(X)
    num_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])

    logging.info(f"Model type: {model_name}")
    logging.info(f"Dataset size: {num_samples} samples, {num_features} features")

    # Check for tree-based models
    tree_keywords = ['forest', 'tree', 'xgb', 'lgbm', 'catboost', 'gradientboosting']
    if any(kw in model_name.lower() for kw in tree_keywords):
        logging.info(f"Using shap.TreeExplainer for tree-based model '{model_name}'.")
        explainer = shap.TreeExplainer(model)
    else:
        logging.info(f"Using shap.KernelExplainer for model '{model_name}'.")
        background = shap.sample(X, min(100, len(X)))
        predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
        explainer = shap.KernelExplainer(predict_fn, background)

    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value

    logging.info("SHAP values computed successfully.")
    return shap_values, expected_value


def get_feature_importance(shap_values, feature_names):
    """
    Computes mean absolute SHAP value per feature, sorted descending.

    Args:
        shap_values: SHAP values array (or list for multiclass).
        feature_names: List of feature names.

    Returns:
        dict: Feature importance dictionary sorted by importance.
    """
    if isinstance(shap_values, list):
        # Multiclass: average across classes
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # 3D array (samples x features x classes): average across samples, then classes
        mean_abs = np.abs(shap_values).mean(axis=0).mean(axis=1)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    # Ensure each value is a scalar
    mean_abs = np.ravel(mean_abs)

    importance = {str(name): float(val) for name, val in zip(feature_names, mean_abs)}
    importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))

    logging.info(f"Feature importance computed for {len(importance)} features.")
    return importance


def explain_prediction(model, X, shap_values, index):
    """
    Extracts SHAP values for a specific row to explain an individual prediction.

    Args:
        model: Trained ML model.
        X (pd.DataFrame): Feature dataset.
        shap_values: SHAP values array.
        index (int): Row index to explain.

    Returns:
        dict: Feature contribution dictionary for the selected row.
    """
    if index < 0 or index >= len(X):
        raise ValueError(f"Index {index} is out of bounds for dataset of size {len(X)}.")

    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"Feature_{i}" for i in range(X.shape[1])]

    # Handle multiclass shap_values (list of arrays)
    if isinstance(shap_values, list):
        row_shap = shap_values[1][index]  # Use class 1 for binary
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # 3D array: take class 1 (positive class) for the given row
        row_shap = shap_values[index, :, 1]
    else:
        row_shap = shap_values[index]

    # Ensure each value is a scalar
    row_shap = np.ravel(row_shap)

    contributions = {str(name): float(val) for name, val in zip(feature_names, row_shap)}
    contributions = dict(sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True))

    logging.info(f"Prediction explanation generated for row {index}.")
    return contributions

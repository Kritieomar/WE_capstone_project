<<<<<<< HEAD
"""
explanation_engine.py

Purpose:
Generate explainability insights using SHAP.
Provides functions for computing SHAP values, extracting feature importance,
and explaining individual predictions.
"""
import shap
=======
import pandas as pd
>>>>>>> 7f29cbc1e1f2545912475f976ad96d49f465b44d
import numpy as np
import pandas as pd
import logging
from explainerdashboard.explainers import ClassifierExplainer, RegressionExplainer
from sklearn.base import is_classifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

<<<<<<< HEAD

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
        # 3D array (samples × features × classes): average across samples, then classes
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
=======
def _get_explainer(model, X):
    """
    Returns an explainerdashboard explainer.
    We pass a dummy y array since we only care about explanations and what-if.
    """
    y_dummy = np.zeros(len(X))
    if is_classifier(model) or hasattr(model, 'predict_proba'):
        logging.info(f"Model type '{type(model).__name__}' recognized as classifier. Using ClassifierExplainer.")
        # Some models fail in shap="guess", so we let explainerdashboard handle it.
        return ClassifierExplainer(model, X, y_dummy, shap='guess')
    else:
        logging.info(f"Model type '{type(model).__name__}' recognized as regressor. Using RegressionExplainer.")
        return RegressionExplainer(model, X, y_dummy, shap='guess')

def generate_global_explanation(model, X):
    logging.info("Generating global SHAP explanation using explainerdashboard...")
    
    # Subsample for faster metrics
    if len(X) > 1000:
        X_sample = X.sample(n=1000, random_state=42)
    else:
        X_sample = X
        
    explainer = _get_explainer(model, X_sample)
    
    # 1. Feature Importance
    try:
        imp_df = explainer.get_mean_abs_shap_df()
        feature_importance = {str(row['Feature']): float(row['MEAN_ABS_SHAP']) for _, row in imp_df.iterrows()}
    except Exception as e:
        logging.warning(f"Could not use get_mean_abs_shap_df: {e}. Falling back to importances().")
        imp_df = explainer.importances()
        feature_importance = {str(idx): float(val) for idx, val in imp_df.items()}

    # 2. Global SHAP values (optional, for summary plots)
    try:
        shap_vals = explainer.get_shap_values()
        if isinstance(shap_vals, list):
            shap_list = [sv.tolist() for sv in shap_vals]
        elif isinstance(shap_vals, np.ndarray):
            shap_list = shap_vals.tolist()
        else:
            shap_list = []
    except Exception as e:
        logging.warning(f"Failed to extract raw shap values: {e}")
        shap_list = []

    logging.info("Global SHAP explanation generated successfully.")
    
    return {
        'shap_values': shap_list,
        'feature_importance': feature_importance
    }

def generate_local_explanation(model, X, index):
    logging.info(f"Generating local SHAP explanation for index {index}... using explainerdashboard")
    
    if index < 0 or index >= len(X):
        error_msg = f"Index {index} is out of bounds for dataset of size {len(X)}."
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Use a small background sample + the specific row
    X_sample_bg = X.sample(n=min(100, len(X)), random_state=42)
    row_df = X.iloc[[index]].copy()
    
    # Combine background and row
    X_combined = pd.concat([X_sample_bg, row_df], ignore_index=True)
    target_idx_in_combined = len(X_combined) - 1
    
    explainer = _get_explainer(model, X_combined)
    
    # Extract Base Value
    try:
        base_val = explainer.expected_value
    except Exception:
        try:
            base_val = explainer.shap_base_value()
        except Exception:
            base_val = 0.0

    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])
    elif hasattr(base_val, "__float__"):
        base_val = float(base_val)

    # Extract Local SHAP Values
    try:
        shap_vals = explainer.get_shap_row(target_idx_in_combined)
        if isinstance(shap_vals, pd.DataFrame):
            shap_list = shap_vals.values[0].tolist()
        elif isinstance(shap_vals, pd.Series):
            shap_list = shap_vals.tolist()
        else:
            shap_list = shap_vals.tolist() if hasattr(shap_vals, 'tolist') else list(shap_vals)
    except Exception as e:
        logging.warning(f"Failed to get_shap_row: {e}")
        shap_list = [0.0] * len(X.columns)
        
    # Extract Prediction (What-if compatible)
    try:
        if is_classifier(model) or hasattr(model, 'predict_proba'):
            # explainer.predict returns prediction probabilities for classifier
            pred_arr = explainer.predict(target_idx_in_combined)
            pred = float(pred_arr[1]) if isinstance(pred_arr, (list, np.ndarray)) and len(pred_arr) > 1 else float(pred_arr)
        else:
            pred_arr = explainer.predict(target_idx_in_combined)
            pred = float(pred_arr) if not isinstance(pred_arr, (list, np.ndarray)) else float(pred_arr[0])
    except Exception as e:
        logging.warning(f"Failed to predict via explainer: {e}. Falling back to model.predict.")
        pred = float(model.predict(row_df)[0])

    feature_values = row_df.iloc[0].to_dict()

    logging.info(f"Local SHAP explanation for index {index} generated successfully.")
    
    return {
        'shap_values': shap_list,
        'base_value': base_val,
        'feature_values': feature_values,
        'prediction': pred
    }

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    
    print("--- Testing SHAP Explanations ---")
    np.random.seed(42)
    X_dummy = pd.DataFrame(np.random.rand(100, 3), columns=['Feature_A', 'Feature_B', 'Feature_C'])
    y_dummy = X_dummy['Feature_A'] * 2 + X_dummy['Feature_B'] - X_dummy['Feature_C'] + np.random.randn(100) * 0.1
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)
    
    global_exp = generate_global_explanation(model, X_dummy)
    print("Feature Importance:")
    print(global_exp['feature_importance'])
    
    local_exp = generate_local_explanation(model, X_dummy, index=10)
    print("\nLocal Explanation for Index 10:")
    print(f"Base Value: {local_exp['base_value']}")
    print(f"Feature Values: {local_exp['feature_values']}")
    print(f"SHAP Values: {local_exp['shap_values']}")
    print(f"Prediction: {local_exp['prediction']}")
>>>>>>> 7f29cbc1e1f2545912475f976ad96d49f465b44d

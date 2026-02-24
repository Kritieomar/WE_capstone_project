import shap
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _get_explainer(model, X):
    """
    Detects the model type and returns the appropriate SHAP explainer.
    
    SHAP (SHapley Additive exPlanations) works by computing the contribution of each 
    feature to the prediction for a specific instance. It is based on game theory.
    - TreeExplainer is fast and exact for tree-based models (RandomForest, XGBoost, etc.).
    - LinearExplainer is used for linear models (LinearRegression, LogisticRegression).
    - KernelExplainer is a model-agnostic fallback that approximates SHAP values.
    """
    model_name = type(model).__name__.lower()
    
    # Check for tree-based models
    if any(tree_kw in model_name for tree_kw in ['forest', 'tree', 'xgb', 'lgbm', 'catboost', 'gradientboosting']):
        logging.info(f"Model type '{type(model).__name__}' detected as tree-based. Using TreeExplainer.")
        return shap.TreeExplainer(model)
        
    # Check for linear models
    elif any(linear_kw in model_name for linear_kw in ['linear', 'logistic', 'ridge', 'lasso', 'elasticnet']):
        logging.info(f"Model type '{type(model).__name__}' detected as linear. Using LinearExplainer.")
        # Background dataset can be passed; shap.sample is used to cap background size to 1000
        background = shap.sample(X, min(1000, len(X)))
        return shap.LinearExplainer(model, background)
        
    # Fallback to KernelExplainer
    else:
        logging.info(f"Model type '{type(model).__name__}' not explicitly recognized. Falling back to KernelExplainer.")
        # Background dataset is required for KernelExplainer to represent expected values.
        background_data = shap.sample(X, min(1000, len(X)))
        # Note: we need a predict function or predict_proba for KernelExplainer
        predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
        return shap.KernelExplainer(predict_fn, background_data)

def generate_global_explanation(model, X):
    """
    Generates global explanations (feature importance) using SHAP.
    
    SHAP global explanation aggregats local SHAP values to understand the overall 
    importance of features across the entire dataset. It calculates the mean absolute 
    SHAP value for each feature.
    
    Args:
        model: Trained machine learning model.
        X (pd.DataFrame): Dataset representing the features.
        
    Returns:
        dict: A dictionary containing 'shap_values' and 'feature_importance'.
    """
    logging.info("Generating global SHAP explanation...")
    
    # Optimize performance: Sample dataset if it's very large (> 1000 rows)
    if len(X) > 1000:
        logging.info(f"Dataset has {len(X)} rows. Sampling 1000 rows for SHAP explanation to optimize performance.")
        X_sample = X.sample(n=1000, random_state=42) if isinstance(X, pd.DataFrame) else shap.sample(X, 1000)
    else:
        X_sample = X
    
    explainer = _get_explainer(model, X_sample)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # Compute feature importance (mean absolute SHAP values across all samples)
    if isinstance(shap_values, list):
        # Handle multiclass case where shap_values is a list of arrays
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # Standard case
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
    # Create feature importance dictionary
    feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"Feature_{i}" for i in range(X.shape[1])]
    
    feature_importance = {
        str(name): float(importance) for name, importance in zip(feature_names, mean_abs_shap)
    }
    
    # Sort feature importance descending
    feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
    
    logging.info("Global SHAP explanation generated successfully.")
    
    return {
        'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else [sv.tolist() for sv in shap_values],
        'feature_importance': feature_importance
    }

def generate_local_explanation(model, X, index):
    """
    Generates a SHAP explanation for a specific row/instance.
    
    Local SHAP values tell us how much each feature contributed to pushing the 
    prediction for this specific instance away from the base value (expected value).
    
    Args:
        model: Trained machine learning model.
        X (pd.DataFrame): Dataset representing the features.
        index (int): The index of the row in X to explain.
        
    Returns:
        dict: A dictionary containing local 'shap_values', 'base_value', and 'feature_values'.
    """
    logging.info(f"Generating local SHAP explanation for index {index}...")
    
    if index < 0 or index >= len(X):
        error_msg = f"Index {index} is out of bounds for dataset of size {len(X)}."
        logging.error(error_msg)
        raise ValueError(error_msg)
        
    row = X.iloc[[index]] if isinstance(X, pd.DataFrame) else X[[index]]
    
    # Get explainer based on full X (or sampled for background if > 1000)
    background_data = X.sample(n=min(1000, len(X)), random_state=42) if isinstance(X, pd.DataFrame) else shap.sample(X, min(1000, len(X)))
    explainer = _get_explainer(model, background_data)
    
    # Calculate SHAP values for the specific row
    shap_values = explainer.shap_values(row)
    
    # Base value (expected value) is the average model output over the background dataset
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value.tolist()
    elif hasattr(base_value, '__float__'):
        base_value = float(base_value)
        
    logging.info(f"Local SHAP explanation for index {index} generated successfully.")
    
    return {
        'shap_values': shap_values[0].tolist() if isinstance(shap_values, np.ndarray) else [sv[0].tolist() for sv in shap_values],
        'base_value': base_value,
        'feature_values': row.iloc[0].to_dict() if isinstance(X, pd.DataFrame) else row[0].tolist()
    }

if __name__ == "__main__":
    # Small test block to verify independently
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    
    print("--- Testing SHAP Explanations ---")
    
    # Dummy data
    np.random.seed(42)
    X_dummy = pd.DataFrame(np.random.rand(100, 3), columns=['Feature_A', 'Feature_B', 'Feature_C'])
    y_dummy = X_dummy['Feature_A'] * 2 + X_dummy['Feature_B'] - X_dummy['Feature_C'] + np.random.randn(100) * 0.1
    
    # Train dummy model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)
    
    # Test Global Explanation
    global_exp = generate_global_explanation(model, X_dummy)
    print("Feature Importance:")
    print(global_exp['feature_importance'])
    
    # Test Local Explanation
    local_exp = generate_local_explanation(model, X_dummy, index=10)
    print("\nLocal Explanation for Index 10:")
    print(f"Base Value: {local_exp['base_value']}")
    print(f"Feature Values: {local_exp['feature_values']}")
    print(f"SHAP Values: {local_exp['shap_values']}")

from explainerdashboard import ClassifierExplainer
from explainerdashboard import RegressionExplainer
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_explainer(model, X, y):
    """
    Creates an ExplainerDashboard explainer object based on the model type.
    
    Args:
        model: Trained machine learning model.
        X (pd.DataFrame): Dataset representing the features.
        y (pd.Series or np.array): Target labels.
        
    Returns:
        Explainer: An ExplainerDashboard explainer object.
    """
    model_type = type(model).__name__
    num_samples = len(X)
    num_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
    
    logging.info(f"Generating explainer for model type: {model_type}")
    logging.info(f"Dataset has {num_samples} samples and {num_features} features.")
    
    if hasattr(model, 'predict_proba'):
        logging.info("Model has predict_proba(). Using ClassifierExplainer.")
        explainer = ClassifierExplainer(model, X, y)
    else:
        logging.info("Model does not have predict_proba(). Using RegressionExplainer.")
        explainer = RegressionExplainer(model, X, y)
        
    logging.info("Explainer generated successfully.")
    
    return explainer

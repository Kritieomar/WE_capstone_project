import numpy as np
import logging
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_classification(model, X, y):
    """
    Computes classification metrics for a given model.
    
    Args:
        model: Trained classification model with a predict method.
        X (pd.DataFrame or np.ndarray): Feature dataset.
        y (pd.Series or np.ndarray): True labels.
        
    Returns:
        dict: A dictionary containing accuracy, precision, recall, f1_score, and confusion_matrix.
    """
    logging.info("Evaluating classification model...")
    y_pred = model.predict(X)
    
    # average='weighted' handles multiclass classification gracefully
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()  # Convert numpy array to list for JSON serialization
    }
    
    logging.info(f"Classification metrics computed: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    return metrics

def evaluate_regression(model, X, y):
    """
    Computes regression metrics for a given model.
    
    Args:
        model: Trained regression model with a predict method.
        X (pd.DataFrame or np.ndarray): Feature dataset.
        y (pd.Series or np.ndarray): True continuous target values.
        
    Returns:
        dict: A dictionary containing rmse, mae, and r2_score.
    """
    logging.info("Evaluating regression model...")
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2)
    }
    
    logging.info(f"Regression metrics computed: RMSE={rmse:.4f}, R2={r2:.4f}")
    return metrics

def evaluate_model(model, X, y):
    """
    Evaluates a model by automatically detecting if it's performing classification or regression.
    
    Detection Strategy:
        If the model has a `predict_proba()` method, it's treated as a classifier.
        Otherwise, it's treated as a regressor.
        
    Args:
        model: Trained model with a predict method.
        X: Feature dataset.
        y: True targets.
        
    Returns:
        dict: A dictionary containing the appropriate evaluation metrics.
    """
    # Determine the task type based on the presence of 'predict_proba'
    is_classification = hasattr(model, 'predict_proba')
    
    if is_classification:
        logging.info("Detected classification model based on 'predict_proba' method.")
        return evaluate_classification(model, X, y)
    else:
        logging.info("Detected regression model (no 'predict_proba' method).")
        return evaluate_regression(model, X, y)

if __name__ == "__main__":
    # Small test block to verify independently
    class DummyClassifier:
        def predict_proba(self, X):
            return np.array([[0.1, 0.9], [0.8, 0.2]])
        def predict(self, X):
            return np.array([1, 0])
            
    class DummyRegressor:
        def predict(self, X):
            return np.array([1.5, 3.2])
            
    # Dummy data
    X_dummy = np.array([[1, 2], [3, 4]])
    y_class = np.array([1, 0])
    y_reg = np.array([1.0, 3.0])
    
    print("--- Testing Classification ---")
    clf = DummyClassifier()
    clf_metrics = evaluate_model(clf, X_dummy, y_class)
    print(clf_metrics)
    
    print("\n--- Testing Regression ---")
    reg = DummyRegressor()
    reg_metrics = evaluate_model(reg, X_dummy, y_reg)
    print(reg_metrics)

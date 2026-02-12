import pandas as pd
import logging
import os

# Configure logging to output to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(csv_path):

    if not os.path.exists(csv_path):
        error_msg = f"Error loading dataset: The file {csv_path} was not found."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        logging.info(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        logging.info(f"Dataset successfully loaded. Shape: {df.shape}")
        return df
    except pd.errors.EmptyDataError as e:
        error_msg = f"Error loading dataset: The file {csv_path} is empty."
        logging.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"An unexpected error occurred while loading the dataset: {e}"
        logging.error(error_msg)
        raise Exception(error_msg) from e

def prepare_features(df, target_column):
    """
    Separates the dataset into features (X) and target (y), dropping the target 
    column from the features side.
    
    Args:
        df (pd.DataFrame): The dataset.
        target_column (str): The name of the target column.
        
    Returns:
        tuple: (X, y) where X is the features DataFrame and y is the target Series.
        
    Raises:
        ValueError: If the target column is not found in the dataset.
    """
    if target_column not in df.columns:
        error_msg = f"Target column '{target_column}' not found in the dataset."
        logging.error(error_msg)
        raise ValueError(error_msg)
        
    logging.info(f"Separating features and target column: '{target_column}'")
    
    # Separate target y
    y = df[target_column]
    
    # Drop target column to get features X
    X = df.drop(columns=[target_column])
    
    logging.info(f"Features prepared. X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def validate_feature_count(model, X):
    """
    Checks if the number of features the model was trained on matches 
    the number of features in the dataset.
    
    Args:
        model: The trained machine learning model.
        X (pd.DataFrame): The features DataFrame.
        
    Raises:
        ValueError: If there is a mismatch in feature count.
    """
    # Check if the model has 'n_features_in_' attribute (standard in scikit-learn)
    if not hasattr(model, 'n_features_in_'):
        logging.warning("The model does not have 'n_features_in_' attribute. Skipping feature count validation.")
        return
        
    expected_features = model.n_features_in_
    actual_features = X.shape[1]
    
    if expected_features != actual_features:
        error_msg = f"Feature count mismatch! Model expects {expected_features} features, but dataset has {actual_features} features."
        logging.error(error_msg)
        raise ValueError(error_msg)
        
    logging.info("Feature count validation passed. Model and dataset feature counts match.")

if __name__ == "__main__":
    # Small test block to verify independently
    try:
        # Create a dummy CSV for testing
        test_csv = "test_dataset_dummy.csv"
        df_dummy = pd.DataFrame({
            'feature_A': [10, 20, 30],
            'feature_B': [4.5, 5.5, 6.5],
            'label': [0, 1, 0]
        })
        df_dummy.to_csv(test_csv, index=False)
        print("--- Testing load_dataset ---")
        df = load_dataset(test_csv)
        print(df.head())
        
        print("\n--- Testing prepare_features ---")
        X, y = prepare_features(df, 'label')
        print("X (features):")
        print(X)
        print("y (target):")
        print(y)
        
        print("\n--- Testing validate_feature_count ---")
        # Dummy model class simulating scikit-learn
        class DummyModel:
            def __init__(self, n_features):
                self.n_features_in_ = n_features
                
        # Matching features -> passes
        valid_model = DummyModel(n_features=2)
        validate_feature_count(valid_model, X)
        print("Passed validation for matching feature counts.")
        
        # Mismatch features -> raises ValueError
        invalid_model = DummyModel(n_features=3)
        try:
            validate_feature_count(invalid_model, X)
        except ValueError as e:
            print(f"Correctly caught mismatch error: {e}")
            
    finally:
        # Cleanup dummy test file
        if os.path.exists(test_csv):
            os.remove(test_csv)

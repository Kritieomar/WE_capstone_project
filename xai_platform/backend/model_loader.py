"""
model_loader.py

Purpose:
Handles loading the trained sklearn-compatible ML model from disk.
Supports .pkl and .joblib formats.
"""

import os
import pickle
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_model(file_path):
    """
    Load a trained machine learning model from a given file path.
    Supported formats: .pkl, .joblib

    Args:
        file_path (str): The path to the model file.

    Returns:
        model: The loaded machine learning model.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported or the model lacks a predict method.
        RuntimeError: If loading the model fails.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Model file not found at {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext not in ['.pkl', '.joblib']:
        logger.error(f"Unsupported file format: {ext}")
        raise ValueError(f"Unsupported file format: {ext}. Only .pkl and .joblib are supported.")

    try:
        logger.info(f"Loading model from {file_path}...")
        if ext == '.pkl':
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
        elif ext == '.joblib':
            model = joblib.load(file_path)
            
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model from {file_path}: {e}")

    # Validate that the model has a predict method
    if not hasattr(model, 'predict') or not callable(getattr(model, 'predict')):
        logger.error("Loaded model does not have a callable 'predict' method.")
        raise ValueError("Loaded model is not compatible. It must have a 'predict()' method.")

    return model

if __name__ == "__main__":
    # Small test example
    import tempfile
    
    class DummyModel:
        def predict(self, X):
            return [1] * len(X)
            
    # Create a dummy model
    dummy = DummyModel()
    
    print("Testing joblib load:")
    try:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_joblib:
            tmp_joblib_name = tmp_joblib.name
            joblib.dump(dummy, tmp_joblib_name)
            
        loaded_joblib = load_model(tmp_joblib_name)
        print(f"Joblib load successful. Predict result: {loaded_joblib.predict([0, 0])}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists(tmp_joblib_name):
            os.remove(tmp_joblib_name)

    print("\n" + "-" * 30 + "\n")

    print("Testing pickle load:")
    try:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_pkl:
            tmp_pkl_name = tmp_pkl.name
            with open(tmp_pkl_name, "wb") as f:
                pickle.dump(dummy, f)
                
        loaded_pkl = load_model(tmp_pkl_name)
        print(f"Pickle load successful. Predict result: {loaded_pkl.predict([0, 0])}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists(tmp_pkl_name):
            os.remove(tmp_pkl_name)

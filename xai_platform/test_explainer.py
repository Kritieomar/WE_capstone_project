import sys
import os

# Ensure the backend module can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.explanation_engine import create_explainer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Generate some dummy data
    X = pd.DataFrame(np.random.rand(100, 3), columns=['Feature_A', 'Feature_B', 'Feature_C'])
    y = np.random.randint(2, size=100)
    
    # Train a dummy model
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Create the explainer
    explainer = create_explainer(model, X, y)
    
    print("\n--- TEST: Object Details ---")
    print(explainer)
    print("Test passed. Explainer object created successfully.")

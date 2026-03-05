import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import shap
import numpy as np
import pandas as pd
import joblib

# Load the regenerated model and dataset
model = joblib.load("test_model.joblib")
df = pd.read_csv("test_dataset.csv")

# Assume target is last column
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Model type: {type(model).__name__}")
print(f"X shape: {X.shape}")

# Compute SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X.head(10))

print(f"\ntype(shap_values): {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"len(shap_values): {len(shap_values)}")
    for i, sv in enumerate(shap_values):
        print(f"  shap_values[{i}].shape: {np.array(sv).shape}")
elif isinstance(shap_values, np.ndarray):
    print(f"shap_values.shape: {shap_values.shape}")
    print(f"shap_values.ndim: {shap_values.ndim}")

# Try mean abs
if isinstance(shap_values, list):
    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    mean_abs = np.abs(shap_values).mean(axis=0).mean(axis=1)
else:
    mean_abs = np.abs(shap_values).mean(axis=0)

print(f"\nmean_abs type: {type(mean_abs)}")
print(f"mean_abs shape: {np.array(mean_abs).shape}")
print(f"mean_abs: {mean_abs}")

# Try float conversion
for i, val in enumerate(mean_abs):
    try:
        f = float(val)
        print(f"  float(mean_abs[{i}]) = {f}")
    except Exception as e:
        print(f"  float(mean_abs[{i}]) FAILED: {e}, type={type(val)}, val={val}")

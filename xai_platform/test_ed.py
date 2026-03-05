from sklearn.ensemble import RandomForestRegressor
from explainerdashboard import RegressionExplainer
import pandas as pd
import numpy as np

# Dummy data
np.random.seed(42)
X_dummy = pd.DataFrame(np.random.rand(100, 3), columns=['Feature_A', 'Feature_B', 'Feature_C'])
y_dummy = X_dummy['Feature_A'] * 2 + X_dummy['Feature_B'] - X_dummy['Feature_C'] + np.random.randn(100) * 0.1

# Train model
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_dummy, y_dummy)

# Dashboard explainer
explainer = RegressionExplainer(model, X_dummy, y_dummy)

# Get feature importances
print("Importances:")
print(explainer.importances())

# Get local components
index = 10
print("\nPredict:")
print(explainer.predict(index))

# Note: In RegressionExplainer, expected_value might be expected_value() or explainer.expected_value
print("\nExpected Value:")
try:
    print(explainer.expected_value)
except AttributeError:
    try:
         print(explainer.shap_base_value())
    except Exception as e:
         print("Base value error:", e)

# Get SHAP values for the specific index
print("\nSHAP Row:")
try:
    print(explainer.get_shap_row(index))
except Exception as e:
    print("SHAP row error:", e)

# Let's see what get_shap_values does
print("\nSHAP Values:")
print(type(explainer.get_shap_values()))

# What-if extraction
print("\nWhat-if Extract Predict:")
try:
    df_whatif = X_dummy.iloc[[10]].copy()
    df_whatif['Feature_A'] = 0.5
    print(explainer.model.predict(df_whatif))
except Exception as e:
    print("What-if predict error:", e)

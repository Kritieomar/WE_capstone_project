import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Loading dataset...")

# Load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Combine into a single dataframe for CSV
df = pd.concat([X, y], axis=1)

# Save dataset
dataset_path = "test_dataset.csv"
df.to_csv(dataset_path, index=False)

print(f"Dataset saved as {dataset_path}")

# Train model
print("Training RandomForest model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
model_path = "test_model.joblib"
joblib.dump(model, model_path)

print(f"Model saved as {model_path}")

print("Setup complete!")

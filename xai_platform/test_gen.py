import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# LOAD DATASET
df = pd.read_csv("d:/hello/loan_preprocessed_dataset.csv")

# DROP ID COLUMN If present
if "Loan_ID" in df.columns:
    df = df.drop(columns=["Loan_ID"])

target_column = "Loan_Status"

X = df.drop(target_column, axis=1)
y = df[target_column]

# Identify categorical and numerical columns
# For this dataset, let's treat object columns as categorical, and number columns as numeric
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing for numerical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# Save dataset to the local folder for easy uploading
df.to_csv("loan_preprocessed_dataset_cleaned.csv", index=False)

# Save model
joblib.dump(pipeline, "loan_model.joblib")

print("Created:")
print("loan_preprocessed_dataset_cleaned.csv")
print("loan_model.joblib")
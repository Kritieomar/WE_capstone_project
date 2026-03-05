"""
api_server.py

Flask REST API for the XAI Model Explanation Platform.
Serves the frontend and exposes endpoints for model analysis,
SHAP explanations, what-if simulations, and AI insights.
"""
import os
import sys
import json
import tempfile
import logging
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.model_loader import load_model
from backend.data_handler import load_dataset, prepare_features, validate_feature_count, extract_model, extract_feature_names
from backend.metrics_engine import evaluate_model
from backend.explanation_engine import compute_shap_values, get_feature_importance, explain_prediction
from backend.ai_insight_engine import generate_ai_insights

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__, static_folder=os.path.join(project_root, 'frontend'))
CORS(app)

# In-memory storage for current session
session_data = {}


# ==================== Serve Frontend ====================

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


# ==================== API Endpoints ====================

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload model and dataset files."""
    try:
        if 'model' not in request.files or 'dataset' not in request.files:
            return jsonify({'error': 'Both model and dataset files are required.'}), 400

        model_file = request.files['model']
        dataset_file = request.files['dataset']

        # Save to temp files
        model_ext = os.path.splitext(model_file.filename)[1]
        dataset_ext = os.path.splitext(dataset_file.filename)[1]

        temp_model = tempfile.NamedTemporaryFile(delete=False, suffix=model_ext)
        model_file.save(temp_model.name)
        temp_model.close()

        temp_dataset = tempfile.NamedTemporaryFile(delete=False, suffix=dataset_ext)
        dataset_file.save(temp_dataset.name)
        temp_dataset.close()

        # Load model and dataset
        model = load_model(temp_model.name)
        df = load_dataset(temp_dataset.name)

        # Store in session
        session_data['model'] = model
        session_data['df'] = df
        session_data['model_name'] = model_file.filename
        session_data['dataset_name'] = dataset_file.filename
        session_data['temp_model_path'] = temp_model.name
        session_data['temp_dataset_path'] = temp_dataset.name

        columns = df.columns.tolist()
        preview = df.head(10).to_dict(orient='records')

        return jsonify({
            'success': True,
            'model_type': type(model).__name__,
            'model_name': model_file.filename,
            'dataset_name': dataset_file.filename,
            'rows': df.shape[0],
            'columns': columns,
            'num_columns': df.shape[1],
            'preview': preview
        })

    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Run full analysis: metrics + SHAP."""
    try:
        if 'model' not in session_data or 'df' not in session_data:
            return jsonify({'error': 'Please upload model and dataset first.'}), 400

        data = request.get_json()
        target_col = data.get('target_column')

        if not target_col:
            return jsonify({'error': 'Target column is required.'}), 400

        model = session_data['model']
        df = session_data['df']

        # Prepare features
        X, y = prepare_features(df, target_col)
        validate_feature_count(model, X)

        feature_names = extract_feature_names(model, X.columns.tolist())
        core_model = extract_model(model)

        # If it's a pipeline, transform the features first for SHAP
        if hasattr(model, 'steps'):
            X_trans_arr = model.steps[0][1].transform(X)
            # Ensure dense format
            if hasattr(X_trans_arr, 'toarray'): X_trans_arr = X_trans_arr.toarray()
            X_transformed = pd.DataFrame(X_trans_arr, columns=feature_names, index=X.index)
        else:
            X_transformed = X.copy()
            X_transformed.columns = feature_names

        # Store for later use
        session_data['X'] = X # Original features for prediction
        session_data['X_transformed'] = X_transformed # Transformed features for explanation
        session_data['y'] = y
        session_data['target_col'] = target_col
        session_data['feature_names'] = feature_names

        # Compute metrics using outer model to ensure pipeline transformations happen normally
        metrics = evaluate_model(model, X, y)

        # Compute SHAP (sample for performance) using inner model on transformed data
        sample_size = min(500, len(X_transformed))
        X_sample_trans = X_transformed.sample(n=sample_size, random_state=42)
        X_sample_orig = X.loc[X_sample_trans.index]
        shap_values, expected_value = compute_shap_values(core_model, X_sample_trans)

        # Store SHAP values
        session_data['shap_values'] = shap_values
        session_data['X_sample'] = X_sample_orig # Return original samples to UI
        session_data['X_sample_trans'] = X_sample_trans

        # Feature importance
        feature_importance = get_feature_importance(shap_values, feature_names)

        # Compute full SHAP for local explanations
        shap_values_full, _ = compute_shap_values(core_model, X_transformed)
        session_data['shap_values_full'] = shap_values_full

        # SHAP values for beeswarm data (2D slice)
        if isinstance(shap_values, list):
            shap_2d = np.array(shap_values[1])  # class 1 for binary
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_2d = shap_values[:, :, 1]
        else:
            shap_2d = np.array(shap_values)

        # Build SHAP summary data for plotting using original raw X data for UI
        shap_summary_data = []
        for i, fname in enumerate(feature_names):
            for j in range(min(200, len(shap_2d))):
                shap_summary_data.append({
                    'feature': fname,
                    'shap_value': float(shap_2d[j][i]),
                    'feature_value': float(X_sample_trans.iloc[j, i]) if j < len(X_sample_trans) else 0
                })

        # Build Class Balance
        class_balance = y.value_counts().to_dict()
        class_balance = {str(k): int(v) for k, v in class_balance.items()}

        # Build Feature Distributions (Sampled up to 1000 rows for UI performance)
        feature_distributions = {}
        sample_size_dist = min(1000, len(X))
        if len(X) > sample_size_dist:
            dist_sample_X = X.sample(n=sample_size_dist, random_state=42)
            dist_sample_y = y.loc[dist_sample_X.index]
        else:
            dist_sample_X = X
            dist_sample_y = y

        for col in X.columns:
            dist_by_class = {}
            for cls_val in dist_sample_y.unique():
                # Extract values for this feature where the target equals cls_val
                vals = dist_sample_X[dist_sample_y == cls_val][col].dropna().tolist()
                # Ensure they are standard python floats/ints/strings for JSON
                dist_by_class[str(cls_val)] = [float(v) if isinstance(v, (np.floating, float)) else v for v in vals]
            feature_distributions[col] = dist_by_class

        # Build Data Statistics
        data_stats = []
        for col in df.columns:
            stats = {
                'column': col, 
                'dtype': str(df[col].dtype),
                'missing': int(df[col].isnull().sum()),
                'unique': int(df[col].nunique())
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_val = df[col].mean()
                min_val = df[col].min()
                max_val = df[col].max()
                stats['mean'] = round(float(mean_val), 3) if not pd.isna(mean_val) else '-'
                stats['min'] = round(float(min_val), 3) if not pd.isna(min_val) else '-'
                stats['max'] = round(float(max_val), 3) if not pd.isna(max_val) else '-'
            else:
                stats['mean'] = '-'
                stats['min'] = '-'
                stats['max'] = '-'
            data_stats.append(stats)

        # Build New Dataset Stats for Panel
        total_rows = len(df)
        feature_columns = df.shape[1] - 1
        missing_values = int(df.isnull().sum().sum())
        missing_pct = round((missing_values / (total_rows * df.shape[1])) * 100, 1) if total_rows > 0 else 0
        duplicate_rows = int(df.duplicated().sum())
        duplicate_pct = round((duplicate_rows / total_rows) * 100, 1) if total_rows > 0 else 0
        
        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(exclude=["number"]).columns
        numeric_features = int(len(numeric_cols))
        categorical_features = int(len(categorical_cols))
        
        if target_col in numeric_cols: numeric_features = max(0, numeric_features - 1)
        if target_col in categorical_cols: categorical_features = max(0, categorical_features - 1)

        dataset_stats_panel = {
            'total_rows': total_rows,
            'feature_columns': feature_columns,
            'missing_values': missing_values,
            'missing_pct': missing_pct,
            'duplicate_rows': duplicate_rows,
            'duplicate_pct': duplicate_pct,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features
        }

        # Numeric feature mean & count for new Bar Chart
        numeric_X = X.select_dtypes(include=["number"])
        numeric_feature_dist = []
        for col in numeric_X.columns:
             mean_val = numeric_X[col].mean()
             count_val = numeric_X[col].count()
             numeric_feature_dist.append({
                 'feature': col,
                 'mean': float(mean_val) if not pd.isna(mean_val) else 0.0,
                 'count': int(count_val)
             })

        return jsonify({
            'success': True,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'feature_names': feature_names,
            'shap_summary': shap_summary_data,
            'class_balance': class_balance,
            'feature_distributions': feature_distributions,
            'data_stats': data_stats,
            'dataset_stats_panel': dataset_stats_panel,
            'numeric_feature_dist': numeric_feature_dist,
            'num_samples': len(X_transformed),
            'num_features': X_transformed.shape[1],
            'model_type': type(core_model).__name__
        })

    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/explain/<int:index>', methods=['GET'])
def explain_local(index):
    """Get local explanation for a specific row."""
    try:
        if 'shap_values_full' not in session_data:
            return jsonify({'error': 'Run analysis first.'}), 400

        model = session_data['model']
        core_model = extract_model(model)
        X = session_data['X'] # original
        X_transformed = session_data['X_transformed'] # transformed features
        shap_values_full = session_data['shap_values_full']

        # Explain the prediction using transformed features for SHAP alignment
        contributions = explain_prediction(core_model, X_transformed, shap_values_full, index)

        # Return actual un-transformed feature values for the UI to edit
        row_data = X.iloc[index].to_dict()
        prediction = model.predict(X.iloc[[index]])[0]

        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X.iloc[[index]])[0]
            probabilities = {f"Class {i}": float(p) for i, p in enumerate(probs)}

        return jsonify({
            'success': True,
            'contributions': contributions,
            'feature_values': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in row_data.items()},
            'prediction': int(prediction) if isinstance(prediction, (np.integer, int)) else float(prediction),
            'probabilities': probabilities
        })

    except Exception as e:
        logging.error(f"Local explanation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/whatif', methods=['POST'])
def whatif_simulation():
    """What-if simulation: modify features, get new prediction."""
    try:
        if 'model' not in session_data:
            return jsonify({'error': 'Run analysis first.'}), 400

        data = request.get_json()
        modified_features = data.get('features', {})

        model = session_data['model']
        core_model = extract_model(model)
        X = session_data['X']
        feature_names = session_data['feature_names']

        # Create a single-row DataFrame with modified values
        # UI returns keys that match raw dataset
        raw_feature_names = X.columns.tolist()
        row = {name: float(modified_features.get(name, 0)) for name in raw_feature_names if name in modified_features}
        # Backfill unchanged string types correctly
        for name in raw_feature_names:
             if name not in modified_features:
                  row[name] = X.iloc[0][name] # Fallback, though UI should pass everything
             elif isinstance(X.iloc[0][name], str):
                  # what-if UI passes input fields. For categorical drops, read original or passed
                  row[name] = modified_features.get(name, X.iloc[0][name])

        row_df = pd.DataFrame([row])

        # Predict using outer pipeline
        prediction = model.predict(row_df)[0]

        # Calculate SHAP values for the modified inputs via inner core model
        if hasattr(model, 'steps'):
             row_trans_arr = model.steps[0][1].transform(row_df)
             if hasattr(row_trans_arr, 'toarray'): row_trans_arr = row_trans_arr.toarray()
             row_trans_df = pd.DataFrame(row_trans_arr, columns=feature_names)
        else:
             row_trans_df = row_df.copy()
             row_trans_df.columns = feature_names

        shap_values_modified, _ = compute_shap_values(core_model, row_trans_df)
        contributions = explain_prediction(core_model, row_trans_df, shap_values_modified, 0)

        result = {
            'success': True,
            'prediction': int(prediction) if isinstance(prediction, (np.integer, int)) else float(prediction),
            'contributions': contributions
        }

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(row_df)[0]
            result['probabilities'] = {f"Class {i}": float(p) for i, p in enumerate(probs)}

        return jsonify(result)

    except Exception as e:
        logging.error(f"What-if error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai-insights', methods=['POST'])
def ai_insights():
    """Generate AI insights using Gemini."""
    try:
        data = request.get_json()
        feature_importance = data.get('feature_importance', {})
        api_key = data.get('api_key', '')

        explanation = generate_ai_insights(feature_importance, api_key=api_key)

        return jsonify({
            'success': True,
            'explanation': explanation
        })

    except Exception as e:
        logging.error(f"AI insights error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

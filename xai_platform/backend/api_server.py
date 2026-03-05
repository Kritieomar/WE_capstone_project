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
from backend.data_handler import load_dataset, prepare_features, validate_feature_count
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

        # Store for later use
        session_data['X'] = X
        session_data['y'] = y
        session_data['target_col'] = target_col

        # Compute metrics
        metrics = evaluate_model(model, X, y)

        # Compute SHAP (sample for performance)
        sample_size = min(500, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        shap_values, expected_value = compute_shap_values(model, X_sample)

        # Store SHAP values
        session_data['shap_values'] = shap_values
        session_data['X_sample'] = X_sample

        # Feature importance
        feature_names = X.columns.tolist()
        feature_importance = get_feature_importance(shap_values, feature_names)

        # Compute full SHAP for local explanations
        shap_values_full, _ = compute_shap_values(model, X)
        session_data['shap_values_full'] = shap_values_full

        # SHAP values for beeswarm data (2D slice)
        if isinstance(shap_values, list):
            shap_2d = np.array(shap_values[1])  # class 1 for binary
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_2d = shap_values[:, :, 1]
        else:
            shap_2d = np.array(shap_values)

        # Build SHAP summary data for plotting
        shap_summary_data = []
        for i, fname in enumerate(feature_names):
            for j in range(min(200, len(shap_2d))):
                shap_summary_data.append({
                    'feature': fname,
                    'shap_value': float(shap_2d[j][i]),
                    'feature_value': float(X_sample.iloc[j, i]) if j < len(X_sample) else 0
                })

        return jsonify({
            'success': True,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'feature_names': feature_names,
            'shap_summary': shap_summary_data,
            'num_samples': len(X),
            'num_features': X.shape[1],
            'model_type': type(model).__name__
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
        X = session_data['X']
        shap_values_full = session_data['shap_values_full']

        contributions = explain_prediction(model, X, shap_values_full, index)

        # Also return the actual feature values and prediction
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
        X = session_data['X']

        # Create a single-row DataFrame with modified values
        feature_names = X.columns.tolist()
        row = {name: float(modified_features.get(name, 0)) for name in feature_names}
        row_df = pd.DataFrame([row])

        # Predict
        prediction = model.predict(row_df)[0]

        result = {
            'success': True,
            'prediction': int(prediction) if isinstance(prediction, (np.integer, int)) else float(prediction)
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

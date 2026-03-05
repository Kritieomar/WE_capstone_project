"""
app.py

Purpose:
Streamlit frontend application for the XAI Model Explanation Platform.
Provides the UI to upload models, datasets, and view SHAP explanations
with AI-powered natural language insights via Gemini.
"""
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import os
import sys
import shap

# Include the project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import backend modules
from backend.model_loader import load_model
from backend.data_handler import load_dataset, prepare_features, validate_feature_count
from backend.metrics_engine import evaluate_model
from backend.explanation_engine import compute_shap_values, get_feature_importance, explain_prediction
from backend.session_manager import save_session
from backend.ai_insight_engine import generate_ai_insights


def save_uploaded_file(uploaded_file):
    """Saves an uploaded file to a temporary file and returns the path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    uploaded_file.seek(0)
    temp_file.write(uploaded_file.getbuffer())
    temp_file.close()
    return temp_file.name


st.set_page_config(page_title="XAI Model Explanation Platform", layout="wide")

st.title("XAI Model Explanation Platform")
st.markdown("""
Upload your trained ML model and dataset to generate **SHAP-based explainability insights** 
and **AI-powered natural language explanations** using Google Gemini.
""")

st.divider()

# ===================== SIDEBAR =====================
st.sidebar.header("Configuration")

# 1. Upload Model
st.sidebar.subheader("1. Upload Model")
model_file = st.sidebar.file_uploader("Upload Trained Model (.pkl, .joblib)", type=["pkl", "joblib"])

# 2. Upload Dataset
st.sidebar.subheader("2. Upload Dataset")
data_file = st.sidebar.file_uploader("Upload Tabular Dataset (.csv)", type=["csv"])

# Gemini API Key
st.sidebar.subheader("AI Insights (Optional)")
gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if model_file and data_file:
    # 3. Select Target Column
    df_preview = pd.read_csv(data_file, nrows=5)
    st.sidebar.subheader("3. Settings")

    target_col = st.sidebar.selectbox("Select Target Column", options=df_preview.columns)

    # Row index for local explanation
    data_file.seek(0)
    df_full = pd.read_csv(data_file)
    max_idx = max(0, len(df_full) - 1)

    local_index = st.sidebar.number_input(
        "Select Row Index for Local Explanation",
        min_value=0,
        max_value=max_idx,
        value=0,
        step=1
    )

    # 4. Run Analysis Button
    run_analysis = st.sidebar.button("Run Analysis", type="primary")

    if run_analysis:
        with st.spinner("Running Analysis Pipeline..."):
            try:
                # Save to temp files
                temp_model_path = save_uploaded_file(model_file)
                temp_data_path = save_uploaded_file(data_file)

                # Load model and dataset
                model = load_model(temp_model_path)
                df = load_dataset(temp_data_path)

                # Validate & Prepare features
                X, y = prepare_features(df, target_col)
                validate_feature_count(model, X)

                # Compute metrics
                metrics = evaluate_model(model, X, y)

                # Save session
                save_session(
                    model_name=model_file.name,
                    dataset_name=data_file.name,
                    metrics=metrics
                )

                st.success("Analysis Complete!")

                # ===================== MODEL SUMMARY =====================
                st.header("Model Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Model Type", type(model).__name__)
                col2.metric("Dataset Rows", df.shape[0])
                col3.metric("Dataset Columns", df.shape[1])
                st.write(f"**Target Column:** {target_col}")

                st.divider()

                # ===================== DATASET PREVIEW =====================
                st.subheader("Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)

                st.divider()

                # ===================== MODEL METRICS =====================
                st.header("Model Metrics")
                metric_cols = st.columns(len(metrics))

                for idx, (m_name, m_val) in enumerate(metrics.items()):
                    if m_name == 'confusion_matrix':
                        metric_cols[idx].metric(m_name, "See Table")
                    elif isinstance(m_val, (int, float)):
                        metric_cols[idx].metric(m_name, f"{m_val:.4f}")
                    else:
                        metric_cols[idx].metric(m_name, str(m_val))

                if 'confusion_matrix' in metrics:
                    st.write("**Confusion Matrix:**")
                    st.dataframe(metrics['confusion_matrix'])

                st.divider()

                # ===================== SHAP COMPUTATION =====================
                st.header("Feature Importance")
                st.write("*Computing SHAP values...*")

                # Sample for performance
                sample_size = min(1000, len(X))
                X_sample = X.sample(n=sample_size, random_state=42)

                shap_values, expected_value = compute_shap_values(model, X_sample)

                # Get feature importance
                feature_names = X.columns.tolist()
                feature_importance = get_feature_importance(shap_values, feature_names)

                # Display bar chart
                importance_df = pd.DataFrame({
                    "Feature": list(feature_importance.keys()),
                    "Importance": list(feature_importance.values())
                }).set_index("Feature")
                st.bar_chart(importance_df)

                st.divider()

                # ===================== SHAP SUMMARY PLOT =====================
                st.header("SHAP Summary Plot")

                fig_summary = plt.figure(figsize=(10, 6))
                # Convert 3D array to 2D for summary_plot
                if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    shap_vals_plot = shap_values[:, :, 1]  # Use class 1 for binary
                elif isinstance(shap_values, list):
                    shap_vals_plot = shap_values[1]  # Use class 1 for binary
                else:
                    shap_vals_plot = shap_values
                shap.summary_plot(shap_vals_plot, X_sample, show=False)
                st.pyplot(fig_summary)

                st.divider()

                # ===================== LOCAL EXPLANATION =====================
                st.header(f"Explain Individual Prediction (Row {local_index})")

                # Compute full dataset SHAP for local explanation
                shap_values_full, _ = compute_shap_values(model, X)
                prediction_explanation = explain_prediction(model, X, shap_values_full, local_index)

                st.write(f"**Feature contributions for Row {local_index}:**")
                explanation_df = pd.DataFrame({
                    "Feature": list(prediction_explanation.keys()),
                    "SHAP Contribution": list(prediction_explanation.values())
                })
                st.dataframe(explanation_df, use_container_width=True)

                # Plot local contributions
                fig_local, ax_local = plt.subplots(figsize=(8, 5))
                top_features = explanation_df.head(10)
                colors = ['crimson' if val > 0 else 'dodgerblue' for val in top_features['SHAP Contribution']]
                ax_local.barh(
                    top_features['Feature'][::-1],
                    top_features['SHAP Contribution'][::-1],
                    color=colors[::-1]
                )
                ax_local.set_xlabel("SHAP Value")
                ax_local.set_title(f"Top 10 Feature Contributions for Row {local_index}")
                st.pyplot(fig_local)

                st.divider()

                # ===================== AI INSIGHTS (GEMINI) =====================
                st.header("AI Model Explanation")
                st.write("*Generating AI-powered explanation using Google Gemini...*")

                ai_explanation = generate_ai_insights(feature_importance, api_key=gemini_api_key)
                st.write(ai_explanation)

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")

            finally:
                # Cleanup temp files
                if 'temp_model_path' in locals() and os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
                if 'temp_data_path' in locals() and os.path.exists(temp_data_path):
                    os.remove(temp_data_path)

else:
    st.info("👈 Please upload both a trained ML model and a tabular CSV dataset via the sidebar to begin.")

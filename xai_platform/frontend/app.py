"""
app.py

Purpose:
Streamlit frontend application for the XAI Model Explanation Platform.
Provides the UI to upload models, datasets, and view SHAP explanations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import os
import sys
import shap

# Include the project project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import backend modules
from backend.model_loader import load_model
from backend.data_handler import load_dataset, prepare_features, validate_feature_count
from backend.metrics_engine import evaluate_model
from backend.explanation_engine import _get_explainer, generate_global_explanation, generate_local_explanation
from backend.session_manager import save_session

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

st.sidebar.header("Configuration")

# 1. Upload Model
st.sidebar.subheader("1. Upload Model")
model_file = st.sidebar.file_uploader("Upload Trained Model (.pkl, .joblib)", type=["pkl", "joblib"])

# 2. Upload Dataset
st.sidebar.subheader("2. Upload Dataset")
data_file = st.sidebar.file_uploader("Upload Tabular Dataset (.csv)", type=["csv"])

if model_file and data_file:
    # 3. Select Target Column
    df_preview = pd.read_csv(data_file, nrows=5)
    st.sidebar.subheader("3. Settings")
    
    target_col = st.sidebar.selectbox("Select Target Column", options=df_preview.columns)
    
    # Needs to know max row index
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
    run_analysis = st.sidebar.button("4. Run Analysis", type="primary")

    if run_analysis:
        with st.spinner("Running Analysis Pipeline..."):
            try:
                # Save models to temp files
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
                
                # Display success
                st.success("Analysis Complete!")
                
                # -------------- Model Summary --------------
                st.header("Model Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Model Type", type(model).__name__)
                col2.metric("Dataset Rows", df.shape[0])
                col3.metric("Dataset Columns", df.shape[1])
                st.write(f"**Target Column:** {target_col}")
                
                st.divider()

                # -------------- Model Metrics --------------
                st.header("Model Metrics")
                metric_cols = st.columns(len(metrics))
                
                # Filter out confusion_matrix for separate table display 
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

                # -------------- Feature Importance --------------
                st.header("Feature Importance (SHAP summary plot)")
                
                # We fetch the explainer internally from the explanation_engine module
                # SHAP works fastest on smaller subsamples for computing summaries
                sample_size = min(1000, len(X))
                X_sample = X.sample(n=sample_size, random_state=42)
                
                st.write(f"*Generating global explanations based on {sample_size} sample background rows...*")
                
                explainer = _get_explainer(model, X_sample)
                shap_values_raw = explainer.shap_values(X_sample)
                
                fig_summary = plt.figure(figsize=(10, 6))
                if isinstance(shap_values_raw, list):
                    shap.summary_plot(shap_values_raw, X_sample, plot_type="bar", show=False)
                else:
                    shap.summary_plot(shap_values_raw, X_sample, show=False)
                st.pyplot(fig_summary)
                
                st.divider()

                # -------------- Local Explanation --------------
                st.header(f"Local Explanation (Row {local_index})")
                
                st.write(f"*Generating local explanations for the specific instance.*")
                local_exp = generate_local_explanation(model, X, local_index)
                
                base_val = local_exp['base_value']
                
                if isinstance(base_val, list):
                    base_val_str = ', '.join([f"{v:.4f}" for v in base_val])
                    st.write(f"**Base Value (Expected Pct/Val):** {base_val_str}")
                else:
                    st.write(f"**Base Value (Expected Pct/Val):** {base_val:.4f}" if isinstance(base_val, float) else f"**Base Value:** {base_val}")
                
                local_shap_vals = local_exp['shap_values']
                feature_vals = local_exp['feature_values']
                
                # Handle multiclass shap values by defaulting to class 1
                if isinstance(local_shap_vals[0], list):
                    st.warning("Multiclass prediction detected. Displaying local SHAP values for first class.")
                    local_shap_vals = [sv[0] for sv in local_shap_vals]
                
                # Create detailed breakdown
                feature_names_list = list(feature_vals.keys())
                shap_df = pd.DataFrame({
                    "Feature": feature_names_list,
                    "Value": [feature_vals[f] for f in feature_names_list],
                    "SHAP Value": local_shap_vals
                })
                shap_df['Abs_SHAP'] = shap_df['SHAP Value'].abs()
                shap_df = shap_df.sort_values(by="Abs_SHAP", ascending=False).drop(columns=['Abs_SHAP'])
                
                # Plot bar chart showing direction of impact 
                fig_local, ax_local = plt.subplots(figsize=(8, 5))
                top_shap_features = shap_df.head(10)
                
                colors = ['crimson' if val > 0 else 'dodgerblue' for val in top_shap_features['SHAP Value']]
                ax_local.barh(top_shap_features['Feature'][::-1], top_shap_features['SHAP Value'][::-1], color=colors[::-1])
                ax_local.set_xlabel("SHAP Value")
                ax_local.set_title(f"Top 10 Feature Contributions for Row {local_index}")
                
                st.write("**Top 10 Feature Contributions:**")
                st.pyplot(fig_local)
                
                st.write("**Detailed SHAP Feature Breakdown:**")
                st.dataframe(shap_df, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
            
            finally:
                # Cleanup temp files
                if 'temp_model_path' in locals() and os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
                if 'temp_data_path' in locals() and os.path.exists(temp_data_path):
                    os.remove(temp_data_path)

else:
    st.info("👈 Please upload both a compiled ML model and a tabular CSV dataset via the sidebar to begin.")

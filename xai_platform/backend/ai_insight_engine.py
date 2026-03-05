"""
ai_insight_engine.py

Purpose:
Use Google Gemini to interpret feature importance and provide
natural language explanations of model behavior.
"""
import google.generativeai as genai
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_ai_insights(feature_importance, api_key=None):
    """
    Uses Gemini to generate a natural language explanation of model behavior
    based on feature importance values.

    Args:
        feature_importance (dict): Dictionary of feature names to importance values.
        api_key (str, optional): Google Gemini API key.

    Returns:
        str: AI-generated explanation text.
    """
    # Configure API key
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return "⚠️ No Gemini API key provided. Set the GOOGLE_API_KEY environment variable or enter it in the sidebar."

    genai.configure(api_key=key)

    # Initialize model
    model = genai.GenerativeModel("gemini-flash-latest")

    # Build prompt
    prompt = f"""Explain the behavior of this machine learning model based on these feature importances:
{feature_importance}

Explain which features influence predictions most and why. 
Provide a clear, concise explanation suitable for a non-technical audience.
Use bullet points for each key feature and its impact."""

    logging.info("Sending feature importance to Gemini for AI insights...")

    try:
        response = model.generate_content(prompt)
        explanation = response.text
        logging.info("AI insights generated successfully.")
        return explanation
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return f"⚠️ Error generating AI insights: {str(e)}"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import pandas as pd
import json

from backend.model_loader import load_model
from backend.data_handler import load_dataset, prepare_features, validate_feature_count
from backend.metrics_engine import evaluate_model
from backend.explanation_engine import generate_global_explanation, generate_local_explanation

app = FastAPI(title="XAI Platform API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory state for a single-user application
app_state = {
    "model_path": None,
    "data_path": None,
    "target_col": None,
}

@app.post("/api/upload")
async def upload_files(
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...),
    target_col: str = Form(...)
):
    try:
        # Save model
        model_suffix = os.path.splitext(model_file.filename)[1]
        temp_model = tempfile.NamedTemporaryFile(delete=False, suffix=model_suffix)
        content = await model_file.read()
        temp_model.write(content)
        temp_model.close()
        
        # Save dataset
        data_suffix = os.path.splitext(data_file.filename)[1]
        temp_data = tempfile.NamedTemporaryFile(delete=False, suffix=data_suffix)
        content = await data_file.read()
        temp_data.write(content)
        temp_data.close()

        app_state["model_path"] = temp_model.name
        app_state["data_path"] = temp_data.name
        app_state["target_col"] = target_col

        # Load data to get basic info
        df = load_dataset(temp_data.name)
        
        return {
            "message": "Files uploaded successfully",
            "columns": list(df.columns),
            "num_rows": len(df),
            "num_cols": len(df.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/explain/global")
def get_global_explanation():
    if not app_state["model_path"] or not app_state["data_path"] or not app_state["target_col"]:
        raise HTTPException(status_code=400, detail="Please upload files first.")
        
    try:
        model = load_model(app_state["model_path"])
        df = load_dataset(app_state["data_path"])
        X, y = prepare_features(df, app_state["target_col"])
        validate_feature_count(model, X)
        
        metrics = evaluate_model(model, X, y)
        
        # Replace NaN/Infinity in metrics for JSON serialization
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, pd.DataFrame):
                clean_metrics[k] = v.to_dict(orient="records")
            else:
                clean_metrics[k] = v
                
        # Generate global explanation
        global_exp = generate_global_explanation(model, X)
        
        return {
            "metrics": clean_metrics,
            "feature_importance": global_exp["feature_importance"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/explain/local/{index}")
def get_local_explanation(index: int):
    if not app_state["model_path"] or not app_state["data_path"] or not app_state["target_col"]:
        raise HTTPException(status_code=400, detail="Please upload files first.")
        
    try:
        model = load_model(app_state["model_path"])
        df = load_dataset(app_state["data_path"])
        X, y = prepare_features(df, app_state["target_col"])
        
        if index < 0 or index >= len(X):
            raise HTTPException(status_code=400, detail=f"Index must be between 0 and {len(X)-1}")
            
        local_exp = generate_local_explanation(model, X, index)
        
        return local_exp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

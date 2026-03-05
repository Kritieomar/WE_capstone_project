from backend.model_loader import load_model
from backend.data_handler import load_dataset, prepare_features, validate_feature_count
from backend.metrics_engine import evaluate_model
from backend.explanation_engine import generate_global_explanation
import json

try:
    model_path = "C:/Users/kriti/OneDrive/Desktop/capstone/WE_capstone_project/xai_platform/test_model.joblib"
    data_path = "C:/Users/kriti/OneDrive/Desktop/capstone/WE_capstone_project/xai_platform/test_dataset.csv"
    target_col = "target"
    
    model = load_model(model_path)
    df = load_dataset(data_path)
    X, y = prepare_features(df, target_col)
    validate_feature_count(model, X)
    
    metrics = evaluate_model(model, X, y)
    
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, type(df)):
            clean_metrics[k] = v.to_dict(orient="records")
        else:
            clean_metrics[k] = v
            
    global_exp = generate_global_explanation(model, X)
    
    result = {
        "metrics": clean_metrics,
        "feature_importance": global_exp["feature_importance"]
    }
    
    print("Trying to json dump:")
    print(json.dumps(result))
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
